"""Reading and writing score banks.

A score bank on disk is a directory:

    <bank>/
      manifest.json      run provenance, environment, operator registry snapshot
      scores.parquet     the long-format table defined in schema.py
      recons/            optional float16 posterior samples, for figures only

While a run is in progress it may also hold ``scores.partial.parquet`` and
``manifest.partial.json`` — every row finished so far, rewritten every
``run.checkpoint_every`` items (P-24). A finished run replaces them with the two
files above and removes them. A directory holding only partial files is an
interrupted run, and :func:`read_bank` refuses it unless asked for explicitly.

The manifest embeds the operator registry so that a bank is self-describing:
Track 3 can compute operator distances and Track 1 can calibrate from a copied
directory, with no access to the machine that produced it.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import schema

PARTIAL_SCORES = "scores.partial.parquet"
PARTIAL_MANIFEST = "manifest.partial.json"


def _atomic_write_bytes(path: Path, write) -> None:
    """Write via a sibling temp file and rename, so a crash never leaves half a file.

    A checkpoint that is itself corrupted by the crash it exists to survive would
    be worse than none: it would look like a result.
    """
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_snapshot() -> dict[str, Any]:
    """Recorded in every manifest so a bank can be reproduced (Sec. 17)."""
    try:
        import torch

        torch_version = torch.__version__
        cuda = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - environment capture must never fail a run
        torch_version, cuda = None, None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch_version,
        "gpu": cuda,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git_sha": _git_sha(),
    }


class ScoreBankWriter:
    """Accumulates rows in memory and flushes one parquet file per bank."""

    def __init__(
        self,
        root: str | Path,
        run_id: str,
        config: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
    ):
        self.root = Path(root)
        self.run_id = run_id
        self.config = config or {}
        # Substrate and back-end provenance, promoted to a top-level manifest
        # key rather than left implicit in the config. A bank produced from
        # phantoms or from the M4Raw stopgap is a valid bank and looks exactly
        # like a reportable one; the distinction has to be recorded where a
        # reader will actually see it.
        self.provenance = provenance or {}
        self._rows: list[dict[str, Any]] = []
        self.root.mkdir(parents=True, exist_ok=True)

    def add_item(
        self,
        base: dict[str, Any],
        region_rows: list[dict[str, Any]],
        arrays: dict[str, Any],
    ) -> None:
        """Append one item's regions.

        ``base`` holds the columns constant across the item (identity, operator,
        solver); ``region_rows`` is the static region geometry; ``arrays`` maps
        per-region column names to length-``n_regions`` sequences.
        """
        n = len(region_rows)
        for name, values in arrays.items():
            if len(values) != n:
                raise ValueError(f"array {name!r} has length {len(values)}, expected {n}")
        for i, region in enumerate(region_rows):
            row = {**base, **region}
            for name, values in arrays.items():
                row[name] = float(values[i])
            self._rows.append(row)

    def __len__(self) -> int:
        return len(self._rows)

    def to_frame(self) -> pd.DataFrame:
        if not self._rows:
            return schema.empty_frame()
        df = pd.DataFrame(self._rows)
        df["schema_version"] = schema.SCHEMA_VERSION
        df["run_id"] = self.run_id
        return schema.coerce(df)

    def _manifest(self, df: pd.DataFrame, registry_descriptors, problems, complete: bool):
        return {
            "schema_version": schema.SCHEMA_VERSION,
            "run_id": self.run_id,
            # Absent from every bank written before P-24; readers treat a missing
            # key as complete, so none of those banks changes meaning.
            "complete": complete,
            "n_rows": int(len(df)),
            "n_items": int(
                df[["dataset", "sample_id", "slice_idx", "operator_id"]].drop_duplicates().shape[0]
            )
            if len(df)
            else 0,
            "operators": registry_descriptors or [],
            "provenance": self.provenance,
            "config": self.config,
            "environment": environment_snapshot(),
            "validation_problems": problems,
        }

    def checkpoint(self, registry_descriptors: list[dict[str, Any]] | None = None) -> Path:
        """Write every row so far as a *partial* bank; returns the bank root (P-24).

        Rewrites the whole partial table each time rather than appending. At the
        sizes this project produces (tens of thousands of rows) that costs well
        under a second, and it means the partial file is always a complete,
        valid table on its own rather than a sequence of fragments to reassemble.
        """
        df = self.to_frame()
        _atomic_write_bytes(
            self.root / PARTIAL_SCORES, lambda tmp: df.to_parquet(tmp, index=False)
        )
        manifest = self._manifest(df, registry_descriptors, [], complete=False)
        _atomic_write_bytes(
            self.root / PARTIAL_MANIFEST,
            lambda tmp: tmp.write_text(json.dumps(manifest, indent=2, default=str),
                                       encoding="utf-8"),
        )
        return self.root

    def flush(self, registry_descriptors: list[dict[str, Any]] | None = None) -> Path:
        """Write ``scores.parquet`` and ``manifest.json``; returns the bank root."""
        df = self.to_frame()
        problems = schema.validate(df, strict=False)
        _atomic_write_bytes(
            self.root / "scores.parquet", lambda tmp: df.to_parquet(tmp, index=False)
        )
        manifest = self._manifest(df, registry_descriptors, problems, complete=True)
        _atomic_write_bytes(
            self.root / "manifest.json",
            lambda tmp: tmp.write_text(json.dumps(manifest, indent=2, default=str),
                                       encoding="utf-8"),
        )
        # Only once the finished bank is safely on disk: removing the checkpoint
        # first would open a window in which a crash loses both.
        for name in (PARTIAL_SCORES, PARTIAL_MANIFEST):
            (self.root / name).unlink(missing_ok=True)
        return self.root


def read_bank(
    root: str | Path, validate: bool = True, allow_partial: bool = False
) -> pd.DataFrame:
    """Load a score bank; CPU-only, and the entry point for all downstream work.

    A directory holding only a checkpoint is an interrupted run (P-24). It is
    refused by default, because a partial bank is a perfectly valid table and
    nothing about its rows says the run never finished — exactly the kind of
    silent difference D-018 exists to make visible. Pass ``allow_partial=True``
    to recover the rows deliberately.
    """
    root = Path(root)
    path = root / "scores.parquet" if root.is_dir() else root
    if root.is_dir() and not path.is_file() and (root / PARTIAL_SCORES).is_file():
        if not allow_partial:
            raise schema.SchemaError(
                f"{root} holds only a checkpoint: the run that produced it is incomplete "
                "(it was interrupted, or is still running). Pass allow_partial=True to "
                "read the rows it finished."
            )
        path = root / PARTIAL_SCORES
    if not path.is_file():
        raise FileNotFoundError(f"no score bank at {path}")
    df = pd.read_parquet(path)
    if validate:
        schema.validate(df, strict=False)
    return df


def read_manifest(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"no manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def merge_banks(roots: list[str | Path], validate: bool = True) -> pd.DataFrame:
    """Concatenate several banks — e.g. the MRI grid and the CT grid for Week 9."""
    frames = [read_bank(r, validate=False) for r in roots]
    df = pd.concat(frames, ignore_index=True)
    if validate:
        schema.validate(df, strict=False)
    return df


def save_reconstructions(
    root: str | Path,
    key: str,
    samples: Any,
    ground_truth: Any = None,
) -> Path:
    """Cache posterior draws as float16 for figures and re-scoring.

    Optional and off by default: a full grid of float32 posterior samples is
    orders of magnitude larger than the score table and would defeat the point
    of caching. Enable it for a small, fixed subset of items.
    """
    root = Path(root) / "recons"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{key}.npz"
    payload = {"samples": np.asarray(samples, dtype=np.float16)}
    if ground_truth is not None:
        payload["ground_truth"] = np.asarray(ground_truth, dtype=np.float16)
    np.savez_compressed(path, **payload)
    return path
