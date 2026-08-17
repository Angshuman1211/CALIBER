"""Score-bank schema v0 — the project's single shared data interface.

Blueprint Sec. 17 and Sec. 19: reconstructions and scores are computed once on
a GPU and cached; every subsequent calibration experiment, coverage curve,
ablation, and numerical check reads these cached values and needs no GPU at
all. All three tracks consume this one format.

Three rules shape the schema, and each exists to prevent re-running GPU work:

1. **Store components, not conclusions.** The three score components are
   separate columns, never a pre-combined total, so Track 1 can re-weight the
   score (T1.1) from cache.
2. **Store continuous quantities, not labels.** Region error magnitudes and
   annotation coverage are stored raw, so the ground-truth criterion (O2) can
   be re-thresholded from cache.
3. **Store provenance on every row.** Operator, solver, checkpoint, seed, and
   config hash travel with the data, so a figure can always be traced back to
   the run that produced it.

Long format: one row per (item, operator, solver, region). Bumping
``SCHEMA_VERSION`` is a cross-track decision and must be recorded in
``docs/decision_log.md``.
"""

from __future__ import annotations

import pandas as pd

SCHEMA_VERSION = "0.1.0"

# column name -> pandas dtype
COLUMNS: dict[str, str] = {
    # -- provenance ---------------------------------------------------------
    "schema_version": "string",
    "run_id": "string",
    "config_hash": "string",
    # -- item identity ------------------------------------------------------
    "dataset": "string",
    "split": "string",
    "sample_id": "string",
    "slice_idx": "int32",
    # -- acquisition operator (joins to the operator registry) --------------
    "operator_id": "string",
    "modality": "string",
    "operator_family": "string",
    "calib_group": "string",  # the group for group-conditional calibration
    # -- solver -------------------------------------------------------------
    "solver_id": "string",
    "solver_family": "string",
    "solver_checkpoint": "string",
    "n_posterior": "int16",
    "seed": "int32",
    # -- region -------------------------------------------------------------
    "region_id": "int32",
    "region_row": "int16",
    "region_col": "int16",
    "region_area": "int32",
    # -- score components (rule 1: recombine without a GPU) -----------------
    "score_version": "string",
    "s_free": "float32",
    "s_disagree": "float32",
    "s_residual": "float32",
    # -- ground truth (rule 2: re-threshold without a GPU) ------------------
    "criterion_version": "string",
    "err_total": "float32",
    "err_free": "float32",
    "ann_coverage": "float32",
    "ref_intensity": "float32",
}

# Uniquely identifies a row; duplicates indicate a double-write.
KEY_COLUMNS = (
    "dataset",
    "split",
    "sample_id",
    "slice_idx",
    "operator_id",
    "solver_id",
    "seed",
    "region_id",
)

# Written as parquet partitions so a modality or operator can be read alone.
PARTITION_COLUMNS = ("modality", "operator_id")


class SchemaError(RuntimeError):
    """Raised when a score bank does not conform to the shared format."""


def empty_frame() -> pd.DataFrame:
    """An empty frame with exactly the schema's columns and dtypes."""
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in COLUMNS.items()})


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder to the canonical column order and cast to the declared dtypes."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"score bank is missing required columns: {missing}")
    extra = [c for c in df.columns if c not in COLUMNS]
    out = df[list(COLUMNS) + extra].copy()
    for name, dtype in COLUMNS.items():
        try:
            out[name] = out[name].astype(dtype)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"column {name!r} cannot be cast to {dtype}: {exc}") from exc
    return out


def validate(df: pd.DataFrame, strict: bool = True) -> list[str]:
    """Check a score bank against the shared format. Returns a list of problems."""
    problems: list[str] = []

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
        if strict:
            raise SchemaError(problems[-1])
        return problems

    versions = set(df["schema_version"].dropna().unique())
    if versions - {SCHEMA_VERSION}:
        problems.append(f"mixed schema versions present: {sorted(versions)}")

    dupes = int(df.duplicated(subset=list(KEY_COLUMNS)).sum())
    if dupes:
        problems.append(f"{dupes} duplicate rows for key {KEY_COLUMNS}")

    for col in ("s_free", "s_disagree", "s_residual", "err_total", "err_free"):
        n_bad = int(df[col].isna().sum())
        if n_bad:
            problems.append(f"{n_bad} null values in {col}")

    if (df["n_posterior"] < 1).any():
        problems.append("n_posterior < 1 present")

    # A coverage curve must vary the operator alone (Sec. 13). More than one
    # solver inside a single run is the signature of a violated control.
    per_run_solvers = df.groupby("run_id")["solver_id"].nunique()
    offenders = per_run_solvers[per_run_solvers > 1]
    if len(offenders):
        problems.append(
            f"runs mixing multiple solvers (violates controlled comparison): "
            f"{list(offenders.index)}"
        )

    if strict and problems:
        raise SchemaError("; ".join(problems))
    return problems
