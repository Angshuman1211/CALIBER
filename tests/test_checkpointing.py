"""A run that dies keeps what it had already done (P-24).

The runner used to hold every row in memory and write the bank once, at the end.
The Week 3 bank took 7.2 hours; P-24 then showed that at 256² this machine can
kill the solver with a silent segfault whenever other applications leave too
little memory. Together that meant a crash at hour seven lost seven hours, with
no traceback and no partial output — a log that simply stopped.

These tests pin three properties:

1. a run that fails part-way leaves its completed rows on disk;
2. a partial bank can never be mistaken for a finished one — `read_bank` refuses
   it unless the caller asks for it explicitly;
3. a run that finishes leaves no partial files behind, and banks written before
   any of this existed still read as complete.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from caliber.config import load_config
from caliber.runner import run
from caliber.scorebank import read_bank, read_manifest
from caliber.scorebank.io import PARTIAL_MANIFEST, PARTIAL_SCORES, ScoreBankWriter
from caliber.scorebank.schema import SchemaError

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE = REPO_ROOT / "configs/experiments/w01_smoke_phantom.yaml"


def _cfg(out: Path, *extra: str) -> dict:
    return load_config(
        str(SMOKE),
        [
            "run.limit=3",
            "experiment.n_posterior=2",
            "run.cache_reconstructions=0",
            # See test_analyse_bank: logic under test, not the machine's memory (D-041).
            "run.skip_memory_check=true",
            f"run.output_root={out.as_posix()}",
            *extra,
        ],
    )


# -- a completed run -----------------------------------------------------------


def test_a_finished_run_is_marked_complete_and_leaves_no_partial_files(tmp_path):
    bank = run(_cfg(tmp_path, "run.checkpoint_every=1"))
    assert (bank / "scores.parquet").is_file()
    assert not (bank / PARTIAL_SCORES).exists()
    assert not (bank / PARTIAL_MANIFEST).exists()
    assert read_manifest(bank)["complete"] is True


def test_checkpointing_does_not_change_the_bank(tmp_path):
    """Writing checkpoints along the way must not alter a single row."""
    plain = read_bank(run(_cfg(tmp_path / "plain")))
    checkpointed = read_bank(run(_cfg(tmp_path / "ckpt", "run.checkpoint_every=1")))
    key = ["sample_id", "operator_id", "region_id"]
    # run_id and config_hash legitimately differ: checkpoint_every is part of the config.
    ignore = ["run_id", "config_hash"]
    pd.testing.assert_frame_equal(
        plain.drop(columns=ignore).sort_values(key).reset_index(drop=True),
        checkpointed.drop(columns=ignore).sort_values(key).reset_index(drop=True),
        check_like=True,
    )


# -- a run that dies part-way --------------------------------------------------


def test_a_run_that_fails_part_way_keeps_its_completed_rows(tmp_path, monkeypatch):
    """The whole point: a crash at item N keeps items 1..N-1."""
    from caliber.solvers.reference import NullSpaceNoiseSampler

    real_sample = NullSpaceNoiseSampler.sample
    calls = {"n": 0}

    def dies_on_the_fourth_item(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 4:
            raise RuntimeError("simulated crash mid-run")
        return real_sample(self, *args, **kwargs)

    monkeypatch.setattr(NullSpaceNoiseSampler, "sample", dies_on_the_fourth_item)

    cfg = _cfg(tmp_path, "run.checkpoint_every=1")
    with pytest.raises(RuntimeError, match="simulated crash"):
        run(cfg)

    bank = next(tmp_path.iterdir())
    assert not (bank / "scores.parquet").exists(), "an interrupted run must not look finished"
    assert (bank / PARTIAL_SCORES).is_file()

    partial = read_bank(bank, allow_partial=True)
    n_regions = partial["region_id"].nunique()
    assert len(partial) == 3 * n_regions  # three (item, operator) pairs finished
    manifest = json.loads((bank / PARTIAL_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert manifest["n_rows"] == len(partial)


def test_a_partial_bank_is_refused_unless_asked_for(tmp_path):
    writer = ScoreBankWriter(tmp_path / "bank", run_id="r")
    writer.add_item(
        base={"dataset": "d", "split": "calib", "sample_id": "s", "slice_idx": 0,
              "operator_id": "o", "modality": "mri", "operator_family": "f",
              "calib_group": "g", "solver_id": "x", "solver_family": "reference",
              "solver_checkpoint": None, "n_posterior": 1, "seed": 0,
              "score_version": "v", "criterion_version": "v", "config_hash": "h",
              "ref_intensity": 1.0},
        region_rows=[{"region_id": 0, "region_row": 0, "region_col": 0, "region_area": 256}],
        arrays={k: [0.1] for k in ("s_free", "s_disagree", "s_residual", "err_total",
                                    "err_free", "free_content", "ann_coverage")},
    )
    writer.checkpoint()

    with pytest.raises(SchemaError, match="incomplete"):
        read_bank(tmp_path / "bank")
    assert len(read_bank(tmp_path / "bank", allow_partial=True)) == 1


def test_checkpoint_every_zero_writes_nothing_until_the_end(tmp_path, monkeypatch):
    """Off by default costs nothing and changes nothing."""
    seen = []
    real = ScoreBankWriter.checkpoint
    monkeypatch.setattr(ScoreBankWriter, "checkpoint",
                        lambda self, *a, **k: (seen.append(1), real(self, *a, **k))[1])
    run(_cfg(tmp_path))
    assert seen == []


# -- banks from before this existed --------------------------------------------


def test_a_bank_without_a_complete_key_still_reads_as_complete(tmp_path):
    """Every bank written before P-24 lacks the key; none of them may break."""
    bank = run(_cfg(tmp_path))
    manifest_path = bank / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("complete")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert len(read_bank(bank)) > 0


# -- resume (D-040) ------------------------------------------------------------


def test_a_resumed_run_produces_exactly_the_uninterrupted_bank(tmp_path, monkeypatch):
    """Crash, restart the same config, and get the bank a clean run would have made.

    Only possible because draws no longer depend on the process (D-039): the pairs
    computed after the restart come out identical to the ones a single run computes.
    """
    from caliber.solvers.reference import NullSpaceNoiseSampler

    real_sample = NullSpaceNoiseSampler.sample
    calls = {"n": 0, "crash_at": 4}

    def sample(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == calls["crash_at"]:
            raise RuntimeError("simulated crash mid-run")
        return real_sample(self, *args, **kwargs)

    monkeypatch.setattr(NullSpaceNoiseSampler, "sample", sample)

    cfg = _cfg(tmp_path / "resumed", "run.checkpoint_every=1")
    with pytest.raises(RuntimeError, match="simulated crash"):
        run(cfg)
    recovered = len(read_bank(next((tmp_path / "resumed").iterdir()), allow_partial=True))

    calls.update(n=0, crash_at=-1)
    bank = run(_cfg(tmp_path / "resumed", "run.checkpoint_every=1"))
    total_pairs = len(read_bank(bank)) // read_bank(bank)["region_id"].nunique()
    n_regions = read_bank(bank)["region_id"].nunique()
    assert calls["n"] == total_pairs - recovered // n_regions, "resume must skip finished pairs"
    assert read_manifest(bank)["complete"] is True
    assert not (bank / PARTIAL_SCORES).exists()

    monkeypatch.setattr(NullSpaceNoiseSampler, "sample", real_sample)
    clean = read_bank(run(_cfg(tmp_path / "clean", "run.checkpoint_every=1")))

    key = ["sample_id", "operator_id", "region_id"]
    ignore = ["run_id", "config_hash"]  # output_root differs, and it is part of the config
    pd.testing.assert_frame_equal(
        read_bank(bank).drop(columns=ignore).sort_values(key).reset_index(drop=True),
        clean.drop(columns=ignore).sort_values(key).reset_index(drop=True),
    )


def test_a_checkpoint_from_a_different_config_is_refused(tmp_path):
    """Two runs' rows in one bank look exactly like one run's — so never merge them."""
    root = tmp_path / "bank"
    first = ScoreBankWriter(root, run_id="r", config={"_meta": {"config_hash": "aaaa"}})
    first._rows.append({"sample_id": "s", "slice_idx": 0, "operator_id": "o"})
    pd.DataFrame(first._rows).to_parquet(root / PARTIAL_SCORES, index=False)
    (root / PARTIAL_MANIFEST).write_text(
        json.dumps({"config": {"_meta": {"config_hash": "aaaa"}}}), encoding="utf-8"
    )

    other = ScoreBankWriter(root, run_id="r", config={"_meta": {"config_hash": "bbbb"}})
    with pytest.raises(SchemaError, match="Refusing to mix two runs"):
        other.resume_from_checkpoint()

    same = ScoreBankWriter(root, run_id="r", config={"_meta": {"config_hash": "aaaa"}})
    assert same.resume_from_checkpoint() == {("s", 0, "o")}


def test_no_checkpoint_means_nothing_to_resume(tmp_path):
    writer = ScoreBankWriter(tmp_path / "bank", run_id="r", config={"_meta": {"config_hash": "x"}})
    assert writer.resume_from_checkpoint() == set()
