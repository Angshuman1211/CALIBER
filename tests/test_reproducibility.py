"""A bank is determined by its config — across processes, not only within one (P-25).

Item seeds used to come from ``abs(hash((sample_id, slice_idx, operator_id)))``.
Python salts string hashing per process unless ``PYTHONHASHSEED`` is set, and nothing
set it, so every run drew different posterior samples for the same item. Every test of
determinism in the suite still passed, because each one compared two runs inside a
single pytest process — where the salt is shared and the bug cannot show.

So the central test here runs the pipeline in **separate processes with deliberately
different hash seeds** and requires the banks to be identical. That is the only form of
the check that can fail for this bug.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from caliber.runner import item_seed
from caliber.scorebank import read_bank

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE = REPO_ROOT / "configs/experiments/w01_smoke_phantom.yaml"


def test_item_seed_is_pinned():
    """The derivation itself, frozen. Changing it changes every bank's draws (D-041)."""
    assert item_seed(0, "file1000022", 18, "mri.random1d.R4.c0.08.s0") == 777145
    assert item_seed(7, "file1000022", 18, "mri.random1d.R8.c0.08.s0") == 898317


def test_item_seed_separates_items_operators_and_the_base_seed():
    base = item_seed(0, "a", 0, "op")
    assert item_seed(0, "b", 0, "op") != base
    assert item_seed(0, "a", 1, "op") != base
    assert item_seed(0, "a", 0, "op2") != base
    assert item_seed(1, "a", 0, "op") == base + 1


def test_item_seed_does_not_confuse_field_boundaries():
    """("ab", 1) and ("a", "b1")-style collisions are what a naive concatenation gives."""
    assert item_seed(0, "a1", 0, "op") != item_seed(0, "a", 10, "op")


def _run_in_subprocess(out: Path, hash_seed: str) -> Path:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    code = (
        "from caliber.config import load_config; from caliber.runner import run; "
        f"cfg = load_config(r'{SMOKE}', ['run.limit=2', 'experiment.n_posterior=2', "
        f"'run.cache_reconstructions=0', 'run.skip_memory_check=true', "
        f"'run.output_root={out.as_posix()}']); "
        "print(run(cfg))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return Path(result.stdout.strip().splitlines()[-1])


def test_two_processes_with_different_hash_seeds_produce_the_same_bank(tmp_path):
    """The regression. Fails on the old derivation; passes only if a bank is truly
    a function of its config."""
    first = read_bank(_run_in_subprocess(tmp_path / "a", "1"))
    second = read_bank(_run_in_subprocess(tmp_path / "b", "987654"))

    key = ["sample_id", "operator_id", "region_id"]
    # The output directory is part of the config, so run_id and config_hash
    # legitimately differ between the two runs. Every data column must not.
    ignore = ["run_id", "config_hash"]
    a = first.drop(columns=ignore).sort_values(key).reset_index(drop=True)
    b = second.drop(columns=ignore).sort_values(key).reset_index(drop=True)
    assert len(a) > 0
    pd.testing.assert_frame_equal(a, b)
