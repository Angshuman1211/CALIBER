"""The memory preflight (P-24, D-041).

A run whose allocations do not fit can segfault instead of raising, so the runner refuses
to start one. That makes the estimate load-bearing in both directions: too low and it
lets a doomed run start, too high and it blocks a run that would have worked.

The estimate is a fit to measurements taken on this machine, so the first test checks it
against those measurements — including the two configurations that actually failed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from caliber.config import ConfigError, load_config
from caliber.resources import (
    available_commit_gb,
    estimate_peak_commit_gb,
    memory_preflight,
)
from caliber.runner import run

REPO_ROOT = Path(__file__).resolve().parents[1]

# (image size, draws, measured peak commit GB) on ddpm-celebahq-256, torch 2.13.0+cpu.
MEASURED = [
    (64, 1, 2.75), (96, 1, 2.79), (128, 1, 3.00), (160, 1, 3.22),
    (192, 1, 4.20), (256, 1, 5.99), (96, 4, 4.14), (128, 2, 4.09), (128, 4, 5.98),
]


@pytest.mark.parametrize(("size", "draws", "measured"), MEASURED)
def test_estimate_matches_what_was_measured(size, draws, measured):
    """Within 0.4 GB of every point it was fitted to."""
    assert estimate_peak_commit_gb((size, size), draws) == pytest.approx(measured, abs=0.4)


def test_estimate_predicts_the_configurations_that_actually_failed():
    """128²×8 segfaulted with 7.6 GB free; 256²×4 raised out-of-memory with ~5 GB free."""
    assert estimate_peak_commit_gb((128, 128), 8) > 7.6
    assert estimate_peak_commit_gb((256, 256), 4) > 5.0


def test_estimate_grows_with_pixels_and_with_draws():
    base = estimate_peak_commit_gb((128, 128), 1)
    assert estimate_peak_commit_gb((256, 256), 1) > base
    assert estimate_peak_commit_gb((128, 128), 4) > base


def test_a_backward_pass_costs_more_than_a_forward_one():
    """DPS differentiates through the UNet; DDNM does not."""
    shape = (128, 128)
    assert estimate_peak_commit_gb(shape, 2, backward=True) > estimate_peak_commit_gb(shape, 2)


def test_availability_is_either_a_positive_number_or_unknown():
    available = available_commit_gb()
    assert available is None or available > 0


# -- verdicts -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("needed", "available", "expected"),
    [(1.0, 100.0, "ok"), (9.0, 10.0, "tight"), (11.0, 10.0, "too_big")],
)
def test_verdicts(monkeypatch, needed, available, expected):
    monkeypatch.setattr("caliber.resources.available_commit_gb", lambda: available)
    monkeypatch.setattr("caliber.resources.estimate_peak_commit_gb", lambda *a, **k: needed)
    assert memory_preflight((128, 128), 1)["verdict"] == expected


def test_unknown_availability_never_blocks(monkeypatch):
    """A check that cannot read the memory must not stop a run."""
    monkeypatch.setattr("caliber.resources.available_commit_gb", lambda: None)
    report = memory_preflight((256, 256), 64)
    assert report["verdict"] == "unknown"
    assert report["available_gb"] is None


# -- the runner acts on it ------------------------------------------------------


def _cfg(tmp_path, *extra):
    return load_config(
        str(REPO_ROOT / "configs/experiments/w01_smoke_phantom.yaml"),
        ["run.limit=1", "experiment.n_posterior=1", "run.cache_reconstructions=0",
         f"run.output_root={tmp_path.as_posix()}", *extra],
    )


def test_the_runner_refuses_a_run_that_cannot_fit(tmp_path, monkeypatch):
    monkeypatch.setattr("caliber.runner.memory_preflight",
                        lambda *a, **k: {"needed_gb": 99.0, "available_gb": 1.0,
                                         "verdict": "too_big"})
    with pytest.raises(ConfigError, match="committed memory"):
        run(_cfg(tmp_path))


def test_the_refusal_can_be_overridden(tmp_path, monkeypatch):
    """The estimate is a fit to one machine. Being wrong must cost a flag, not a week."""
    monkeypatch.setattr("caliber.runner.memory_preflight",
                        lambda *a, **k: {"needed_gb": 99.0, "available_gb": 1.0,
                                         "verdict": "too_big"})
    bank = run(_cfg(tmp_path, "run.skip_memory_check=true"))
    assert (bank / "scores.parquet").is_file()


def test_a_tight_run_warns_but_proceeds(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("caliber.runner.memory_preflight",
                        lambda *a, **k: {"needed_gb": 9.0, "available_gb": 10.0,
                                         "verdict": "tight"})
    bank = run(_cfg(tmp_path))
    assert (bank / "scores.parquet").is_file()
    assert "within 20% of available memory" in capsys.readouterr().err
