"""The analysis entry point runs, and keeps running (P-19).

`scripts/analyse_bank.py` is where the two Week 3 baselines come from, and it sat
broken for two weeks: `RegionTruth` gained a required `free_content` field when
A1 / D21 landed, and this caller was never updated. Nothing caught it because
nothing ever executed the script — the unit tests exercised the *library* the
script calls, which is not the same thing.

So the test here is deliberately end-to-end and deliberately cheap: build a real
bank on disk with the reference sampler, then run `main()` over it exactly as a
person would. It asserts almost nothing about the numbers, because its job is to
fail when the script stops running at all.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

from caliber.config import load_config
from caliber.runner import run
from caliber.scorebank import read_bank
from caliber.scoring.labels import RegionTruth

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyse_bank  # noqa: E402


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    """A real two-operator bank on disk, from the phantom smoke config."""
    out = tmp_path_factory.mktemp("banks")
    cfg = load_config(
        str(REPO_ROOT / "configs/experiments/w01_smoke_phantom.yaml"),
        [
            "run.limit=4",
            "experiment.n_posterior=4",
            "run.cache_reconstructions=0",
            # The preflight (D-041) guards long runs; these tests exercise logic and
            # must not fail because of whatever else the machine happens to be running.
            "run.skip_memory_check=true",
            f"run.output_root={out.as_posix()}",
        ],
    )
    return run(cfg)


def test_the_bank_carries_every_field_the_criterion_needs(bank):
    """A RegionTruth field with no bank column is the P-19 bug, structurally.

    If someone adds a field to the criterion, this fails until the bank records
    it — which is the check that was missing when `free_content` was added.
    """
    columns = set(read_bank(bank).columns)
    required = {f.name for f in dataclasses.fields(RegionTruth)}
    assert required <= columns, f"criterion needs columns the bank lacks: {required - columns}"


def test_analyse_bank_runs_to_completion(bank, capsys):
    """The regression itself: the script executed, end to end, exit 0."""
    assert analyse_bank.main([str(bank), "--label-threshold", "0.05"]) == 0
    out = capsys.readouterr().out
    assert "coverage by operator" in out
    for method in ("naive_transferred", "matched_per_operator", "group_conditional"):
        assert method in out


def test_analyse_bank_runs_with_the_relative_criterion(bank, capsys):
    """The A1 / D21 form has to work from the script, not only from the library."""
    assert analyse_bank.main([str(bank), "--relative", "--label-threshold", "1.0"]) == 0
    assert "err_free / free_content" in capsys.readouterr().out


def test_the_score_is_standardised_not_a_raw_sum(bank, capsys):
    """D26: an unstandardised 1,1,1 sum silently reduces the score to `s_free`.

    The script must say which combination produced its table, so a number can
    never be quoted without it.
    """
    analyse_bank.main([str(bank), "--label-threshold", "0.05"])
    assert "standardised affine" in capsys.readouterr().out


def test_a_bank_with_no_positives_is_refused_rather_than_divided_by_zero(bank, capsys):
    """An impossible threshold must exit non-zero, not produce an empty table."""
    assert analyse_bank.main([str(bank), "--label-threshold", "1e9"]) == 1
    assert "no positive regions" in capsys.readouterr().out


def test_weights_must_form_a_valid_combination(bank):
    """Negative weights break monotonicity and the T3.3 Lipschitz argument (D25)."""
    with pytest.raises(ValueError, match="non-negative"):
        analyse_bank.main([str(bank), "--w-free", "-1", "--w-disagree", "1", "--w-residual", "1"])


# -- every threshold is held out, not only the naive one (P-23) ----------------


@pytest.fixture(scope="module")
def two_split_bank(bank, tmp_path_factory):
    """The phantom bank with half its volumes relabelled `test`.

    The phantom substrate only supports a fixed split, so the two-split bank is
    derived rather than generated; the split column is all that the held-out path
    reads, and it is assigned per volume so no volume straddles the sides.
    """
    import shutil

    out = tmp_path_factory.mktemp("two_split") / "bank"
    shutil.copytree(bank, out)
    df = read_bank(out)
    volumes = sorted(df["sample_id"].unique())
    test_side = set(volumes[: len(volumes) // 2])
    df["split"] = df["sample_id"].map(lambda v: "test" if v in test_side else "calib")
    df.to_parquet(out / "scores.parquet", index=False)
    return out


def test_every_threshold_is_fitted_on_calibration_rows_and_scored_on_test_rows(
    two_split_bank, monkeypatch
):
    """The regression P-23 is about.

    The first held-out fix calibrated the naive threshold on `split=calib` but fitted
    the matched and group-conditional thresholds on the evaluation rows themselves.
    Their coverage then sat on nominal by construction, and a held-out method was
    compared against two in-sample ones. Spy on every fitting and scoring call and
    check the split of the rows each one actually receives.
    """
    bank_df = read_bank(two_split_bank)
    fitted_on, scored_on = [], []

    real_split_conformal = analyse_bank.split_threshold
    real_mondrian = analyse_bank.mondrian_calibration_results
    real_evaluate = analyse_bank.evaluate_groups

    def spy_split_conformal(scores, labels, alpha):
        fitted_on.append(set(bank_df.loc[scores.index, "split"]))
        return real_split_conformal(scores, labels, alpha)

    def spy_mondrian(df, score_col, label_col, *args, **kwargs):
        fitted_on.append(set(df["split"]))
        return real_mondrian(df, score_col, label_col, *args, **kwargs)

    def spy_evaluate(df, *args, **kwargs):
        scored_on.append(set(df["split"]))
        return real_evaluate(df, *args, **kwargs)

    monkeypatch.setattr(analyse_bank, "split_threshold", spy_split_conformal)
    monkeypatch.setattr(analyse_bank, "mondrian_calibration_results", spy_mondrian)
    monkeypatch.setattr(analyse_bank, "evaluate_groups", spy_evaluate)

    assert analyse_bank.main([str(two_split_bank), "--label-threshold", "0.05"]) == 0

    # naive once, matched once per operator, mondrian once
    n_ops = bank_df["operator_id"].nunique()
    assert len(fitted_on) == 1 + n_ops + 1
    assert all(sides == {"calib"} for sides in fitted_on), fitted_on
    assert len(scored_on) == 3
    assert all(sides == {"test"} for sides in scored_on), scored_on
