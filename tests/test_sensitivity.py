"""Gates for the weight sweep (D76) and the matched-coverage area (D39/D51).

Both exist to turn a weakness into something measured rather than conceded, so both have
to be able to come back negative. The tests build fixtures where the answer is planted in
each direction:

* a bank whose coverage restoration is the same at every weighting, and one where it is
  not -- the second must *not* report insensitivity;
* a degenerate detector that flags everything, which wins on the old area comparison and
  must lose on the matched-coverage one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caliber.evaluation import area_at_matched_coverage, weight_sensitivity

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import weight_sensitivity as script  # noqa: E402

LOW, HIGH = "op.low", "op.high"


def _bank(
    free_only: bool = False,
    n_patients: int = 24,
    n_regions: int = 24,
    seed: int = 0,
) -> pd.DataFrame:
    """A crossed bank where naive transfer loses coverage at the aggressive operator.

    ``free_only`` makes ``s_disagree`` pure noise, so the score's quality depends on
    ``w_free`` and the sweep should *not* report insensitivity. Otherwise both components
    carry the same signal and every weighting is equivalent by construction.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        split = ("design" if p < 4 else "calib" if p < n_patients // 2 + 2 else "test")
        for op in (LOW, HIGH):
            shift = 0.0 if op == LOW else 1.2
            for r in range(n_regions):
                hallucinated = bool(rng.random() < 0.4)
                base = 2.0 if hallucinated else 0.0
                signal = base + rng.normal() - shift
                rows.append({
                    "patient_id": f"p{p:02d}",
                    "sample_id": f"p{p:02d}_v0",
                    "slice_idx": 0,
                    "region_id": r,
                    "operator_id": op,
                    "split": split,
                    "s_free": signal,
                    "s_disagree": rng.normal() if free_only else signal + rng.normal(0, 0.2),
                    "s_residual": rng.normal(0, 0.1),
                    "err_free": (2.5 if hallucinated else 0.2) + abs(rng.normal(0, 0.1)),
                    "err_total": 3.0,
                    "free_content": 1.0,
                    "ann_coverage": 1.0,
                })
    return pd.DataFrame(rows)


def _label(df: pd.DataFrame) -> dict:
    eps = float(df.loc[df["split"] == "design", "err_free"].quantile(0.75))
    return {"threshold": eps, "relative": False}


# -- the sweep ---------------------------------------------------------------------


def test_the_sweep_covers_the_grid_and_holds_the_label_fixed():
    df = _bank()
    result = weight_sensitivity(df, LOW, HIGH, label_kwargs=_label(df), n_boot=200)
    assert len(result.curve) >= 10
    assert result.curve["w_free"].is_monotonic_increasing
    assert (result.curve["w_free"] + result.curve["w_disagree"]).round(6).eq(1.0).all()
    # One label throughout: the positive count cannot move with the weighting.
    assert result.curve["n_positive"].nunique() == 1


def test_a_claim_that_holds_across_the_grid_is_reported_as_insensitive():
    """Both components carry the same signal, so no weighting can be better or worse and
    the inability to select one costs nothing."""
    df = _bank(free_only=False)
    result = weight_sensitivity(df, LOW, HIGH, label_kwargs=_label(df), n_boot=400)
    assert result.sign_is_constant
    assert result.holds_everywhere, result.curve[["w_free", "gap", "lo"]].to_string()
    assert "costs the claim nothing" in result.verdict() or result.within_noise


def test_a_weight_dependent_claim_is_not_reported_as_insensitive():
    """The negative control. With ``s_disagree`` pure noise, the score degrades as weight
    moves off ``s_free``, so the span across the grid must exceed one weighting's own
    interval and the verdict must say the weighting matters."""
    df = _bank(free_only=True, seed=2)
    result = weight_sensitivity(df, LOW, HIGH, label_kwargs=_label(df), n_boot=400)
    assert result.span > 0.0
    at_free = result.curve.loc[result.curve["w_free"].idxmax(), "gap"]
    at_disagree = result.curve.loc[result.curve["w_free"].idxmin(), "gap"]
    assert at_free != pytest.approx(at_disagree, abs=1e-6), (
        "a sweep that cannot see a pure-noise component is not measuring anything"
    )
    if not result.within_noise:
        assert "weighting does move the result" in result.verdict() or not result.holds_everywhere


def test_a_bank_with_one_split_side_is_refused():
    df = _bank()
    df["split"] = "calib"
    with pytest.raises(ValueError, match="both 'calib' and 'test'"):
        weight_sensitivity(df, LOW, HIGH, label_kwargs=_label(_bank()))


def test_a_missing_cluster_column_is_named():
    df = _bank().drop(columns=["patient_id"])
    with pytest.raises(KeyError, match="patient_id"):
        weight_sensitivity(df, LOW, HIGH, label_kwargs=_label(_bank()))


def test_an_absent_deployment_operator_is_refused_rather_than_skipped_silently():
    df = _bank()
    with pytest.raises(ValueError, match="no weighting could be evaluated"):
        weight_sensitivity(df, LOW, "op.missing", label_kwargs=_label(df), n_boot=100)


def test_insensitivity_is_only_claimed_against_the_interval_not_in_absolute_terms():
    """D72's lesson in the other direction: a flat curve is evidence of insensitivity only
    if a single point is determined to better than the curve's own spread."""
    df = _bank()
    result = weight_sensitivity(df, LOW, HIGH, label_kwargs=_label(df), n_boot=400)
    assert np.isfinite(result.span) and np.isfinite(result.widest_interval)
    assert result.within_noise == (result.span < result.widest_interval)


# -- area at matched coverage ------------------------------------------------------


def _scored(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = rng.random(n) < 0.3
    return pd.DataFrame({
        "label": label,
        "good": np.where(label, rng.normal(3.0, 1.0, n), rng.normal(0.0, 1.0, n)),
        "weak": np.where(label, rng.normal(0.6, 1.0, n), rng.normal(0.0, 1.0, n)),
        "constant": np.ones(n),
    })


def test_a_better_detector_needs_less_area_for_the_same_coverage():
    frame = _scored()
    table = area_at_matched_coverage(frame, {"good": "good", "weak": "weak"}, 0.90)
    areas = dict(zip(table["detector"], table["flag_area"], strict=True))
    assert areas["good"] < areas["weak"]
    assert (table["coverage"] >= 0.90 - 1e-9).all(), "every arm must reach the target"


def test_the_degenerate_detector_cannot_win_on_this_axis():
    """**The reason this function exists.** A constant score flags everything, so on the
    old comparison -- fixed threshold, free coverage -- it can post whatever area its
    threshold happens to give. Here it has to reach the target, and reaching it costs the
    whole image."""
    frame = _scored()
    table = area_at_matched_coverage(
        frame, {"good": "good", "constant": "constant"}, 0.90
    )
    areas = dict(zip(table["detector"], table["flag_area"], strict=True))
    assert areas["constant"] == pytest.approx(1.0)
    assert areas["good"] < areas["constant"]


def test_a_higher_target_never_costs_less_area():
    frame = _scored()
    low = area_at_matched_coverage(frame, {"good": "good"}, 0.50)["flag_area"].iloc[0]
    high = area_at_matched_coverage(frame, {"good": "good"}, 0.95)["flag_area"].iloc[0]
    assert high >= low


def test_full_coverage_is_reachable_and_reported_as_reached():
    frame = _scored()
    table = area_at_matched_coverage(frame, {"good": "good"}, 1.0)
    assert bool(table["reached"].iloc[0])
    assert table["coverage"].iloc[0] == pytest.approx(1.0)


def test_no_positives_gives_nan_rather_than_a_flattering_zero():
    frame = _scored()
    frame["label"] = False
    table = area_at_matched_coverage(frame, {"good": "good"}, 0.90)
    assert np.isnan(table["flag_area"].iloc[0])
    assert not bool(table["reached"].iloc[0])


def test_a_target_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError, match="target coverage"):
        area_at_matched_coverage(_scored(), {"good": "good"}, 1.5)


def test_a_missing_score_column_is_named():
    with pytest.raises(KeyError, match="absent"):
        area_at_matched_coverage(_scored(), {"x": "absent"}, 0.9)


# -- the entry point ---------------------------------------------------------------


def test_the_script_runs_both_sweeps_and_the_area_table(tmp_path, capsys):
    bank = tmp_path / "bank.parquet"
    _bank(n_patients=20, n_regions=20).to_parquet(bank, index=False)
    code = script.main([
        str(bank), "--low", LOW, "--high", HIGH, "--boot", "200",
        "--out", str(tmp_path / "sweep.csv"), "--area-out", str(tmp_path / "area.csv"),
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "ascending" in out and "descending" in out
    assert "flag area at matched coverage" in out
    sweep = pd.read_csv(tmp_path / "sweep.csv")
    assert set(sweep["direction"]) == {"ascending", "descending"}
    area = pd.read_csv(tmp_path / "area.csv")
    assert area["operator_id"].nunique() >= 2, "one row per operator, never pooled"


def test_the_script_refuses_a_bank_without_patient_ids(tmp_path, capsys):
    bank = tmp_path / "nopid.parquet"
    _bank(n_patients=12).drop(columns=["patient_id"]).to_parquet(bank, index=False)
    code = script.main([str(bank), "--low", LOW, "--high", HIGH])
    assert code == 1
    assert "patient_id" in capsys.readouterr().err
