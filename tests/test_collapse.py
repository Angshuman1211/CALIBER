"""Gates for the curve-collapse test (T1.4, Sec. 21 R5, D72).

Three things need pinning here, in descending order of how badly a mistake would hurt:

* **the F survival function**, which is hand-rolled to keep scipy off the analysis path and
  is therefore a liability unless it is checked against published values;
* **that the verdict rests on magnitude rather than significance** — the control that
  exposed this had slopes of 0.3025 and 0.3029 and the F-test rejected at p = 0.031;
* **the two controls**, one curve fitting both modalities and one not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caliber.evaluation.collapse import (
    NOISE_FLOOR,
    _f_sf,
    curve_collapse_test,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import curve_collapse  # noqa: E402

#: Published upper-tail critical values of F at alpha = 0.05.
F_CRITICAL = [(1, 10, 4.9646), (2, 10, 4.1028), (2, 20, 3.4928), (3, 15, 3.2874),
              (1, 1, 161.4476), (5, 30, 2.5336), (2, 13, 3.8056)]


def _points(slope_ct: float, slope_mri: float, noise: float = 0.01,
            n_ct: int = 9, n_mri: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for modality, slope, xs in (
        ("ct", slope_ct, np.linspace(0.1, 0.9, n_ct)),
        ("mri", slope_mri, np.linspace(0.1, 0.9, n_mri)),
    ):
        for x in xs:
            rows.append(
                {"modality": modality, "distance": float(x),
                 "gap": slope * x + noise * rng.normal()}
            )
    return pd.DataFrame(rows)


# -- the hand-rolled F tail ----------------------------------------------------


@pytest.mark.parametrize("d1,d2,crit", F_CRITICAL)
def test_the_f_survival_function_matches_published_critical_values(d1, d2, crit):
    """Written out rather than imported, so it has to be checked rather than trusted.

    scipy is a declared dependency and a heavyweight import for one survival function; it
    has also broken at least one development install in this project, which is reason enough
    not to put it on the analysis path.
    """
    assert _f_sf(crit, d1, d2) == pytest.approx(0.05, abs=5e-3)


def test_the_f_tail_decreases_with_the_statistic():
    values = [_f_sf(f, 2, 10) for f in (0.5, 1.0, 2.0, 4.0, 8.0)]
    assert values == sorted(values, reverse=True)


def test_a_zero_statistic_is_certainty_not_an_undefined_value():
    """The separate fits improved on the pooled one not at all, so there is no evidence
    against collapse: p = 1, not NaN."""
    assert _f_sf(0.0, 2, 10) == 1.0
    assert np.isnan(_f_sf(float("nan"), 2, 10))
    assert np.isnan(_f_sf(1.0, 0, 10))


# -- the verdict rests on magnitude -------------------------------------------


def test_a_negligible_difference_collapses_even_when_the_f_test_rejects():
    """The control that set the design.

    Slopes 0.3025 and 0.3029 with overlapping intervals, and the F-test rejects at
    p = 0.031 — because the residuals are near zero, so a trivial intercept difference fills
    the denominator. Statistical significance is not the question; whether one curve is
    wrong by more than the measurement noise is.
    """
    result = curve_collapse_test(_points(0.3, 0.3, seed=0))
    assert result.slopes_overlap
    assert result.f_p_value < 0.05, "this control needs the F-test to reject"
    assert result.excess_rmse < NOISE_FLOOR
    assert result.collapses, "the magnitude test must override the significance test"


def test_a_real_difference_does_not_collapse():
    result = curve_collapse_test(_points(0.5, 0.02, seed=1))
    assert not result.slopes_overlap
    assert result.excess_rmse > NOISE_FLOOR
    assert not result.collapses
    assert "DOES NOT COLLAPSE" in result.summary()


def test_the_noise_floor_is_the_lever_and_is_stated():
    """Raising the floor should eventually accept any discrepancy, which is why the floor is
    a parameter with a documented default rather than a constant buried in the verdict."""
    points = _points(0.5, 0.30, seed=2)
    strict = curve_collapse_test(points, noise_floor=0.001)
    lenient = curve_collapse_test(points, noise_floor=1.0)
    assert not strict.collapses
    assert lenient.slopes_overlap is strict.slopes_overlap
    assert lenient.excess_rmse == pytest.approx(strict.excess_rmse)


# -- the caveats the result has to carry --------------------------------------


def test_non_overlapping_distance_ranges_are_flagged():
    """A collapse across disjoint ranges extrapolates one modality's law into the other's,
    so a positive result there is weak evidence and the summary must say so."""
    rows = []
    for x in np.linspace(0.05, 0.20, 6):
        rows.append({"modality": "mri", "distance": float(x), "gap": 0.3 * x})
    for x in np.linspace(0.60, 0.90, 6):
        rows.append({"modality": "ct", "distance": float(x), "gap": 0.3 * x})
    result = curve_collapse_test(pd.DataFrame(rows))
    assert not result.ranges_overlap
    assert "CAUTION" in result.summary()


def test_a_modality_with_too_few_points_is_noted_rather_than_fitted():
    result = curve_collapse_test(_points(0.3, 0.3, n_mri=2, seed=3))
    assert any("below" in n for n in result.notes)
    assert np.isnan(result.per_modality["mri"]["slope_lo"])


def test_one_modality_is_refused():
    single = _points(0.3, 0.3).query("modality == 'ct'")
    with pytest.raises(ValueError, match="compares modalities"):
        curve_collapse_test(single)


def test_a_missing_column_is_refused():
    with pytest.raises(KeyError):
        curve_collapse_test(_points(0.3, 0.3).drop(columns=["gap"]))


# -- the entry point (P-19: executing the script is not the same as the library) ----


def _write_inputs(tmp_path: Path) -> dict[str, str]:
    coverage, distances = {}, {}
    for modality, slope, ops in (
        ("mri", 0.02, ["mri.op1", "mri.op2", "mri.op3", "mri.op4"]),
        ("ct", 0.50, ["ct.op1", "ct.op2", "ct.op3", "ct.op4"]),
    ):
        cal = f"{modality}.cal"
        d = np.linspace(0.2, 0.8, len(ops))
        pd.DataFrame(
            [
                {"operator_id": o, "method": "naive_transferred",
                 "coverage": 0.90 - slope * float(x)}
                for o, x in zip(ops, d, strict=True)
            ]
        ).to_csv(tmp_path / f"{modality}_coverage.csv", index=False)
        pd.DataFrame(
            [
                {"calibration": cal, "deployment": o, "measure": "null_space_growth",
                 "value": float(x), "signed": True, "seconds": 1.0, "note": ""}
                for o, x in zip(ops, d, strict=True)
            ]
        ).to_csv(tmp_path / f"{modality}_distances.csv", index=False)
        coverage[modality] = str(tmp_path / f"{modality}_coverage.csv")
        distances[modality] = str(tmp_path / f"{modality}_distances.csv")
    return {"cov": coverage, "dist": distances}


def test_the_script_runs_end_to_end_and_reports_the_weak_form(tmp_path, capsys):
    paths = _write_inputs(tmp_path)
    code = curve_collapse.main([
        "--mri-coverage", paths["cov"]["mri"], "--mri-distances", paths["dist"]["mri"],
        "--ct-coverage", paths["cov"]["ct"], "--ct-distances", paths["dist"]["ct"],
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "does one curve fit both modalities" in out
    assert "DOES NOT COLLAPSE" in out, "the fixture has a 25x slope difference"
    assert "weak form" in out


def test_the_script_refuses_a_measure_that_is_not_in_the_csv(tmp_path, capsys):
    paths = _write_inputs(tmp_path)
    code = curve_collapse.main([
        "--mri-coverage", paths["cov"]["mri"], "--mri-distances", paths["dist"]["mri"],
        "--ct-coverage", paths["cov"]["ct"], "--ct-distances", paths["dist"]["ct"],
        "--measure", "spectral_divergence",
    ])
    assert code == 1
    assert "spectral_divergence" in capsys.readouterr().err


# -- each row takes the distance of its own (calibration, deployment) pair ------------


def _two_direction_inputs(tmp_path: Path) -> tuple[str, str]:
    """A Week 8-style table, both directions, against a CSV holding both calibrations."""
    coverage = pd.DataFrame([
        {"direction": "ascending", "calibrated_at": "ct.v180", "deployed_at": "ct.v90",
         "method": "naive", "coverage": 0.85},
        {"direction": "descending", "calibrated_at": "ct.v30", "deployed_at": "ct.v90",
         "method": "naive", "coverage": 0.95},
        {"direction": "descending", "calibrated_at": "ct.v30", "deployed_at": "ct.v180",
         "method": "naive", "coverage": 0.97},
        {"direction": "ascending", "calibrated_at": "ct.v180", "deployed_at": "ct.v90",
         "method": "group", "coverage": 0.90},
    ])
    distances = pd.DataFrame([
        {"calibration": "ct.v180", "deployment": "ct.v90", "measure": "null_space_growth",
         "value": 0.40},
        {"calibration": "ct.v30", "deployment": "ct.v90", "measure": "null_space_growth",
         "value": -0.30},
        # stored only in the other order: the reader must negate it
        {"calibration": "ct.v180", "deployment": "ct.v30", "measure": "null_space_growth",
         "value": 0.70},
    ])
    cov, dist = tmp_path / "cov.csv", tmp_path / "dist.csv"
    coverage.to_csv(cov, index=False)
    distances.to_csv(dist, index=False)
    return str(cov), str(dist)


def test_a_two_direction_table_is_matched_by_pair_not_by_deployment_alone(tmp_path):
    """The bug: matching on the deployment operator took the first calibration in the CSV,
    so v30 -> v90 was plotted at v180 -> v90's distance."""
    cov, dist = _two_direction_inputs(tmp_path)
    points = curve_collapse.points_for(cov, dist, "ct", "naive_transferred",
                                       "null_space_growth", 0.90)
    got = {(r.calibration, r.operator_id): r.distance for r in points.itertuples()}
    assert got[("ct.v180", "ct.v90")] == pytest.approx(0.40)
    assert got[("ct.v30", "ct.v90")] == pytest.approx(-0.30)
    assert got[("ct.v30", "ct.v180")] == pytest.approx(-0.70), "reverse order is negated"
    assert len(points) == 3, "the group arm is not the naive arm"


def test_one_direction_can_be_selected(tmp_path):
    cov, dist = _two_direction_inputs(tmp_path)
    points = curve_collapse.points_for(cov, dist, "ct", "naive_transferred",
                                       "null_space_growth", 0.90, direction="descending")
    assert set(points["direction"]) == {"descending"} and len(points) == 2


def test_a_table_that_does_not_name_its_calibration_is_refused_when_that_is_ambiguous(
    tmp_path,
):
    cov, dist = _two_direction_inputs(tmp_path)
    legacy = tmp_path / "legacy.csv"
    pd.DataFrame([{"operator_id": "ct.v90", "method": "naive_transferred",
                   "coverage": 0.85}]).to_csv(legacy, index=False)
    with pytest.raises(ValueError, match="calibrated_at"):
        curve_collapse.points_for(str(legacy), dist, "ct", "naive_transferred",
                                  "null_space_growth", 0.90)
