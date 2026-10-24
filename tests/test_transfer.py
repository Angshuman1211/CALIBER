"""Gates for the coverage-axis baseline comparison (T1.4, D73).

The module exists to answer one question -- whose threshold survives a change of operator
-- so the tests are built around fixtures where **the answer is known by construction**:
one detector's scores drift between operators and the other's do not. A module that cannot
recover a planted answer cannot be trusted with a real one.

The rest of the file is the list of ways this project has previously got a transfer number
wrong, each turned into a test:

* a calibration frame sharing patients with the frame it is scored on (P-23, and again in
  ``tool2_holdout.py``), which reads as valid when it is not;
* pooling across operators, which averaged two opposite biases into "exactly chance"
  (D60, retracted);
* a direction-sensitive quantity checked in one direction only (D65);
* a verdict resting on significance rather than magnitude (D72);
* a detector reaching coverage by flagging everything (D51's validity gate).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caliber.evaluation.transfer import (
    PENALTY_FLOOR,
    TRIVIAL_FLAG_AREA,
    detector_penalty,
    detector_transfer,
    transfer_table,
    transfer_verdict,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import transfer_baseline  # noqa: E402

CALIB_OP = "op.low"
DEPLOY_OPS = ("op.mid", "op.high")
#: Enough positives that a 10th-percentile threshold is not set by three regions.
N_PATIENTS, N_REGIONS = 20, 24


def _bank(
    caliber_drift: float = 0.0,
    baseline_drift: float = 0.0,
    seed: int = 0,
    n_patients: int = N_PATIENTS,
    positive_rate: float = 0.4,
) -> pd.DataFrame:
    """A crossed bank with a planted answer.

    ``*_drift`` is how far that detector's scores move **down** at every deployment
    operator, in units of the positive/negative separation. A threshold fitted at
    ``op.low`` then sits too high for the deployment operator's scores and misses
    positives, which is a coverage penalty. Zero drift means the threshold transfers
    perfectly.

    Downward is the direction that matters: drifting up makes a transferred threshold
    over-flag, which costs efficiency rather than validity, and would plant the wrong
    answer for a test about coverage.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        split = "calib" if p < n_patients // 2 else "test"
        for op in (CALIB_OP, *DEPLOY_OPS):
            shift = 0.0 if op == CALIB_OP else 1.0
            for r in range(N_REGIONS):
                label = bool(rng.random() < positive_rate)
                base = 2.0 if label else 0.0
                rows.append({
                    "patient_id": f"p{p:02d}",
                    "sample_id": f"p{p:02d}_v0",
                    "slice_idx": 0,
                    "region_id": r,
                    "operator_id": op,
                    "split": split,
                    "label": label,
                    "score": base + rng.normal() - caliber_drift * shift,
                    "s_dispersion": base + rng.normal() - baseline_drift * shift,
                })
    return pd.DataFrame(rows)


def _sides(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[df["split"] == "calib"], df[df["split"] == "test"]


# -- the planted answers ------------------------------------------------------------


def test_a_detector_whose_scores_do_not_drift_keeps_its_coverage():
    """Centred on zero across seeds, not zero in any one of them.

    A single no-drift fixture returns a penalty of up to about 0.05 at this size, because
    the two thresholds are order statistics of different samples of a few dozen positives.
    The claim that holds is about the mean, and it is the reason a *per-detector* penalty
    is not evidence of transfer failure on its own -- only the difference between two
    detectors measured on the same patients is.
    """
    penalties = []
    for seed in range(12):
        calib, test = _sides(_bank(seed=seed))
        r = detector_penalty(calib, test, "score", CALIB_OP, "op.high", n_boot=200, seed=seed)
        assert r.reaches_nominal, "the in-group arm must be calibrated to compare against"
        penalties.append(r.penalty)
    mean = float(np.mean(penalties))
    assert abs(mean) < PENALTY_FLOOR, f"nothing drifted, so no coverage should be lost: {mean}"


def test_a_detector_whose_scores_drift_loses_coverage():
    calib, test = _sides(_bank(baseline_drift=1.5))
    r = detector_penalty(calib, test, "s_dispersion", CALIB_OP, "op.high")
    assert r.penalty > PENALTY_FLOOR, (
        f"a threshold 1.5 separations too high must miss positives, got {r.penalty}"
    )
    assert r.coverage_naive < r.coverage_group


def test_the_comparison_recovers_the_planted_winner():
    """The headline. CALIBER's scores hold still, the baseline's drift; the module must
    say so, and with an interval that excludes zero."""
    df = _bank(caliber_drift=0.0, baseline_drift=1.5)
    calib, test = _sides(df)
    c = detector_transfer(calib, test, "s_dispersion", "score", CALIB_OP, "op.high")
    assert c.interpretable
    assert c.delta > PENALTY_FLOOR
    assert c.excludes_zero
    assert c.verdict == "caliber_transfers_better"


def test_the_comparison_recovers_the_planted_loser_too():
    """D65's lesson: a direction-sensitive verdict checked in one direction only was wrong.
    Swapping which detector drifts must swap the verdict, not merely weaken it."""
    df = _bank(caliber_drift=1.5, baseline_drift=0.0)
    calib, test = _sides(df)
    c = detector_transfer(calib, test, "s_dispersion", "score", CALIB_OP, "op.high")
    assert c.delta < -PENALTY_FLOOR
    assert c.verdict == "baseline_transfers_better"


def test_two_detectors_that_both_hold_still_are_not_called_a_winner():
    """The null. The point estimate is near zero, but the interval at this size is far
    wider than the floor, so the honest verdict is 'undetermined' -- never a win."""
    calib, test = _sides(_bank(caliber_drift=0.0, baseline_drift=0.0))
    c = detector_transfer(calib, test, "s_dispersion", "score", CALIB_OP, "op.high")
    assert c.interpretable
    assert abs(c.delta) < 0.10
    assert c.verdict in {"undetermined", "equivalent"}
    assert c.verdict not in {"caliber_transfers_better", "baseline_transfers_better"}


def test_equivalence_needs_the_whole_interval_inside_the_floor_not_the_estimate():
    """The trap this gate exists for: ``abs(delta) <= floor`` would call the null
    'equivalent' off a point estimate while the interval spans +-0.15. Equivalence is a
    claim, and it has to be earned by the interval."""
    calib, test = _sides(_bank())
    c = detector_transfer(calib, test, "s_dispersion", "score", CALIB_OP, "op.high")
    assert abs(c.delta) < 0.10, "the null's point estimate is small"
    assert c.half_width > PENALTY_FLOOR, "and its interval is much wider than the floor"
    assert not c.equivalent, "so 'alike' is not established"
    wide = detector_transfer(calib, test, "s_dispersion", "score", CALIB_OP, "op.high",
                             floor=1.0)
    assert wide.equivalent and wide.verdict == "equivalent"


def test_both_drifting_alike_is_not_credited_to_either_detector():
    """Transfer failure that both detectors share is a property of the shift, not of
    either detector, and must not become a win for one."""
    calib, test = _sides(_bank(caliber_drift=1.5, baseline_drift=1.5))
    c = detector_transfer(calib, test, "s_dispersion", "score", CALIB_OP, "op.high")
    assert c.baseline.penalty > PENALTY_FLOOR and c.caliber.penalty > PENALTY_FLOOR
    assert c.verdict not in {"caliber_transfers_better", "baseline_transfers_better"}


def test_the_floor_needed_for_equivalence_is_reported_when_undetermined():
    """The actionable number: whether the gap to a conclusion is a few patients or an
    order of magnitude."""
    calib, test = _sides(_bank())
    c = detector_transfer(calib, test, "s_dispersion", "score", CALIB_OP, "op.high")
    assert np.isfinite(c.floor_for_equivalence)
    assert c.floor_for_equivalence >= abs(c.delta)
    relaxed = detector_transfer(calib, test, "s_dispersion", "score", CALIB_OP, "op.high",
                                floor=c.floor_for_equivalence * 1.01)
    assert relaxed.equivalent, "the reported floor must be the one that actually suffices"


# -- magnitude over significance (D72) ---------------------------------------------


def test_the_floor_is_a_verdict_and_never_moves_the_estimate():
    """D72's control: with slopes 0.3025 and 0.3029 an F-test rejected at p = 0.031. The
    floor is what stops that reading as a finding, so moving it must change the verdict
    and must not touch the measurement."""
    calib, test = _sides(_bank(baseline_drift=0.35, seed=4))
    tight = detector_transfer(calib, test, "s_dispersion", "score", CALIB_OP, "op.high",
                              floor=0.0)
    loose = detector_transfer(calib, test, "s_dispersion", "score", CALIB_OP, "op.high",
                              floor=1.0)
    assert loose.delta == pytest.approx(tight.delta), "the floor is a verdict, not an estimate"
    assert loose.lo == pytest.approx(tight.lo) and loose.hi == pytest.approx(tight.hi)
    assert loose.verdict == "equivalent" and tight.verdict != "equivalent"


def test_a_real_difference_too_few_patients_to_place_is_undetermined_not_a_win():
    """Four patients cannot place a difference. 'undetermined' is the answer; a win is
    not, and neither is 'equivalent'."""
    calib, test = _sides(_bank(baseline_drift=1.5, n_patients=4, seed=7))
    c = detector_transfer(calib, test, "s_dispersion", "score", CALIB_OP, "op.high")
    assert c.verdict in {"undetermined", "not_interpretable"}


# -- the interval is calibrated, which is the gate the first version failed ---------


def test_under_the_null_the_interval_covers_zero_and_no_fixture_reads_as_a_win():
    """**The gate that caught the original bug.**

    The first version resampled only the test patients and held the thresholds fixed. Its
    interval therefore measured patient variation and ignored the variation in the
    threshold itself -- a single order statistic of a few dozen positives, and the larger
    term. Against a 0.02 floor it called pure threshold noise a CALIBER win: ``|delta|``
    under the null averaged 0.039 and reached 0.128.

    Resampling the calibration side as well puts that noise in the interval. Twelve null
    fixtures, nothing drifting: the interval must cover zero in essentially all of them
    and no fixture may read as a directional win.
    """
    covered, verdicts = 0, []
    for seed in range(12):
        calib, test = _sides(_bank(seed=seed))
        c = detector_transfer(calib, test, "s_dispersion", "score", CALIB_OP, "op.high",
                              n_boot=400, seed=seed)
        covered += int(c.lo <= 0.0 <= c.hi)
        verdicts.append(c.verdict)
    assert covered >= 11, f"a 95% interval covered zero only {covered}/12 times under the null"
    assert not any(v.endswith("transfers_better") for v in verdicts), (
        f"threshold noise read as a directional result: {verdicts}"
    )


def test_the_planted_difference_is_still_found_once_the_interval_is_honest():
    """The other half of the calibration: an interval wide enough to cover zero under the
    null must still be narrow enough to exclude it when a real difference is planted.
    Otherwise the fix to the null would have been bought by making the test useless."""
    found = 0
    for seed in range(8):
        calib, test = _sides(_bank(baseline_drift=1.5, seed=seed))
        c = detector_transfer(calib, test, "s_dispersion", "score", CALIB_OP, "op.high",
                              n_boot=400, seed=seed)
        found += int(c.verdict == "caliber_transfers_better")
    assert found >= 7, f"a 1.5-separation drift was detected in only {found}/8 fixtures"


def test_the_interval_narrows_as_patients_are_added():
    """A sanity check on the whole construction: if the interval did not respond to the
    patient count, it would not be measuring sampling variation at all."""
    widths = []
    for n_patients in (20, 80):
        calib, test = _sides(_bank(baseline_drift=1.5, n_patients=n_patients, seed=3))
        c = detector_transfer(calib, test, "s_dispersion", "score", CALIB_OP, "op.high",
                              n_boot=400, seed=3)
        widths.append(c.hi - c.lo)
    assert widths[1] < widths[0], f"width did not fall with more patients: {widths}"


# -- the in-sample trap (P-23, and tool2_holdout.py) -------------------------------


def test_a_shared_patient_between_calibration_and_test_is_refused():
    df = _bank()
    calib, test = _sides(df)
    leaked = pd.concat([test, calib[calib["patient_id"] == "p00"]], ignore_index=True)
    with pytest.raises(ValueError, match="both the calibration and test frames"):
        detector_penalty(calib, leaked, "score", CALIB_OP, "op.high")


def test_a_bank_with_no_test_side_is_refused_rather_than_scored_in_sample():
    df = _bank()
    with pytest.raises(ValueError, match="no calib/test split"):
        transfer_table(df[df["split"] == "calib"], "s_dispersion", "score", CALIB_OP)


def test_a_missing_column_is_named():
    calib, test = _sides(_bank())
    with pytest.raises(KeyError, match="s_dispersion"):
        detector_penalty(calib, test.drop(columns=["s_dispersion"]), "s_dispersion",
                         CALIB_OP, "op.high")


def test_an_operator_with_no_calibration_side_cannot_be_a_group_reference():
    """Without in-group calibration rows there is no 'what coverage should have been', so
    the cell is refused rather than compared against the transferred arm alone."""
    df = _bank()
    thin = df[~((df["split"] == "calib") & (df["operator_id"] == "op.high"))]
    calib, test = _sides(thin)
    with pytest.raises(KeyError, match="group-conditional"):
        detector_penalty(calib, test, "score", CALIB_OP, "op.high")


# -- the validity gate (D51) and the trivial solution ------------------------------


def test_flagging_everything_transfers_perfectly_and_is_refused_as_a_result():
    """A constant detector has a zero penalty and full coverage. Without the gate it would
    be the best-transferring detector in the table."""
    df = _bank()
    df["s_dispersion"] = 1.0
    calib, test = _sides(df)
    r = detector_penalty(calib, test, "s_dispersion", CALIB_OP, "op.high")
    assert r.flag_area_group >= TRIVIAL_FLAG_AREA
    assert r.trivial and not r.interpretable
    c = detector_transfer(calib, test, "s_dispersion", "score", CALIB_OP, "op.high")
    assert c.verdict == "not_interpretable"


# -- never pooled across operators (D60, retracted) --------------------------------


def test_the_table_has_one_row_per_deployment_operator_and_omits_the_calibration_one():
    df = _bank(baseline_drift=1.5)
    table, results = transfer_table(df, "s_dispersion", "score", CALIB_OP)
    assert set(table["deployed_at"]) == set(DEPLOY_OPS)
    assert CALIB_OP not in set(table["deployed_at"]), "a gap against itself is zero by construction"
    assert len(results) == len(DEPLOY_OPS)


def test_the_verdict_counts_operators_rather_than_averaging_their_deltas():
    """Two operators with opposite signs must read as a split, not as no effect -- which is
    exactly what averaging produced in D60."""
    df = _bank(baseline_drift=1.5)
    down = df["operator_id"] == "op.mid"
    df.loc[down, "score"] = df.loc[down, "score"] - 3.0
    df.loc[down, "s_dispersion"] = df.loc[down, "s_dispersion"] + 3.0
    _, results = transfer_table(df, "s_dispersion", "score", CALIB_OP)
    verdict = transfer_verdict(results)
    assert verdict["n_operators"] == len(DEPLOY_OPS)
    assert "counts" in verdict and sum(verdict["counts"].values()) == len(DEPLOY_OPS)
    if verdict["caliber_better"] and verdict["baseline_better"]:
        assert "do not average" in verdict["headline"]


def test_a_verdict_with_nothing_interpretable_says_so():
    df = _bank()
    df["s_dispersion"] = 1.0
    df["score"] = 1.0
    _, results = transfer_table(df, "s_dispersion", "score", CALIB_OP)
    verdict = transfer_verdict(results)
    assert verdict["n_interpretable"] == 0
    assert "nothing is" in verdict["headline"] or "no operator" in verdict["headline"]


# -- pairing is retained, and measured rather than claimed -------------------------


def test_the_paired_interval_is_reported_against_the_unpaired_one():
    """paired.py's rule: the variance reduction is printed so it can come back at 1 and be
    believed. A difference-of-differences that silently assumed the gain would be the same
    error one level out."""
    calib, test = _sides(_bank(baseline_drift=1.5))
    c = detector_transfer(calib, test, "s_dispersion", "score", CALIB_OP, "op.high")
    assert np.isfinite(c.se_paired) and np.isfinite(c.se_unpaired)
    assert c.se_paired > 0


def test_no_positive_region_yields_nan_rather_than_a_confident_zero():
    df = _bank()
    df["label"] = False
    calib, test = _sides(df)
    r = detector_penalty(calib, test, "score", CALIB_OP, "op.high")
    assert r.n_positive == 0
    assert np.isnan(r.penalty)
    assert not r.interpretable


# -- the entry point (P-19: the library passing is not the script working) ----------


def _write_bank(tmp_path: Path, **kwargs) -> Path:
    df = _bank(**kwargs)
    # the design split a quantile label needs, taken from patients the others never see
    design = _bank(n_patients=6, seed=99, **kwargs)
    design["split"] = "design"
    design["patient_id"] = "d" + design["patient_id"]
    design["sample_id"] = "d" + design["sample_id"]
    full = pd.concat([df, design], ignore_index=True)
    for col, src in (("err_free", "score"), ("err_total", "score")):
        full[col] = full[src].abs() + 1.0
    full["free_content"] = 1.0
    full["ann_coverage"] = 1.0
    full["s_free"] = full["score"]
    full["s_disagree"] = full["score"]
    full["s_residual"] = full["score"]
    out = tmp_path / "bank.parquet"
    full.to_parquet(out, index=False)
    return out


def test_the_script_runs_end_to_end_on_a_bank_column(tmp_path, capsys):
    bank = _write_bank(tmp_path, baseline_drift=1.5)
    code = transfer_baseline.main([
        str(bank), "--low", CALIB_OP, "--high", "op.high",
        "--baseline", "s_dispersion", "--boot", "200",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "ascending" in out and "descending" in out
    assert "penalty" in out


def test_the_script_refuses_a_bank_with_no_patient_id(tmp_path, capsys):
    df = _bank()
    bank = tmp_path / "nopid.parquet"
    df.drop(columns=["patient_id"]).to_parquet(bank, index=False)
    code = transfer_baseline.main([str(bank), "--low", CALIB_OP, "--high", "op.high"])
    assert code == 1
    assert "patient_id" in capsys.readouterr().err


def test_the_script_refuses_a_dispersion_file_that_does_not_match_the_bank(tmp_path, capsys):
    bank = _write_bank(tmp_path)
    bogus = tmp_path / "disp.parquet"
    pd.DataFrame({
        "sample_id": ["nope"], "slice_idx": [0], "operator_id": ["op.low"],
        "region_id": [0], "s_dispersion": [1.0],
    }).to_parquet(bogus, index=False)
    code = transfer_baseline.main([
        str(bank), "--low", CALIB_OP, "--high", "op.high", "--dispersion", str(bogus),
    ])
    err = capsys.readouterr().err
    assert code == 1
    assert "no bank row matches" in err or "neither direction" in err
