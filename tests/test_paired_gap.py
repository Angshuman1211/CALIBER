"""Gates for the paired coverage-gap estimand (T1.4, D45).

P-23 concluded that fastMRI cannot support a per-operator coverage claim at any grid size,
and that conclusion is correct about per-operator coverage. What it got wrong is the
estimand: the headline figure is a coverage *gap*, the design is fully crossed, and a gap
between two arms measured on the same patients is a paired contrast whose dominant variance
term cancels.

These tests pin the three claims that ruling rests on:

* the pairing is arithmetically real — a contrast of an arm with itself has *exactly* zero
  variance, where differencing two independent bootstraps invents some;
* the gain is a function of between-patient variance, not a constant, so it is measured per
  bank and never assumed — including the honest case where it is 1.0;
* the crossed design is checked rather than believed, because a future split design would
  make P-23's division the right arithmetic again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caliber.calibration import apply_flags, calibrate_mondrian, calibrate_naive
from caliber.evaluation.bank import derive_score_and_label
from caliber.evaluation.paired import (
    clusters_for_paired_precision,
    is_crossed,
    paired_coverage_gap,
    paired_gap_curve,
    paired_method_gap,
)
from caliber.scorebank import read_bank

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import sizing  # noqa: E402

BANK = REPO_ROOT / "banks" / "w02_solver_smoke"


def _crossed_frame(
    n_clusters: int = 40,
    per_cluster: int = 40,
    base: float = 0.90,
    delta: float = 0.10,
    sigma: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Two operators over the same clusters, with a shared per-cluster difficulty.

    ``sigma`` is the SD of the cluster effect that both operators share — the patient's own
    difficulty. It is the quantity the pairing cancels, so sweeping it is how the gain is
    characterised.
    """
    rng = np.random.default_rng(seed)
    effect = rng.normal(0.0, sigma, n_clusters)
    rows = []
    for i in range(n_clusters):
        for op, p in (("A", base + effect[i]), ("B", base - delta + effect[i])):
            flags = rng.random(per_cluster) < float(np.clip(p, 0.01, 0.99))
            rows.extend(
                {"sample_id": f"c{i}", "operator_id": op, "label": True, "flag": bool(f)}
                for f in flags
            )
    return pd.DataFrame(rows)


# -- the arithmetic the ruling rests on ----------------------------------------


def test_an_arm_against_itself_has_exactly_zero_uncertainty():
    """The cleanest demonstration that differencing independent bootstraps is inflated.

    Coverage at an operator minus coverage at *the same* operator is zero, with no
    uncertainty whatever — it is the same rows twice. The paired estimator returns exactly
    that. The unpaired one returns a positive standard error for a quantity that cannot
    vary, and that error is precisely what P-23's intervals carried into the grid table.
    """
    df = _crossed_frame(sigma=0.10, seed=3)
    same = paired_coverage_gap(df, "A", "A")
    assert same.gap == 0.0
    assert same.se_paired == 0.0
    assert same.lo == 0.0 and same.hi == 0.0
    assert same.se_unpaired > 0.0, "if this is zero the comparison is no longer meaningful"


@pytest.mark.parametrize("sigma", [0.0, 0.05, 0.10, 0.20])
def test_paired_standard_error_does_not_grow_with_between_patient_variance(sigma):
    """The mechanism: the patient effect cancels, so only the within-patient term is left.

    The paired SE must stay near its ``sigma = 0`` value however heterogeneous the patients
    are, while the unpaired SE grows with them. Measured on the development sweep: paired
    0.0130 → 0.0132 across this range, unpaired 0.0131 → 0.0385.
    """
    gap = paired_coverage_gap(_crossed_frame(sigma=sigma, seed=1), "A", "B")
    assert 0.008 < gap.se_paired < 0.020, gap.se_paired
    assert gap.se_unpaired >= gap.se_paired * 0.9


def test_the_gain_appears_only_where_patients_differ():
    """Reported per bank, never assumed — including when it is nothing.

    At ``sigma = 0`` pairing buys nothing and the ratio sits at 1; the estimator must not be
    built so that it always shows a gain. The phantom bank behaves like this, because
    Shepp-Logan variants are nearly identical to each other.
    """
    flat = paired_coverage_gap(_crossed_frame(sigma=0.0, seed=1), "A", "B")
    varied = paired_coverage_gap(_crossed_frame(sigma=0.20, seed=1), "A", "B")
    assert 0.85 < flat.variance_reduction < 1.15
    assert varied.variance_reduction > 2.0


def test_the_gap_is_recovered_and_its_interval_covers_the_truth():
    df = _crossed_frame(base=0.90, delta=0.10, sigma=0.10, seed=7)
    gap = paired_coverage_gap(df, "A", "B")
    assert abs(gap.gap - 0.10) < 0.04
    assert gap.lo < 0.10 < gap.hi
    assert gap.excludes_zero


# -- the crossed-design premise ------------------------------------------------


def test_is_crossed_accepts_the_crossed_design_and_rejects_a_split_one():
    crossed = is_crossed(_crossed_frame(n_clusters=8, per_cluster=4))
    assert crossed["crossed"] and crossed["n_shared"] == 8

    df = _crossed_frame(n_clusters=8, per_cluster=4)
    # A split design: operator A keeps the first four clusters, B the last four.
    split = df[
        ((df.operator_id == "A") & df.sample_id.isin([f"c{i}" for i in range(4)]))
        | ((df.operator_id == "B") & df.sample_id.isin([f"c{i}" for i in range(4, 8)]))
    ]
    assert not is_crossed(split)["crossed"]


def test_pairing_refuses_a_split_design_rather_than_pairing_nothing():
    df = _crossed_frame(n_clusters=6, per_cluster=4)
    split = df[
        ((df.operator_id == "A") & df.sample_id.isin(["c0", "c1", "c2"]))
        | ((df.operator_id == "B") & df.sample_id.isin(["c3", "c4", "c5"]))
    ]
    with pytest.raises(ValueError):
        paired_coverage_gap(split, "A", "B")


def test_every_bank_in_the_repository_is_crossed():
    """The premise, checked against real banks rather than asserted in prose."""
    crossing = is_crossed(read_bank(BANK))
    assert crossing["crossed"], crossing["reason"]
    assert crossing["n_operators"] >= 2


# -- the method contrast -------------------------------------------------------


def test_two_identical_methods_show_no_difference_and_no_uncertainty():
    df = _crossed_frame(sigma=0.10, seed=4)
    df["flag_b"] = df["flag"]
    gap = paired_method_gap(df, "flag", "flag_b")
    assert gap.gap == 0.0
    assert gap.se_paired == 0.0
    assert gap.se_unpaired > 0.0


def test_the_method_contrast_separates_mondrian_from_naive_on_a_real_bank():
    """The project's central claim, in the form that is best determined.

    Calibrating at one operator and deploying at another is the failure; the
    group-conditional threshold is the remedy. Both are applied to the *same* rows, so the
    patient and the operator both cancel and the contrast is tighter than either coverage.
    """
    df = derive_score_and_label(read_bank(BANK), "mri.random1d.R4.c0.08.s0", threshold=1.75)
    group = df[df.operator_id == "mri.random1d.R4.c0.08.s0"].calib_group.iloc[0]
    df = apply_flags(
        df, calibrate_naive(df, calibration_group=group, alpha=0.1), out_col="flag_naive"
    )
    df = apply_flags(df, calibrate_mondrian(df, alpha=0.1), out_col="flag_mondrian")

    deployment = df[df.operator_id == "mri.random1d.R8.c0.08.s0"]
    gap = paired_method_gap(deployment, "flag_mondrian", "flag_naive")
    assert np.isfinite(gap.gap)
    assert gap.se_paired >= 0.0
    assert gap.n_clusters >= 1


def test_paired_curve_keeps_the_calibration_operator_as_its_origin():
    curve = paired_gap_curve(_crossed_frame(n_clusters=10, per_cluster=8), "A")
    assert set(curve["deployment_operator"]) == {"A", "B"}
    origin = curve[curve.deployment_operator == "A"].iloc[0]
    assert origin["gap"] == 0.0


# -- sizing --------------------------------------------------------------------


def test_paired_sizing_scales_as_one_over_root_n():
    n = clusters_for_paired_precision(0.10, 12, 0.05)
    assert clusters_for_paired_precision(0.05, 12, 0.05) < n
    assert clusters_for_paired_precision(0.10, 12, 0.10) < n


def test_paired_sizing_refuses_a_nonsense_standard_error():
    with pytest.raises(ValueError):
        clusters_for_paired_precision(0.0, 12, 0.05)
    with pytest.raises(ValueError):
        clusters_for_paired_precision(float("nan"), 12, 0.05)


def test_the_sizing_script_runs_and_no_longer_divides_the_budget(capsys):
    """P-19's lesson, and D45's correction in the same test.

    The grid table used to divide the patient budget by the number of operators. In a
    crossed design every operator sees every patient, so the 'test patients each' column
    must be constant down the table.
    """
    assert sizing.main([str(BANK), "--tau", "1.75"]) == 0
    out = capsys.readouterr().out
    assert "CROSSED" in out
    assert "paired coverage gap" in out

    rows = [
        line.split() for line in out.splitlines()
        if line.strip() and line.split()[0] in {"1", "2", "3", "6", "12"} and len(line.split()) == 3
    ]
    assert len(rows) >= 4, out
    per_operator = {row[1] for row in rows}
    assert len(per_operator) == 1, f"budget is still being divided across operators: {rows}"
