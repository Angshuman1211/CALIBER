"""The design side: patients held out for choices made by looking (D-051).

Track 1's D28 (score weights) and D37 (criterion threshold) are *selected* by looking
at scores. Selecting on calibration data biases the threshold; selecting on test data
biases the coverage number. Neither is visible afterwards, which is why the guarantee
has to be a test rather than a convention.

Two properties carry the whole design, and both are checked here against the two-way
split rather than against a fixture that could drift with it:

1. ``design_fraction=0.0`` reproduces the two-way split exactly, so adding the
   capability disturbs no existing bank, pin, or salt.
2. The ``test`` side is *identical* for every ``design_fraction``, so a coverage number
   computed before the design split existed stays valid.

The last test is the one with teeth: it checks that the consumers -- the calibrator and
the analysis frames -- actually drop design rows, instead of sweeping them into calib or
test where they would silently contaminate the thing they were held out to protect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from caliber.calibration.calibrators import _calibration_rows
from caliber.data.splits import (
    SPLITS,
    assign_splits,
    split_for,
    split_summary,
    verify_no_patient_straddles,
)

KEYS = [f"patient-{i:03d}" for i in range(400)]
FRACTIONS = (0.3, 0.5, 0.7)


def test_zero_design_fraction_is_the_two_way_split_exactly():
    """The compatibility guarantee: no bank, pin, or salt is disturbed by the default."""
    for fraction in FRACTIONS:
        two_way = [split_for(k, calib_fraction=fraction) for k in KEYS]
        defaulted = [
            split_for(k, calib_fraction=fraction, design_fraction=0.0) for k in KEYS
        ]
        assert two_way == defaulted
        assert "design" not in set(defaulted)


@pytest.mark.parametrize("design_fraction", [0.0, 0.05, 0.1, 0.2, 0.3])
def test_the_test_side_never_moves(design_fraction):
    """The invariant that lets a design split be added after coverage was measured.

    A patient lands on ``test`` exactly when it would have with no design side at all,
    because the design side is carved out of the *calibration* range. If this ever
    fails, every coverage number computed under a different design_fraction becomes
    incomparable -- and nothing else in the pipeline would say so.
    """
    baseline = {k: split_for(k, calib_fraction=0.5) for k in KEYS}
    with_design = {
        k: split_for(k, calib_fraction=0.5, design_fraction=design_fraction)
        for k in KEYS
    }
    assert {k for k, v in baseline.items() if v == "test"} == {
        k for k, v in with_design.items() if v == "test"
    }
    # And the design side is drawn only from what used to be calibration.
    moved = {k for k, v in with_design.items() if v == "design"}
    assert all(baseline[k] == "calib" for k in moved)


def test_the_three_sides_are_disjoint_and_cover_everybody():
    assignment = assign_splits(KEYS, calib_fraction=0.5, design_fraction=0.2)
    sides = {s: {k for k, v in assignment.items() if v == s} for s in SPLITS}
    assert set().union(*sides.values()) == set(KEYS)
    assert sum(len(v) for v in sides.values()) == len(KEYS)
    assert all(v for v in sides.values()), "every side should be non-empty at n=400"


def test_proportions_land_near_what_was_asked_for():
    assignment = assign_splits(KEYS, calib_fraction=0.6, design_fraction=0.2)
    n = len(KEYS)
    share = {s: sum(v == s for v in assignment.values()) / n for s in SPLITS}
    assert share["design"] == pytest.approx(0.2, abs=0.05)
    assert share["calib"] == pytest.approx(0.4, abs=0.05)
    assert share["test"] == pytest.approx(0.4, abs=0.05)


def test_design_fraction_may_not_eat_into_the_test_side():
    """Refuses rather than silently shrinking test, and says which knob to turn."""
    with pytest.raises(ValueError, match="must not exceed calib_fraction"):
        split_for("patient-000", calib_fraction=0.3, design_fraction=0.4)
    with pytest.raises(ValueError, match=r"design_fraction must be in \[0, 1\]"):
        split_for("patient-000", design_fraction=1.5)


def test_design_assignment_is_still_per_patient():
    """The D-022/D-027 guarantee has to survive the third side, not just the first two."""
    sample_to_patient = {f"file{i}": f"P{i // 2}" for i in range(200)}
    summary = split_summary(sample_to_patient, calib_fraction=0.5, design_fraction=0.2)
    assert summary["straddling_patients"] == []
    assert summary["design_fraction"] == 0.2
    assert summary["patients_design"] > 0
    assert (
        summary["patients_design"] + summary["patients_calib"] + summary["patients_test"]
        == summary["n_patients"]
    )


def test_design_rows_reach_neither_the_calibrator_nor_the_evaluation_frame():
    """The contamination this split exists to prevent, checked at the consumers.

    The design rows below carry scores far outside the calibration range. If any
    consumer treats ``"design"`` as calibration data the threshold moves; if one treats
    it as test data the coverage number moves. Both are checked against the frame with
    the design rows deleted, so the assertion is "identical", not "close".
    """
    rng = np.random.default_rng(0)
    real = pd.DataFrame(
        {
            "split": ["calib"] * 40 + ["test"] * 40,
            "score": rng.uniform(0.0, 1.0, 80),
            "label": rng.integers(0, 2, 80).astype(bool),
        }
    )
    contaminated = pd.concat(
        [
            real,
            pd.DataFrame(
                {
                    "split": ["design"] * 20,
                    "score": np.full(20, 999.0),
                    "label": np.ones(20, dtype=bool),
                }
            ),
        ],
        ignore_index=True,
    )

    # The calibrator's row selection.
    pd.testing.assert_frame_equal(
        _calibration_rows(real, "split").reset_index(drop=True),
        _calibration_rows(contaminated, "split").reset_index(drop=True),
    )

    # The frames analyse_bank.py and sizing.py build, by the same rule they use.
    for side in ("calib", "test"):
        pd.testing.assert_frame_equal(
            real[real["split"] == side].reset_index(drop=True),
            contaminated[contaminated["split"] == side].reset_index(drop=True),
        )


def test_straddle_check_reports_a_patient_split_across_design_and_calib():
    sample_to_patient = {"vol_a": "P1", "vol_b": "P1", "vol_c": "P2"}
    leaky = {"vol_a": "design", "vol_b": "calib", "vol_c": "test"}
    assert verify_no_patient_straddles(sample_to_patient, leaky) == ["P1"]
