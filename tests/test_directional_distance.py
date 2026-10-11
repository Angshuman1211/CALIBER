"""Gates for the signed null-space growth (T1.3, D47).

D-052 measured that four of the five registered distances are the *same function* on MRI
mask operators — agreeing to 1.1e-16 — and that a mask-family change saturates all of them
at ~0.65, non-monotonically in acceleration. Its conclusion was that the MRI grid cannot
rank distance measures and the comparison must wait for CT.

The missing observation is that all five are **metrics**, and the coverage gap is not a
symmetric function of the operator pair: the same two operators in the two directions give
opposite-signed gaps with non-overlapping intervals. So a sixth quantity was needed, and it
is not another metric.

These tests pin what makes it worth having, and what it cannot do.
"""

from __future__ import annotations

import pytest

from caliber.operators.distance import (
    SIGNED_MEASURES,
    compute_distance,
    distance_matrix,
    null_space_growth,
)
from caliber.operators.mri import MRIUndersamplingOperator

FAMILIES = ("random1d", "equispaced1d", "variable_density2d")


def op(family: str = "random1d", acceleration: float = 4.0, size: int = 128, seed: int = 0):
    return MRIUndersamplingOperator(
        (size, size), family=family, acceleration=acceleration,
        center_fraction=0.08, mask_seed=seed,
    )


def growth(a, b) -> float:
    return compute_distance(a, b, measure="null_space_growth", normalise="ambient")


def symmetric(a, b) -> float:
    return compute_distance(a, b, measure="null_space_projection", normalise="ambient")


# -- the property the symmetric measures cannot have ---------------------------


def test_the_measure_is_antisymmetric_where_the_metrics_are_symmetric():
    """The whole reason it exists: direction survives into the number."""
    low, high = op(acceleration=2), op(acceleration=8)
    assert growth(low, high) == pytest.approx(-growth(high, low), abs=1e-12)
    assert growth(low, high) > 0, "deploying to a more aggressive operator hides more"
    assert symmetric(low, high) == pytest.approx(symmetric(high, low), abs=1e-12)


def test_nested_masks_reproduce_the_symmetric_measure_up_to_sign():
    """D19's interpretation survives on the graded within-family axis.

    Where the deployment operator samples a subset of the calibration operator's lines,
    nothing is newly revealed, so the signed measure's magnitude *is* the ratified
    distance. This is what makes the new axis a restoration of direction rather than a
    different quantity.
    """
    low, high = op(acceleration=2), op(acceleration=8)
    assert abs(growth(low, high)) == pytest.approx(symmetric(low, high), rel=1e-9)


def test_it_never_exceeds_the_symmetric_distance():
    """Why T3.3's bound is untouched: a bound in the metric stays valid here.

    ``|growth| <= null_space_projection`` always, so a Lipschitz bound proved against the
    symmetric measure still bounds the gap when the gap is plotted against this one — it is
    merely loose in the benign direction.
    """
    for family in FAMILIES:
        for a in (2, 4, 8):
            for b in (2, 4, 8):
                left, right = op(acceleration=a), op(family=family, acceleration=b)
                assert abs(growth(left, right)) <= symmetric(left, right) + 1e-9


# -- what it fixes, and what it does not ---------------------------------------


def test_it_orders_the_cross_family_grid_that_the_symmetric_axis_saturates():
    """D-052's saturation, and the de-saturation.

    From `random1d R2`, the six cross-family deployment operators sit within 0.05 of each
    other on the symmetric axis and are not even monotone in acceleration. On the signed
    axis they span the full range and are monotone.
    """
    cal = op(family="random1d", acceleration=2, size=256)
    grid = [op(family=f, acceleration=r, size=256) for f in FAMILIES[1:] for r in (2, 4, 8)]
    sym = [symmetric(cal, o) for o in grid]
    sgn = [growth(cal, o) for o in grid]
    assert max(sym) - min(sym) < 0.10, "the symmetric axis is supposed to be saturated here"
    assert max(sgn) - min(sgn) > 0.50
    assert (max(sgn) - min(sgn)) > 5 * (max(sym) - min(sym))


def test_acceleration_is_monotone_on_the_signed_axis_within_every_family():
    cal = op(family="random1d", acceleration=2, size=256)
    for family in FAMILIES:
        values = [growth(cal, op(family=family, acceleration=r, size=256)) for r in (2, 4, 8)]
        assert values == sorted(values), f"{family}: {values}"


def test_equal_line_counts_give_exactly_zero_whatever_the_pattern():
    """The limitation, stated as a test rather than left to be discovered.

    Two masks that keep the *same number* of lines in different places hide as much as they
    reveal, so the net is exactly zero — the measure collapses the family axis entirely. It
    therefore orders the grid but is not injective on it, and the honest reading is that
    operator shift here has two components: how much is newly hidden (graded, directional)
    and whether the pattern changed (categorical). This measure carries only the first.

    The prediction this licensed — equal gaps at matched acceleration — was **tested against
    the Week 4 grid and falsified**: at growth 0 the gaps are +0.038, +0.142 and +0.070, a
    spread of 0.104 against a total gap range of 0.242. The measure is kept because it
    orders the grid far better than the symmetric axis (Spearman −0.751 against +0.305, and
    the symmetric one does not even have a coherent sign), not because the family change is
    negligible. See D47.
    """
    cal = op(family="random1d", acceleration=4, size=256)
    for family in FAMILIES[1:]:
        assert growth(cal, op(family=family, acceleration=4, size=256)) == pytest.approx(
            0.0, abs=1e-12
        )


def test_an_operator_against_itself_is_zero():
    assert growth(op(), op()) == pytest.approx(0.0, abs=1e-12)


# -- guards --------------------------------------------------------------------


def test_a_signed_measure_is_refused_as_a_symmetric_matrix():
    """Mirroring an antisymmetric quantity would record half the pairs backwards."""
    ops = [op(acceleration=r) for r in (2, 4, 8)]
    assert "null_space_growth" in SIGNED_MEASURES
    with pytest.raises(ValueError):
        distance_matrix(ops, measure="null_space_growth")
    # The symmetric ones still work, so the guard is specific rather than a blanket.
    assert distance_matrix(ops, measure="null_space_projection").shape == (3, 3)


def test_it_is_registered_under_its_name_so_a_bank_can_record_it():
    ops = (op(acceleration=2), op(acceleration=8))
    assert compute_distance(*ops, measure="null_space_growth") == pytest.approx(
        null_space_growth(*ops), rel=1e-12
    )
