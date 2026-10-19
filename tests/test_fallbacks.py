"""Gates for fallback [7], group-weighted calibration with truncation (T1.2, D58/D66).

Tool 2 fails on an ordered grid because the alpha-quantile is a lower-tail statistic, so it
is set by the lowest-scoring regime in the mixture, and an exponential kernel down-weights
smoothly without ever excluding (D52). [7]'s two answers are hard truncation and equal mass
per group, and the tests below pin each separately -- the second is the one that cannot be
reproduced by any choice of Tool 2's bandwidth, so it is what makes [7] a different method
rather than a retuning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from caliber.calibration import calibrate_weighted
from caliber.calibration.fallbacks import (
    calibrate_group_weighted,
    group_weights,
    nearest_certifiable_set,
)

NEAR, MID, FAR = "g_near", "g_mid", "g_far"
DISTANCES = {NEAR: 0.1, MID: 0.4, FAR: 0.9}


def _frame(sizes=None, offsets=None, seed: int = 0) -> pd.DataFrame:
    """Calibration regions in three groups, each group's scores shifted by ``offsets``.

    The near group's scores are the ones a correct method should follow; the far group's are
    deliberately much lower, which is the contamination D52 identified.
    """
    sizes = sizes or {NEAR: 200, MID: 200, FAR: 200}
    offsets = offsets or {NEAR: 1.0, MID: 0.7, FAR: 0.2}
    rng = np.random.default_rng(seed)
    parts = []
    for group, n in sizes.items():
        scores = offsets[group] + 0.10 * rng.normal(size=n)
        parts.append(
            pd.DataFrame(
                {
                    "calib_group": group,
                    "score": scores,
                    "label": rng.random(n) < 0.5,
                    "split": "calib",
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


# -- truncation ----------------------------------------------------------------


def test_a_group_outside_the_retained_set_gets_exactly_zero():
    """The thing a smooth kernel cannot do, and the whole point of [7]."""
    w = group_weights(DISTANCES, retain=[NEAR, MID])
    assert w[FAR] == 0.0
    assert w[NEAR] > 0 and w[MID] > 0
    assert pytest.approx(sum(w.values()), abs=1e-12) == 1.0


def test_the_nearest_group_carries_more_weight_than_a_farther_one():
    w = group_weights(DISTANCES)
    assert w[NEAR] > w[MID] > w[FAR] > 0


def test_uniform_drops_the_distance_ordering_on_purpose():
    """D-052 measured that a mask-family change saturates every registered distance, so on a
    cross-family grid the distances carry an ordering they do not really have."""
    w = group_weights(DISTANCES, uniform=True)
    assert len(set(round(v, 12) for v in w.values())) == 1


def test_radius_truncation_keeps_only_what_is_inside():
    report = calibrate_group_weighted(
        _frame(), deployment_group="deploy", distance_of=DISTANCES, radius=0.5,
        alpha=0.1, split_col="split",
    )
    assert set(report.truncation.retained) == {NEAR, MID}


def test_a_radius_that_retains_nothing_is_refused_with_the_nearest_distance():
    with pytest.raises(ValueError, match="nearest"):
        calibrate_group_weighted(
            _frame(), "deploy", DISTANCES, radius=0.01, alpha=0.1, split_col="split"
        )


# -- the distinctive property: mass per group, not per region ------------------


def test_a_large_distant_group_cannot_outvote_a_small_near_one_on_size():
    """[7]'s second change, and the one no bandwidth can reproduce.

    Tool 2 weights every *region*, so a group with ten times the regions contributes ten
    times the mass and a large distant regime outvotes a small near one on size alone. Here
    each retained group contributes a fixed share, divided among its own regions.
    """
    lopsided = _frame(sizes={NEAR: 20, MID: 20, FAR: 600})
    group_weighted = calibrate_group_weighted(
        lopsided, "deploy", DISTANCES, retain=[NEAR, MID, FAR], alpha=0.1, split_col="split"
    )
    tool2 = calibrate_weighted(
        lopsided, deployment_group="deploy", distance_of=DISTANCES, alpha=0.1,
        split_col="split",
    )
    near_scores = lopsided.loc[lopsided.calib_group == NEAR, "score"]
    gw = next(iter(group_weighted.thresholds.values()))
    t2 = next(iter(tool2.thresholds.values()))
    # [7] stays nearer the near group's own scale; Tool 2 is dragged toward the big far one.
    assert abs(gw - near_scores.mean()) < abs(t2 - near_scores.mean())


def test_with_equal_groups_and_no_truncation_it_tracks_tool_two():
    """A strict generalisation: nothing truncated and equal sizes should agree closely."""
    equal = _frame(sizes={NEAR: 200, MID: 200, FAR: 200})
    gw = next(iter(calibrate_group_weighted(
        equal, "deploy", DISTANCES, retain=list(DISTANCES), alpha=0.1, split_col="split"
    ).thresholds.values()))
    t2 = next(iter(calibrate_weighted(
        equal, deployment_group="deploy", distance_of=DISTANCES, alpha=0.1, split_col="split"
    ).thresholds.values()))
    assert abs(gw - t2) < 0.15


# -- the nearest-certifiable rule ---------------------------------------------


def test_it_stops_at_the_first_set_that_certifies():
    """D52's tension resolved rather than named: the nearest groups give the least
    contaminated tail, and D32's floor says when enough have been added."""
    choice = nearest_certifiable_set(_frame(), DISTANCES, alpha=0.1, split_col="split")
    assert choice.certifiable
    assert choice.retained == (NEAR,), "one group of 200 is plenty at alpha=0.1"
    assert not choice.exhausted


def test_it_adds_groups_in_distance_order_when_the_nearest_is_too_thin():
    thin = _frame(sizes={NEAR: 4, MID: 200, FAR: 200})
    choice = nearest_certifiable_set(thin, DISTANCES, alpha=0.1, split_col="split")
    assert choice.retained[0] == NEAR, "order is by distance, nearest first"
    assert len(choice.retained) >= 2, "the nearest alone cannot certify at this size"


def test_exhausting_every_group_reports_no_admissible_set():
    tiny = _frame(sizes={NEAR: 2, MID: 2, FAR: 2})
    choice = nearest_certifiable_set(tiny, DISTANCES, alpha=0.1, split_col="split")
    assert choice.exhausted and not choice.certifiable
    assert "no admissible calibration set" in choice.note()


def test_a_plentiful_set_does_not_degenerate_to_minus_infinity():
    """Regression for a scaling bug of my own (D66).

    A first version passed `1 / n_groups` as the test-point weight against group masses
    normalised to sum to one, making the test atom's own probability 0.33 against
    alpha = 0.10. The -inf atom then satisfies the quantile condition alone and every
    threshold came back -inf -- which reads as "the method refused to certify" rather than as
    a scaling error, and the note said so.
    """
    choice = nearest_certifiable_set(_frame(), DISTANCES, alpha=0.1, split_col="split")
    assert np.isfinite(choice.result.threshold), "the threshold must not be -inf here"
    assert choice.result.n_positives > 9


# -- refusals ------------------------------------------------------------------


def test_an_empty_retained_set_is_refused():
    with pytest.raises(ValueError, match="nothing to calibrate"):
        group_weights(DISTANCES, retain=[])


def test_a_retained_group_with_no_distance_is_refused():
    with pytest.raises(KeyError):
        group_weights(DISTANCES, retain=[NEAR, "g_unknown"])


def test_groups_with_distances_but_no_calibration_rows_are_refused():
    with pytest.raises(ValueError, match="calibration rows"):
        nearest_certifiable_set(
            _frame().assign(calib_group="something_else"), DISTANCES, alpha=0.1,
            split_col="split",
        )
