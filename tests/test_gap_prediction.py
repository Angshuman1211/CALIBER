"""T1-B: the gap curve, its baselines, and the three ways this comparison can flatter itself.

The claim T1-B defends is narrow and the objections to it are specific, so most of these
tests are about the *comparison* rather than the fit:

1. A baseline run through weaker machinery than the method is not a baseline. Every
   predictor here goes through the same isotonic fit on a different ``x``.
2. Two error rates computed on two different samples are not comparable. The acquisition
   parameter is undefined across families *by construction*, so an unrestricted comparison
   would score the distance on hard pairs and the baseline on easy ones.
3. 72 ordered pairs are 9 operators. An interval that resamples pairs describes a sample
   size the design does not have.

The last group of tests is behavioural: given data where the gap is driven by the knob and
not by the distance, the machinery must *decline* to report the distance as predictive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caliber.evaluation import gap as G
from caliber.evaluation import prereg

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_MANIFEST = REPO_ROOT / "runs" / "t1a_ct_distances" / "manifest.json"

#: A grid with the shape that matters: one family ordered by a knob, a second family whose
#: members are indistinguishable on null-space dimension, and no knob spanning the two.
META = {
    "sv.a": {"family": "sparse_view", "views": 180, "arc_deg": 180, "null_space_dim": 10},
    "sv.b": {"family": "sparse_view", "views": 90, "arc_deg": 180, "null_space_dim": 20},
    "sv.c": {"family": "sparse_view", "views": 30, "arc_deg": 180, "null_space_dim": 30},
    "la.a": {"family": "limited_angle", "views": 90, "arc_deg": 150, "null_space_dim": 20},
    "la.b": {"family": "limited_angle", "views": 90, "arc_deg": 90, "null_space_dim": 20},
}


def _pairs(include_diagonal: bool = False) -> pd.DataFrame:
    """Ordered pairs with a signed distance that is a clean total order on the grid."""
    index = {"sv.a": 0.0, "sv.b": 1.0, "sv.c": 2.0, "la.a": 0.5, "la.b": 1.5}
    rows = []
    for a in META:
        for b in META:
            if a == b and not include_diagonal:
                continue
            rows.append({
                "calibration": a,
                "deployment": b,
                "signed_distance": index[b] - index[a],
                "coverage": 0.90 - 0.08 * max(0.0, index[b] - index[a]),
            })
    return pd.DataFrame(rows)


# -- the fit ---------------------------------------------------------------------------


def test_pava_pools_the_violators_and_nothing_else():
    out = G._pava(np.array([1.0, 3.0, 2.0, 4.0]), np.ones(4))
    assert out.tolist() == [1.0, 2.5, 2.5, 4.0]


def test_an_already_monotone_sequence_is_returned_unchanged():
    y = np.array([0.1, 0.2, 0.5, 0.9])
    assert G._pava(y, np.ones(4)).tolist() == y.tolist()


def test_a_decreasing_sequence_collapses_to_its_mean():
    out = G._pava(np.array([4.0, 3.0, 2.0, 1.0]), np.ones(4))
    assert out.tolist() == [2.5] * 4


def test_the_fit_is_monotone_on_random_data():
    rng = np.random.default_rng(0)
    for _ in range(25):
        x = rng.normal(size=40)
        fit = G.isotonic(x, rng.normal(size=40))
        assert np.all(np.diff(fit.y) >= -1e-12)


def test_ties_in_x_become_one_knot_so_input_order_cannot_be_read_as_structure():
    """The null-space baseline ties four of nine operators, so this is load-bearing."""
    a = G.isotonic([0.0, 0.0, 0.0, 1.0], [5.0, 1.0, 3.0, 9.0])
    b = G.isotonic([0.0, 0.0, 0.0, 1.0], [1.0, 3.0, 5.0, 9.0])
    assert a.x.tolist() == [0.0, 1.0]
    assert a.y.tolist() == pytest.approx(b.y.tolist())
    assert a.y[0] == pytest.approx(3.0)


def test_prediction_is_clamped_outside_the_training_range():
    fit = G.isotonic([0.0, 1.0], [0.0, 1.0])
    assert fit.predict([-5.0])[0] == pytest.approx(0.0)
    assert fit.predict([5.0])[0] == pytest.approx(1.0)


def test_a_single_knot_predicts_its_own_value_everywhere():
    fit = G.isotonic([2.0, 2.0], [0.3, 0.5])
    assert fit.predict([-1.0, 2.0, 99.0]).tolist() == pytest.approx([0.4, 0.4, 0.4])


# -- the response ----------------------------------------------------------------------


def test_the_frozen_gap_definition_needs_the_matched_diagonal():
    """Reading one definition as the other would change the response under a result."""
    with pytest.raises(KeyError, match="matched diagonal"):
        G.gap_column(_pairs(include_diagonal=False))


def test_the_matched_gap_vanishes_on_the_diagonal():
    frame = _pairs(include_diagonal=True)
    gaps = G.gap_column(frame)
    diag = frame["calibration"] == frame["deployment"]
    assert gaps[diag].abs().max() == pytest.approx(0.0)


def test_the_two_gap_definitions_differ_by_the_matched_level():
    frame = _pairs(include_diagonal=True)
    matched = G.gap_column(frame, "matched_minus_transfer")
    nominal = G.gap_column(frame, "nominal_minus_transfer")
    offset = (nominal - matched).to_numpy()
    # per deployment operator the offset is NOMINAL - coverage(b -> b), a constant
    for _, part in frame.assign(off=offset).groupby("deployment"):
        assert part["off"].std(ddof=0) == pytest.approx(0.0, abs=1e-12)


def test_an_unnamed_gap_definition_is_refused():
    with pytest.raises(ValueError, match="not a gap definition"):
        G.gap_column(_pairs(True), "whatever_makes_the_number_work")


def test_a_deployment_operator_with_no_matched_cell_raises_rather_than_dropping():
    frame = _pairs(include_diagonal=True)
    frame = frame[~((frame["calibration"] == "sv.c") & (frame["deployment"] == "sv.c"))]
    with pytest.raises(KeyError, match="no matched cell"):
        G.gap_column(frame)


# -- the predictors --------------------------------------------------------------------


def test_the_acquisition_knob_is_undefined_across_families_by_construction():
    assert np.isnan(G.acquisition_parameter("sv.a", "la.a", META))
    assert G.acquisition_parameter("sv.a", "sv.c", META) == pytest.approx(150.0)
    assert G.acquisition_parameter("la.a", "la.b", META) == pytest.approx(60.0)


def test_null_space_dimension_cannot_separate_the_limited_angle_family():
    """D84's point: it is the only parameter baseline defined everywhere, and degenerate."""
    assert G.null_space_dimension("la.a", "la.b", META) == pytest.approx(0.0)
    assert G.null_space_dimension("sv.a", "sv.c", META) == pytest.approx(20.0)


def test_an_operator_off_the_grid_raises_instead_of_reading_as_out_of_scope():
    """A NaN here would move that pair out of the denominator silently."""
    with pytest.raises(KeyError, match="not in the operator grid"):
        G.acquisition_parameter("sv.a", "typo", META)
    with pytest.raises(KeyError, match="not in the operator grid"):
        G.null_space_dimension("typo", "sv.a", META)


def test_an_unknown_predictor_name_is_refused():
    with pytest.raises(ValueError, match="not a predictor"):
        G.predictor_column(_pairs(), "distance_that_works", META)


# -- the protocols ---------------------------------------------------------------------


def test_cross_family_trains_inside_one_family_and_tests_everything_else():
    pairs = _pairs()
    got = dict((name, (train.sum(), test.sum())) for name, train, test in
               G.folds(pairs, "cross_family", META))
    assert got["train_sparse_view"] == (6, len(pairs) - 6)      # 3 sv operators -> 6 pairs
    assert got["train_limited_angle"] == (2, len(pairs) - 2)    # 2 la operators -> 2 pairs
    for _, train, test in G.folds(pairs, "cross_family", META):
        assert not (train & test).any(), "a pair cannot be trained and tested at once"


def test_leave_one_operator_out_holds_out_every_pair_touching_that_operator():
    pairs = _pairs()
    for name, train, test in G.folds(pairs, "leave_one_operator_out", META):
        op = name.removeprefix("holdout_")
        touching = (pairs["calibration"] == op) | (pairs["deployment"] == op)
        assert test.tolist() == touching.tolist()
        assert not (train & test).any()
    assert len(G.folds(pairs, "leave_one_operator_out", META)) == len(META)


def test_an_unnamed_protocol_is_refused():
    with pytest.raises(ValueError, match="not a frozen protocol"):
        G.folds(_pairs(), "whichever_wins", META)


def test_pair_scope_splits_within_from_across_family():
    scope = G.pair_scope(_pairs(), META)
    assert set(scope) == {"within_family", "cross_family"}
    assert int((scope == "cross_family").sum()) == 2 * 3 * 2


# -- the comparison --------------------------------------------------------------------


def _fitted(gaps=None):
    pairs = _pairs(include_diagonal=True)
    pairs = pairs[pairs["calibration"] != pairs["deployment"]].reset_index(drop=True)
    frame = _pairs(include_diagonal=True)
    pairs["gap"] = (
        G.gap_column(frame)[frame["calibration"] != frame["deployment"]].to_numpy()
        if gaps is None else gaps
    )
    return pairs


def test_the_comparison_is_restricted_to_pairs_both_predictors_can_score():
    out = G.gap_prediction(_fitted(), META, n_boot=50, seed=0)
    comp = out["comparisons"]
    acq = comp[(comp["baseline"] == "acquisition_parameter")
               & (comp["scope"] == "cross_family")]
    assert (acq["verdict"] == "undefined_here").all()
    assert (acq["n_dropped"] > 0).all(), "the dropped rows must be counted, not hidden"


def test_the_verdict_is_conjunctive():
    """One surviving baseline withholds the claim: the objection is 'any knob would do'."""
    out = G.gap_prediction(_fitted(), META, n_boot=50, seed=0)
    for _, row in out["verdict"].iterrows():
        assert row["reportable"] == (row["n_baselines_defined"] > 0 and not row["unbeaten"])


def test_both_family_scopes_are_always_reported():
    assert prereg.GAP_PREDICTION_REPORT_BY_FAMILY_SCOPE is True
    out = G.gap_prediction(_fitted(), META, n_boot=50, seed=0)
    for frame in (out["accuracy"], out["verdict"]):
        assert set(frame["scope"]) == {"within_family", "cross_family"}


def test_both_directions_of_the_cross_family_fit_are_reported_separately():
    """'Predict limited-angle, then reverse' is two claims; averaging them hides which held."""
    out = G.gap_prediction(_fitted(), META, n_boot=50, seed=0)
    folds = set(out["accuracy_by_fold"].query("protocol == 'cross_family'")["fold"])
    assert folds == {"train_sparse_view", "train_limited_angle"}


def test_extrapolation_is_reported_because_a_clamped_fit_is_a_constant():
    out = G.gap_prediction(_fitted(), META, n_boot=50, seed=0)
    assert "extrapolated" in out["accuracy"].columns
    const = out["accuracy"].query("predictor == 'constant'")
    assert (const["extrapolated"] == 0.0).all(), "a constant cannot extrapolate"


# -- the interval ----------------------------------------------------------------------


def test_the_bootstrap_resamples_operators_not_pairs():
    """Nine operators, 72 pairs. Weighting pairs by the product is what makes it dyadic."""
    rng = np.random.default_rng(3)
    counts = G.operator_weights(rng, sorted(META))
    assert sum(counts.values()) == len(META)
    assert all(v >= 0 for v in counts.values())
    assert prereg.GAP_CLUSTER_UNIT == "operator"


def test_an_operator_drawn_twice_doubles_its_pairs_in_that_replicate():
    err = np.array([1.0, 1.0])
    assert G._weighted_mae(err, np.array([2.0, 1.0])) == pytest.approx(1.0)
    assert G._weighted_mae(np.array([1.0, 0.0]), np.array([2.0, 1.0])) == pytest.approx(2 / 3)


def test_a_zero_weight_replicate_does_not_crash_the_interval():
    assert np.isnan(G._weighted_mae(np.array([1.0]), np.array([0.0])))


# -- behaviour: it must be able to say no ----------------------------------------------


def test_the_distance_wins_when_the_gap_really_is_monotone_in_it():
    pairs = _fitted()
    rng = np.random.default_rng(1)
    pairs["gap"] = (
        0.4 / (1 + np.exp(-2.0 * pairs["signed_distance"].to_numpy()))
        + rng.normal(0, 0.004, len(pairs))
    )
    out = G.gap_prediction(pairs, META, n_boot=400, seed=2)
    comp = out["comparisons"].query(
        "protocol == 'leave_one_operator_out' and baseline == 'constant'"
    )
    assert (comp["delta_mae"] > 0).all()
    assert comp["beaten"].any()


def test_the_distance_is_not_credited_when_the_knob_explains_the_gap():
    """The test that matters: given knob-driven gaps, the machinery must decline."""
    pairs = _fitted()
    rng = np.random.default_rng(4)
    knob = np.array([
        float(META[a]["views"] - META[b]["views"])
        for a, b in zip(pairs["calibration"], pairs["deployment"])
    ])
    pairs["gap"] = 0.001 * knob + rng.normal(0, 0.02, len(pairs))
    out = G.gap_prediction(pairs, META, n_boot=400, seed=5)
    within = out["verdict"].query("scope == 'within_family'")
    assert not within["reportable"].all(), (
        "knob-driven gaps must not be reported as a win for the distance everywhere"
    )


# -- the real grid ---------------------------------------------------------------------


def test_the_real_ct_grid_has_the_degeneracy_d84_claims():
    if not REAL_MANIFEST.exists():
        pytest.skip("the T1-0 distance run is not present")
    meta = G.operator_metadata(REAL_MANIFEST)
    assert len(meta) == 9
    dims = [m["null_space_dim"] for m in meta.values()]
    assert sum(1 for d in dims if d == 64000) == 4, "D84: four operators at 120 views tie"
    la = [o for o, m in meta.items() if m["family"] == "limited_angle"]
    assert len({meta[o]["null_space_dim"] for o in la}) == 1, "all limited-angle tie"
    for a in la:
        for b in meta:
            if meta[b]["family"] == "sparse_view":
                assert np.isnan(G.acquisition_parameter(a, b, meta))
