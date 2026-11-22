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


# -- the direction, and why a reversal must be visible ---------------------------------
#
# The frozen form is a NON-DECREASING fit. Fitted to a decreasing truth it returns the
# training mean, so before D89 a perfectly reversed relation and no relation at all
# produced the same output. These tests pin the difference and pin that showing the
# reversal cannot change a verdict.


def test_the_frozen_direction_reports_a_perfect_reversal_as_no_information():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [4.0, 3.0, 2.0, 1.0]
    frozen = G.isotonic(x, y)
    assert frozen.direction == prereg.GAP_PREDICTION_DIRECTION == "increasing"
    # every knot equal to the mean: indistinguishable from fitting pure noise
    assert np.allclose(frozen.y, np.mean(y))
    assert np.isclose(G.rank_correlation(x, y), -1.0)


def test_the_opposite_direction_recovers_what_the_frozen_one_flattens():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [4.0, 3.0, 2.0, 1.0]
    flipped = G.isotonic(x, y, "decreasing")
    assert np.allclose(flipped.y, y)
    assert np.all(np.diff(flipped.y) <= 0)


def test_the_opposite_direction_is_the_l2_optimal_non_increasing_fit():
    rng = np.random.default_rng(3)
    x = np.arange(12, dtype=float)
    y = -0.4 * x + rng.normal(scale=0.3, size=12)
    flipped = G.isotonic(x, y, "decreasing")
    assert np.all(np.diff(flipped.y) <= 1e-12)
    # no non-increasing step function on this grid beats it in squared error
    best = float(np.sum((flipped.predict(x) - y) ** 2))
    for _ in range(200):
        cand = np.sort(rng.permutation(flipped.y))[::-1]
        assert float(np.sum((cand - y) ** 2)) >= best - 1e-9


def test_an_unnamed_direction_is_refused():
    with pytest.raises(ValueError, match="not a frozen direction"):
        G.isotonic([1.0, 2.0], [1.0, 2.0], "u_shaped")


def test_the_reversal_frame_names_the_reversal_without_touching_the_verdict():
    pairs = _pairs(include_diagonal=True)
    # a world where the gap FALLS with the distance: the CT bank's shape
    pairs["coverage"] = 0.90 + 0.05 * pairs["signed_distance"]
    pairs["gap"] = G.gap_column(pairs)
    out = G.gap_prediction(pairs, META, n_boot=200, seed=0)

    rev = out["reversal"]
    assert not rev.empty
    assert (rev["rho_distance_gap"] < 0).all()
    assert rev["opposite_fits_better"].any()

    # the verdict is built from the frozen direction alone
    assert not out["verdict"]["reportable"].any()
    flipped_rows = out["accuracy"][~out["accuracy"]["pre_registered"]]
    assert not flipped_rows.empty
    assert (flipped_rows["direction"] == "decreasing").all()
    assert set(out["comparisons"]["scope"]) <= {"within_family", "cross_family"}


def test_showing_the_reversal_cannot_turn_a_withheld_claim_into_a_reported_one():
    pairs = _pairs(include_diagonal=True)
    pairs["coverage"] = 0.90 + 0.05 * pairs["signed_distance"]
    pairs["gap"] = G.gap_column(pairs)
    out = G.gap_prediction(pairs, META, n_boot=200, seed=0)
    verdict = out["verdict"].set_index(["protocol", "scope"])["reportable"]
    # the opposite-direction fit is near-perfect here, and changes no verdict
    assert not verdict.any()


# -- the threshold identity: a check on the bank, never a predictor --------------------


def _threshold_bank(n: int = 4000, reverse: bool = False) -> pd.DataFrame:
    """Pairs whose coverages are *computed from* thresholds, as a real bank's are.

    The score scale grows with ill-posedness, so tau does too -- which is the whole of
    D89's account of the sign reversal.
    """
    rng = np.random.default_rng(11)
    order = list(META)
    scales = np.linspace(1.0, 3.0, len(order))
    if reverse:
        scales = scales[::-1]
    scores = {op: rng.gamma(2.0, s, size=n) for op, s in zip(order, scales)}
    tau = {op: float(np.quantile(scores[op], prereg.ALPHA)) for op in order}
    # the distance always points the same way: positive when b is further down the grid
    index = {op: float(i) for i, op in enumerate(order)}
    rows = []
    for a in order:
        for b in order:
            rows.append({
                "calibration": a,
                "deployment": b,
                "signed_distance": index[b] - index[a],
                "threshold": tau[a],
                "coverage": float((scores[b] >= tau[a]).mean()),
            })
    pairs = pd.DataFrame(rows)
    pairs["gap"] = G.gap_column(pairs)
    return pairs


def test_the_gap_sign_is_the_threshold_difference_at_every_pair():
    pairs = _threshold_bank()
    check = G.gap_sign_from_thresholds(pairs)
    live = check[check["decidable"]]
    assert len(live) >= 18
    assert live["agrees"].all(), "sign(gap) == sign(tau_a - tau_b) is a theorem"


def test_the_identity_catches_a_bank_that_contradicts_its_own_coverage():
    pairs = _threshold_bank()
    # sv.a is the least ill-posed operator, so tau_a < tau_c and the identity forces
    # gap < 0 at sv.a -> sv.c. Put the transfer cell BELOW its matched cell instead.
    bad = (pairs["calibration"] == "sv.a") & (pairs["deployment"] == "sv.c")
    matched = float(pairs.loc[
        (pairs["calibration"] == "sv.c") & (pairs["deployment"] == "sv.c"), "coverage"
    ].iloc[0])
    pairs.loc[bad, "coverage"] = matched - 0.2
    pairs["gap"] = G.gap_column(pairs)
    assert float(pairs.loc[bad, "gap"].iloc[0]) > 0
    check = G.gap_sign_from_thresholds(pairs)
    row = check[(check["calibration"] == "sv.a") & (check["deployment"] == "sv.c")]
    assert bool(row["decidable"].iloc[0])
    assert not bool(row["agrees"].iloc[0])


def test_a_calibration_operator_with_two_thresholds_raises():
    pairs = _threshold_bank()
    idx = pairs.index[pairs["calibration"] == "sv.a"][0]
    pairs.loc[idx, "threshold"] = float(pairs.loc[idx, "threshold"]) * 1.5
    with pytest.raises(ValueError, match="differing thresholds"):
        G.gap_sign_from_thresholds(pairs)


def test_the_identity_refuses_the_nominal_gap_definition():
    pairs = _threshold_bank()
    with pytest.raises(ValueError, match="matched_minus_transfer"):
        G.gap_sign_from_thresholds(pairs, definition="nominal_minus_transfer")


def test_the_identity_is_not_available_as_a_baseline():
    # the bar exists so a tautology cannot be entered as a competitor
    for barred in prereg.GAP_BARRED_PREDICTORS:
        assert barred not in prereg.GAP_PREDICTION_BASELINES
        assert barred not in G.PREDICTORS


def test_the_threshold_ordering_is_what_sets_the_sign_of_the_relation():
    """D89's account as a test, and the reasoning behind the face-bank prediction.

    The identity fixes the gap's relation to *tau*; nothing in it fixes the gap's relation
    to the *distance*. The sign of the measured relation is therefore set entirely by which
    way tau runs along the grid -- so two banks can disagree on the sign while agreeing on
    the mechanism, which is what :data:`prereg.GAP_FACE_BANK_PREDICTION` asserts.
    """
    rising = _threshold_bank(reverse=False)    # tau grows with ill-posedness, as on CT
    falling = _threshold_bank(reverse=True)    # tau shrinks with it, as predicted on faces

    def rhos(bank: pd.DataFrame) -> tuple[float, float]:
        off = bank[bank["calibration"] != bank["deployment"]]
        return (
            G.rank_correlation(off["signed_distance"], off["threshold"]),
            G.rank_correlation(off["signed_distance"], off["gap"]),
        )

    rho_tau_up, rho_gap_up = rhos(rising)
    rho_tau_dn, rho_gap_dn = rhos(falling)

    # A high tau_a goes with a calibration operator late in the grid, which is where the
    # signed distance is most negative -- so the two correlations must share their sign.
    assert rho_tau_up < 0 and rho_gap_up < 0, (rho_tau_up, rho_gap_up)
    assert rho_tau_dn > 0 and rho_gap_dn > 0, (rho_tau_dn, rho_gap_dn)

    # The mechanism is one mechanism; only the threshold ordering differs.
    assert rho_gap_up < 0 < rho_gap_dn


# -- the pre-registration record itself ------------------------------------------------


def test_amendments_stay_in_chronological_order():
    """They print as a provenance line, so an out-of-order entry is a visible defect."""
    dates = [d for d, _ in prereg.AMENDMENTS]
    assert dates == sorted(dates)


def test_the_face_bank_prediction_is_recorded_before_the_number_exists():
    assert len(prereg.GAP_FACE_BANK_PREDICTION) >= 3
    assert any("falsified" in line for line in prereg.GAP_FACE_BANK_PREDICTION)


# --------------------------------------------------------------------------- #
# D91: the direction is read off the thresholds, and the arms' tables are tied
# --------------------------------------------------------------------------- #


def test_the_direction_rule_reads_ct_and_face_orderings_oppositely():
    """One rule, two banks, two answers -- which is why a constant could not work."""
    ct = G.direction_from_thresholds(_threshold_bank(), bank="synthetic_ct")
    face = G.direction_from_thresholds(_threshold_bank(reverse=True),
                                       bank="synthetic_face")
    assert ct.direction == "decreasing"
    assert face.direction == "increasing"
    assert ct.rho > 0 > face.rho


def test_the_direction_rule_agrees_with_the_sign_the_gaps_actually_have():
    for reverse in (False, True):
        pairs = _threshold_bank(reverse=reverse)
        call = G.direction_from_thresholds(pairs)
        moving = pairs[pairs["calibration"] != pairs["deployment"]]
        rho_gap = G.rank_correlation(moving["signed_distance"], moving["gap"])
        truth = "decreasing" if rho_gap < 0 else "increasing"
        assert call.direction == truth, (reverse, call.rho, rho_gap)


def test_the_direction_rule_never_touches_a_coverage_or_a_gap():
    """Scramble both outcome columns; the call must not move."""
    pairs = _threshold_bank()
    before = G.direction_from_thresholds(pairs)
    wrecked = pairs.copy()
    wrecked["coverage"] = 0.5
    wrecked["gap"] = -wrecked["gap"]
    after = G.direction_from_thresholds(wrecked)
    assert (after.direction, after.rho) == (before.direction, before.rho)


def test_the_rule_is_not_reportable_on_a_bank_whose_gaps_were_already_seen():
    pairs = _threshold_bank()
    barred = prereg.GAP_DIRECTION_BANKS_BARRED[0]
    seen = G.direction_from_thresholds(
        pairs, tau_noise=1e-6, bank=barred, meta=META)
    fresh = G.direction_from_thresholds(
        pairs, tau_noise=1e-6, bank="face_prior", meta=META)
    assert seen.determinate is True and seen.reportable is False
    assert fresh.determinate is True and fresh.reportable is True


def test_an_undetermined_tau_ordering_is_not_reportable_on_any_bank():
    pairs = _threshold_bank()
    span = pairs["threshold"].max() - pairs["threshold"].min()
    loud = G.direction_from_thresholds(pairs, tau_noise=float(span), bank="face_prior")
    assert loud.determinate is False
    assert loud.reportable is False
    assert loud.tau_range / span < prereg.GAP_DIRECTION_DETERMINACY_MIN_RATIO


def test_the_determinacy_verdict_is_withheld_until_the_noise_is_measured():
    call = G.direction_from_thresholds(_threshold_bank(), bank="face_prior")
    assert call.tau_noise is None
    assert call.determinate is None
    assert call.reportable is False


def _two_arms(warp: float, tau_shift: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two label definitions over one operator grid, as registry/across-split are."""
    rng = np.random.default_rng(5)
    order = list(META)
    index = {op: float(i) for i, op in enumerate(order)}
    scales = np.linspace(1.0, 3.0, len(order))
    base = {op: rng.gamma(2.0, s, size=4000) for op, s in zip(order, scales)}
    out = []
    for mult, shift in ((1.0, 0.0), (warp, tau_shift)):
        scores = {op: base[op] * mult for op in order}
        tau = {op: float(np.quantile(base[op], prereg.ALPHA)) * (1.0 + shift)
               for op in order}
        rows = []
        for a in order:
            for b in order:
                rows.append({
                    "calibration": a,
                    "deployment": b,
                    "signed_distance": index[b] - index[a],
                    "threshold": tau[a],
                    "coverage": float((scores[b] >= tau[a]).mean()),
                })
        frame = pd.DataFrame(rows)
        frame["gap"] = G.gap_column(frame)
        out.append(frame)
    return out[0], out[1]


def test_the_gap_difference_table_is_the_level_difference_table_minus_its_diagonal():
    a, b = _two_arms(warp=1.3, tau_shift=0.02)
    moving_a = a[a["calibration"] != a["deployment"]]
    moving_b = b[b["calibration"] != b["deployment"]]
    diag_a = a[a["calibration"] == a["deployment"]]
    diag_b = b[b["calibration"] == b["deployment"]]
    got = G.arm_difference_identity(moving_a, moving_b, diag_a, diag_b)
    assert got["residual"] < 1e-12, got["residual"]


def _identity(warp: float, tau_shift: float) -> dict[str, float]:
    a, b = _two_arms(warp=warp, tau_shift=tau_shift)
    return G.arm_difference_identity(
        a[a["calibration"] != a["deployment"]],
        b[b["calibration"] != b["deployment"]],
        a[a["calibration"] == a["deployment"]],
        b[b["calibration"] == b["deployment"]],
    )


def test_the_correlation_bound_holds_without_an_equal_variance_assumption():
    """The bound must sit below the measured correlation on this fixture.

    It is not a tight bound here *because* the fixture's two displacements have wildly
    unequal spreads -- which is the whole reason the bound is used instead of inverting
    the ratio directly.
    """
    got = _identity(warp=1.3, tau_shift=0.02)
    assert got["common_mode_correlation_lower_bound"] <= got["common_mode_correlation"]


def test_inverting_the_ratio_directly_would_overstate_the_correlation():
    got = _identity(warp=1.3, tau_shift=0.02)
    naive = 1.0 - got["sd_difference_ratio"] ** 2 / 2.0   # the equal-variance shortcut
    assert naive > got["common_mode_correlation"] + 0.1, (naive, got)


def test_the_bound_is_what_the_algebra_says_and_is_never_breached():
    for s_ratio in (0.0, 0.2, 0.5, 0.75, 0.95, 1.0):
        bound = np.sqrt(max(1.0 - min(s_ratio, 1.0) ** 2, 0.0))
        # every (spread ratio, correlation) pair reaching this sd ratio has r >= bound
        for s in np.linspace(0.01, 4.0, 60):
            r = (s * s + 1.0 - s_ratio * s_ratio) / (2.0 * s)
            if -1.0 <= r <= 1.0:
                assert r >= bound - 1e-9, (s_ratio, s, r, bound)


def test_at_zero_common_mode_the_gap_difference_is_no_smaller_than_the_level():
    """The statistic D86 needs can fail; this is the value at which it does."""
    got = _identity(warp=1.3, tau_shift=0.02)
    assert got["ratio_at_zero_common_mode"] == 1.0
    assert -1.0 <= got["common_mode_correlation"] <= 1.0


def test_gap_sign_agreement_between_arms_survives_total_non_cancellation():
    """Why the 72-of-72 count is barred from D86's support set.

    The two arms here share a threshold vector exactly and differ only in the score
    scale, so every gap difference is large and nothing cancels -- yet the sign agrees
    everywhere, because by D89 both signs are ``sign(tau_a - tau_b)``.
    """
    a, b = _two_arms(warp=4.0, tau_shift=0.0)
    moving_a = a[a["calibration"] != a["deployment"]].reset_index(drop=True)
    moving_b = b[b["calibration"] != b["deployment"]].reset_index(drop=True)
    agree = int(np.sum(np.sign(moving_a["gap"]) == np.sign(moving_b["gap"])))
    assert agree == len(moving_a)
    got = G.arm_difference_identity(moving_a, moving_b)
    assert got["mean_abs_gap_difference"] > 0.05, got


def test_gap_sign_agreement_breaks_under_perfect_cancellation_if_tau_transposes():
    """The other half: the count falls when nothing at all fails to cancel."""
    pairs = _threshold_bank()
    moving = pairs[pairs["calibration"] != pairs["deployment"]].reset_index(drop=True)
    order = list(META)
    tau = {op: float(pairs[pairs["calibration"] == op]["threshold"].iloc[0])
           for op in order}
    swapped = dict(tau)
    swapped[order[3]], swapped[order[4]] = tau[order[4]], tau[order[3]]
    sign_a = np.sign(moving["calibration"].map(tau).to_numpy()
                     - moving["deployment"].map(tau).to_numpy())
    sign_b = np.sign(moving["calibration"].map(swapped).to_numpy()
                     - moving["deployment"].map(swapped).to_numpy())
    assert int(np.sum(sign_a == sign_b)) < len(moving)


def test_the_arm_sign_agreement_bar_names_its_replacement():
    text = " ".join(prereg.GAP_ARM_SIGN_AGREEMENT_BARRED)
    assert "D86" in text
    assert "level_difference" in text


def test_amendment_seven_admits_that_it_widens_what_may_be_claimed():
    widening = [a for a in prereg.AMENDMENTS if "WIDENS" in a[1]]
    assert len(widening) == 1
    assert "GAP_DIRECTION_BANKS_BARRED" in widening[0][1]


# -- amendment ten: the direction is read within family cells, not pooled ------


def _simpson_bank() -> tuple[pd.DataFrame, dict]:
    """A bank whose pooled rho is ~0 while every family cell agrees -- the CT pattern.

    Six sparse-view operators and three limited-angle, with the limited-angle thresholds
    offset upward as they are on the real bank. Within each family tau rises with the
    distance; pooled, the between-family level difference swamps it.
    """
    sv = [f"sv.{i}" for i in range(6)]
    la = [f"la.{i}" for i in range(3)]
    order = sv + la
    meta = {op: {"family": "sparse_view" if op.startswith("sv") else "limited_angle",
                 "views": 180 - 15 * i, "arc_deg": 180, "null_space_dim": 10 + i}
            for i, op in enumerate(order)}
    tau = dict(zip(order, [0.00, -0.08, -0.04, 0.03, 0.06, 0.09, 0.40, 0.45, 0.50]))
    dist = dict(zip(order, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.15, 0.25, 0.35]))
    rows = [{"calibration": a, "deployment": b,
             "signed_distance": dist[b] - dist[a], "threshold": tau[a]}
            for a in order for b in order if a != b]
    return pd.DataFrame(rows), meta


def test_a_washed_out_pooled_rho_does_not_block_a_unanimous_cell_verdict():
    """Amendment ten. Track 2 found pooled +0.05 against cells +0.55 to +1.00."""
    pairs, meta = _simpson_bank()
    call = G.direction_from_thresholds(
        pairs, tau_noise=1e-6, bank="face_prior", meta=meta)
    assert abs(call.pooled_rho) < 0.25, (
        "fixture must reproduce the washed-out pool, got %.3f" % call.pooled_rho)
    cells = [r for _, r in call.cell_rho]
    assert len(cells) >= 3
    assert all(r > 0.4 for r in cells), cells
    assert call.cells_unanimous is True
    assert call.reportable is True, "unanimous cells must survive a washed-out pool"
    assert call.direction == "decreasing", "cells are positive, so the gap falls"


def test_the_pooled_rho_would_have_returned_the_other_direction():
    """The point of the amendment: the pool and the cells disagree on this fixture."""
    pairs, meta = _simpson_bank()
    pooled = G.direction_from_thresholds(pairs, tau_noise=1e-6, bank="face_prior")
    celled = G.direction_from_thresholds(
        pairs, tau_noise=1e-6, bank="face_prior", meta=meta)
    assert pooled.direction != celled.direction, (
        "fixture no longer discriminates the two readings")
    assert pooled.reportable is False, "a pooled-only call may never be reported"


def test_the_direction_rule_is_not_reportable_without_the_family_cells():
    pairs = _threshold_bank()
    call = G.direction_from_thresholds(pairs, tau_noise=1e-6, bank="face_prior")
    assert call.determinate is True, "tau's range still clears its noise"
    assert call.cells_unanimous is None, "no meta, so unanimity is unknown"
    assert call.reportable is False, "unknown is not the same as satisfied"


def test_cells_that_disagree_withhold_the_direction():
    pairs, meta = _simpson_bank()
    flipped = pairs.copy()
    sel = flipped["calibration"].str.startswith("la")
    flipped.loc[sel, "signed_distance"] = -flipped.loc[sel, "signed_distance"]
    call = G.direction_from_thresholds(
        flipped, tau_noise=1e-6, bank="face_prior", meta=meta)
    assert call.cells_unanimous is False
    assert call.reportable is False


def test_the_determinacy_ratio_alone_was_not_sufficient():
    """Both banks clear the ratio; the CT bank cleared it with an unresolved rho."""
    pairs, meta = _simpson_bank()
    call = G.direction_from_thresholds(
        pairs, tau_noise=1e-6, bank="face_prior", meta=meta)
    assert call.tau_range / 1e-6 > prereg.GAP_DIRECTION_DETERMINACY_MIN_RATIO
    assert call.determinate is True
    assert prereg.GAP_DIRECTION_DETERMINACY_SCOPE == "within_family_cell"


def test_amendment_ten_narrows_rather_than_widens():
    entry = [t for d, t in prereg.AMENDMENTS if "GAP_DIRECTION_DETERMINACY_SCOPE" in t]
    assert len(entry) == 1
    assert "NARROWS WHAT MAY BE CLAIMED" in entry[0]
    assert "both banks pass it" in entry[0]
    widening = [t for _, t in prereg.AMENDMENTS if "WIDENS WHAT MAY BE CLAIMED" in t]
    assert len(widening) == 1, "only amendment seven may widen"
