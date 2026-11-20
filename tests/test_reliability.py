"""T1-D: the Beta law, the worst-cell bound, and what the right spread reference actually is.

The load-bearing tests here are :func:`test_measured_spread_exceeds_the_closed_form_beta_sd`
and :func:`test_the_correction_formula_tracks_the_simulation`. They exist because the first
version of this module asserted the wrong thing: that overlapping splits would make the
observed spread come out *below* the closed-form Beta SD. These tests measured it instead,
and the dominant effect runs the other way -- a finite test set inflates the spread about
36%, while the overlap deflates it about 2%. Both are checked, in the directions and the
magnitudes that were measured.
"""

from __future__ import annotations

import numpy as np
import pytest

from caliber.evaluation import prereg
from caliber.evaluation import reliability as R


# -- the Beta law --------------------------------------------------------------


def test_beta_law_matches_the_textbook_case():
    r = R.beta_reference(99, alpha=0.1)
    assert r.k == 10, "floor(0.1 * 100) = 10"
    assert (r.a, r.b) == (90.0, 10.0)
    assert r.mean == pytest.approx(0.90, abs=1e-12)
    assert r.sd == pytest.approx(np.sqrt(0.9 * 0.1 / 101), rel=1e-6)
    assert r.reportable


def test_beta_mean_is_never_below_nominal_and_the_slack_is_one_over_n_plus_one():
    """The flooring rule's whole content: conservative by at most 1/(n+1)."""
    for n in (30, 49, 50, 99, 100, 199, 500):
        r = R.beta_reference(n, alpha=0.1)
        assert r.mean >= 0.90 - 1e-12, f"n={n} mean {r.mean} below nominal"
        assert r.mean - 0.90 <= 1.0 / (n + 1) + 1e-12


def test_beta_sd_shrinks_with_n():
    sds = [R.beta_reference(n).sd for n in (30, 100, 300, 1000)]
    assert sds == sorted(sds, reverse=True)


def test_thin_cells_are_refused_a_reference():
    assert not R.beta_reference(5).reportable
    assert not R.beta_reference(5).certifiable, "floor(0.1*6) = 0, so nothing is certifiable"
    assert not R.beta_reference(R.MIN_POSITIVES_FOR_REFERENCE - 1).reportable
    assert R.beta_reference(R.MIN_POSITIVES_FOR_REFERENCE).reportable


def test_beta_quantile_fallback_agrees_with_scipy(monkeypatch):
    """The no-scipy path exists for the partner's machine; it must give the same number."""
    import builtins

    real_import = builtins.__import__

    def no_scipy(name, *a, **k):
        if name.startswith("scipy"):
            raise ImportError("scipy disabled for this test")
        return real_import(name, *a, **k)

    reference = R._beta_ppf(90.0, 10.0, 0.01)
    monkeypatch.setattr(builtins, "__import__", no_scipy)
    fallback = R._beta_ppf(90.0, 10.0, 0.01)
    assert fallback == pytest.approx(reference, abs=5e-4)


def test_coverage_inside_and_outside_the_band():
    r = R.beta_reference(99)
    assert r.contains(0.90)
    assert not r.contains(0.70), "well below the 1st percentile"
    assert not r.contains(float("nan"))


# -- worst-cell bounds ---------------------------------------------------------


def _cells(coverages, n=100):
    import pandas as pd

    return pd.DataFrame(
        [
            {"operator_id": f"op{i}", "n_caught": int(round(c * n)), "n_hallucinated": n}
            for i, c in enumerate(coverages)
        ]
    )


def test_the_three_bounds_are_ordered_as_claimed():
    """simultaneous <= pointwise <= observed. Reversing any of these would be a bug."""
    w = R.worst_operator_coverage(_cells([0.95, 0.92, 0.86, 0.91]))
    assert w.cell == "op2"
    assert w.observed_min == pytest.approx(0.86)
    assert w.simultaneous_lower <= w.pointwise_lower <= w.observed_min


def test_a_well_covered_grid_certifies_and_a_poor_one_does_not():
    good = R.worst_operator_coverage(_cells([0.99, 0.99, 0.985, 0.99], n=4000))
    assert good.certifiable_at_nominal
    bad = R.worst_operator_coverage(_cells([0.95, 0.92, 0.72, 0.91]))
    assert not bad.certifiable_at_nominal


def test_more_cells_lower_the_simultaneous_bound_at_fixed_coverage():
    """The Bonferroni price for scanning a bigger grid, which must be visible."""
    few = R.worst_operator_coverage(_cells([0.90] * 3))
    many = R.worst_operator_coverage(_cells([0.90] * 30))
    assert many.simultaneous_lower < few.simultaneous_lower


def test_cells_with_no_positives_are_dropped_not_scored_as_zero():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {"operator_id": "a", "n_caught": 90, "n_hallucinated": 100},
            {"operator_id": "b", "n_caught": 0, "n_hallucinated": 0},
        ]
    )
    w = R.worst_operator_coverage(frame)
    assert w.n_cells == 1 and w.cell == "a"


def test_clopper_pearson_edges():
    assert R.clopper_pearson_lower(100, 100, 0.05) == pytest.approx(0.05 ** 0.01, rel=1e-9)
    assert R.clopper_pearson_lower(0, 50, 0.05) == 0.0
    assert np.isnan(R.clopper_pearson_lower(0, 0, 0.05))


# -- the claim about overlapping splits ----------------------------------------


def _disjoint_splits(n_patients=30, n_splits=3):
    pts = [f"p{i:02d}" for i in range(n_patients)]
    per = n_patients // n_splits
    calib, test = [], []
    for s in range(n_splits):
        block = pts[s * per : (s + 1) * per]
        calib.append(block[: per // 2])
        test.append(block[per // 2 :])
    return calib, test


def _overlapping_splits(n_patients=30, n_splits=13, seed=0):
    """The project's actual shape: 30 patients re-cut 13 ways, half calibration each time."""
    rng = np.random.default_rng(seed)
    pts = np.array([f"p{i:02d}" for i in range(n_patients)])
    calib, test = [], []
    for _ in range(n_splits):
        perm = rng.permutation(pts)
        calib.append(list(perm[: n_patients // 2]))
        test.append(list(perm[n_patients // 2 :]))
    return calib, test


def test_measured_spread_exceeds_the_closed_form_beta_sd():
    """The correction that matters, and the one the first draft of this module got backwards.

    Beta(n+1-k, k) is the law of coverage against an infinite test set. A split measures
    coverage on a finite one, so the measured spread is strictly larger. At the project's
    balanced split sizes the inflation is around a third, which means a direct comparison
    against the closed form would report a correctly calibrated method as over-dispersed.
    """
    calib, test = _overlapping_splits()
    counts = {p: 8 for split in calib for p in split}
    sim, (lo, hi) = R.simulate_split_spread(calib, test, counts, n_sim=500, seed=1)
    mean_n = int(np.mean([sum(counts[p] for p in c) for c in calib]))
    beta_sd = R.beta_reference(mean_n).sd

    assert np.isfinite(sim)
    assert sim > beta_sd, (
        f"simulated SD {sim:.4f} is not above the closed-form Beta SD {beta_sd:.4f}; the "
        "module's stated correction would then be wrong"
    )
    assert 1.2 <= sim / beta_sd <= 1.6, f"inflation {sim / beta_sd:.2f} outside the measured range"
    # The interval is wide enough to reach below beta_sd, which is the limitation below.
    assert lo < beta_sd < hi


@pytest.mark.parametrize("per_patient", [8, 80, 400])
def test_the_correction_formula_tracks_the_simulation(per_patient):
    """``sqrt(Var_Beta + p(1-p)/n_test)`` against the simulation, over a 50x size range.

    This is what lets the module publish a closed-form correction a reader can check by
    hand instead of having to rerun a simulation.
    """
    calib, test = _overlapping_splits()
    counts = {p: per_patient for split in calib for p in split}
    sim, _ = R.simulate_split_spread(calib, test, counts, n_sim=400, seed=1)

    chk = R.split_spread_check(
        [0.9] * len(calib), calib, test, counts, n_sim=1, seed=1
    )
    assert chk.beta_sd_corrected == pytest.approx(sim, rel=0.10)


def test_overlap_deflates_the_spread_slightly_at_matched_sizes():
    """The second-order effect, isolated: same split sizes, shared patients or not.

    Reported because it is real and in the predicted direction, not because it changes a
    conclusion -- it is roughly fifteen times smaller than the test-set inflation above.
    """
    counts = {f"p{i:02d}": 8 for i in range(30)}
    calib_o, test_o = _overlapping_splits()
    sim_overlap, _ = R.simulate_split_spread(calib_o, test_o, counts, n_sim=800, seed=4)

    # Thirteen splits over disjoint patient pools of the same per-split size.
    big = {f"q{i:03d}": 8 for i in range(13 * 30)}
    calib_i, test_i = [], []
    for s in range(13):
        block = [f"q{i:03d}" for i in range(s * 30, (s + 1) * 30)]
        calib_i.append(block[:15])
        test_i.append(block[15:])
    sim_independent, _ = R.simulate_split_spread(calib_i, test_i, big, n_sim=800, seed=4)

    assert sim_overlap < sim_independent
    assert 0.93 <= sim_overlap / sim_independent <= 1.0, "the effect is a few per cent, not large"


def test_spread_check_flags_the_overlap_and_points_at_the_right_ratio():
    calib, test = _overlapping_splits()
    counts = {p: 8 for split in calib for p in split}
    # A spread drawn to match the simulated reference, so the verdict should be consistent.
    sim, _ = R.simulate_split_spread(calib, test, counts, n_sim=300, seed=2)
    rng = np.random.default_rng(3)
    cov = list(0.90 + rng.normal(0.0, sim, size=len(calib)))

    chk = R.split_spread_check(cov, calib, test, counts, n_sim=300, seed=2)
    assert chk.overlapping
    assert chk.mean_shared_fraction > 0.1
    assert "ratio_to_simulated" in chk.note
    assert np.isfinite(chk.ratio_to_beta) and np.isfinite(chk.ratio_to_simulated)
    assert chk.beta_sd_corrected > chk.beta_sd, "the correction must add variance, not remove it"
    assert chk.n_test > 0
    assert chk.verdict in {
        "consistent_with_exchangeability",
        "over_dispersed",
        "under_dispersed",
    }
    assert chk.as_row()["prereg"] == prereg.fingerprint()


def test_an_over_dispersed_method_is_caught():
    """A method whose splits scatter far more than exchangeability allows must not pass."""
    calib, test = _overlapping_splits()
    counts = {p: 8 for split in calib for p in split}
    rng = np.random.default_rng(5)
    cov = list(0.90 + rng.normal(0.0, 0.15, size=len(calib)))
    chk = R.split_spread_check(cov, calib, test, counts, n_sim=300, seed=2)
    assert chk.verdict == "over_dispersed"
    assert not chk.consistent


def test_disjoint_splits_are_named_as_such_but_still_get_the_correction():
    """No shared patients removes the 2% term; it does not remove the test-set term."""
    calib, test = _disjoint_splits()
    counts = {p: 8 for split in calib + test for p in split}
    chk = R.split_spread_check([0.9, 0.88, 0.92], calib, test, counts, n_sim=200)
    assert not chk.overlapping
    assert "splits are disjoint" in chk.note
    assert chk.beta_sd_corrected > chk.beta_sd


def test_one_split_is_refused():
    calib, test = _disjoint_splits(n_splits=3)
    counts = {p: 8 for split in calib + test for p in split}
    with pytest.raises(ValueError, match="at least two splits"):
        R.split_spread_check([0.9], calib[:1], test[:1], counts, n_sim=50)


def test_thirteen_splits_are_reported_as_underpowered():
    """A passing spread check at 13 splits must not read as evidence of calibration.

    The sample SD of 13 coverages carries about 20% relative noise, so the reference
    interval spans roughly 0.62x to 1.49x the correct SD -- wider than the 36% correction
    this module exists to apply. The verdict has to say so.
    """
    calib, test = _overlapping_splits(n_splits=13)
    counts = {p: 8 for split in calib for p in split}
    sim, _ = R.simulate_split_spread(calib, test, counts, n_sim=400, seed=1)
    # A spread drawn at the correct SD, so the check genuinely passes and the only
    # question left is whether it had the power to fail.
    rng = np.random.default_rng(21)
    cov = list(0.90 + rng.normal(0.0, sim, size=13))
    chk = R.split_spread_check(cov, calib, test, counts, n_sim=400, seed=1)

    lo, hi = chk.power_band
    assert hi > R.INFORMATIVE_BAND, f"band {hi:.2f} unexpectedly tight at 13 splits"
    assert not chk.informative
    assert chk.verdict == "consistent_but_underpowered"
    assert "weak evidence" in chk.note


def test_a_gross_mis_calibration_is_still_caught_at_thirteen_splits():
    """Low power is not no power: a factor-of-three spread must still be flagged."""
    calib, test = _overlapping_splits(n_splits=13)
    counts = {p: 8 for split in calib for p in split}
    rng = np.random.default_rng(11)
    cov = list(0.90 + rng.normal(0.0, 0.11, size=13))
    chk = R.split_spread_check(cov, calib, test, counts, n_sim=400, seed=1)
    assert chk.verdict == "over_dispersed"


# -- the clustered control (D85) ----------------------------------------------


def _clustered_side(n_subjects, per_subject, rng, subject_sd=0.9, within_sd=0.35):
    """Scores with a strong subject effect: regions within a subject are correlated."""
    return [
        rng.normal(rng.normal(0.0, subject_sd), within_sd, per_subject)
        for _ in range(n_subjects)
    ]


def test_the_cluster_bootstrap_recovers_the_design_effect():
    """The defect this was written for: region counts are not independent units.

    T2-A's CT bank has ~16,400 calibration regions across 41 subjects. Handing that region
    count to the Beta law gives an interval of about +/-0.011 and rejects a correctly
    calibrated bank. The cluster bootstrap must instead recover an effective sample size of
    the order of the subject count.
    """
    rng = np.random.default_rng(0)
    c = R.clustered_coverage(
        _clustered_side(41, 100, rng), _clustered_side(51, 100, rng), n_boot=400
    )
    assert c.n_positives_calib == 4100
    assert 10 <= c.effective_n <= 300, f"effective_n {c.effective_n:.0f} not subject-scale"
    assert c.design_effect > 10, "regions within a subject must not look independent"
    beta_sd = R.beta_reference(c.n_positives_calib).sd
    assert c.sd > 5 * beta_sd, "the clustered interval must be far wider than the Beta band"


def test_independent_regions_give_a_design_effect_near_one():
    """The control for the test above: with no clustering, the design effect must be ~1.

    This is also the test that caught the first definition of ``design_effect``, which
    compared against the Beta SD alone and so reported 2.6 on this fixture -- the extra
    factor being the test-set term, not clustering. The reference is now both terms.
    """
    rng = np.random.default_rng(1)
    singletons_cal = [rng.normal(0, 1, 1) for _ in range(800)]
    singletons_tst = [rng.normal(0, 1, 1) for _ in range(800)]
    c = R.clustered_coverage(singletons_cal, singletons_tst, n_boot=400)
    assert 0.5 <= c.design_effect <= 2.0, f"design effect {c.design_effect:.2f} should be ~1"
    # And the reference is strictly wider than the Beta SD, which is the correction itself.
    assert c.independent_sd > R.beta_reference(c.n_positives_calib).sd


def test_resampling_the_calibration_side_widens_the_interval():
    """Why a test-side-only interval understates: the threshold varies too.

    This is the error that made T2-A's control read as a failure at 5 of 9 operators.
    """
    rng = np.random.default_rng(2)
    cal = _clustered_side(41, 100, rng)
    tst = _clustered_side(51, 100, rng)
    both = R.clustered_coverage(cal, tst, n_boot=600, seed=3)

    # The same bootstrap with the calibration side held fixed.
    pooled = np.concatenate(cal)
    k = int(np.floor(0.1 * (pooled.size + 1)))
    tau = np.sort(pooled)[k - 1]
    gen = np.random.default_rng(3)
    reps = [
        float((np.concatenate([tst[i] for i in gen.integers(0, len(tst), len(tst))]) >= tau).mean())
        for _ in range(600)
    ]
    test_only_sd = float(np.std(reps, ddof=1))
    assert both.sd > test_only_sd, (
        f"moving both sides ({both.sd:.4f}) must give a wider interval than moving the "
        f"test side alone ({test_only_sd:.4f})"
    )


def test_a_genuinely_miscalibrated_cell_still_fails_the_clustered_control():
    """Widening the interval must not make the control unable to reject anything."""
    rng = np.random.default_rng(4)
    cal = _clustered_side(41, 100, rng)
    # Test scores shifted far below the calibration threshold: real under-coverage.
    tst = [s - 3.0 for s in _clustered_side(51, 100, rng)]
    c = R.clustered_coverage(cal, tst, n_boot=400)
    assert c.coverage < 0.5
    assert not c.contains_nominal


def test_an_empty_side_returns_nan_not_a_number():
    c = R.clustered_coverage([], [np.array([1.0, 2.0])], n_boot=50)
    assert np.isnan(c.coverage) and not c.contains_nominal
