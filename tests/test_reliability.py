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

import math

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


# -- D86: the draw-marginal coverage the unsafe label is read from ------------


def _recuts(n_subjects: int = 100, n_calib: int = 41, n_splits: int = 13, seed: int = 0):
    """Re-cuts of one fixed cohort, which is how this study's splits are actually made."""
    rng = np.random.default_rng(seed)
    ids = [f"s{i:03d}" for i in range(n_subjects)]
    return [list(rng.permutation(ids)[:n_calib]) for _ in range(n_splits)]


def test_overlap_measure_is_the_correlation_not_the_jaccard():
    """A third shared gives a correlation of 0.33 and a Jaccard of 0.20; we need the former."""
    a = [f"s{i}" for i in range(30)]
    b = [f"s{i}" for i in range(20, 50)]          # 10 of 30 shared
    assert R.mean_pairwise_overlap([a, b]) == pytest.approx(10 / 30, abs=1e-9)
    assert R._shared_fraction([a, b]) == pytest.approx(10 / 50, abs=1e-9)
    # the degenerate ends
    assert R.mean_pairwise_overlap([a, a]) == pytest.approx(1.0)
    assert R.mean_pairwise_overlap([a, [f"t{i}" for i in range(30)]]) == 0.0


def test_disjoint_splits_get_no_overlap_correction():
    """With independent draws the honest standard error is the naive one."""
    est = R.draw_marginal_coverage([0.88, 0.90, 0.92, 0.89, 0.91, 0.90, 0.905, 0.895],
                                   overlap=0.0)
    assert est.n_eff == pytest.approx(est.n_splits)
    assert est.overlap_inflation == pytest.approx(1.0, abs=1e-9)
    assert est.se == pytest.approx(est.se_naive, abs=1e-12)


def test_overlapping_recuts_lose_most_of_their_effective_n():
    """13 re-cuts of 100 subjects taking 41 each are worth about 2.6 independent draws."""
    splits = _recuts()
    rho = R.mean_pairwise_overlap(splits)
    assert rho == pytest.approx(41 / 100, abs=0.05)      # overlap ~ the calibration fraction
    est = R.draw_marginal_coverage([0.9] * 12 + [0.8], calib_splits=splits)
    assert 2.0 < est.n_eff < 4.0
    assert est.overlap_inflation > 1.8
    assert est.df < 3.0
    # the Student form is reported, never applied
    assert est.t_inflation > 2.0
    assert est.interval()[1] == pytest.approx(est.coverage + R._Z975 * est.se, abs=1e-9)


def test_too_few_splits_is_refused_with_the_reason():
    est = R.draw_marginal_coverage([0.9, 0.91, 0.89], overlap=0.3)
    assert est.n_splits < prereg.UNSAFE_LABEL_MIN_SPLITS
    assert not est.reportable
    assert "below the frozen minimum" in est.note


def test_non_finite_splits_are_dropped_not_credited():
    est = R.draw_marginal_coverage([0.9, np.nan, 0.88, float("inf"), 0.92], overlap=0.0)
    assert est.n_splits == 3
    assert est.coverage == pytest.approx(np.mean([0.9, 0.88, 0.92]))


def test_across_split_columns_match_what_the_label_source_asks_for():
    """The contract between the two modules, pinned so a rename cannot silently break it."""
    import pandas as pd

    from caliber.evaluation import safety

    wanted = safety.LABEL_SOURCE_COLUMNS[prereg.UNSAFE_LABEL_SOURCE]
    rows = [
        {"calibration": "a", "deployment": b, "split": i, "coverage": cov}
        for b, base in (("b", 0.91), ("c", 0.70))
        for i, cov in enumerate([base - 0.01, base, base + 0.01] * 4 + [base])
    ]
    out = R.across_split_columns(pd.DataFrame(rows), overlap=1 / 3)
    assert set(wanted) <= set(out.columns)
    assert len(out) == 2
    assert out["n_splits"].eq(13).all()


def test_the_aggregated_columns_feed_the_safety_rule_end_to_end():
    """D86's two halves actually join: aggregate, merge, label, with no fall-back raised."""
    import pandas as pd

    from caliber.evaluation import safety

    ops = [f"op{i}" for i in range(5)]
    sev = {o: i for i, o in enumerate(ops)}
    long_rows, reg_rows = [], []
    for a in ops:
        for b in ops:
            if a == b:
                continue
            base = 0.70 if sev[b] > sev[a] else 0.93
            for i in range(13):
                long_rows.append({"calibration": a, "deployment": b, "split": i,
                                  "coverage": base + 0.004 * (i % 3 - 1)})
            reg_rows.append({"calibration": a, "deployment": b,
                             "signed_distance": float(sev[b] - sev[a]),
                             "coverage": base - 0.07, "coverage_hi": base - 0.07})
    across = R.across_split_columns(pd.DataFrame(long_rows), overlap=1 / 3)
    frame = pd.DataFrame(reg_rows).merge(across, on=["calibration", "deployment"])

    rule = safety.safety_rule(frame, with_null=False)
    assert rule.label_source == prereg.UNSAFE_LABEL_SOURCE
    assert rule.n_pairs == 20
    assert rule.false_safe_rate == 0.0          # perfect on the draw-marginal label
    both = safety.label_source_sensitivity(frame)
    assert both["false_safe_rate"].notna().all()
    # the single draw depresses every pair, so it manufactures unsafe cells
    single = both.loc[both["label_source"] == "registry_split"].iloc[0]
    assert single["n_unsafe"] > both.loc[both["frozen_primary"]].iloc[0]["n_unsafe"]


# -- the contract between whoever measures coverage and whoever reads the label (D86) ----


def _per_split(n_splits: int = 13, coverages=None, pairs=(("a", "b"), ("b", "a"))):
    """A long per-split table shaped like ``pair_coverage_by_split.py``'s output."""
    import pandas as pd

    from caliber.data.splits import PROBE_SALTS

    vals = coverages if coverages is not None else [0.88 + 0.004 * i for i in range(n_splits)]
    return pd.DataFrame([
        {"salt": PROBE_SALTS[i], "calibration": a, "deployment": b, "coverage": v}
        for i, v in enumerate(vals)
        for a, b in pairs
    ])


def _registry(pairs=(("a", "b"), ("b", "a")), coverage=0.84, hi=0.89):
    import pandas as pd

    return pd.DataFrame([
        {"calibration": a, "deployment": b, "coverage": coverage, "lo": 0.79, "hi": hi}
        for a, b in pairs
    ])


def test_pair_coverage_from_splits_carries_both_label_sources():
    """One frame has to score the frozen arm and the arm it displaced, or there is no
    sensitivity table to report."""
    from caliber.evaluation.reliability import pair_coverage_from_splits
    from caliber.evaluation.safety import LABEL_SOURCE_COLUMNS

    frame = pair_coverage_from_splits(_per_split(), _registry(), overlap=0.41)
    for cols in LABEL_SOURCE_COLUMNS.values():
        for col in cols:
            assert col in frame.columns, f"{col} missing; {sorted(frame.columns)}"
    assert frame["coverage"].eq(0.84).all()
    assert frame["coverage_across"].round(6).eq(round(0.88 + 0.004 * 6, 6)).all()


def test_the_across_split_interval_is_rebuilt_not_read():
    """The decisive property: a stored ``sd / sqrt(n)`` ignores that the splits overlap.

    Accepting one would narrow the interval, and a narrower interval calls more pairs
    unsafe -- so the error would move the primary endpoint, in a direction nobody notices.
    """
    import numpy as np

    from caliber.evaluation.reliability import pair_coverage_from_splits

    frame = pair_coverage_from_splits(_per_split(), _registry(), overlap=0.41)
    row = frame.iloc[0]
    assert row["overlap"] == 0.41
    assert row["overlap_inflation"] > 2.0, row["overlap_inflation"]
    # the honest half-width is the inflated one, not sd/sqrt(13)
    naive_hi = row["coverage_across"] + 1.959964 * row["across_sd"] / np.sqrt(row["n_splits"])
    assert row["coverage_across_hi"] > naive_hi + 1e-6


def test_a_pair_in_one_table_only_keeps_its_row():
    """A dropped pair shrinks the false-safe denominator, which is this project's own
    recurring failure: a statistic that improves because its sample changed."""
    import numpy as np

    from caliber.evaluation.reliability import pair_coverage_from_splits

    frame = pair_coverage_from_splits(
        _per_split(pairs=(("a", "b"), ("b", "a"))),
        _registry(pairs=(("a", "b"), ("b", "a"), ("a", "c"))),
        overlap=0.41,
    )
    assert len(frame) == 3
    only_registry = frame[(frame["calibration"] == "a") & (frame["deployment"] == "c")]
    assert len(only_registry) == 1
    assert np.isnan(float(only_registry["coverage_across"].iloc[0]))


def test_pair_coverage_from_splits_names_the_columns_it_is_missing():
    import pytest

    from caliber.evaluation.reliability import pair_coverage_from_splits

    with pytest.raises(KeyError, match="salt"):
        pair_coverage_from_splits(
            _per_split().drop(columns=["salt"]), _registry(), overlap=0.41
        )
    with pytest.raises(KeyError, match="hi"):
        pair_coverage_from_splits(_per_split(), _registry().drop(columns=["hi"]), overlap=0.41)


def test_the_overlap_can_be_measured_from_the_splits_instead_of_passed():
    from caliber.data.splits import calib_sets_for_salts
    from caliber.evaluation.reliability import pair_coverage_from_splits

    splits = calib_sets_for_salts(
        [f"LIDC-IDRI-{i:04d}" for i in range(1, 101)], design_fraction=0.10
    )
    frame = pair_coverage_from_splits(_per_split(), _registry(), calib_splits=splits)
    assert 0.2 < float(frame["overlap"].iloc[0]) < 0.6


def test_a_short_run_is_marked_unreportable_rather_than_averaged_anyway():
    """Below the frozen minimum the mean is noisier than the single-draw bias it removes."""
    from caliber.evaluation import prereg
    from caliber.evaluation.reliability import pair_coverage_from_splits

    n = prereg.UNSAFE_LABEL_MIN_SPLITS - 1
    frame = pair_coverage_from_splits(
        _per_split(n_splits=n, coverages=[0.88 + 0.004 * i for i in range(n)]),
        _registry(),
        overlap=0.41,
    )
    assert not frame["reportable"].any()
    assert "below the frozen minimum" in frame["note"].iloc[0]


# -- the Beta law at a transfer cell (D92) -------------------------------------


def _gamma(scale: float, n: int = 60000, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).gamma(2.0, scale, size=n)


def test_the_matched_cell_recovers_the_classical_beta_law():
    """G is the identity when both operators are the same, so nothing may change."""
    a = _gamma(1.0)
    ref = R.transfer_beta_reference(a, a, n_positives=120, alpha=0.1)
    assert abs(ref.mean - ref.matched.mean) < 5e-3, (ref.mean, ref.matched.mean)
    assert abs(ref.sd_ratio - 1.0) < 0.05, ref.sd_ratio
    assert abs(ref.slope - 1.0) < 0.05, ref.slope


def test_the_probability_map_is_the_identity_on_one_sample():
    a = _gamma(1.0)
    u = np.array([0.05, 0.1, 0.25, 0.5, 0.9])
    got = R.probability_map(a, a, u)
    assert np.max(np.abs(got - u)) < 2e-3, got


def test_the_transfer_spread_is_the_matched_spread_times_the_density_ratio():
    """The first-order claim, over a 7x range of spreads. Measured agreement is ~2%."""
    a = _gamma(1.0)
    seen = []
    for scale in (0.7, 1.6, 2.4):
        ref = R.transfer_beta_reference(a, _gamma(scale, seed=4), 120, alpha=0.1)
        assert abs(ref.sd_ratio / ref.slope - 1.0) < 0.04, (scale, ref.as_row())
        seen.append(ref.sd_ratio)
    # the range has to be wide, or agreeing with it says nothing
    assert max(seen) / min(seen) > 5.0, seen


def test_the_matched_band_can_be_too_narrow_as_well_as_too_wide():
    """A certificate quoting the matched Beta band is wrong in BOTH directions."""
    a = _gamma(1.0)
    wider = R.transfer_beta_reference(a, _gamma(0.7, seed=4), 120, alpha=0.1)
    narrower = R.transfer_beta_reference(a, _gamma(2.4, seed=4), 120, alpha=0.1)
    assert wider.sd_ratio > 1.0 and wider.matched_band_is_too_narrow
    assert narrower.sd_ratio < 1.0 and not narrower.matched_band_is_too_narrow
    assert wider.sd_ratio / narrower.sd_ratio > 3.0, (wider.sd_ratio,
                                                      narrower.sd_ratio)


def test_the_LESS_ill_posed_deployment_operator_is_the_one_that_loses_coverage():
    """D89's counterintuitive direction, in the law rather than in the bank.

    A deployment operator whose scores run *high* clears an inherited low threshold more
    easily, so it OVER-covers; the operator that loses coverage is the easier one. This
    is why an operator distance that tracks ill-posedness gets the sign of the gap
    backwards, and it is the whole of why T1-A failed.
    """
    a = _gamma(1.0)
    matched = R.transfer_beta_reference(a, a, 120, alpha=0.1).mean
    harder = R.transfer_beta_reference(a, _gamma(2.4, seed=4), 120, alpha=0.1)
    easier = R.transfer_beta_reference(a, _gamma(0.7, seed=4), 120, alpha=0.1)
    assert harder.mean > matched > easier.mean, (harder.mean, matched, easier.mean)
    # stated as the gap, which is the quantity T1-B predicts
    assert matched - harder.mean < 0.0 < matched - easier.mean


def test_one_calibration_draw_moves_every_cell_in_its_row_the_same_way():
    """D86's mechanism, as a theorem: comonotone within a row, exactly.

    One draw of ``U`` is pushed through several deployment operators' maps; because every
    map is non-decreasing, the resulting coverages are a monotone function of that one
    scalar, so their ranks across draws agree perfectly.
    """
    a = _gamma(1.0)
    rng = np.random.default_rng(7)
    ref = R.beta_reference(120, alpha=0.1)
    u = rng.beta(ref.b, ref.a, size=4000)
    rows = [1.0 - R.probability_map(a, _gamma(s, seed=4), u)
            for s in (0.7, 1.0, 1.6, 2.4)]
    for other in rows[1:]:
        order_a = np.argsort(np.argsort(rows[0]))
        order_b = np.argsort(np.argsort(other))
        assert np.corrcoef(order_a, order_b)[0, 1] > 0.999


def test_nine_independent_rows_would_not_give_a_one_directional_displacement():
    """So 72-of-72 is evidence about the cohort, not about the within-row mechanism."""
    rng = np.random.default_rng(11)
    ref = R.beta_reference(120, alpha=0.1)
    trials = 20000
    u = rng.beta(ref.b, ref.a, size=(trials, 9))
    side = np.sign(u - np.median(u, axis=0, keepdims=True))
    same = float(np.mean(np.abs(side.sum(axis=1)) == 9))
    assert same < 0.01, same
    assert abs(same - 2 * 0.5 ** 9) < 0.004, same


def test_a_cell_with_no_certifiable_threshold_yields_no_transfer_law():
    a = _gamma(1.0)
    ref = R.transfer_beta_reference(a, _gamma(1.6, seed=4), n_positives=3, alpha=0.1)
    assert ref.matched.k < 1
    assert ref.reportable is False
    assert ref.samples.size == 0


def test_the_transfer_spread_prediction_is_frozen_and_falsifiable():
    text = " ".join(prereg.TRANSFER_SPREAD_PREDICTION)
    assert "falsified" in text
    assert "CALIBRATION operator" in text
    assert any(a[0] == "2026-11-22" for a in prereg.AMENDMENTS)


def test_either_matched_cell_gives_the_same_spread_ratio_at_equal_counts():
    """Why the choice of reference cell is principled but not load-bearing here.

    Only cell (a, a) shares the calibration draw, and it is the right reference when the
    two operators' positive counts differ. But the SPREAD ratio divides out the Beta's
    scale, so at equal counts the deployment operator's matched cell gives the same
    number. Recorded because the frozen prediction first claimed otherwise.
    """
    a = _gamma(1.0)
    b = _gamma(1.6, seed=4)
    rng = np.random.default_rng(13)
    ref = R.beta_reference(120, alpha=0.1)
    u_a = rng.beta(ref.b, ref.a, size=20000)
    u_b = rng.beta(ref.b, ref.a, size=20000)          # an INDEPENDENT draw
    cov_ab = 1.0 - R.probability_map(a, b, u_a)
    right = float(np.std(cov_ab) / np.std(1.0 - u_a))  # vs cell (a, a)
    wrong = float(np.std(cov_ab) / np.std(1.0 - u_b))  # vs an independent draw
    g = R.transfer_beta_reference(a, b, 120, alpha=0.1).slope
    assert abs(right / g - 1.0) < 0.05, (right, g)
    # the independent-draw version is not wrong by much here, which is the point:
    # it agrees in SPREAD and is still the wrong comparison, because it throws away
    # the pairing that makes the ratio a statement about one draw
    assert abs(wrong / g - 1.0) < 0.05, (wrong, g)
    assert np.corrcoef(cov_ab, 1.0 - u_a)[0, 1] > 0.99
    assert abs(np.corrcoef(cov_ab, 1.0 - u_b)[0, 1]) < 0.05


# -- amendments eleven and twelve: the Beta's sample size ----------------------


def test_a_transfer_band_is_refused_without_the_independent_count():
    """Amendment eleven. The interval I shipped on 22 Nov carried no unit."""
    a = _gamma(1.0)
    ref = R.transfer_beta_reference(a, _gamma(1.6, seed=4), 10762, alpha=0.1)
    assert ref.reportable is True, "the dimensionless results are still available"
    assert ref.rescalable is False, "but an interval is not"
    try:
        ref.interval()
    except ValueError as exc:
        assert "n_independent" in str(exc)
    else:
        raise AssertionError("interval() must refuse without the independent count")
    assert ref.contains(0.9) is False, "containment is a certificate too"


def test_the_independent_count_widens_the_band_by_the_design_effect():
    a = _gamma(1.0)
    b = _gamma(1.6, seed=4)
    n_pos, n_ind = 10762, 41
    ref = R.transfer_beta_reference(a, b, n_pos, alpha=0.1, n_independent=n_ind)
    assert ref.rescalable is True
    lo, hi = ref.interval()
    raw = np.quantile(ref.samples, [0.01, 0.99])
    widened = (hi - lo) / float(raw[1] - raw[0])
    expected = math.sqrt(n_pos / n_ind)
    assert widened > 5.0, "a design effect of order 100 is not a rounding correction"
    # clipping at 1.0 can only shorten the band, never lengthen it
    assert widened <= expected * 1.02, (widened, expected)
    assert 0.0 <= lo <= hi <= 1.0, "coverage is a proportion"


def test_the_spread_ratio_is_blind_to_the_reference_count():
    """Amendment twelve, as a test. This is why rho +0.977 certified nothing."""
    a = _gamma(1.0)
    b = _gamma(1.6, seed=4)
    wrong = R.transfer_beta_reference(a, b, 10762, alpha=0.1, n_independent=10762)
    right = R.transfer_beta_reference(a, b, 10762, alpha=0.1, n_independent=41)
    assert wrong.sd_ratio == right.sd_ratio, (
        "the predicted ratio cannot see the unit error standing next to it")
    assert wrong.slope == right.slope
    assert wrong.interval() != right.interval(), (
        "while the quantity a certificate quotes moves by the design effect")


def test_no_beta_band_is_a_gate_at_either_count():
    text = " ".join(prereg.BETA_REGION_COUNT_USE)
    assert "idealisation" in text
    assert "no default" in text
    assert "cluster" in text and "bootstrap" in text, (
        "the rule has to name the path that IS the certificate")
    assert "accepts almost anything" in text, (
        "the subject count must be barred as a gate too, not just the region count")
    assert prereg.BETA_REFERENCE_UNIT == "calibration_subject"


def test_amendment_eleven_does_not_claim_the_unit_finding_as_new():
    # Keyed on "BETA_REFERENCE_UNIT added", not on the bare constant name: later
    # amendments may legitimately mention the constant, and D96's did.
    entry = [t for _, t in prereg.AMENDMENTS if "BETA_REFERENCE_UNIT added" in t]
    assert len(entry) == 1
    assert "not written up as new" in entry[0]
    assert "ClusteredCoverage" in entry[0], (
        "the amendment must point at the place that already recorded it")


def test_the_independent_count_is_never_defaulted_from_the_positive_count():
    """The defect was a silent default, so the absence of one is the thing to pin."""
    a = _gamma(1.0)
    ref = R.transfer_beta_reference(a, _gamma(1.6, seed=4), 10762, alpha=0.1)
    assert ref.n_independent is None, (
        "filling this in from n_positives is exactly the error being fixed")
    assert ref.as_row()["n_independent"] is None
    assert ref.as_row()["rescalable"] is False
    assert np.isnan(ref.as_row()["beta_lo"] if "beta_lo" in ref.as_row()
                    else ref.as_row()["transfer_lo"]), (
        "a row emitted without the unit must carry no band")


def test_no_effective_count_is_frozen_because_the_code_measures_it():
    """The count belongs to the bank, and ClusteredCoverage already computes it per cell."""
    assert not hasattr(prereg, "BETA_REFERENCE_EFFECTIVE_COUNT_BRACKET"), (
        "a frozen effective count would be a number invented for convenience")
    for name in ("design_effect", "effective_n", "independent_sd"):
        assert hasattr(R.ClusteredCoverage, name), (
            "the measured path must exist before the Beta path is allowed to refuse")


def test_the_equal_count_clause_is_retracted_visibly():
    text = " ".join(prereg.TRANSFER_SPREAD_PREDICTION)
    assert "RETRACTED" in text, "a silently rewritten freeze is not a freeze"
    assert "10,760 is their MEAN" in text
    assert "3,157 to 21,762" in text
    assert "+0.977" in text and "+0.88" in text, "both references' measured values"


def test_the_dimensionless_warning_was_frozen_with_a_dimensioned_remedy():
    text = " ".join(prereg.DIMENSIONLESS_PASS_WARNING)
    assert "does not certify the reference" in text
    assert "divides out" in text
    assert "dimensioned check" in text, (
        "a warning with no remedy beside it is a note, not a rule")


# -- D94: the dimensioned check, and the regime the ratio is n-free in ---------


def test_the_level_range_brackets_the_count_it_was_drawn_from():
    """Recovers n from a PURE calibration draw -- no test-side term.

    This is the only regime the method is exact in, and D96 found it is not the
    regime either bank is in: the computable level is test-side, which adds a
    second variance term. Kept as the method's floor, not as a bank result.
    """
    for n in (41, 200, 5000):
        k = int(np.floor(0.1 * (n + 1)))
        levels = np.random.default_rng(n).beta(k, n + 1 - k, 13)
        got = R.count_from_level_range(levels, seed=2)
        lo, hi = got.bracket
        assert lo <= n <= hi, f"bracket {got.bracket} missed the true count {n}"


def test_the_level_range_cannot_exclude_a_region_scale_calibration_side():
    """The test D94 owed itself and did not run: does the check reject a truth?

    D94 read Track 2's level range as excluding the region count "by a factor of
    order 300" and took that for a pass. Build the world the check is supposed to
    reject -- calibration side AT the region count -- and read it through the
    test-side level, which is the only computable one. The check still excludes
    the region count, because the subject-scale test term dominates the variance.
    A check that returns the same verdict whatever the truth is not a check.
    """
    rng = np.random.default_rng(0)
    n_cal, n_test, alpha = 5000, 50, 0.1
    k = int(np.floor(alpha * (n_cal + 1)))
    levels = [rng.binomial(n_test, np.sort(rng.random(n_cal))[k - 1]) / n_test
              for _ in range(13)]
    got = R.count_from_level_range(levels, alpha=alpha, candidates=(
        *R.DEFAULT_LEVEL_RANGE_CANDIDATES, n_cal))
    assert got.excludes(n_cal), (
        "the point of the test: the region count IS the calibration side here, and "
        f"the check rejects it anyway (bracket {got.bracket})")
    lo, hi = got.bracket
    assert hi < n_cal, f"and the bracket sits far below the truth, at {got.bracket}"


def test_the_retraction_is_recorded_where_the_method_is_frozen():
    """COUNT_FROM_LEVEL_RANGE must not read as a usable dimensioned check."""
    text = " ".join(prereg.COUNT_FROM_LEVEL_RANGE)
    assert "RETRACTED" in text, "the retraction belongs in the frozen tuple, not only prose"
    assert "region count" in text, "what it may not be cited against must be named"
    entry = [t for _, t in prereg.AMENDMENTS if "COUNT_FROM_LEVEL_RANGE RETRACTED" in t]
    assert len(entry) == 1, "the retraction must appear exactly once in the audit trail"
    assert "200 of 200" in entry[0], "the measurement that forced it must be in the record"


def test_excludes_refuses_a_count_it_never_tried():
    """A bank's region count is never a round number, so a silent False is a trap."""
    levels = [0.03, 0.05, 0.07, 0.08, 0.09, 0.09, 0.10,
              0.11, 0.12, 0.13, 0.15, 0.16, 0.18]
    got = R.count_from_level_range(levels, seed=1)
    with pytest.raises(ValueError, match="not among the candidates"):
        got.excludes(5021)
    assert got.excludes(20000) is True, "an on-grid count still answers"


def test_the_level_range_yields_a_bracket_and_refuses_to_resolve_more():
    """D93 Sec. 3 read 'near 45' off this statistic. It does not carry a point estimate."""
    levels = [0.03, 0.05, 0.07, 0.08, 0.09, 0.09, 0.10,
              0.11, 0.12, 0.13, 0.15, 0.16, 0.18]
    got = R.count_from_level_range(levels, seed=1)
    assert 35 in got.consistent and 50 in got.consistent, (
        "a statistic that cannot separate 35 from 50 must report both as consistent")
    assert len(got.consistent) > 1, "a single surviving candidate would read as a count"


def test_no_count_is_read_off_the_level_range_when_nothing_fits():
    narrow = np.full(13, 0.10) + np.linspace(0.0, 1e-5, 13)
    got = R.count_from_level_range(narrow, seed=1)
    assert got.consistent == ()
    assert got.bracket is None, "a failed fit must return no number at all"


def test_the_level_range_needs_levels_and_more_than_one_split():
    with pytest.raises(ValueError):
        R.count_from_level_range([0.1])


def test_the_level_range_bracket_carries_no_verdict():
    """Not a gate: prereg.COUNT_FROM_LEVEL_RANGE says so and the row must not imply one."""
    levels = np.random.default_rng(0).beta(4, 38, 13)
    row = R.count_from_level_range(levels, seed=1).as_row()
    assert not any(str(k).startswith(("pass", "certif", "gate", "contains")) for k in row), (
        f"a bracket must not emit a verdict field: {sorted(row)}")
    assert row["unit"] == prereg.BETA_REFERENCE_UNIT, "the unit travels with the number"


def test_the_spread_ratio_is_n_free_only_where_the_slope_is_not_tiny():
    """Amendment 13's measurement. Amendment 12 claimed this held unconditionally."""
    out = {}
    for name, scale_b in (("big", 0.35), ("small", 3.0)):
        a, b = _gamma(1.0), _gamma(scale_b, seed=9)
        r41 = R.transfer_beta_reference(a, b, 41, alpha=0.1, n_draws=60000, seed=3)
        r2k = R.transfer_beta_reference(a, b, 21762, alpha=0.1, n_draws=60000, seed=3)
        out[name] = (abs(r41.slope), r41.sd_ratio, r2k.sd_ratio)

    slope_big, big41, big2k = out["big"]
    assert slope_big > 1.0, "this pair must sit in the large-slope regime"
    assert abs(big41 / big2k - 1.0) < 0.10, (
        f"where |G'| >= 1 the two counts agree: {big41:.4f} vs {big2k:.4f}")

    slope_small, small41, small2k = out["small"]
    assert slope_small < 0.3, "this pair must sit in the small-slope regime"
    assert abs(small41 / small2k - 1.0) > 0.10, (
        "where |G'| << 1 the ratio is NOT n-free, which is what amendment 12 missed: "
        f"{small41:.4f} vs {small2k:.4f}")
    assert max(small41, small2k) < 1.0, (
        "and such a cell is harmless -- its spread is below the matched cell's either way")


def test_the_dimensionless_warning_now_states_the_regime_it_holds_in():
    text = " ".join(prereg.DIMENSIONLESS_PASS_WARNING)
    assert "DIMENSIONLESS_PASS_REGIME" in text, (
        "amendment 12's premise must point at the measurement that narrowed it")
    regime = " ".join(prereg.DIMENSIONLESS_PASS_REGIME)
    assert "linear" in regime and "U's range" in regime, (
        "the stated reason the ratio is not exactly dimensionless")


def test_the_level_range_method_was_frozen_before_the_second_bank():
    text = " ".join(prereg.COUNT_FROM_LEVEL_RANGE)
    assert "bracket, never a count" in text
    assert "gate" in text, "the method must be frozen together with its refusal to gate"
    assert "count_from_level_range" in prereg.as_dict(), "the freeze must be exported"
    assert prereg.as_dict()["count_from_level_range"] == list(prereg.COUNT_FROM_LEVEL_RANGE)


def test_the_beta_unit_no_longer_cites_the_retracted_cross_check():
    """BETA_REFERENCE_UNIT leaned on the level range; that support is withdrawn."""
    doc = R.count_from_level_range.__doc__ or ""
    assert doc, "the function must keep documenting itself"
    assert prereg.BETA_REFERENCE_UNIT == "calibration_subject", (
        "the UNIT is unchanged by D96; it was never the thing in doubt")
    text = " ".join(prereg.COUNT_FROM_LEVEL_RANGE)
    assert "harmonic" in text or "min(n_calibration" in text, (
        "the estimand it actually measures must be stated where it is frozen")


def test_amendment_thirteen_narrows_rather_than_widens():
    """Keyed on the entry's own text, not on its index -- the index moves every round."""
    dates = [d for d, _ in prereg.AMENDMENTS]
    assert dates == sorted(dates), "the audit trail must read in order"
    widening = [t for _, t in prereg.AMENDMENTS if "LOWERS THE PRIMARY ENDPOINT" in t]
    assert len(widening) == 1, (
        "exactly one amendment has ever moved an endpoint; the rest only narrow")
    entry = [t for _, t in prereg.AMENDMENTS if "DIMENSIONLESS_PASS_REGIME" in t]
    assert len(entry) == 1, "amendment 13 must appear exactly once"
    assert "both narrow" in entry[0]
    assert "0.03 and 0.18" in entry[0], "the number that did the work must be in the record"


# -- D97: what the grid leaves the comparison room to show ---------------------


def test_the_knob_baseline_is_not_independent_of_the_distance_on_this_grid():
    """T1-B's within-family margin is not identified, and the freeze must say so."""
    text = " ".join(prereg.BASELINE_COLLINEARITY)
    assert "+0.92" in text and "+0.97" in text, "the measured range must be in the record"
    assert "cross-family" in text, (
        "the freeze must name where T1-B's claim can still live")
    assert "grid change and not an analysis change" in text, (
        "the remedy must be stated, or this reads as a reason to re-fit")
    assert prereg.BASELINE_SCOPE["acquisition_parameter"].startswith("within-family"), (
        "the collinearity bites exactly where the baseline is defined")


def test_a_one_sided_partial_correlation_may_not_settle_attribution():
    """The pre-specification passed and the claim is still refused."""
    text = " ".join(prereg.SPREAD_RESIDUAL_UNATTRIBUTABLE)
    assert "+0.068" in text and "+0.128" in text, (
        "both directions must be recorded, which is the whole point")
    assert "NOT shown to be explained" in text
    entry = [t for _, t in prereg.AMENDMENTS if "BASELINE_COLLINEARITY" in t]
    assert len(entry) == 1
    assert "pre-specification was defective" in entry[0], (
        "the defect was in the test design, not the statistic, and must be legible")
    assert "both NARROW" in entry[0]


def test_the_sixteenth_amendment_keeps_the_single_widening_entry_unique():
    dates = [d for d, _ in prereg.AMENDMENTS]
    assert dates == sorted(dates), "the audit trail must read in order"
    widening = [t for _, t in prereg.AMENDMENTS if "LOWERS THE PRIMARY ENDPOINT" in t]
    assert len(widening) == 1, "still exactly one amendment has moved an endpoint"
    for key in ("baseline_collinearity", "spread_residual_unattributable"):
        assert key in prereg.as_dict(), f"{key} must be exported in the fingerprint"
