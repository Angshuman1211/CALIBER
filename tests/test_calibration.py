"""T1.2 correctness gates for shift-robust calibration.

A conformal implementation is only worth as much as its measured coverage, so
the central tests here are Monte-Carlo: draw exchangeable scores, calibrate,
and check the realised miss rate against the promise. A threshold formula that
looks right and over-covers by 5x is the failure mode that survives code review
and dies in front of an examiner.

The second theme is *refusal*. Several tests assert that the calibrator declines
to certify rather than returning a plausible-looking number — thin groups, an
over-heavy test atom, no positives at all. Silent over-promising is the specific
way a distribution-free guarantee stops being one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from caliber.calibration import (
    apply_flags,
    calibrate_mondrian,
    calibrate_naive,
    calibrate_weighted,
    distance_weights,
    effective_sample_size,
    min_positives,
    split_threshold,
    weighted_threshold,
)

ALPHA = 0.1


def _positives(rng, n, loc=0.0, scale=1.0):
    """Scores for hallucinated regions only — labels all True."""
    return rng.normal(loc, scale, n), np.ones(n, dtype=bool)


def test_min_positives_matches_the_arithmetic():
    assert min_positives(0.10) == 9  # need n+1 >= 1/alpha
    assert min_positives(0.05) == 19
    with pytest.raises(ValueError):
        min_positives(0.0)


def test_split_threshold_achieves_the_promised_miss_rate():
    """The gate that matters: measured miss rate must not exceed alpha."""
    rng = np.random.default_rng(0)
    n_cal, trials = 200, 400
    misses = 0
    for _ in range(trials):
        s, lab = _positives(rng, n_cal)
        tau = split_threshold(s, lab, ALPHA).threshold
        misses += int(rng.normal() < tau)  # one fresh exchangeable positive
    rate = misses / trials
    assert rate <= ALPHA + 0.03, f"miss rate {rate:.3f} exceeds alpha={ALPHA}"
    assert rate > ALPHA / 3, f"miss rate {rate:.3f} is far below alpha — over-conservative"


def test_thin_groups_are_refused_not_certified():
    """Flooring is what makes this refuse; a ceiling would certify n=1."""
    rng = np.random.default_rng(1)
    s, lab = _positives(rng, 5)
    res = split_threshold(s, lab, ALPHA)
    assert not res.certifiable
    assert res.threshold == -np.inf and res.flags_everything

    s, lab = _positives(rng, 9)  # exactly min_positives(0.1)
    assert split_threshold(s, lab, ALPHA).certifiable


def test_a_ceiling_rule_would_over_promise_on_one_positive():
    """Documents why `split_threshold` floors — the failure it avoids is 5x."""
    rng = np.random.default_rng(2)
    trials, misses = 2000, 0
    for _ in range(trials):
        s = rng.normal(size=1)
        tau_ceiling = float(np.sort(s)[0])  # what ceil((n+1)*alpha)=1 would give
        misses += int(rng.normal() < tau_ceiling)
    assert misses / trials > 0.4, "expected ~0.5 miss rate from a single-positive threshold"
    assert not split_threshold(s, np.ones(1, dtype=bool), ALPHA).certifiable


def test_no_positives_refuses():
    res = split_threshold(np.array([0.3, 0.7]), np.zeros(2, dtype=bool), ALPHA)
    assert not res.certifiable and res.n_positives == 0


# --- the central claim, in miniature -----------------------------------------


def _shifted_bank(rng, n_per_group=300):
    """Calibration regime scores high, deployment regime scores low.

    A threshold transferred from the first to the second sits far too high, so
    the flag misses most hallucinations — the coverage collapse of Sec. 16.
    """
    rows = []
    for group, loc in [("mri/random1d/R2", 2.0), ("mri/random1d/R8", 0.0)]:
        s, lab = _positives(rng, n_per_group, loc=loc)
        rows.append(
            pd.DataFrame(
                {"score": s, "label": lab, "calib_group": group, "split": "calib"}
            )
        )
    return pd.concat(rows, ignore_index=True)


def _miss_rate(df, report, group):
    part = df[df["calib_group"] == group]
    flagged = apply_flags(part, report)
    return 1.0 - flagged["flag"].mean()


def test_naive_transfer_loses_coverage_and_mondrian_restores_it():
    rng = np.random.default_rng(3)
    df = _shifted_bank(rng)

    naive = calibrate_naive(df, calibration_group="mri/random1d/R2", alpha=ALPHA)
    mondrian = calibrate_mondrian(df, alpha=ALPHA)

    naive_miss = _miss_rate(df, naive, "mri/random1d/R8")
    mondrian_miss = _miss_rate(df, mondrian, "mri/random1d/R8")

    assert naive_miss > 0.5, (
        f"naive transfer should collapse, got miss rate {naive_miss:.3f} at nominal {ALPHA}"
    )
    assert mondrian_miss <= ALPHA + 0.05, (
        f"Mondrian should restore the promised rate, got {mondrian_miss:.3f}"
    )
    assert mondrian.reportable


def test_mondrian_reports_thin_groups_instead_of_hiding_them():
    rng = np.random.default_rng(4)
    df = _shifted_bank(rng, n_per_group=200)
    thin = pd.DataFrame(
        {
            "score": rng.normal(size=4),
            "label": np.ones(4, dtype=bool),
            "calib_group": "mri/random1d/R16",
            "split": "calib",
        }
    )
    report = calibrate_mondrian(pd.concat([df, thin], ignore_index=True), alpha=ALPHA)
    assert "mri/random1d/R16" in report.uncertifiable_groups
    assert not report.reportable


# --- Tool 2 -------------------------------------------------------------------


def test_distance_weights_decay_and_reject_negatives():
    w = distance_weights(np.array([0.0, 0.5, 1.0]))
    assert w[0] == pytest.approx(1.0)
    assert w[0] > w[1] > w[2] > 0
    with pytest.raises(ValueError):
        distance_weights(np.array([-1.0]))


def test_effective_sample_size_tracks_concentration():
    assert effective_sample_size(np.ones(20)) == pytest.approx(20.0)
    assert effective_sample_size(np.array([1.0] + [1e-9] * 19)) == pytest.approx(1.0, abs=1e-3)


def test_test_atom_makes_the_weighted_threshold_conservative():
    """Dropping the -inf atom would raise tau, i.e. flag less. That is the bug."""
    rng = np.random.default_rng(5)
    s, lab = _positives(rng, 100)
    w = np.ones(100)

    with_atom = weighted_threshold(s, lab, w, ALPHA, test_weight=1.0).threshold
    tiny_atom = weighted_threshold(s, lab, w, ALPHA, test_weight=1e-9).threshold
    assert with_atom <= tiny_atom, "the test atom must push the threshold down, never up"


def test_weighted_achieves_the_promised_miss_rate_without_shift():
    """Sanity floor: with all weights equal, Tool 2 must match Tool 0's guarantee."""
    rng = np.random.default_rng(6)
    trials, misses = 400, 0
    for _ in range(trials):
        s, lab = _positives(rng, 200)
        tau = weighted_threshold(s, lab, np.ones(200), ALPHA).threshold
        misses += int(rng.normal() < tau)
    rate = misses / trials
    assert rate <= ALPHA + 0.03, f"weighted miss rate {rate:.3f} exceeds alpha"


def test_weighted_serves_a_thin_group_that_mondrian_cannot():
    """Q4's case: the only honest argument for an approximate method over an exact one."""
    rng = np.random.default_rng(7)
    dense = pd.DataFrame(
        {
            "score": rng.normal(0.0, 1.0, 300),
            "label": np.ones(300, dtype=bool),
            "calib_group": "mri/random1d/R6",
            "split": "calib",
        }
    )
    thin = pd.DataFrame(
        {
            "score": rng.normal(0.0, 1.0, 3),
            "label": np.ones(3, dtype=bool),
            "calib_group": "mri/random1d/R8",
            "split": "calib",
        }
    )
    df = pd.concat([dense, thin], ignore_index=True)

    mondrian = calibrate_mondrian(df, alpha=ALPHA)
    assert "mri/random1d/R8" in mondrian.uncertifiable_groups

    weighted = calibrate_weighted(
        df,
        deployment_group="mri/random1d/R8",
        distance_of={"mri/random1d/R6": 0.05, "mri/random1d/R8": 0.0},
        alpha=ALPHA,
    )
    res = weighted.results["mri/random1d/R8"]
    assert res.ess >= min_positives(ALPHA), f"ESS {res.ess:.1f} too low to certify"
    assert res.certifiable and np.isfinite(res.threshold)


def test_weighted_requires_a_distance_for_every_group():
    rng = np.random.default_rng(8)
    df = _shifted_bank(rng, n_per_group=20)
    with pytest.raises(KeyError):
        calibrate_weighted(
            df,
            deployment_group="mri/random1d/R8",
            distance_of={"mri/random1d/R8": 0.0},  # R2 missing
            alpha=ALPHA,
        )


def test_pooling_every_regime_understates_the_naive_failure():
    """Guards the trap that `calibration_group` exists to prevent."""
    rng = np.random.default_rng(11)
    df = _shifted_bank(rng)
    pooled = _miss_rate(df, calibrate_naive(df, alpha=ALPHA), "mri/random1d/R8")
    honest = _miss_rate(
        df, calibrate_naive(df, calibration_group="mri/random1d/R2", alpha=ALPHA), "mri/random1d/R8"
    )
    assert honest > pooled + 0.3, (
        "pooling the deployment regime into calibration hides the collapse "
        f"(pooled {pooled:.3f} vs transferred {honest:.3f})"
    )


def test_unknown_group_is_flagged_wholesale_not_left_unflagged():
    rng = np.random.default_rng(9)
    df = _shifted_bank(rng, n_per_group=100)
    report = calibrate_mondrian(df, alpha=ALPHA)
    unseen = pd.DataFrame(
        {"score": [0.0, 5.0], "label": [True, True], "calib_group": ["mri/new/R32"] * 2}
    )
    assert apply_flags(unseen, report)["flag"].all()


# -- D-026: the reference wrappers delegate instead of rounding up -------------


def test_the_reference_wrapper_no_longer_certifies_a_single_positive():
    """The case D-026 names, and the reason it mattered.

    The old implementation took the `ceil((n+1)*alpha)`-th smallest positive score. At
    `n=1, alpha=0.1` that is the single positive score itself, so a new positive falls
    below it half the time: a true miss rate of 1/2 against a promised 0.1, returned as
    an ordinary float with nothing marking it as junk. Flooring sends it to -inf, which
    flags everything -- valid, useless, and visibly so.
    """
    from caliber.calibration import split_conformal_threshold

    assert split_conformal_threshold(np.array([5.0]), np.array([True]), 0.1) == -np.inf


def test_no_positives_flags_everything_rather_than_nothing():
    """The other behaviour that changed, and the more dangerous of the two.

    With no hallucinated calibration regions the old code returned `+inf`, flagging
    nothing -- a 100% miss rate presented as a successful calibration. `-inf` is the
    conservative direction: catch everything, certify nothing.
    """
    from caliber.calibration import split_conformal_threshold

    assert split_conformal_threshold(np.array([5.0]), np.array([False]), 0.1) == -np.inf


def test_the_wrapper_and_the_real_implementation_cannot_drift():
    """Two implementations of one rule was the whole defect; assert there is one."""
    from caliber.calibration import split_conformal_threshold
    from caliber.calibration.conformal import split_threshold

    rng = np.random.default_rng(7)
    for n in (12, 60, 500):
        scores = rng.normal(size=n)
        labels = rng.random(n) < 0.5
        for alpha in (0.05, 0.1, 0.2):
            assert split_conformal_threshold(scores, labels, alpha) == (
                split_threshold(scores, labels, alpha=alpha).threshold
            )


def test_mondrian_results_expose_the_flag_the_float_dict_cannot():
    """A thin group has to be reportable as thin, not just quietly wrong."""
    import pandas as pd

    from caliber.calibration import mondrian_calibration_results, mondrian_thresholds
    from caliber.calibration.conformal import min_positives

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "score": rng.normal(size=200),
            "label": np.ones(200, dtype=bool),
            "calib_group": ["thick"] * 195 + ["thin"] * 5,
        }
    )
    results = mondrian_calibration_results(df, "score", "label")
    assert results["thick"].certifiable
    assert not results["thin"].certifiable
    assert results["thin"].n_positives < min_positives(0.1)

    # and the float form stays consistent with it
    taus = mondrian_thresholds(df, "score", "label")
    assert taus == {g: r.threshold for g, r in results.items()}
    assert taus["thin"] == -np.inf
