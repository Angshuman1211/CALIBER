"""T3.5: the distance estimators, the corrected Step 1 chain, and the scripts that print them.

Three kinds of gate, following P-19's lesson (exercising a library is not executing the
script that calls it):

* estimator correctness on laws with known answers;
* the *claims* T3.5 makes, on a controlled synthetic pair -- ``ks <= tv``, the corrected chain
  ``gap <= ks``, and the Kolmogorov/TV distinction behind ``sqrt(D * W1)``;
* the committed-bank script actually runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from caliber.calibration.conformal import split_threshold
from caliber.scoring.score import ScoreWeights
from caliber.theory import (
    distribution_distances,
    ks_distance,
    tv_distance,
    w1_distance,
)
from caliber.theory.harness import HarnessConfig, frozen_scales, make_mri, run_trials

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def test_w1_of_a_translation_is_the_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, 20_000)
    assert w1_distance(a, a + 0.7) == pytest.approx(0.7, abs=1e-9)


def test_identical_samples_are_at_distance_zero():
    a = np.random.default_rng(1).normal(size=2000)
    rep = distribution_distances(a, a.copy())
    assert rep.ks == 0.0 and rep.w1 == 0.0 and rep.tv == pytest.approx(0.0, abs=1e-9)


def test_kolmogorov_never_exceeds_total_variation():
    """The distinction T3.3 sec. 3(iii) missed: sup|G| is ks, not tv."""
    rng = np.random.default_rng(2)
    for shift in (0.1, 0.5, 1.5):
        a, b = rng.normal(0, 1, 6000), rng.normal(shift, 1.3, 6000)
        assert ks_distance(a, b) <= tv_distance(a, b) + 0.03  # KDE smoothing tolerance


def test_the_tent_bound_holds_for_ks_on_a_density_pair():
    """sqrt(D * W1) bounds the Kolmogorov distance (it is the tent argument's conclusion)."""
    rng = np.random.default_rng(3)
    for shift in (0.2, 0.6, 1.0):
        rep = distribution_distances(rng.normal(0, 1, 8000), rng.normal(shift, 1, 8000))
        assert rep.ks_bound_holds


def test_total_variation_can_exceed_sqrt_d_w1():
    """High-frequency density difference: tv fixed, ks and w1 shrink, D fixed (analytic)."""
    x = np.linspace(0, 1, 200_001)
    dx = x[1] - x[0]
    eps, k = 0.5, 20
    qa, qb = np.ones_like(x), 1 + eps * np.sin(2 * np.pi * k * x)
    g = np.cumsum(qa - qb) * dx
    ks, w1 = np.abs(g).max(), np.abs(g).sum() * dx
    tv, d = 0.5 * np.abs(qa - qb).sum() * dx, np.abs(qa - qb).max()
    assert ks <= np.sqrt(d * w1)
    assert tv > np.sqrt(d * w1)


def test_corrected_chain_gap_is_bounded_by_ks():
    """coverage_dep >= (1 - alpha) - ks, with the repo's own calibrator, both directions."""
    rng = np.random.default_rng(4)
    alpha = 0.1
    cal = rng.normal(0.0, 1.0, 40_000)
    for shift in (0.3, 1.0):
        dep = rng.normal(shift, 1.0, 40_000)
        for src, dst in ((cal, dep), (dep, cal)):
            gaps = []
            for _ in range(300):
                pick = rng.choice(src.size, size=300, replace=False)
                res = split_threshold(src[pick], np.ones(300, dtype=bool), alpha)
                gaps.append((1 - alpha) - float((dst >= res.threshold).mean()))
            assert np.mean(gaps) <= ks_distance(src, dst) + 0.02


def test_calibrating_on_few_dependent_clusters_undercovers_without_any_shift():
    """T3.3 sec. 4b item 2: regions of one image are not exchangeable with a fresh image's."""
    rng = np.random.default_rng(5)
    alpha, per_image = 0.1, 16

    def images(n):
        return rng.normal(0, 1, (n, 1)) + 0.5 * rng.normal(0, 1, (n, per_image))

    covs = []
    for _ in range(600):
        cal, test = images(2).ravel(), images(50).ravel()
        res = split_threshold(cal, np.ones(cal.size, dtype=bool), alpha)
        covs.append(float((test >= res.threshold).mean()))
    assert np.mean(covs) < 0.86  # nominal 0.90; observed ~0.77 on the real harness (T3.5 sec. 4)


@pytest.fixture(scope="module")
def small_cfg():
    return HarnessConfig(image_shape=(64, 64), region_size=16, n_scale_items=10, n_draws=4)


def test_harness_is_paired_and_reproducible(small_cfg):
    op = make_mri(small_cfg, "random1d", 4)
    a, b = run_trials(op, small_cfg, 3), run_trials(op, small_cfg, 3)
    assert np.array_equal(a.components.numpy(), b.components.numpy())


def test_free_component_ratio_follows_the_sqrt_null_frac_shape(small_cfg):
    """T3.4 sec. 2b: measured / closed form is near-constant across R on nested masks."""
    cfg = small_cfg
    from caliber.operators.distance import null_space_projection_distance as dop

    cal = make_mri(cfg, "random1d", 2)
    scales = frozen_scales(cal, cfg)
    base = run_trials(cal, cfg, 12)
    w = ScoreWeights.only("free")
    bs = base.scores(w, scales)
    a = cal.descriptor().null_space_dim / (64 * 64)
    quotients = []
    for r in (4, 8):
        dep = make_mri(cfg, "random1d", r)
        d = dop(cal, dep)
        nf = dep.descriptor().null_space_dim / (64 * 64)
        closed = (nf**0.5 - a**0.5) / d
        measured = (run_trials(dep, cfg, 12).scores(w, scales) - bs).abs().mean().item() / d
        quotients.append(measured / closed)
    assert max(quotients) / min(quotients) < 1.25
    assert all(1.0 < q < 2.0 for q in quotients)


def test_bank_check_script_runs_on_the_committed_bank(capsys):
    import t3_5_bank_check

    t3_5_bank_check.main([str(REPO_ROOT / "banks" / "w02_solver_smoke")])
    out = capsys.readouterr().out
    assert "R4->R8" in out and "bound_agreement" in out


def test_cluster_proposition_grid_step_holds_with_the_stated_probability():
    """T3.3 sec. 2b(c): the grid deviation exceeds eps with probability <= 2K exp(-2 m eps^2)."""
    from scipy.stats import norm

    rng = np.random.default_rng(6)
    m, r, K, eps = 60, 16, 20, 0.25
    grid = norm.ppf(np.arange(1, K) / K)  # F_cal(t_k) = k/K for the common marginal N(0, 1.25)
    scale = np.sqrt(1.0 + 0.25)
    bad = 0
    reps = 400
    for _ in range(reps):
        centres = rng.normal(0, 1.0, (m, 1))  # shared per-image component -> dependent regions
        scores = (centres + 0.5 * rng.normal(0, 1, (m, r))).ravel()
        t = grid * scale
        emp = (scores[None, :] <= t[:, None]).mean(axis=1)
        dev = np.abs(emp - np.arange(1, K) / K)
        # between grid points monotonicity adds at most 1/K to the sup deviation
        if dev.max() > eps:
            bad += 1
    assert bad / reps <= 2 * K * np.exp(-2 * m * eps**2) + 0.02


def test_label_conditional_decomposition_is_exact_algebra():
    """T3.3 sec. 3b: d_K^+ <= (sup|G_A - G_B| + |pi_A - pi_B|) / pi_A for any coupled sample."""
    from t3_5_conditional_link import sup_joint_gap

    rng = np.random.default_rng(7)
    for _ in range(40):
        n = 600
        sa = rng.normal(0, 1, n)
        sb = sa + rng.normal(0.3, 0.5, n)
        la = rng.random(n) < 0.3 + 0.4 * (sa > 0.5)
        lb = rng.random(n) < 0.3 + 0.4 * (sb > 0.2)
        if la.sum() < 5 or lb.sum() < 5:
            continue
        alg = (sup_joint_gap(sa, la, sb, lb) + abs(la.mean() - lb.mean())) / la.mean()
        assert ks_distance(sa[la], sb[lb]) <= alg + 1e-9


def test_ratio_of_sums_lemma_is_exact_algebra():
    """|A_bar/p_hat - A/p_bar| <= (|A_bar - A| + |p_bar - p_hat|)/p_hat whenever A <= p_bar."""
    rng = np.random.default_rng(8)
    for _ in range(2000):
        p_bar, p_hat = rng.uniform(0.05, 1, 2)
        a = rng.uniform(0, p_bar)
        a_bar = rng.uniform(0, p_hat)
        lhs = abs(a_bar / p_hat - a / p_bar)
        assert lhs <= (abs(a_bar - a) + abs(p_bar - p_hat)) / p_hat + 1e-12


def test_ratio_of_sums_hoeffding_events_for_variable_positives_per_image():
    """T3.3 sec. 2b(d): the Hoeffding events fail with probability <= 2(K+1) exp(-2 m eta^2)."""
    rng = np.random.default_rng(9)
    m, r, K, eta = 400, 16, 10, 0.1

    def draw(n_img):
        centre = rng.normal(0, 1, (n_img, 1))
        s = centre + 0.5 * rng.normal(0, 1, (n_img, r))
        lab = rng.random((n_img, r)) < 1 / (1 + np.exp(-(centre + 0.5 * s)))
        return s, lab

    s0, l0 = draw(20_000)
    p_bar = l0.mean()
    tk = np.quantile(s0, np.arange(1, K + 1) / (K + 1))  # grid in score space
    a_pop = np.array([((s0 <= t) & l0).mean() for t in tk])
    bad, reps = 0, 300
    for _ in range(reps):
        s, lab = draw(m)
        a_hat = np.array([((s <= t) & lab).mean() for t in tk])
        if np.abs(a_hat - a_pop).max() > eta or abs(lab.mean() - p_bar) > eta:
            bad += 1
    assert bad / reps <= 2 * (K + 1) * np.exp(-2 * m * eta**2) + 0.01
