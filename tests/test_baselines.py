"""The non-conformal reference detectors and the ranking metrics they are scored with.

Every metric is pinned to a value worked out by hand, because these numbers go into the
midterm comparison and a ranking metric that is subtly wrong (ties, empty classes) would
mislead without raising.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from caliber.evaluation.baselines import (
    auroc,
    average_precision,
    cluster_bootstrap_metric,
    hallucination_index,
    posterior_dispersion,
)
from caliber.scoring.regions import RegionGrid


def test_auroc_matches_the_hand_count():
    # pairs (pos, neg): (0.35>0.1) (0.35<0.4) (0.8>0.1) (0.8>0.4) -> 3 of 4
    assert auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)


def test_auroc_extremes_and_ties():
    assert auroc([0, 1, 2, 3], [0, 0, 1, 1]) == 1.0
    assert auroc([3, 2, 1, 0], [0, 0, 1, 1]) == 0.0
    assert auroc([5, 5, 5, 5], [0, 1, 0, 1]) == pytest.approx(0.5), "a tie is worth one half"


def test_auroc_is_undefined_without_both_classes():
    assert math.isnan(auroc([1, 2], [1, 1]))
    assert math.isnan(auroc([1, 2], [0, 0]))


def test_average_precision_matches_the_hand_count():
    # ranked 0.9(+) 0.8(-) 0.7(+) 0.6(-): precision 1/1 at the first hit, 2/3 at the second
    assert average_precision([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([0.9, 0.8, 0.1], [1, 1, 0]) == 1.0


def test_average_precision_breaks_ties_against_the_detector():
    """A detector that cannot separate two regions is not credited with luck."""
    assert average_precision([5, 5], [1, 0]) == pytest.approx(0.5)
    assert math.isnan(average_precision([1, 2], [0, 0]))


def test_identical_draws_have_no_dispersion():
    grid = RegionGrid(image_shape=(8, 8), region_size=4)
    draws = torch.rand(1, 8, 8).repeat(3, 1, 1)
    assert torch.equal(posterior_dispersion(draws, grid, ref=1.0), torch.zeros(4))


def test_dispersion_is_the_region_mean_of_the_pixel_std_over_ref():
    """Two draws differing by d in one region: unbiased std is |d|/sqrt(2) per pixel."""
    grid = RegionGrid(image_shape=(8, 8), region_size=4)
    base = torch.zeros(8, 8)
    other = base.clone()
    other[:4, :4] = 2.0
    got = posterior_dispersion(torch.stack([base, other]), grid, ref=0.5)
    expected = 2.0 / math.sqrt(2.0) / 0.5
    assert got[0].item() == pytest.approx(expected)
    assert torch.equal(got[1:], torch.zeros(3))


def test_dispersion_refuses_a_single_draw():
    grid = RegionGrid(image_shape=(8, 8), region_size=4)
    with pytest.raises(ValueError, match="at least two"):
        posterior_dispersion(torch.rand(1, 8, 8), grid, ref=1.0)


def test_the_cluster_bootstrap_brackets_its_point_estimate():
    rng = np.random.default_rng(1)
    labels = rng.random(600) < 0.5
    scores = labels + rng.normal(0, 1.0, 600)  # informative but noisy
    clusters = np.repeat(np.arange(30), 20)
    r = cluster_bootstrap_metric(auroc, scores, labels, clusters, n_boot=200)
    assert r["n_clusters"] == 30
    assert r["lo"] <= r["value"] <= r["hi"]
    assert 0.5 < r["value"] < 1.0


# -- the Hallucination Index [22] --------------------------------------------


def test_the_closed_form_is_the_hellinger_distance_between_equal_variance_gaussians():
    """Checked against the definition, integrated numerically, not against itself."""
    mu_p, mu_q, sigma = 0.7, 0.0, 0.5
    x = np.linspace(-6, 7, 200001)

    def pdf(m):
        return np.exp(-((x - m) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

    integrate = getattr(np, "trapezoid", None) or np.trapz  # renamed in NumPy 2
    bc = integrate(np.sqrt(pdf(mu_p) * pdf(mu_q)), x)
    hellinger = np.sqrt(1 - bc)
    # one pixel, and so many draws that the Monte Carlo correction vanishes
    hi, hnr = hallucination_index([abs(mu_p - mu_q)], [sigma], n_pixels=1, n_draws=10**9)
    assert hi[0] == pytest.approx(hellinger, rel=1e-6)
    assert hnr[0] == pytest.approx((mu_p - mu_q) ** 2 / sigma**2, rel=1e-6)


def test_a_mean_off_only_by_its_monte_carlo_error_scores_zero():
    """The mean of N draws misses the truth by sigma/sqrt(N) even with no hallucination."""
    sigma, n = 0.2, 4
    hi, hnr = hallucination_index([sigma / np.sqrt(n)], [sigma], n_pixels=256, n_draws=n)
    assert hnr[0] == 0.0 and hi[0] == 0.0


def test_unbiased_draws_average_to_no_hallucination():
    """Simulated: draws = truth + noise. The corrected ratio sits near zero, not near 1/N."""
    rng = np.random.default_rng(0)
    sigma, n, pix = 0.1, 4, 256
    draws = rng.normal(0.0, sigma, size=(2000, n, pix))
    err = np.sqrt((draws.mean(axis=1) ** 2).mean(axis=1))
    std = draws.std(axis=1, ddof=1).mean(axis=1)
    _, hnr = hallucination_index(err, std, n_pixels=pix, n_draws=n)
    assert float(np.mean(hnr)) < 0.05


def test_the_index_is_bounded_and_grows_with_the_bias():
    hi, hnr = hallucination_index([0.05, 0.1, 0.2, 0.4], [0.1] * 4, n_pixels=16, n_draws=4)
    assert np.all((hi >= 0) & (hi < 1))
    assert np.all(np.diff(hnr) > 0) and np.all(np.diff(hi) >= 0)


@pytest.mark.parametrize("std, n_pixels, n_draws", [([0.0], 1, 4), ([0.1], 0, 4), ([0.1], 1, 1)])
def test_degenerate_inputs_are_refused(std, n_pixels, n_draws):
    with pytest.raises(ValueError):
        hallucination_index([0.1], std, n_pixels=n_pixels, n_draws=n_draws)
