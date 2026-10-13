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
