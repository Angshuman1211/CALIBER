"""Non-conformal reference detectors, for context (Blueprint §15.1).

The Blueprint lists "posterior-sample dispersion and published hallucination indices" as
detectors that come with **no coverage guarantee**. They are not competitors to the
calibrated methods; they answer a narrower question -- how well does a popular uncertainty
signal rank hallucinated regions, before anyone calibrates anything?

**Posterior dispersion is not ``s_disagree``.** CALIBER's ``s_disagree`` is the spread of
the posterior draws in the operator's *null space* only (T1.1), a deliberate choice: spread
in the measured component is solver noise, not invented content. The detector the
literature uses is the spread of the *whole image*. Reporting ``s_disagree`` as the
baseline would compare CALIBER against one of its own components, so the full-image
version is computed here, with the same per-region reduction and the same reference scale
so the two stay directly comparable.

**The published Hallucination Index needs the ground truth** (:func:`hallucination_index`),
so it enters as a reference measure of hallucination -- a yardstick for the project's own
label and its score -- not as a detector to compete with them.

The ranking metrics are written out rather than imported: the project has no scikit-learn,
and two short, tested functions are cheaper than a dependency.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from torch import Tensor

from ..scoring.regions import RegionGrid


def posterior_dispersion(samples: Tensor, grid: RegionGrid, ref: float) -> Tensor:
    """Per-region mean of the pixelwise standard deviation across posterior draws.

    ``samples`` is ``(n_posterior, H, W)``, as the runner passes it to the score. Divided by
    ``ref`` -- the item's ``ref_intensity`` -- exactly as ``s_disagree`` is, so the two
    differ only in *which* spread they measure: the whole image here, the null space there.
    """
    if samples.dim() != 3:
        raise ValueError(f"expected (n_posterior, H, W), got {tuple(samples.shape)}")
    if samples.shape[0] < 2:
        raise ValueError("dispersion needs at least two posterior draws")
    draws = samples.abs() if samples.is_complex() else samples
    spread = draws.std(dim=0)
    return grid.reduce(spread, how="mean") / (ref or 1.0)


def hallucination_index(err_rms, noise_std, n_pixels: int, n_draws: int):
    """Tivnan et al.'s Hallucination Index [22], per region, from cached statistics.

    **The published index** is the Hellinger distance between the distribution of a
    generative model's reconstructions and a *zero-hallucination reference*: the ground truth
    plus noise at the reconstructions' own apparent signal-to-noise ratio. Both are taken as
    Gaussian, the reconstructions' covariance as shift-invariant (one noise power spectrum),
    and the index reads as the share of the error that is hallucination rather than noise.
    It needs the ground truth, so it is an evaluation metric, never a deployable detector.

    **Adaptation, stated rather than hidden.**

    - The reference shares the reconstructions' covariance: "the same apparent noise" taken
      literally. With equal covariances the closed form reduces to
      ``H = sqrt(1 - exp(-D2 / 8))`` with ``D2 = (mu - x0)' S^-1 (mu - x0)``; the
      log-determinant terms cancel.
    - The covariance is locally stationary: one variance per region (``noise_std``,
      the region's posterior spread), the regional form of the paper's per-image spectrum.
    - The mean comes from ``n_draws`` samples (4 here, against the paper's 64), so the
      observed ``err_rms**2`` carries ``noise_std**2 / n_draws`` of Monte Carlo variance.
      That is subtracted, so a hallucination-free region scores 0 on average rather than
      ``n_pixels / n_draws``.

    Returns ``(hi, hnr)``. ``hnr`` is the hallucination-to-noise ratio, squared bias over noise
    variance per pixel. ``D2 = n_pixels * hnr``, so ``hi`` is a monotone function of ``hnr``.
    Over a 256-pixel region ``hi`` saturates near 1 once ``hnr`` passes ~0.2 -- Hellinger
    distances do in high dimension -- so rank by ``hnr``, which loses nothing.
    """
    err = np.asarray(err_rms, dtype=float)
    std = np.asarray(noise_std, dtype=float)
    if n_pixels < 1 or n_draws < 2:
        raise ValueError("need at least one pixel and two draws")
    if (std <= 0).any():
        raise ValueError("noise_std must be positive: a region with no spread has no index")
    bias2 = np.maximum(err**2 - std**2 / n_draws, 0.0)
    hnr = bias2 / std**2
    hi = np.sqrt(1.0 - np.exp(-n_pixels * hnr / 8.0))
    return hi, hnr


def auroc(scores, labels) -> float:
    """Area under the ROC curve, as the Mann-Whitney U statistic; ties count one half.

    The probability that a randomly chosen hallucinated region outscores a randomly chosen
    clean one. ``nan`` when either class is empty.
    """
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels).astype(bool).ravel()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Average ranks over ties, so a tied pair contributes one half. Vectorised: this runs
    # inside a cluster bootstrap over ~23k regions per operator, where a Python loop over
    # ties would cost hours.
    ranks = rankdata(s, method="average")
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def average_precision(scores, labels) -> float:
    """Average precision: mean precision at the rank of each hallucinated region.

    Ties are broken pessimistically (clean regions first within a tie), so a detector
    that cannot separate classes is not credited with a lucky ordering. ``nan`` when there
    are no hallucinated regions.
    """
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels).astype(bool).ravel()
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.lexsort((y.astype(int), -s))  # score descending; within a tie, clean first
    hits = y[order]
    precision_at_hit = np.cumsum(hits)[hits] / (np.flatnonzero(hits) + 1.0)
    return float(precision_at_hit.mean())


def cluster_bootstrap_metric(metric, scores, labels, clusters, n_boot: int = 500,
                             seed: int = 0) -> dict[str, float]:
    """``metric`` with a 95% interval from resampling whole clusters (D-027: patients)."""
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels).astype(bool).ravel()
    c = np.asarray(clusters).ravel()
    ids, inverse = np.unique(c, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(ids))]
    rng = np.random.default_rng(seed)
    point = metric(s, y)
    draws = []
    for _ in range(n_boot):
        pick = np.concatenate([members[k] for k in rng.integers(0, len(ids), len(ids))])
        draws.append(metric(s[pick], y[pick]))
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    lo, hi = (np.quantile(draws, [0.025, 0.975]) if len(draws) else (np.nan, np.nan))
    return {"value": point, "lo": float(lo), "hi": float(hi), "n_clusters": len(ids)}


__all__ = ["auroc", "average_precision", "cluster_bootstrap_metric", "hallucination_index",
           "posterior_dispersion"]
