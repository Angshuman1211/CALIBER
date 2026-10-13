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


__all__ = ["auroc", "average_precision", "cluster_bootstrap_metric", "posterior_dispersion"]
