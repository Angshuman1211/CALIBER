"""T1.2 — shift-robust calibration: scores in, per-region flags out.

Three tools in increasing ambition (Blueprint Sec. 9), all operating on cached
score-bank values with no GPU (Sec. 17):

* **Tool 0 — split conformal.** One threshold for everything. Present as the
  *baseline that fails*: calibrated at one operator and transferred to another,
  it is the coverage collapse this project exists to demonstrate. Never a remedy.
* **Tool 1 — group-conditional (Mondrian).** One threshold per acquisition
  regime. Exact within-group when the deployment operator belongs to a known,
  finite set, with no assumption about the shift. The guaranteed backbone (D3).
* **Tool 2 — operator-distance weighted recalibration.** Reweights calibration
  regions by how far their operator sits from the deployment operator. The
  primary novel contribution (D4), and *approximate* — see `weighted_threshold`.

What is controlled
------------------
The **region-level miss rate**: among genuinely hallucinated regions, the
fraction the flag fails to raise. This is risk control (D9), not marginal
coverage over an interval. A region is flagged when ``score >= tau``, so a miss
is a hallucinated region with ``score < tau``, and calibration picks the largest
``tau`` whose miss rate is provably at most ``alpha``.

The direction matters and is easy to get backwards: the flag must be
*conservative downward*. When in doubt the threshold drops, more regions are
flagged, and efficiency suffers rather than validity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CalibrationResult",
    "distance_weights",
    "effective_sample_size",
    "min_positives",
    "split_threshold",
    "weighted_threshold",
]


def min_positives(alpha: float) -> int:
    """Fewest hallucinated calibration regions that can certify level ``alpha``.

    With ``n`` positives the smallest achievable miss-rate bound is ``1/(n+1)``
    — reached by thresholding at the minimum score. So certifying ``alpha``
    needs ``n + 1 >= 1/alpha``. At the usual ``alpha = 0.1`` that is **9
    positives**, per group.

    This is D3's "thin groups give noisy thresholds" made checkable. Below this
    count no threshold is valid at the requested level, and the honest answer is
    to say so rather than return a number that looks like a guarantee.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return max(1, math.ceil(1.0 / alpha) - 1)


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size, ``(sum w)^2 / sum w^2``.

    The honest accounting of what reweighting costs (Sec. 15.2). Equals ``n``
    for uniform weights and collapses toward 1 as the weight mass concentrates
    on a few regions — at which point Tool 2's threshold is being set by a
    handful of examples regardless of how many were nominally available.
    """
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0:
        return 0.0
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = w.sum()
    if total <= 0:
        return 0.0
    return float(total**2 / np.square(w).sum())


@dataclass(frozen=True)
class CalibrationResult:
    """A threshold plus everything needed to judge whether to trust it."""

    threshold: float
    n_positives: int
    alpha: float
    certifiable: bool
    ess: float | None = None

    @property
    def flags_everything(self) -> bool:
        return not np.isfinite(self.threshold) and self.threshold < 0

    def __post_init__(self) -> None:
        if self.n_positives < 0:
            raise ValueError("n_positives must be non-negative")


def split_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.1,
) -> CalibrationResult:
    """Unweighted split-conformal threshold controlling the region-level miss rate.

    Take the ``k``-th smallest score among hallucinated calibration regions with
    ``k = floor(alpha * (n + 1))``. For exchangeable scores the new positive's
    rank is uniform on ``1..n+1``, so

        P(miss) = P(S_new < S_(k)) = k / (n + 1) <= alpha

    exactly, with no slack beyond the flooring.

    **Floor, not ceiling.** Rounding ``k`` up would return a threshold from as
    few as one positive: at ``n = 1, alpha = 0.1`` it gives ``k = 1``, so
    ``tau`` is that single score and the true miss rate is ``1/2`` by symmetry —
    five times the promise, reported as if certified. Flooring sends thin groups
    to ``certifiable=False`` instead, which is the whole point of tracking the
    count. See `min_positives`.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels).astype(bool).ravel()
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} and labels {labels.shape} must align")

    positive = np.sort(scores[labels])
    n = int(positive.size)
    k = int(math.floor(alpha * (n + 1)))

    if k < 1:
        # Cannot certify. -inf flags every region: trivially zero misses, so it
        # is valid, and useless in a way the caller can see and report.
        return CalibrationResult(-np.inf, n, alpha, certifiable=False)
    return CalibrationResult(float(positive[k - 1]), n, alpha, certifiable=True)


def distance_weights(
    distances: np.ndarray,
    bandwidth: float | None = None,
) -> np.ndarray:
    """Exponential kernel in operator distance: ``w = exp(-d / h)``.

    ``distances[i]`` is D19's operator distance between region ``i``'s
    calibration operator and the deployment operator; zero means the region was
    acquired under the deployment operator itself and gets full weight.

    **Bandwidth.** ``None`` selects the median of the non-zero distances, which
    is scale-free and adapts to whatever grid the experiment happens to use.
    Fixing ``h`` in absolute units would silently change the method's behaviour
    when the operator grid changes, which is exactly the confound T1.4's grids
    are designed to avoid.

    **Why exponential rather than a hard cut-off.** A cut-off is a Mondrian
    group with extra steps, and it discards the near-neighbours that carry most
    of the usable information about a nearby operator. The exponential decays
    smoothly, so a deployment operator sitting between two calibration regimes
    borrows from both — which is the case Tool 1 structurally cannot serve
    (D4), and therefore the case Tool 2 exists for.
    """
    d = np.asarray(distances, dtype=float).ravel()
    if np.any(d < 0):
        raise ValueError("operator distances must be non-negative")
    if bandwidth is None:
        nonzero = d[d > 0]
        bandwidth = float(np.median(nonzero)) if nonzero.size else 1.0
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    return np.exp(-d / bandwidth)


def weighted_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.1,
    test_weight: float | None = None,
) -> CalibrationResult:
    """Operator-distance weighted threshold — Tool 2, the primary contribution.

    Weighted split conformal in the sense of Tibshirani et al. (2019), mirrored
    to the lower tail. Calibration weights are normalised together with a weight
    for the test point, and the test point contributes an atom at ``-inf``:

        p_i = w_i / (sum_j w_j + w_test),   p_test = w_test / (sum_j w_j + w_test)
        tau = inf { s : p_test + sum_{i: S_i <= s} p_i >= alpha }

    The ``-inf`` atom is what makes this conservative: it consumes ``p_test`` of
    the lower tail before any calibration score counts, pushing ``tau`` down.
    Dropping it would produce a threshold that looks tighter and is not valid.

    ``test_weight`` defaults to 1.0 — the deployment operator's distance to
    itself is zero, so ``exp(0) = 1`` under `distance_weights`.

    Honest scope
    ------------
    Tibshirani's guarantee holds when the weights are the true likelihood ratio
    between deployment and calibration. A kernel in operator distance is **not**
    that ratio; it is a defensible surrogate, and operator shift is a change in
    the measurement process rather than a covariate shift (D6), so no exact
    finite-sample guarantee is claimed here. This is the approximate tool. Tool 1
    is what carries an exact statement, and the residual gap is what Track 3's
    bound quantifies. ``ess`` is reported so the cost is visible rather than
    inferred.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels).astype(bool).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    if not (scores.shape == labels.shape == weights.shape):
        raise ValueError("scores, labels and weights must align")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    pos = labels
    s, w = scores[pos], weights[pos]
    n = int(s.size)
    w_test = 1.0 if test_weight is None else float(test_weight)
    if w_test < 0:
        raise ValueError("test_weight must be non-negative")

    total = w.sum() + w_test
    if n == 0 or total <= 0:
        return CalibrationResult(-np.inf, n, alpha, certifiable=False, ess=0.0)

    ess = effective_sample_size(w)
    order = np.argsort(s, kind="stable")
    s_sorted, p_sorted = s[order], w[order] / total
    p_test = w_test / total

    # Quantile of  p_test*delta_{-inf} + sum_i p_i*delta_{S_i}  at level alpha.
    if p_test >= alpha:
        # The test atom alone fills the lower tail: cannot place a finite
        # threshold at this level. Flag everything — valid, and visibly useless.
        return CalibrationResult(-np.inf, n, alpha, certifiable=False, ess=ess)

    cumulative = p_test + np.cumsum(p_sorted)
    idx = int(np.searchsorted(cumulative, alpha, side="left"))
    if idx >= n:
        # Even the whole calibration mass does not reach alpha: no finite
        # threshold is justified, so fall back to flagging everything.
        return CalibrationResult(-np.inf, n, alpha, certifiable=False, ess=ess)

    # An ESS below what min_positives requires means the threshold rests on
    # fewer effective examples than the level needs, whatever the raw count says.
    certifiable = ess >= min_positives(alpha)
    return CalibrationResult(float(s_sorted[idx]), n, alpha, certifiable, ess=ess)
