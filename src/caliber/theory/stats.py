"""Distances between two one-dimensional score laws, as the T3.3 derivation states them.

Four quantities, all functions of the two CDFs ``F_a, F_b`` and ``G = F_a - F_b``:

* ``ks``  = ``sup_t |G(t)|``              (Kolmogorov distance — events are half-lines)
* ``w1``  = ``int |G(t)| dt``             (Wasserstein-1, exact in one dimension)
* ``tv``  = ``(1/2) int |q_a - q_b| dt``  (total variation — events are *all* sets)
* ``D``   = ``sup_t |q_a(t) - q_b(t)|``   (density-difference sup, T3.3's constant)

``ks <= tv`` always. T3.3 §3(iii) wrote ``tv = sup|G|``, which is ``ks``; the two coincide
only for monotone-density-ratio families, not in general (T3.5 §2). ``tv`` and ``D`` need
densities, so they are estimated with a Gaussian KDE on a shared grid. They carry the
smoothing bias that implies and are reported with the bandwidth used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def _clean(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


def ks_distance(a, b) -> float:
    """Kolmogorov distance between two empirical samples, exact."""
    a, b = np.sort(_clean(a)), np.sort(_clean(b))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def w1_distance(a, b) -> float:
    """Wasserstein-1 between two empirical samples (``int |F_a - F_b|``), exact."""
    a, b = np.sort(_clean(a)), np.sort(_clean(b))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.sort(np.concatenate([a, b]))
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.sum(np.abs(fa - fb)[:-1] * np.diff(grid)))


def _bandwidth(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.concatenate([a, b])
    sd = float(np.std(pooled)) or 1.0
    iqr = float(np.subtract(*np.percentile(pooled, [75, 25]))) / 1.349
    spread = min(sd, iqr) if iqr > 0 else sd
    return 0.9 * spread * pooled.size ** (-0.2)


def _kde(sample: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    out = np.zeros_like(grid)
    for start in range(0, grid.size, 256):
        g = grid[start : start + 256, None]
        out[start : start + 256] = np.exp(-0.5 * ((g - sample[None, :]) / bw) ** 2).sum(axis=1)
    return out / (sample.size * bw * np.sqrt(2 * np.pi))


def kde_density_difference_sup(a, b, bandwidth: float | None = None, n_grid: int = 1024):
    """``(tv, D, bandwidth)`` from Gaussian KDEs on a shared grid."""
    a, b = _clean(a), _clean(b)
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan"), float("nan")
    bw = float(bandwidth) if bandwidth else _bandwidth(a, b)
    lo = min(a.min(), b.min()) - 4 * bw
    hi = max(a.max(), b.max()) + 4 * bw
    grid = np.linspace(lo, hi, n_grid)
    qa, qb = _kde(a, grid, bw), _kde(b, grid, bw)
    diff = np.abs(qa - qb)
    tv = 0.5 * float(np.trapezoid(diff, grid))
    return tv, float(diff.max()), bw


def tv_distance(a, b, bandwidth: float | None = None) -> float:
    return kde_density_difference_sup(a, b, bandwidth)[0]


@dataclass(frozen=True)
class DistanceReport:
    ks: float
    w1: float
    tv: float
    d_sup: float
    bandwidth: float
    n_a: int
    n_b: int

    @property
    def sqrt_dw1(self) -> float:
        """T3.3's ``sqrt(D * W1)``: the value the tent argument bounds ``ks`` by."""
        return float(np.sqrt(self.d_sup * self.w1))

    @property
    def ks_bound_holds(self) -> bool:
        return self.ks <= self.sqrt_dw1 * (1 + 1e-9)

    @property
    def tv_bound_holds(self) -> bool:
        return self.tv <= self.sqrt_dw1 * (1 + 1e-9)

    def as_row(self) -> dict:
        row = asdict(self)
        row.update(
            sqrt_dw1=self.sqrt_dw1,
            ks_bound_holds=self.ks_bound_holds,
            tv_bound_holds=self.tv_bound_holds,
        )
        return row


def distribution_distances(a, b, bandwidth: float | None = None) -> DistanceReport:
    """All four distances for two score samples."""
    a, b = _clean(a), _clean(b)
    tv, d_sup, bw = kde_density_difference_sup(a, b, bandwidth)
    return DistanceReport(
        ks=ks_distance(a, b),
        w1=w1_distance(a, b),
        tv=tv,
        d_sup=d_sup,
        bandwidth=bw,
        n_a=int(a.size),
        n_b=int(b.size),
    )


def joint_cdf_gap(sa, la, sb, lb) -> float:
    """``sup_t |G_A(t) - G_B(t)|`` for sub-CDFs ``G(t) = P(S <= t, label = 1)``.

    ``(sa, la)`` and ``(sb, lb)`` are score and boolean-label arrays over the same ``n`` coupled
    units. O(n log n): both sub-CDFs are evaluated on the union of scores by one sort each.
    """
    sa, sb = np.asarray(sa, float).ravel(), np.asarray(sb, float).ravel()
    la, lb = np.asarray(la, bool).ravel(), np.asarray(lb, bool).ravel()
    n = sa.size
    if n == 0 or sb.size != n:
        return float("nan")
    grid = np.sort(np.concatenate([sa, sb]))
    ia, ib = np.argsort(sa, kind="stable"), np.argsort(sb, kind="stable")
    ca, cb = np.concatenate([[0], np.cumsum(la[ia])]), np.concatenate([[0], np.cumsum(lb[ib])])
    ga = ca[np.searchsorted(sa[ia], grid, side="right")] / n
    gb = cb[np.searchsorted(sb[ib], grid, side="right")] / n
    return float(np.abs(ga - gb).max())


def crossing_probability(sa, sb) -> float:
    """``sup_t P(t lies between S_A and S_B)`` under the pairing (``lo <= t < hi``), O(n log n)."""
    sa, sb = np.asarray(sa, float).ravel(), np.asarray(sb, float).ravel()
    if sa.size == 0 or sb.size != sa.size:
        return float("nan")
    lo, hi = np.sort(np.minimum(sa, sb)), np.sort(np.maximum(sa, sb))
    grid = np.concatenate([lo, hi])
    f_lo = np.searchsorted(lo, grid, side="right") / lo.size
    f_hi = np.searchsorted(hi, grid, side="right") / hi.size
    return float((f_lo - f_hi).max())
