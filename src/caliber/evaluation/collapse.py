"""The curve-collapse test (T1.4, Sec. 21 R5).

**Track 1 owns this module.** The cross-domain claim has a strong form and a weak one, and
this is the test that separates them.

**Strong form.** The two modalities' coverage-gap curves *collapse onto one curve* when
plotted against operator distance. If they do, the gap is a function of how far the operator
moved and not of the modality, which is what "the failure is a property of the acquisition in
general" would mean — a single reliability-versus-distance law for imaging inverse problems.

**Weak form.** The same failure and the same fix appear in both modalities, with the ordering
by distance demonstrated where the distance is a genuine scalar.

D70 already expects the strong form to fail: the gap is ordered by signed distance on CT
(rho = +0.976) and barely on MRI (+0.316), because MRI's worst failures sit at matched
acceleration across mask families where the scalar distance is ~0. This module turns that
expectation into a measurement, because "we looked at the figure" is not a test and a
reviewer will ask for the statistic.

## What is tested, and the honest limits of it

Two nested fits of gap on distance:

* **pooled** — one slope and intercept for both modalities (the strong form);
* **separate** — slope and intercept per modality (the weak form).

Reported together, because they fail differently:

1. **An F-test** on the nested improvement. Parametric, and with ~17 points and 2 extra
   parameters it is the conventional answer, so it is here — with its assumptions named
   rather than buried.
2. **Bootstrap intervals on each modality's slope**, resampled over points within modality.
   This is the one that actually decides it: collapse requires the slopes to be
   indistinguishable, and overlapping intervals is a weaker claim than a non-significant
   F-test but a more robust one at this sample size.

**The limitation that matters.** The two modalities occupy different distance ranges — MRI's
points cluster near the top of its axis, CT's spread along it — so a permutation test on
modality labels is not exchangeable and is deliberately not offered. Non-overlapping ranges
also mean that even a passing test would be extrapolating one modality's law into the other's
range. **A positive result here would be weak evidence; a negative one is strong**, which is
the asymmetry to state in the write-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Bootstrap replicates for the per-modality slope intervals.
N_BOOT = 2000

#: Fewest points in a modality before its slope is estimated at all. Two points fit a line
#: exactly and say nothing about its uncertainty.
MIN_POINTS = 4

#: The coverage noise floor the pooled fit's extra error must stay under for the curves to
#: count as collapsed (D69): the patient split moves coverage by 0.034-0.048, so a
#: discrepancy below ~0.05 is not a discrepancy.
#:
#: This, and not the F-test, carries the verdict. The F-test asks whether the two fits differ
#: *at all*, and it is hypersensitive where the residuals are small: a control with slopes
#: 0.3025 and 0.3029 and overlapping intervals came back at p = 0.031, because RSS_separate
#: was near zero and a negligible intercept difference filled the denominator. Statistical
#: significance is not the question -- whether the single curve is wrong by more than the
#: measurement noise is.
NOISE_FLOOR = 0.05


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Slope, intercept and residual sum of squares for a straight-line fit."""
    if x.size < 2 or np.ptp(x) == 0:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return float(slope), float(intercept), float(np.sum(residual**2))


def _slope_interval(
    x: np.ndarray, y: np.ndarray, n_boot: int, alpha: float, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, x.size, size=x.size)
        slope, _, _ = _ols(x[idx], y[idx])
        if np.isfinite(slope):
            draws.append(slope)
    if len(draws) < n_boot // 10:
        return float("nan"), float("nan")
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


@dataclass(frozen=True)
class CollapseResult:
    """Whether one curve fits both modalities, and the evidence either way."""

    per_modality: dict[str, dict[str, float]]
    rss_pooled: float
    rss_separate: float
    f_statistic: float
    f_p_value: float
    slopes_overlap: bool
    n_points: int
    ranges_overlap: bool
    rmse_pooled: float = float("nan")
    rmse_separate: float = float("nan")
    noise_floor: float = NOISE_FLOOR
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def excess_rmse(self) -> float:
        """How much worse the single curve fits, in coverage units."""
        return float(self.rmse_pooled - self.rmse_separate)

    @property
    def collapses(self) -> bool:
        """Collapse requires indistinguishable slopes *and* a practically adequate fit.

        The magnitude test is the binding one. See `NOISE_FLOOR` for why the F-test is
        reported but does not decide.
        """
        return bool(self.slopes_overlap and self.excess_rmse < self.noise_floor)

    def summary(self) -> str:
        lines = []
        for name, stats in sorted(self.per_modality.items()):
            lines.append(
                f"  {name:<6} slope {stats['slope']:+.4f} "
                f"[{stats['slope_lo']:+.4f}, {stats['slope_hi']:+.4f}]  "
                f"spearman {stats['spearman']:+.3f}  n={int(stats['n'])}"
            )
        verdict = (
            "COLLAPSES: one curve is not rejected -- but read the range caveat below"
            if self.collapses
            else "DOES NOT COLLAPSE: the modalities need separate curves"
        )
        lines.append(
            f"  RMSE pooled {self.rmse_pooled:.4f} vs separate {self.rmse_separate:.4f} "
            f"-- the single curve costs {self.excess_rmse:+.4f} coverage against a "
            f"{self.noise_floor:.2f} floor (D69)"
        )
        lines.append(
            f"  slopes {'overlap' if self.slopes_overlap else 'do not overlap'}; "
            f"F = {self.f_statistic:.2f}, p = {self.f_p_value:.4f} "
            f"(reported, not decisive -- hypersensitive at small residuals)"
        )
        lines.append(f"  {verdict}")
        if not self.ranges_overlap:
            lines.append(
                "  CAUTION: the modalities' distance ranges do not overlap, so a collapse "
                "would be an extrapolation of one law into the other's range. A negative "
                "result here is strong; a positive one is weak."
            )
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def curve_collapse_test(
    points: pd.DataFrame,
    modality_col: str = "modality",
    distance_col: str = "distance",
    gap_col: str = "gap",
    n_boot: int = N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
    noise_floor: float = NOISE_FLOOR,
) -> CollapseResult:
    """Does one gap-against-distance curve fit both modalities?

    ``points`` is one row per (modality, deployment operator) with the signed operator
    distance from that modality's calibration operator and the measured coverage gap.
    """
    for col in (modality_col, distance_col, gap_col):
        if col not in points.columns:
            raise KeyError(f"points has no {col!r} column")
    frame = points.dropna(subset=[distance_col, gap_col])
    modalities = sorted(frame[modality_col].astype(str).unique())
    if len(modalities) < 2:
        raise ValueError(
            f"the collapse test compares modalities and this has {len(modalities)}: "
            f"{modalities}"
        )

    notes: list[str] = []
    per_modality: dict[str, dict[str, float]] = {}
    rss_separate = 0.0
    params_separate = 0
    spans: list[tuple[float, float]] = []

    from ..scoring.attribution import spearman

    for i, name in enumerate(modalities):
        part = frame[frame[modality_col].astype(str) == name]
        x = part[distance_col].to_numpy(dtype=float)
        y = part[gap_col].to_numpy(dtype=float)
        slope, intercept, rss = _ols(x, y)
        if x.size < MIN_POINTS:
            notes.append(
                f"{name} has {x.size} point(s); below {MIN_POINTS} a slope interval is not "
                "estimated and the test is underpowered"
            )
        lo, hi = (
            _slope_interval(x, y, n_boot, alpha, seed + i)
            if x.size >= MIN_POINTS
            else (float("nan"), float("nan"))
        )
        per_modality[name] = {
            "slope": slope, "intercept": intercept, "slope_lo": lo, "slope_hi": hi,
            "spearman": spearman(x, y), "n": float(x.size), "rss": rss,
        }
        rss_separate += 0.0 if np.isnan(rss) else rss
        params_separate += 2
        spans.append((float(x.min()), float(x.max())))

    x_all = frame[distance_col].to_numpy(dtype=float)
    y_all = frame[gap_col].to_numpy(dtype=float)
    _, _, rss_pooled = _ols(x_all, y_all)

    n = int(x_all.size)
    df_num = params_separate - 2
    df_den = n - params_separate
    if df_den > 0 and rss_separate > 0 and np.isfinite(rss_pooled):
        f_stat = ((rss_pooled - rss_separate) / df_num) / (rss_separate / df_den)
        f_p = _f_sf(f_stat, df_num, df_den)
    else:
        f_stat, f_p = float("nan"), float("nan")
        notes.append("too few points for an F-test; read the slope intervals instead")

    overlaps = all(
        not (a[1] < b[0] or b[1] < a[0])
        for a in spans
        for b in spans
    )
    intervals = [
        (v["slope_lo"], v["slope_hi"]) for v in per_modality.values()
        if np.isfinite(v["slope_lo"])
    ]
    slopes_overlap = (
        all(not (a[1] < b[0] or b[1] < a[0]) for a in intervals for b in intervals)
        if len(intervals) == len(modalities)
        else True
    )
    if len(intervals) < len(modalities):
        notes.append(
            "at least one modality has no slope interval, so 'slopes overlap' defaults to "
            "true and the verdict rests on the F-test alone"
        )

    return CollapseResult(
        per_modality=per_modality,
        rmse_pooled=float(np.sqrt(rss_pooled / n)) if n else float("nan"),
        rmse_separate=float(np.sqrt(rss_separate / n)) if n else float("nan"),
        noise_floor=noise_floor,
        rss_pooled=float(rss_pooled),
        rss_separate=float(rss_separate),
        f_statistic=float(f_stat),
        f_p_value=float(f_p),
        slopes_overlap=bool(slopes_overlap),
        n_points=n,
        ranges_overlap=bool(overlaps),
        notes=tuple(notes),
    )


def _f_sf(f: float, d1: int, d2: int) -> float:
    """Upper tail of the F distribution, via the regularised incomplete beta.

    Written out rather than imported so the analysis path does not depend on scipy, which is
    a heavyweight import for one survival function and has broken at least one development
    install in this project.
    """
    if not np.isfinite(f) or f < 0 or d1 <= 0 or d2 <= 0:
        return float("nan")
    if f == 0:
        # The separate fits improved on the pooled one not at all, so there is no evidence
        # against collapse. p = 1, not "undefined".
        return 1.0
    x = d2 / (d2 + d1 * f)
    return float(_betainc(d2 / 2.0, d1 / 2.0, x))


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) by continued fraction (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    from math import exp, lgamma

    front = exp(
        lgamma(a + b) - lgamma(a) - lgamma(b) + a * np.log(x) + b * np.log(1.0 - x)
    )
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)

    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        else:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        d = 1.0 + numerator * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + numerator / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0) / a
