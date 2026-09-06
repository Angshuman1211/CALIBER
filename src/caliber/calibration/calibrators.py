"""T1.2 — the three calibrators, operating on a score-bank DataFrame.

Each takes cached rows and returns per-region flags plus enough bookkeeping to
say whether the result may be reported. Grouping and split semantics follow the
Week 1 sign-off: groups are ``(modality, family, severity)`` and never split by
mask seed (D24); the calibration/test split is fixed per patient in the registry
and is operator-independent (D23).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .conformal import (
    CalibrationResult,
    distance_weights,
    min_positives,
    split_threshold,
    weighted_threshold,
)

__all__ = ["CalibrationReport", "calibrate_mondrian", "calibrate_naive", "calibrate_weighted"]


@dataclass
class CalibrationReport:
    """Outcome of one calibration, with the caveats attached rather than implied."""

    method: str
    alpha: float
    thresholds: dict[str, float]
    results: dict[str, CalibrationResult] = field(default_factory=dict)

    @property
    def uncertifiable_groups(self) -> list[str]:
        """Groups too thin to certify at ``alpha`` — D3's declared trade-off, measured."""
        return sorted(g for g, r in self.results.items() if not r.certifiable)

    @property
    def reportable(self) -> bool:
        """False if any group failed to certify. Check this before quoting coverage."""
        return not self.uncertifiable_groups

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "group": g,
                    "threshold": r.threshold,
                    "n_positives": r.n_positives,
                    "ess": r.ess,
                    "certifiable": r.certifiable,
                }
                for g, r in sorted(self.results.items())
            ]
        )


def _calibration_rows(df: pd.DataFrame, split_col: str) -> pd.DataFrame:
    if split_col not in df.columns:
        return df
    return df[df[split_col] == "calib"]


def calibrate_naive(
    df: pd.DataFrame,
    calibration_group: str | None = None,
    score_col: str = "score",
    label_col: str = "label",
    group_col: str = "calib_group",
    alpha: float = 0.1,
    split_col: str = "split",
) -> CalibrationReport:
    """Tool 0 — one threshold, calibrated at one operator and applied everywhere.

    **This is the baseline that fails, not a method.** Sec. 16's first bullet:
    *calibrate at one acquisition setting and apply at others*. It is here so
    the failure can be measured through the same code path as the remedies;
    reporting it as a remedy would invert the paper's claim.

    ``calibration_group`` names the single regime to calibrate on, and should
    almost always be given. Omitting it pools **every** regime in the frame,
    which produces a mixture threshold that is neither the baseline nor a
    method: pooling a low-scoring deployment regime into the calibration set
    drags the threshold down and makes naive transfer look far healthier than
    it is. That understates the very collapse this baseline exists to expose.
    """
    cal = _calibration_rows(df, split_col)
    if calibration_group is not None:
        cal = cal[cal[group_col].astype(str) == str(calibration_group)]
        if cal.empty:
            raise KeyError(f"no calibration rows for group {calibration_group!r}")
    res = split_threshold(cal[score_col].to_numpy(), cal[label_col].to_numpy(), alpha)
    return CalibrationReport("naive", alpha, {"__all__": res.threshold}, {"__all__": res})


def calibrate_mondrian(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "label",
    group_col: str = "calib_group",
    alpha: float = 0.1,
    split_col: str = "split",
) -> CalibrationReport:
    """Tool 1 — one threshold per acquisition regime. The guaranteed backbone.

    Exact within-group coverage when the deployment operator belongs to a known,
    finite set, with no assumption about the shift (D3). Its structural limits
    are the reason Tool 2 exists: a regime with too few positives cannot certify
    at all, and an operator that falls *between* regimes has no group to be
    calibrated in.

    Thin groups are reported through ``uncertifiable_groups`` rather than
    silently handed a noisy threshold.
    """
    cal = _calibration_rows(df, split_col)
    results = {
        str(g): split_threshold(part[score_col].to_numpy(), part[label_col].to_numpy(), alpha)
        for g, part in cal.groupby(group_col, dropna=False)
    }
    return CalibrationReport(
        "mondrian", alpha, {g: r.threshold for g, r in results.items()}, results
    )


def calibrate_weighted(
    df: pd.DataFrame,
    deployment_group: str,
    distance_of: dict[str, float],
    score_col: str = "score",
    label_col: str = "label",
    group_col: str = "calib_group",
    alpha: float = 0.1,
    bandwidth: float | None = None,
    split_col: str = "split",
) -> CalibrationReport:
    """Tool 2 — operator-distance weighted recalibration. The primary contribution.

    Every calibration region contributes, weighted by how far its acquisition
    operator sits from the deployment operator under D19's distance. Unlike
    Tool 1 this borrows across regimes, which is what lets it serve thin groups
    and off-grid deployment operators — the two cases Q4 requires the experiment
    to demonstrate, and the only honest argument for preferring an approximate
    method over an exact one.

    ``distance_of`` maps each calibration group to its D19 distance from the
    deployment operator. Supplying it rather than recomputing keeps one
    definition of distance across all three tracks (Sec. 19, operator registry).

    Approximate by construction — see `weighted_threshold` for why, and check
    ``ess`` before trusting the number.
    """
    cal = _calibration_rows(df, split_col)
    missing = set(cal[group_col].astype(str)) - set(distance_of)
    if missing:
        raise KeyError(f"no operator distance supplied for groups: {sorted(missing)}")

    distances = cal[group_col].astype(str).map(distance_of).to_numpy(dtype=float)
    weights = distance_weights(distances, bandwidth)
    res = weighted_threshold(
        cal[score_col].to_numpy(),
        cal[label_col].to_numpy(),
        weights,
        alpha,
        test_weight=float(np.exp(-0.0)),  # deployment operator is at distance 0
    )
    return CalibrationReport(
        "weighted", alpha, {deployment_group: res.threshold}, {deployment_group: res}
    )


def apply_flags(
    df: pd.DataFrame,
    report: CalibrationReport,
    score_col: str = "score",
    group_col: str = "calib_group",
    out_col: str = "flag",
) -> pd.DataFrame:
    """Attach boolean flags. A group with no threshold is flagged entirely.

    Flagging an unknown group wholesale is the conservative direction: it costs
    efficiency, never validity. Returning no flag would silently assert that
    nothing in an uncalibrated regime is hallucinated.
    """
    out = df.copy()
    if len(report.thresholds) == 1 and "__all__" in report.thresholds:
        tau = pd.Series(report.thresholds["__all__"], index=out.index)
    elif len(report.thresholds) == 1:
        tau = pd.Series(next(iter(report.thresholds.values())), index=out.index)
    else:
        tau = out[group_col].astype(str).map(report.thresholds).astype(float)
        tau = tau.fillna(-np.inf)
    out[out_col] = out[score_col].to_numpy() >= tau.to_numpy()
    return out


def certification_note(report: CalibrationReport, alpha: float | None = None) -> str:
    """One line stating what may and may not be claimed from this calibration."""
    alpha = report.alpha if alpha is None else alpha
    if report.reportable:
        return (
            f"{report.method}: region-level miss rate certified at alpha={alpha:g} "
            f"for all {len(report.results)} group(s)."
        )
    thin = report.uncertifiable_groups
    return (
        f"{report.method}: NOT reportable at alpha={alpha:g} — "
        f"{len(thin)} of {len(report.results)} group(s) below the "
        f"{min_positives(alpha)}-positive minimum: {', '.join(thin)}."
    )
