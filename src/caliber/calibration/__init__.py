"""Shift-robust calibration — **Track 1 owns this module** (task T1.2).

It lives in the shared package because the evaluation harness has to be able to
call it, but the methods here are Track 1's deliverable. Track 2's job is to
keep the score bank rich enough that everything below runs on cached values
with no GPU.

What lands here over the term:

* Week 2  — group-conditional (Mondrian) calibration skeleton (Tool 1)
* Week 2  — unweighted shift-robust calibration logic (Stage 4)
* Week 5  — fallbacks: doubly-robust [8], adaptive conformal [9],
            group-weighted [7]
* Week 6  — operator-distance-weighted recalibration (Tool 2, the primary
            contribution) with effective-sample-size tracking

``split_conformal_threshold`` and ``mondrian_thresholds`` below are minimal
reference implementations, present only so the Week 4 evaluation harness has
something to exercise before Track 1's versions land. They are marked as
references and must not be used for reported results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_conformal_threshold(
    scores: np.ndarray | pd.Series,
    labels: np.ndarray | pd.Series,
    alpha: float = 0.1,
) -> float:
    """Reference split-conformal threshold controlling the region-level miss rate.

    Chooses the largest threshold whose empirical miss rate on the calibration
    regions is at most ``alpha``, with the standard finite-sample correction:
    the ``ceil((n+1)(1-alpha))``-th smallest score among hallucinated regions.

    Reference implementation. Track 1's T1.2 supersedes this.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels).astype(bool).ravel()
    positive = np.sort(scores[labels])
    n = positive.size
    if n == 0:
        return float("inf")  # nothing to catch; flag nothing
    k = int(np.ceil((n + 1) * alpha)) - 1
    k = min(max(k, 0), n - 1)
    return float(positive[k])


def mondrian_thresholds(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    group_col: str = "calib_group",
    alpha: float = 0.1,
) -> dict[str, float]:
    """One threshold per acquisition regime — the guaranteed backbone (Tool 1).

    Exact within-group coverage when the deployment operator belongs to a
    known, finite set. Thin groups give noisy thresholds, which is the declared
    trade-off and the reason Week 5 stands up the group-weighted fallback [7].

    Reference implementation. Track 1's T1.2 supersedes this.
    """
    return {
        str(group): split_conformal_threshold(part[score_col], part[label_col], alpha)
        for group, part in df.groupby(group_col, dropna=False)
    }


def apply_threshold(
    df: pd.DataFrame,
    thresholds: float | dict[str, float],
    score_col: str,
    group_col: str = "calib_group",
    out_col: str = "flag",
) -> pd.DataFrame:
    """Attach a boolean flag column by comparing scores against the threshold(s)."""
    out = df.copy()
    if isinstance(thresholds, dict):
        default = float("inf")
        mapped = out[group_col].astype(str).map(thresholds).fillna(default)
    else:
        mapped = float(thresholds)
    out[out_col] = out[score_col] >= mapped
    return out


__all__ = ["apply_threshold", "mondrian_thresholds", "split_conformal_threshold"]
