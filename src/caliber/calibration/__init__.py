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

T1.2 has landed. The real implementations are in ``conformal.py`` (threshold
machinery) and ``calibrators.py`` (the three tools over a score bank), and are
re-exported here.

``split_conformal_threshold`` and ``mondrian_thresholds`` remain so the Week 4
harness keeps working, but they are **no longer separate implementations** — D-026
(Track 1's D32) is closed by making them thin wrappers over
``conformal.split_threshold``.

The bug they carried is worth remembering rather than deleting. They took the
``ceil((n+1)*alpha)``-th smallest positive score, while their docstring claimed the
``ceil((n+1)(1-alpha))``-th; neither was right. Rounding *up* certifies thin groups it
should refuse — at ``n=1, alpha=0.1`` it returns that single positive score as a
threshold whose true miss rate is **1/2**, five times the promise, with nothing in the
return value saying so. Flooring sends those groups to ``certifiable=False`` instead.

On the Week 3 bank the two agree to about 0.001 because every group carries thousands
of positives, so nothing reported from it moves. That is why the bug survived four
weeks: it is invisible exactly where the data is thick, and wrong exactly where a
reviewer would look hardest.

These wrappers return a bare ``float`` and so cannot express ``certifiable``. Anything
that needs to *report* a threshold should call ``conformal.split_threshold`` and read
the flag; the scripts now do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_conformal_threshold(
    scores: np.ndarray | pd.Series,
    labels: np.ndarray | pd.Series,
    alpha: float = 0.1,
) -> float:
    """Split-conformal threshold controlling the region-level miss rate.

    Takes the ``floor(alpha * (n + 1))``-th smallest score among hallucinated
    calibration regions, which makes ``P(miss) = k / (n + 1) <= alpha`` exact.

    **This now delegates to** :func:`caliber.calibration.conformal.split_threshold`
    (D-026). It is kept because the Week 4 harness and two scripts call it by this
    name, but it flattens the result to a float and therefore **cannot tell the caller
    that a group was uncertifiable** — a group too thin to certify comes back as
    ``-inf``, which flags every region: valid, useless, and silent about it. Call
    ``split_threshold`` directly wherever that distinction has to be reported.

    .. note::

       Two behaviours changed when this stopped being its own implementation, both
       toward correctness. It used to round ``k`` *up*, certifying thin groups whose
       true miss rate was as high as 1/2. And with no positives at all it returned
       ``+inf`` — flag nothing — which is a 100% miss rate presented as success; the
       delegate returns ``-inf``, which is the conservative direction.
    """
    result = split_threshold(
        np.asarray(scores, dtype=float).ravel(),
        np.asarray(labels).astype(bool).ravel(),
        alpha=alpha,
    )
    return float(result.threshold)


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

    Delegates per group to :func:`caliber.calibration.conformal.split_threshold`
    (D-026). A group with fewer than ``min_positives(alpha)`` positives cannot be
    certified and lands at ``-inf``, flagging everything in that group. Use
    :func:`mondrian_calibration_results` where the per-group flag is needed rather
    than only the threshold.
    """
    return {
        group: result.threshold
        for group, result in mondrian_calibration_results(
            df, score_col, label_col, group_col, alpha
        ).items()
    }


def mondrian_calibration_results(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    group_col: str = "calib_group",
    alpha: float = 0.1,
) -> dict[str, CalibrationResult]:
    """``mondrian_thresholds`` without discarding ``certifiable`` and ``n_positives``.

    The dict-of-floats form cannot say that a group was too thin to certify, and a
    caller that reports its numbers needs to know. D-026 is only half fixed by
    flooring; the other half is giving the caller somewhere to read the flag.
    """
    return {
        str(group): split_threshold(
            np.asarray(part[score_col], dtype=float).ravel(),
            np.asarray(part[label_col]).astype(bool).ravel(),
            alpha=alpha,
        )
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


from .calibrators import (  # noqa: E402
    CalibrationReport,
    apply_flags,
    calibrate_mondrian,
    calibrate_naive,
    calibrate_weighted,
    certification_note,
)
from .fallbacks import (  # noqa: E402
    calibrate_group_weighted,
    group_weights,
    nearest_certifiable_set,
)
from .conformal import (  # noqa: E402
    CalibrationResult,
    distance_weights,
    effective_sample_size,
    min_positives,
    split_threshold,
    weighted_threshold,
)

__all__ = [
    # T1.2 — use these
    "CalibrationReport",
    "CalibrationResult",
    "apply_flags",
    "calibrate_mondrian",
    "calibrate_naive",
    "calibrate_weighted",
    "certification_note",
    "distance_weights",
    "effective_sample_size",
    "min_positives",
    "split_threshold",
    "weighted_threshold",
    # Track 2 reference implementations — not for reported results
    "apply_threshold",
    "mondrian_thresholds",
    "mondrian_calibration_results",
    "split_conformal_threshold",
]
