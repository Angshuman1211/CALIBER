"""T1-C -- the three baselines Tool 2 is scored against (``prereg.TOOL2_BASELINES``).

**Why these three and not ``nearest``.** The leave-one-operator-out harness originally
compared Tool 2 against the single closest calibration operator. That comparison is
contaminated two ways, and D-070 Sec. 3 retired it. First, these grids are dense, so
``nearest`` is already almost ``matched`` and the weighting has nothing left to win --
which says more about the grid than about the method. Second, and worse, ``nearest`` is
*invalid* in 3 of 7 interior folds: it posted the smallest flagged area of any arm while
averaging coverage 0.547, because flagging almost nothing is cheap once you have stopped
catching anything. An arm's area is only efficiency if the arm is valid, so a baseline that
wins on area by being broken is not a baseline.

The three here each answer a different objection:

===========================  =========================================================
``max_quantile``             The **trivially-safe rule**. Valid in every calibration
                             group by construction, so beating it on area is a real
                             efficiency claim rather than an artifact of grid spacing.
``pooled``                   The thing a sensible person does with no grouping at all:
                             one threshold from every calibration operator at once.
``group_conditional_oracle`` The **ceiling**, calibrated on the held-out operator's own
                             rows. Not deployable; reported to show the headroom left.
===========================  =========================================================

**Direction, stated because the name does not fix it.** Flags are ``score >= tau``
(:func:`~caliber.calibration.calibrators.apply_flags`), so a *lower* ``tau`` flags more and
``-inf`` flags everything. "Max-quantile" therefore means the maximum *coverage level*
attainable across the calibration groups -- the **minimum** threshold -- and not the
maximum ``tau``, which would be the least safe choice available. The reading is pinned by
a test asserting the arm is valid wherever any per-group arm is, which is the property the
frozen name is appealing to; taking the max ``tau`` fails it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calibrators import CalibrationReport, _calibration_rows, calibrate_naive
from .conformal import CalibrationResult, split_threshold

__all__ = [
    "calibrate_group_conditional_oracle",
    "calibrate_max_quantile",
    "calibrate_pooled",
]


def calibrate_max_quantile(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "label",
    group_col: str = "calib_group",
    alpha: float = 0.1,
    split_col: str = "split",
) -> CalibrationReport:
    """The trivially-safe rule: the most conservative per-group threshold, applied to all.

    Calibrate every group that can certify, then take the **smallest** threshold among
    them. Because flags are ``score >= tau``, that threshold is at or below every group's
    own, so this arm's coverage is at least each group's -- valid in all of them at once,
    with no assumption about which one the deployment operator resembles.

    **Uncertifiable groups are excluded from the minimum, not treated as ``-inf``.** A
    group too thin to certify has no threshold to be safe against, so letting its ``-inf``
    into the minimum would send this arm to "flag everything" whenever any group anywhere
    was thin -- a degenerate baseline that compares against nothing. They are recorded in
    ``results`` instead, so :attr:`CalibrationReport.reportable` still goes False and the
    caller can see which groups dropped out.
    """
    cal = _calibration_rows(df, split_col)
    if cal.empty:
        raise KeyError("no calibration rows to build the max-quantile baseline from")

    results: dict[str, CalibrationResult] = {}
    for group, rows in cal.groupby(cal[group_col].astype(str), sort=True):
        results[str(group)] = split_threshold(
            rows[score_col].to_numpy(), rows[label_col].to_numpy(), alpha)

    usable = [r.threshold for r in results.values()
              if r.certifiable and np.isfinite(r.threshold)]
    # No group certifies: say so with the conservative threshold rather than a quiet NaN.
    tau = float(min(usable)) if usable else float(-np.inf)
    return CalibrationReport("max_quantile", alpha, {"__all__": tau}, results)


def calibrate_pooled(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "label",
    group_col: str = "calib_group",
    alpha: float = 0.1,
    split_col: str = "split",
) -> CalibrationReport:
    """One threshold from every calibration operator pooled together.

    This is :func:`calibrate_naive` with no ``calibration_group``, which that function's
    docstring warns against -- correctly, in its own setting: pooling a *deployment*
    regime's low-scoring rows into the calibration set drags the threshold down and makes
    naive transfer look healthier than it is.

    **That objection does not apply here, and the reason is structural.** The holdout
    harness has already removed the deployment operator from the frame, so there is no
    deployment regime left to contaminate the pool. Pooling the operators that genuinely
    were calibrated on is the honest form of "ignore the grouping", which is the objection
    this baseline exists to answer. Call it on the held-out pool and nothing else.
    """
    report = calibrate_naive(
        df, calibration_group=None, score_col=score_col, label_col=label_col,
        group_col=group_col, alpha=alpha, split_col=split_col)
    return CalibrationReport("pooled", alpha, dict(report.thresholds),
                             dict(report.results))


def calibrate_group_conditional_oracle(
    df: pd.DataFrame,
    deployment_group: str,
    score_col: str = "score",
    label_col: str = "label",
    group_col: str = "calib_group",
    alpha: float = 0.1,
    split_col: str = "split",
) -> CalibrationReport:
    """The ceiling: calibrated on the held-out operator's own calibration rows.

    **Not deployable, and reported anyway.** It is the number that says how much the
    deployable arms leave on the table, and without it a small area difference between
    two deployable arms cannot be read as near-optimal or as both-bad.

    ``df`` must be the held-out operator's *calibration* side. Passing the rows it is
    scored on makes every metric in-sample, which is the defect D-022 and D-027 record.
    """
    return calibrate_naive(
        df, calibration_group=deployment_group, score_col=score_col,
        label_col=label_col, group_col=group_col, alpha=alpha, split_col=split_col)
