"""Does the headline claim depend on the score weighting at all? (T1.1, D76)

**Track 1 owns this module.** It turns a known weakness into a measurement.

D54 recorded that the design split -- 8 patients -- cannot separate any weighting from any
other, so :class:`caliber.scoring.weights.WeightSelection` reports ``discriminated=False``
and the project falls back to equal weights over the live components (D25). That is
honest, and as it stands it reads as a gap in the method: a score whose weights were never
chosen.

But "the weights could not be selected" and "the result depends on the weights" are
different statements, and only the first has been established. If the coverage claim holds
at **every** weighting on the sweep, then the inability to select one costs the project
nothing, and that is a stronger position than a selected weight would have been -- a
selected weight invites the question of what the result would have been at another.

So this measures the central claim, not the score. For each weight on the grid it
recomputes the quantity the paper leads with -- the coverage that group-conditional
calibration restores relative to a transferred threshold, as a within-patient paired
difference (:func:`caliber.evaluation.paired.paired_method_gap`) -- and reports its range
over the grid.

**The label is held fixed.** Every label form in the project is a function of the cached
truth columns (and, for D74's alignment label, of ``s_free``), none of which depend on the
weighting, so the positives do not move as the weight sweeps. That is asserted rather than
assumed: a label that moved with the weights would make this sweep meaningless, because
each weight would be scored against a different question.

What this cannot show: that the weighting is *irrelevant to detection*. It is a statement
about coverage restoration only, which per D67 is the axis the project can report on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..calibration import apply_threshold
from ..calibration.conformal import split_threshold
from ..scoring.score import ScoreWeights
from ..scoring.weights import DEFAULT_GRID
from .bank import derive_score_and_label
from .paired import DEFAULT_N_BOOT, paired_method_gap


@dataclass(frozen=True)
class WeightSensitivity:
    """The headline claim measured at every weighting on the grid."""

    deployment_operator: str
    curve: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def gaps(self) -> np.ndarray:
        return self.curve["gap"].to_numpy(dtype=float)

    @property
    def sign_is_constant(self) -> bool:
        """Does the claim point the same way at every weighting?"""
        finite = self.gaps[np.isfinite(self.gaps)]
        return bool(finite.size > 0 and (np.all(finite > 0) or np.all(finite < 0)))

    @property
    def holds_everywhere(self) -> bool:
        """Is the restoration positive *and* its interval clear of zero, at every weight?"""
        if self.curve.empty:
            return False
        return bool((self.curve["gap"] > 0).all() and (self.curve["lo"] > 0).all())

    @property
    def span(self) -> float:
        """How much the headline number moves across the whole grid."""
        finite = self.gaps[np.isfinite(self.gaps)]
        return float(finite.max() - finite.min()) if finite.size else float("nan")

    @property
    def widest_interval(self) -> float:
        """The widest per-weight interval, for comparison against :attr:`span`."""
        if self.curve.empty:
            return float("nan")
        return float((self.curve["hi"] - self.curve["lo"]).max())

    @property
    def within_noise(self) -> bool:
        """Is the movement across the grid smaller than one weight's own interval?

        The comparison that makes the claim meaningful. A span of 0.01 is only evidence of
        insensitivity if a single weighting's coverage is determined to better than 0.01 --
        otherwise the flat curve is just a curve nobody can see the shape of, which is
        D72's lesson in the other direction.
        """
        return bool(
            np.isfinite(self.span) and np.isfinite(self.widest_interval)
            and self.span < self.widest_interval
        )

    def verdict(self) -> str:
        if self.curve.empty:
            return "no weighting could be evaluated on this bank"
        lo, hi = float(np.nanmin(self.gaps)), float(np.nanmax(self.gaps))
        if self.holds_everywhere and self.within_noise:
            return (
                f"the restoration holds at all {len(self.curve)} weightings on the grid "
                f"({lo:+.3f} to {hi:+.3f}), and the spread across the grid ({self.span:.3f}) "
                f"is smaller than one weighting's own interval ({self.widest_interval:.3f}). "
                "D54's inability to select a weight therefore costs the claim nothing"
            )
        if self.holds_everywhere:
            return (
                f"the restoration holds at all {len(self.curve)} weightings ({lo:+.3f} to "
                f"{hi:+.3f}), but the spread across the grid ({self.span:.3f}) exceeds one "
                f"weighting's interval ({self.widest_interval:.3f}): the weighting does "
                "move the result, so leaving it unselected is a real limitation"
            )
        failing = int((self.curve["lo"] <= 0).sum())
        return (
            f"the restoration is not established at {failing} of {len(self.curve)} "
            f"weightings (gap {lo:+.3f} to {hi:+.3f}); the claim depends on the weighting "
            "and D54 leaves it unselected, which has to be stated as a limitation"
        )


def weight_sensitivity(
    df: pd.DataFrame,
    calibration_operator: str,
    deployment_operator: str,
    label_kwargs: dict | None = None,
    grid: tuple[float, ...] = DEFAULT_GRID,
    alpha: float = 0.10,
    split_col: str = "split",
    operator_col: str = "operator_id",
    cluster_col: str = "patient_id",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> WeightSensitivity:
    """The paired coverage restoration at each ``w_free`` on the grid.

    ``w_disagree = 1 - w_free`` and ``w_residual = 0``, matching D42 and the sweep
    :func:`caliber.scoring.weights.weight_curve` uses, so the two tables index the same
    grid and can be read side by side.
    """
    label_kwargs = dict(label_kwargs or {})
    for col in (split_col, operator_col, cluster_col):
        if col not in df.columns:
            raise KeyError(f"bank has no {col!r} column")
    sides = set(df[split_col].astype(str).unique())
    if not {"calib", "test"} <= sides:
        raise ValueError(
            f"bank has split side(s) {sorted(sides) or 'none'}; this needs both 'calib' "
            "and 'test' so the threshold is not fitted on the patients it is scored on"
        )

    rows, label_signature = [], None
    for w_free in grid:
        weights = ScoreWeights(free=float(w_free), disagree=1.0 - float(w_free), residual=0.0)
        scored = derive_score_and_label(
            df, calibration_operator, weights=weights, **label_kwargs
        )
        # The label must not move with the weighting, or each row answers a different
        # question. Checked once per weight rather than argued for in a comment.
        signature = tuple(np.flatnonzero(scored["label"].to_numpy().astype(bool)))
        if label_signature is None:
            label_signature = signature
        elif signature != label_signature:
            raise ValueError(
                "the label changed when the score weighting changed, so the weights are "
                "feeding the label and this sweep cannot be interpreted"
            )

        calib = scored[scored[split_col] == "calib"]
        test = scored[scored[split_col] == "test"]
        here = test[test[operator_col].astype(str) == str(deployment_operator)]
        away = calib[calib[operator_col].astype(str) == str(calibration_operator)]
        own = calib[calib[operator_col].astype(str) == str(deployment_operator)]
        if here.empty or away.empty or own.empty:
            continue
        naive = split_threshold(away["score"], away["label"], alpha=alpha).threshold
        group = split_threshold(own["score"], own["label"], alpha=alpha).threshold
        frame = apply_threshold(here, naive, "score", out_col="flag_naive")
        frame = apply_threshold(frame, group, "score", out_col="flag_group")
        gap = paired_method_gap(
            frame, "flag_group", "flag_naive", cluster_col=cluster_col,
            n_boot=n_boot, seed=seed,
        )
        rows.append({
            "w_free": float(w_free),
            "w_disagree": 1.0 - float(w_free),
            "coverage_group": gap.coverage_calibration,
            "coverage_naive": gap.coverage_deployment,
            "gap": gap.gap,
            "lo": gap.lo,
            "hi": gap.hi,
            "se_paired": gap.se_paired,
            "n_clusters": gap.n_clusters,
            "n_positive": gap.n_positive_calibration,
        })
    if not rows:
        raise ValueError(
            f"no weighting could be evaluated: {deployment_operator!r} needs test rows and "
            f"both it and {calibration_operator!r} need calibration rows"
        )
    return WeightSensitivity(str(deployment_operator), pd.DataFrame(rows))
