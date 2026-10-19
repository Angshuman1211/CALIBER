"""Fallback [7] — group-weighted calibration with hard truncation (T1.2, D58).

**Track 1 owns this module.** The risk register listed three fallbacks for the case where
Tool 2 underperforms: [7] group-weighted, [8] doubly-robust, [9] adaptive conformal. D52
measured Tool 2 as valid on 2 of 3 interior folds and no better than the conservative
fallback, so the trigger has fired. This is [7], and it is first because it is the one that
addresses the *measured* failure rather than a hypothetical one.

## What went wrong with Tool 2, in one paragraph

`calibrate_weighted` weights every calibration region by `exp(-d/h)` in its operator's
distance from the deployment operator. On the interior fold of the phantom bank the oracle
threshold is 0.8152 and Tool 2 picks 0.5658 -- *below the deployment operator's minimum
score*, so it flags everything. Sweeping the bandwidth over 0.02-0.8 moves it by 0.05 and
never changes the outcome. The reason is structural: **the alpha-quantile is a lower-tail
statistic, so it is set by the lowest-scoring regime in the mixture, and an exponential
kernel down-weights smoothly without ever excluding.** At `h = 0.08` the far regime still
holds 14% relative weight, which at alpha = 0.10 is enough to occupy the whole tail. Drive
the weight to zero and the effective sample size collapses, at which point D32 refuses to
certify and the threshold returns -inf.

So Tool 2's accuracy wants concentrated weight and its certifiability wants spread weight,
and on an ordered grid no single `h` gives both.

## What [7] does differently

Two changes, and both are the textbook content of "group-weighted" rather than inventions
here.

**1. Hard truncation.** Weight is assigned per group and a group may receive *exactly*
zero. Removing the contaminating regime outright is what a smooth kernel cannot do, and it
is the lead D52 recorded. Tool 2 is the special case of this with nothing truncated, so [7]
is a strict generalisation and not a competitor.

**2. Equal mass per group.** Tool 2 gives every *region* in a group the same weight, so a
group with three times the regions contributes three times the total mass and a large,
distant regime can outvote a small, near one on size alone. Here each retained group
contributes a fixed share of the total, divided among its own regions. On the current MRI
grid every operator sees every item so the two agree; on any bank with unequal groups --
CT's `ct/{family}/v{views}/a{arc}` after P-34, or a grid with a dropped volume -- they do
not.

## How the truncation set is chosen

The tension D52 identified is *resolvable* rather than merely nameable: accuracy wants few
groups, certifiability wants many, and D32 says exactly how many are enough -- the
nine-positive floor at alpha = 0.1. So :func:`nearest_certifiable_set` adds groups in
distance order and stops at the first set that certifies. That is the **tightest**
calibration set that can still carry a guarantee, which is the best available answer to
both requirements at once rather than a compromise between them.

It can fail, and says so: if every group is included and the result still does not certify,
there is no admissible set and the caller gets that rather than a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calibrators import CalibrationReport, _calibration_rows
from .conformal import CalibrationResult, min_positives, weighted_threshold

#: Weight below which a group is treated as excluded outright. Not a tuning knob: it exists
#: so that `radius`-based truncation and the kernel agree about what "zero" means.
ZERO_WEIGHT = 1e-12


def group_weights(
    distance_of: dict[str, float],
    retain: list[str] | None = None,
    bandwidth: float | None = None,
    uniform: bool = False,
) -> dict[str, float]:
    """Per-group weights, zero outside ``retain``, normalised to sum to one.

    ``uniform`` gives every retained group the same weight, which is the honest choice when
    the distances are not trusted to be graded -- D-052 measured that four of the five
    registered measures coincide on MRI masks and that a mask-family change saturates all of
    them, so on a cross-family grid the distances carry an ordering they do not really have.
    The default is still the exponential kernel, so that with nothing truncated this reduces
    to Tool 2.
    """
    groups = sorted(distance_of)
    keep = set(groups if retain is None else retain)
    unknown = keep - set(groups)
    if unknown:
        raise KeyError(f"no distance supplied for retained group(s): {sorted(unknown)}")
    if not keep:
        raise ValueError("the retained set is empty; there is nothing to calibrate on")

    kept = [g for g in groups if g in keep]
    if uniform:
        raw = {g: 1.0 for g in kept}
    else:
        distances = np.array([distance_of[g] for g in kept], dtype=float)
        nonzero = distances[distances > 0]
        h = bandwidth if bandwidth else (float(np.median(nonzero)) if nonzero.size else 1.0)
        h = h if h > 0 else 1.0
        raw = {g: float(np.exp(-distance_of[g] / h)) for g in kept}

    total = sum(raw.values())
    if total <= 0:
        raise ValueError(
            "every retained group received zero weight; the bandwidth is far below the "
            "distances, which is the degenerate case D52 reached at h -> 0"
        )
    out = {g: 0.0 for g in groups}
    out.update({g: w / total for g, w in raw.items()})
    return out


@dataclass(frozen=True)
class TruncationChoice:
    """Which groups were retained, and whether the result could certify."""

    retained: tuple[str, ...]
    result: CalibrationResult
    considered: int
    exhausted: bool

    @property
    def certifiable(self) -> bool:
        return self.result.certifiable

    def note(self) -> str:
        if self.certifiable:
            return (
                f"retained {len(self.retained)} of {self.considered} group(s) in distance "
                f"order: {', '.join(self.retained)}. {self.result.n_positives} positives, "
                f"ESS {self.result.ess:.1f}."
                if self.result.ess is not None
                else f"retained {len(self.retained)} of {self.considered} group(s)."
            )
        floor = min_positives(self.result.alpha)
        if self.result.n_positives < floor:
            cause = (
                f"only {self.result.n_positives} positives against the {floor} needed at "
                f"alpha={self.result.alpha:g} (D32)"
            )
        else:
            cause = (
                f"{self.result.n_positives} positives is above D32's floor of {floor}, so "
                f"the threshold degenerated for another reason -- check the test-point "
                f"weight against the calibration mass before concluding the method refused"
            )
        return (
            f"NOT certifiable after retaining all {self.considered} group(s): {cause}. "
            f"There is no admissible calibration set here."
        )


def nearest_certifiable_set(
    df: pd.DataFrame,
    distance_of: dict[str, float],
    score_col: str = "score",
    label_col: str = "label",
    group_col: str = "calib_group",
    alpha: float = 0.1,
    bandwidth: float | None = None,
    uniform: bool = False,
    split_col: str = "split",
) -> TruncationChoice:
    """Smallest set of nearest groups whose weighted threshold certifies at ``alpha``.

    Groups are added in increasing operator distance and the threshold recomputed after
    each, stopping at the first set that clears D32's floor. This is the direct resolution
    of D52's tension: the nearest groups give the least contaminated lower tail, and the
    floor says when enough of them have been added to carry a guarantee.

    Returns the choice even when it fails, so the caller can report "no admissible set"
    rather than receiving a threshold that cannot be certified.
    """
    cal = _calibration_rows(df, split_col)
    order = sorted(distance_of, key=lambda g: (distance_of[g], g))
    present = set(cal[group_col].astype(str))
    order = [g for g in order if g in present]
    if not order:
        raise ValueError(
            "none of the groups with distances appear in the calibration rows; check the "
            "group column and the split filter"
        )

    last: CalibrationResult | None = None
    for i in range(1, len(order) + 1):
        retained = order[:i]
        weights = group_weights(
            distance_of, retain=retained, bandwidth=bandwidth, uniform=uniform
        )
        res = _weighted_over_groups(cal, weights, score_col, label_col, group_col, alpha)
        last = res
        if res.certifiable:
            return TruncationChoice(tuple(retained), res, len(order), exhausted=False)
    assert last is not None
    return TruncationChoice(tuple(order), last, len(order), exhausted=True)


def _weighted_over_groups(
    cal: pd.DataFrame,
    weights: dict[str, float],
    score_col: str,
    label_col: str,
    group_col: str,
    alpha: float,
) -> CalibrationResult:
    """Threshold from a per-group weight vector, with equal mass per retained group.

    Each group's share is divided among its own regions, so a group's influence is its
    weight and not its size -- the second of [7]'s two changes from Tool 2.
    """
    groups = cal[group_col].astype(str)
    sizes = groups.value_counts().to_dict()
    per_region = groups.map(
        lambda g: weights.get(g, 0.0) / max(sizes.get(g, 1), 1)
    ).to_numpy(dtype=float)

    keep = per_region > ZERO_WEIGHT
    if not keep.any():
        return CalibrationResult(-np.inf, 0, alpha, certifiable=False, ess=0.0)
    # The test point is one region acquired under the deployment operator, which sits at
    # distance zero and so carries the largest per-region weight in play. Tool 2 passes
    # exp(0) = 1.0 against *unnormalised* weights, where that is exactly the maximum.
    #
    # Getting this wrong is not subtle but it is silent: a first version passed
    # `1 / n_groups` against group masses normalised to sum to one, which made the test
    # atom's own probability 0.33 against alpha = 0.10. The -inf atom then satisfies the
    # quantile condition on its own and every threshold comes back -inf -- a degenerate
    # result that reads as "the method refused to certify" rather than as a scaling error.
    return weighted_threshold(
        cal.loc[keep, score_col].to_numpy(),
        cal.loc[keep, label_col].to_numpy(),
        per_region[keep],
        alpha,
        test_weight=float(per_region[keep].max()),
    )


def calibrate_group_weighted(
    df: pd.DataFrame,
    deployment_group: str,
    distance_of: dict[str, float],
    score_col: str = "score",
    label_col: str = "label",
    group_col: str = "calib_group",
    alpha: float = 0.1,
    bandwidth: float | None = None,
    retain: list[str] | None = None,
    radius: float | None = None,
    uniform: bool = False,
    split_col: str = "split",
) -> CalibrationReport:
    """Fallback [7] — group-weighted calibration with truncation.

    ``retain`` names the groups to keep explicitly; ``radius`` keeps those within a distance
    of the deployment operator; with neither, :func:`nearest_certifiable_set` chooses the
    tightest set that certifies.

    Approximate in the same way and for the same reason as Tool 2 -- see
    `weighted_threshold` -- with one addition worth stating: truncation makes the implied
    weight function discontinuous in operator distance, so the Lipschitz argument T3.3 uses
    for the smooth kernel does not transfer to it. **A bound proved for Tool 2 does not
    cover this fallback**, which is a real cost and is why [7] is a fallback rather than a
    replacement.
    """
    cal = _calibration_rows(df, split_col)
    if retain is None and radius is not None:
        retain = [g for g, d in distance_of.items() if d <= radius]
        if not retain:
            raise ValueError(
                f"radius {radius:g} retains no group; the nearest is at "
                f"{min(distance_of.values()):.4f}"
            )

    if retain is None:
        choice = nearest_certifiable_set(
            df, distance_of, score_col=score_col, label_col=label_col,
            group_col=group_col, alpha=alpha, bandwidth=bandwidth, uniform=uniform,
            split_col=split_col,
        )
        report = CalibrationReport(
            "group_weighted", alpha, {deployment_group: choice.result.threshold},
            {deployment_group: choice.result},
        )
        report.truncation = choice  # type: ignore[attr-defined]
        return report

    weights = group_weights(distance_of, retain=retain, bandwidth=bandwidth, uniform=uniform)
    res = _weighted_over_groups(cal, weights, score_col, label_col, group_col, alpha)
    report = CalibrationReport(
        "group_weighted", alpha, {deployment_group: res.threshold}, {deployment_group: res}
    )
    report.truncation = TruncationChoice(  # type: ignore[attr-defined]
        tuple(sorted(retain)), res, len(distance_of), exhausted=False
    )
    return report
