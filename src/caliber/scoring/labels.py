"""Ground-truth hallucination criterion (objective O2, Week 2, Blueprint Sec. 14).

The weekly card for Week 2 makes this a named deliverable: define the
ground-truth hallucination criterion, the region definition, and the
aggregation rule used to compute the region-level false-negative risk, and
document how MRI pathology annotations and CT anatomical annotations support
evaluating it.

Track 1 owns the criterion. Track 2 owns computing and caching the *continuous*
quantities it thresholds, so that changing the criterion later costs no GPU
time. Nothing here is written as a boolean: the bank stores per-region error
magnitudes and annotation coverage, and the label is derived at analysis time.

Note the asymmetry this criterion has to live with. A region can be wrong for
two reasons — the prior invented structure, or the prior smoothed structure
away. Only the first is hallucination in the Blueprint's sense (Sec. 4).
Restricting attention to the free component is what distinguishes them, and is
why ``err_free`` is recorded separately from ``err_total``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..operators.base import ForwardOperator
from .regions import RegionGrid

CRITERION_VERSION = "v0-error-threshold"


@dataclass(frozen=True)
class RegionTruth:
    """Continuous per-region ground-truth quantities; thresholded at analysis time."""

    err_total: Tensor  # region RMSE between reconstruction and truth
    err_free: Tensor  # region RMSE restricted to the free (null-space) component
    free_content: Tensor  # ||P_N x_true|| per region / ref -- the A1 denominator
    ann_coverage: Tensor  # fraction of the region covered by an annotation, or zeros


def compute_region_truth(
    samples: Tensor,
    x_true: Tensor,
    operator: ForwardOperator,
    grid: RegionGrid,
    annotation: Tensor | None = None,
    ref_intensity: float | None = None,
) -> RegionTruth:
    """Per-region error magnitudes for one item, scaled by reference intensity."""
    mean_recon = samples.mean(dim=0)
    x_true = x_true.abs() if x_true.is_complex() else x_true
    ref = ref_intensity or float(x_true.abs().mean()) or 1.0

    err = (mean_recon - x_true).abs()
    err_total = grid.reduce(err, how="rms") / ref

    free_err = operator.project_null(mean_recon - x_true)
    free_err = free_err.abs() if free_err.is_complex() else free_err.abs()
    err_free = grid.reduce(free_err, how="rms") / ref

    # The ground truth's *own* null-space content, per region (A1 / D21).
    # `err_free` alone cannot separate "the reconstruction invented structure"
    # from "this region simply has a lot the operator cannot see": a region of
    # dense anatomy carries large ||P_N x_true|| before any solver runs. Track 1
    # needs the denominator to make the criterion relative (err_free /
    # free_content > tau) instead of an absolute constant, which D-020 showed
    # transfers neither across modalities nor across substrates within one.
    # Recorded, never thresholded here -- the bank stays re-thresholdable.
    free_true = operator.project_null(x_true)
    free_true = free_true.abs() if free_true.is_complex() else free_true.abs()
    free_content = grid.reduce(free_true, how="rms") / ref

    if annotation is None:
        ann = torch.zeros(grid.n_regions, dtype=torch.float32)
    else:
        ann = grid.mask_coverage(annotation).to(torch.float32)

    return RegionTruth(
        err_total=err_total.to(torch.float32),
        err_free=err_free.to(torch.float32),
        free_content=free_content.to(torch.float32),
        ann_coverage=ann,
    )


# Below this much free content a region has essentially nothing the operator
# cannot see, and `err_free / free_content` is 0/0 — an arbitrarily large ratio
# from an empty patch of background. This is a guard against a degenerate
# denominator, *not* a tuning knob: it excludes regions where the relative
# criterion has nothing to say, and it must stay far below any real anatomy.
# Measured on the M4Raw real-prior bank, the smallest region `free_content` is
# 0.162, so this floor is two orders of magnitude clear of live data.
MIN_FREE_CONTENT = 1e-3


def label_hallucinated(
    truth: RegionTruth,
    threshold: float = 0.10,
    use_free_component: bool = True,
    require_annotation: bool = False,
    annotation_min_coverage: float = 0.10,
    relative: bool = False,
    min_free_content: float = MIN_FREE_CONTENT,
) -> Tensor:
    """Derive the boolean hallucination label from cached continuous quantities.

    Two forms, and which one carries the headline claim is Track 1's call (O2):

    * **absolute** (default) — ``err_free > threshold``. This is the v0
      criterion, and D-020 recorded that its threshold transfers neither across
      modalities nor across substrates within one.
    * **relative** (``relative=True``) — ``err_free / free_content > threshold``,
      the form agreed at the Week 1 sign-off (A1 / D21) and the reason
      ``free_content`` was added to the bank. The denominator is the ground
      truth's *own* null-space content, so the criterion asks whether the
      reconstruction put more into the free component than was there to begin
      with, rather than whether it put in more than some fixed constant. A
      region of dense anatomy carries large ``err_free`` before any solver runs;
      dividing by ``free_content`` is what stops that from reading as invention.

    ``require_annotation`` restricts positives to clinically annotated regions,
    which is the stricter reading of "does the flag fire where it matters"
    (Sec. 14). Both readings are reported.

    Note that ``threshold`` means different things in the two modes — an error
    magnitude in one, a dimensionless ratio in the other — so a value tuned for
    one is meaningless in the other. There is deliberately no shared default.
    """
    err = truth.err_free if use_free_component else truth.err_total
    if relative:
        content = truth.free_content
        # Guarded division: regions with no free content to speak of are not
        # candidates at all, rather than being divided into a large ratio.
        usable = content > min_free_content
        ratio = torch.where(usable, err / content.clamp_min(min_free_content),
                            torch.zeros_like(err))
        label = (ratio > threshold) & usable
    else:
        label = err > threshold
    if require_annotation:
        label = label & (truth.ann_coverage >= annotation_min_coverage)
    return label


def positive_rate_sweep(
    err_free: Tensor,
    free_content: Tensor,
    thresholds: Sequence[float],
    min_free_content: float = MIN_FREE_CONTENT,
) -> list[tuple[float, float]]:
    """Positive rate of the relative criterion at each candidate ``tau``.

    The Week 1 sign-off fixed the criterion's *form* and deferred ``tau`` until a
    real prior existed (T2.2), with its positive rate to be **reported, not
    targeted**. This is that report: Track 2 supplies the curve, Track 1 picks
    the point on it.

    Returns ``[(tau, positive_rate), ...]`` over regions with usable free
    content.
    """
    usable = free_content > min_free_content
    if not bool(usable.any()):
        raise ValueError("no region has enough free content to apply the relative criterion")
    ratio = err_free[usable] / free_content[usable]
    return [(float(t), float((ratio > t).to(torch.float32).mean())) for t in thresholds]
