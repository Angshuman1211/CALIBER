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

    if annotation is None:
        ann = torch.zeros(grid.n_regions, dtype=torch.float32)
    else:
        ann = grid.mask_coverage(annotation).to(torch.float32)

    return RegionTruth(
        err_total=err_total.to(torch.float32),
        err_free=err_free.to(torch.float32),
        ann_coverage=ann,
    )


def label_hallucinated(
    truth: RegionTruth,
    threshold: float = 0.10,
    use_free_component: bool = True,
    require_annotation: bool = False,
    annotation_min_coverage: float = 0.10,
) -> Tensor:
    """Derive the boolean hallucination label from cached continuous quantities.

    ``require_annotation`` restricts positives to clinically annotated regions,
    which is the stricter reading of "does the flag fire where it matters"
    (Sec. 14). Both readings are reported; which one carries the headline claim
    is a Week 2 decision to be recorded in the decision log.
    """
    err = truth.err_free if use_free_component else truth.err_total
    label = err > threshold
    if require_annotation:
        label = label & (truth.ann_coverage >= annotation_min_coverage)
    return label
