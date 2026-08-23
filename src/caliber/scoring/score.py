"""Operator-aware hallucination scoring — Stage 3 (Blueprint Sec. 10, objective O1).

The score combines three sources of evidence, exactly as the Blueprint and the
Week 3 card specify:

* **free-component structure** — energy in ``P_N x``, the part of the
  reconstruction the operator cannot see;
* **cross-sample disagreement** — variation across posterior draws, which is
  large where the prior is inventing rather than reporting;
* **measurement residual** — evidence from ``A x - y`` that a region is not
  data-consistent.

Ownership: the *definition* of the final score is Track 1's (T1.1, finalised
Week 3). Track 2 owns making it computable, cacheable, and identical across
modalities. The design decision that matters for that division of labour is
that the three components are **stored separately** in the score bank. Track 1
can then re-weight, re-combine, and re-threshold from cached values with no GPU
time at all, which is the whole point of Sec. 17.

``combine`` below is therefore a v0 placeholder, and is marked as such in the
score bank via ``score_version``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..operators.base import ForwardOperator
from .regions import RegionGrid

SCORE_VERSION = "v0-placeholder"


@dataclass(frozen=True)
class ScoreComponents:
    """Per-region evidence, one entry per region, all in ``(n_regions,)``."""

    s_free: Tensor
    s_disagree: Tensor
    s_residual: Tensor
    ref_intensity: float

    def combine(self, w_free: float = 1.0, w_disagree: float = 1.0, w_residual: float = 1.0):
        """Weighted sum of the three components — a T1.1 decision, not a fixed truth."""
        return w_free * self.s_free + w_disagree * self.s_disagree + w_residual * self.s_residual


def compute_score_components(
    samples: Tensor,
    y: Tensor,
    operator: ForwardOperator,
    grid: RegionGrid,
) -> ScoreComponents:
    """Score one item: ``samples`` is ``(n_posterior, H, W)`` for measurements ``y``.

    Components are scaled by the mean intensity of the pseudo-inverse
    reconstruction so that magnitudes are comparable across items and across
    modalities. The scaling is a fixed function of the *measurements*, never of
    the ground truth, so nothing here leaks information unavailable at
    deployment — which is the entire premise of the method (Sec. 4.1).
    """
    if samples.dim() != 3:
        raise ValueError(
            f"expected samples of shape (n_posterior, H, W), got {tuple(samples.shape)}"
        )

    base = operator.pinv(y)
    base = base.abs() if base.is_complex() else base
    ref = float(base.abs().mean()) or 1.0

    mean_recon = samples.mean(dim=0)

    # 1. Free-component structure: energy the operator cannot see.
    free = operator.project_null(mean_recon)
    free = free.abs() if free.is_complex() else free.abs()
    s_free = grid.reduce(free, how="rms") / ref

    # 2. Cross-sample disagreement, measured in the free component only. Spread
    #    in the measured component reflects solver noise, not invented content.
    if samples.shape[0] > 1:
        free_draws = torch.stack([operator.project_null(s) for s in samples], dim=0)
        free_draws = free_draws.abs() if free_draws.is_complex() else free_draws
        spread = free_draws.std(dim=0)
    else:
        spread = torch.zeros_like(mean_recon)
    s_disagree = grid.reduce(spread, how="mean") / ref

    # 3. Measurement residual, backprojected so it can be attributed to regions.
    residual = operator.residual(mean_recon, y)
    residual_img = operator.adjoint(residual)
    residual_img = residual_img.abs() if residual_img.is_complex() else residual_img.abs()
    s_residual = grid.reduce(residual_img, how="rms") / ref

    return ScoreComponents(
        s_free=s_free.to(torch.float32),
        s_disagree=s_disagree.to(torch.float32),
        s_residual=s_residual.to(torch.float32),
        ref_intensity=ref,
    )
