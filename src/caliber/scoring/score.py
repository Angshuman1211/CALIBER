"""Operator-aware hallucination scoring — Stage 3 (Blueprint Sec. 10, objective O1).

The score combines three sources of evidence, exactly as the Blueprint and the
Week 3 card specify:

* **free-component structure** — energy in ``P_N x``, the part of the
  reconstruction the operator cannot see;
* **cross-sample disagreement** — variation across posterior draws, which is
  large where the prior is inventing rather than reporting;
* **measurement residual** — evidence from ``A x - y`` that a region is not
  data-consistent.

Ownership: the *definition* of the final score is Track 1's (T1.1). Track 2 owns
making it computable, cacheable, and identical across modalities. The design
decision that matters for that division of labour is that the three components
are **stored separately** in the score bank. Track 1 can then re-weight,
re-combine, and re-threshold from cached values with no GPU time at all, which
is the whole point of Sec. 17.

T1.1 (WI-3) ratifies the three component definitions unchanged and replaces the
v0 placeholder combination with a **standardised non-negative affine** rule. Four
decisions, recorded as D25-D28 in ``Documentation/problems_and_decisions.md``:

1. **Affine, not a ratio or a gate.** Any non-negative weighted sum of Lipschitz
   components is Lipschitz with ``L = sum(w_i * L_i)``, which is exactly what
   T3.3's L1 bound needs (``T3.3_stability_result_L1.md`` §3, assumption ii). A
   ratio such as ``s_disagree / s_free`` is *not* Lipschitz — it diverges as the
   denominator vanishes, i.e. in empty background regions. The combination form
   is therefore constrained by the theory track, deliberately.
2. **Standardise first.** The raw components differ by an order of magnitude, so
   the v0 default of ``1,1,1`` was in practice "score = s_free". Each component
   is divided by a fixed reference scale before weighting, so the weights mean
   what they say.
3. **Scales come from the calibration operator and never move.** Re-estimating
   them per deployment operator would pull the score distributions back towards
   each other and shrink the very coverage gap this project exists to measure.
4. **Weights are selected on a design split**, disjoint from calibration and
   test, then frozen. Conformal calibration picks a *threshold*, not the weights;
   fitting weights on calibration data would leak.

``SCORE_VERSION`` bumps to mark the components as ratified. Their arithmetic is
**unchanged** from ``v0-placeholder``, so existing banks are numerically identical
and need no regeneration — only the label differs. ``COMBINATION_VERSION`` is
tracked separately because the combination is applied at analysis time, from
cache, and will be revised more often than the components are.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..operators.base import ForwardOperator
from .regions import RegionGrid

SCORE_VERSION = "v1-components-ratified"
COMBINATION_VERSION = "v1-standardised-affine"


@dataclass(frozen=True)
class ScoreComponents:
    """Per-region evidence, one entry per region, all in ``(n_regions,)``."""

    s_free: Tensor
    s_disagree: Tensor
    s_residual: Tensor
    ref_intensity: float

    def stack(self) -> Tensor:
        """The three components as ``(3, n_regions)``, in canonical order."""
        return torch.stack([self.s_free, self.s_disagree, self.s_residual], dim=0)

    def combine(self, weights: ScoreWeights, scales: ScoreScales) -> Tensor:
        """The T1.1 non-conformity score, one value per region.

        Both arguments are required and neither has a default. A default would
        invite exactly the failure the v0 placeholder had: an unstandardised
        ``1,1,1`` sum that silently reduces to whichever component happens to be
        largest. Build ``scales`` once from the calibration operator
        (:meth:`ScoreScales.from_components`) and reuse it everywhere.
        """
        return combine_components(self, weights, scales)


@dataclass(frozen=True)
class ScoreScales:
    """Fixed per-component reference scales, estimated once and then frozen.

    **These must come from the calibration operator and must not be
    re-estimated at deployment** (D27). Re-standardising per deployment operator
    would map each operator's score distribution onto a common spread, which is
    precisely the difference the coverage-gap experiment is trying to measure.
    Doing so would make the naive-transfer baseline look artificially healthy and
    destroy the headline result.
    """

    free: float
    disagree: float
    residual: float

    @classmethod
    def from_components(
        cls,
        components: ScoreComponents | Sequence[ScoreComponents],
        quantile: float = 0.75,
        floor: float = 1e-8,
    ) -> ScoreScales:
        """Robust scales, pooled over all regions of the design items.

        The quantile defaults to 0.75 rather than the median because medical
        images are background-heavy: a large fraction of regions is near-empty,
        which drags the median of every component towards zero and makes the
        standardisation numerically unstable. A mid-high quantile stays inside
        the anatomy while remaining insensitive to the extreme tail.

        ``floor`` guards the degenerate case where a component is identically
        zero — a single-draw run has no disagreement at all, and dividing by it
        would produce NaNs that propagate silently into the bank.
        """
        items = [components] if isinstance(components, ScoreComponents) else list(components)
        if not items:
            raise ValueError("need at least one ScoreComponents to estimate scales")
        if not 0.0 < quantile <= 1.0:
            raise ValueError(f"quantile must be in (0, 1], got {quantile}")

        def scale_of(index: int) -> float:
            pooled = torch.cat([item.stack()[index].flatten() for item in items]).double()
            return max(float(torch.quantile(pooled.abs(), quantile)), floor)

        return cls(free=scale_of(0), disagree=scale_of(1), residual=scale_of(2))

    def as_tensor(self) -> Tensor:
        return torch.tensor([self.free, self.disagree, self.residual], dtype=torch.float64)


@dataclass(frozen=True)
class ScoreWeights:
    """Non-negative weights on the standardised components, summing to one.

    Non-negativity is not cosmetic. It keeps the score **monotone** in each piece
    of evidence — more unexplained free-component structure can never lower the
    score — and it is what makes the Lipschitz constant of the combination the
    plain weighted sum of the component constants (D25). A negative weight would
    break both, and would let two kinds of evidence cancel into a confident-looking
    zero.

    Summing to one fixes the score's overall scale so that thresholds are
    comparable across weight choices during the ablation.
    """

    free: float = 1 / 3
    disagree: float = 1 / 3
    residual: float = 1 / 3

    def __post_init__(self) -> None:
        values = (self.free, self.disagree, self.residual)
        if any(w < 0 for w in values):
            raise ValueError(f"weights must be non-negative, got {values}")
        total = sum(values)
        if total <= 0:
            raise ValueError("weights must not all be zero")
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1, got {total}")

    @classmethod
    def only(cls, component: str) -> ScoreWeights:
        """Single-component weights, for the T1.4 ablation that isolates each term."""
        fields = {"free", "disagree", "residual"}
        if component not in fields:
            raise ValueError(f"unknown component {component!r}; expected one of {sorted(fields)}")
        return cls(**{name: 1.0 if name == component else 0.0 for name in fields})

    def as_tensor(self) -> Tensor:
        return torch.tensor([self.free, self.disagree, self.residual], dtype=torch.float64)


def combine_components(
    components: ScoreComponents,
    weights: ScoreWeights,
    scales: ScoreScales,
) -> Tensor:
    """Standardise, weight, and sum — the T1.1 non-conformity score.

    ``score(r) = sum_i w_i * (s_i(r) / scale_i)`` over the three components.

    Affine by design, so that a Lipschitz constant for the score follows from
    Lipschitz constants for the components (T3.3 §3). Larger means *more likely
    to contain invented structure*; conformal calibration then places a threshold
    on this quantity, and nothing here consults ground truth.
    """
    standardised = components.stack().double() / scales.as_tensor().unsqueeze(-1)
    return (weights.as_tensor().unsqueeze(-1) * standardised).sum(dim=0).to(torch.float32)


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
