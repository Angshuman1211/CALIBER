"""Reference samplers that need no pretrained checkpoint.

These exist so that the full pipeline — operators, decomposition, scoring,
score bank, calibration, evaluation — is runnable end to end on day one, before
fastMRI access clears and before any diffusion checkpoint is downloaded. Track 1
can develop calibration against a synthetic score bank while Track 2 is still
wiring real solvers.

``NullSpaceNoiseSampler`` is not a strawman. It is a *faithful caricature* of
the failure the project studies: the measured component is pinned to the data
and every posterior draw differs only inside the operator's null space, which
is precisely where the Blueprint says hallucination lives (Sec. 4). Coverage
should therefore degrade under operator shift even with this sampler, which
makes it a useful smoke test for the whole claim.

It is a reference, never a result. No figure or reported number may come from it.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..operators.base import ForwardOperator
from .base import PosteriorSampler, SolverInfo


class NullSpaceNoiseSampler(PosteriorSampler):
    """``x_i = A^+ y + P_N(smooth noise_i)`` — data-consistent, freely hallucinating."""

    family = "reference"
    supports_modalities = ("mri", "ct")

    def __init__(self, hallucination_scale: float = 0.15, smoothness: int = 4, **_):
        self.hallucination_scale = float(hallucination_scale)
        self.smoothness = int(smoothness)

    def info(self) -> SolverInfo:
        return SolverInfo(
            solver_id=f"reference.nullspace_noise.s{self.hallucination_scale:g}",
            family="reference",
            checkpoint=None,
            n_steps=None,
            notes="Synthetic sampler for pipeline smoke tests; not a reportable back-end.",
        )

    def _texture(self, shape: tuple[int, ...], gen: torch.Generator) -> Tensor:
        """Low-frequency noise, so invented structure looks structural not speckled."""
        small = [max(2, s // self.smoothness) for s in shape]
        seed_field = torch.randn(1, 1, *small, generator=gen)
        upsampled = torch.nn.functional.interpolate(
            seed_field, size=shape, mode="bilinear", align_corners=False
        )
        return upsampled.reshape(*shape)

    def sample(
        self, y: Tensor, operator: ForwardOperator, n_samples: int, seed: int = 0
    ) -> Tensor:
        self.check_supports(operator)
        gen = torch.Generator().manual_seed(seed)
        base = operator.pinv(y)
        scale = float(base.abs().mean()) * self.hallucination_scale

        # The draw stays in the operator's own (complex, for MRI) domain, and
        # magnitudes are taken only where something is measured. A random 1-D
        # Cartesian mask is not conjugate-symmetric, so P_N of a real image
        # genuinely has an imaginary part -- 33% of its norm at 128^2, R=4.
        # Casting it away with `.real` put 4.1% of the perturbation back into
        # the *measured* component, so the draws were no longer data-consistent
        # and s_residual was non-zero as a pure artefact. See D-024.
        draws = []
        for _ in range(n_samples):
            texture = self._texture(tuple(operator.image_shape), gen) * scale
            draws.append(base + operator.project_null(texture))
        return torch.stack(draws, dim=0)


class PseudoInverseSampler(PosteriorSampler):
    """Deterministic ``A^+ y``. Zero posterior spread — a degenerate lower reference."""

    family = "reference"
    supports_modalities = ("mri", "ct")

    def info(self) -> SolverInfo:
        return SolverInfo(
            solver_id="reference.pinv",
            family="reference",
            notes="Deterministic pseudo-inverse; no generative prior, no hallucination.",
        )

    def sample(
        self, y: Tensor, operator: ForwardOperator, n_samples: int, seed: int = 0
    ) -> Tensor:
        self.check_supports(operator)
        # Same domain convention as NullSpaceNoiseSampler (D-024): return A^+ y
        # as the operator produces it. Collapsing to magnitude here would make
        # the two reference samplers disagree about what a draw *is*, and would
        # break the data consistency A(A^+ y) = y that the residual score reads.
        base = operator.pinv(y)
        return base.unsqueeze(0).repeat(n_samples, *([1] * base.dim()))
