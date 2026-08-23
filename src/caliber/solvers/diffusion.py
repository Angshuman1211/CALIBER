"""Diffusion-family posterior samplers — DPS [23], DDRM [24], DDNM [25].

Week 2, task T2.2. These wrap *public pretrained checkpoints used unmodified*
(Blueprint Sec. 17); nothing here trains a prior.

Wiring checklist for each back-end, in the order that surfaces problems soonest:

1. Load the checkpoint and confirm unconditional sampling produces sane images
   at the project's reduced resolution before any measurement guidance is added.
2. Express the guidance step through ``operator.forward`` / ``operator.adjoint``
   only. If a solver needs anything else from the operator, that is a signal the
   ``ForwardOperator`` interface is missing something — extend the interface,
   do not special-case the solver.
3. Verify determinism: two calls with the same seed must agree bitwise, or the
   reproducibility package (Week 10) is not reproducible.
4. Record the checkpoint identifier in ``SolverInfo`` so every score-bank row is
   traceable to the exact weights that produced it.

Compute note: the available GPU is 6 GB, so single-slice 2-D at reduced
resolution with a modest posterior-sample count is the operating point
(Sec. 17, risk R7). Budget the sample count before scaling the operator grid.
"""

from __future__ import annotations

from torch import Tensor

from ..operators.base import ForwardOperator
from .base import PosteriorSampler, SolverInfo


class DPSSampler(PosteriorSampler):
    """Diffusion Posterior Sampling (Chung et al., ICLR 2023) — Blueprint ref [23]."""

    family = "diffusion"
    supports_modalities = ("mri", "ct")

    def __init__(self, checkpoint: str | None = None, n_steps: int = 200, step_size: float = 1.0):
        self.checkpoint = checkpoint
        self.n_steps = int(n_steps)
        self.step_size = float(step_size)

    def info(self) -> SolverInfo:
        return SolverInfo(
            solver_id=f"diffusion.dps.{self.checkpoint or 'unset'}.t{self.n_steps}",
            family="diffusion",
            checkpoint=self.checkpoint,
            n_steps=self.n_steps,
        )

    def sample(
        self, y: Tensor, operator: ForwardOperator, n_samples: int, seed: int = 0
    ) -> Tensor:
        raise NotImplementedError(
            "DPS back-end not yet wired (Week 2, T2.2). Use the reference sampler "
            "for pipeline development until a checkpoint is configured."
        )


class DDNMSampler(PosteriorSampler):
    """Denoising Diffusion Null-Space Model (Wang et al., ICLR 2023) — ref [25].

    Of particular interest here: DDNM operates on the same range/null-space
    split this project's score is built from, so its internals and the audit
    layer speak the same language.
    """

    family = "diffusion"
    supports_modalities = ("mri", "ct")

    def __init__(self, checkpoint: str | None = None, n_steps: int = 100):
        self.checkpoint = checkpoint
        self.n_steps = int(n_steps)

    def info(self) -> SolverInfo:
        return SolverInfo(
            solver_id=f"diffusion.ddnm.{self.checkpoint or 'unset'}.t{self.n_steps}",
            family="diffusion",
            checkpoint=self.checkpoint,
            n_steps=self.n_steps,
        )

    def sample(
        self, y: Tensor, operator: ForwardOperator, n_samples: int, seed: int = 0
    ) -> Tensor:
        raise NotImplementedError("DDNM back-end not yet wired (Week 2, T2.2).")
