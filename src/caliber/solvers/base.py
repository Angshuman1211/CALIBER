"""The common solver interface (task T2.2).

Blueprint Sec. 10, Stage 1: a pretrained diffusion or flow-matching solver is
treated as a *black-box posterior sampler*. Nothing downstream may depend on
which family produced the samples — that is what makes the audit layer
solver-agnostic and what makes the Week 10 solver ablation nearly free.

The interface is deliberately fixed before the first solver is wired in
(weekly cards, Week 2: establish solver-agnosticism early, not at ablation
stage). Adding a third family later must not require touching scoring,
calibration, or evaluation.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..operators.base import ForwardOperator


@dataclass(frozen=True)
class SolverInfo:
    """Provenance recorded alongside every score-bank row."""

    solver_id: str
    family: str  # "diffusion" | "flow_matching" | "reference"
    checkpoint: str | None = None
    n_steps: int | None = None
    notes: str = ""

    @property
    def prior_domain(self) -> str:
        """What the prior was trained on -- "natural_image", "mri", ... (P-18).

        Reportability depends on this and not only on the substrate: a real
        generative prior trained on the wrong domain passes both of D-018's
        original gates while still being an inappropriate prior for the images
        it is reconstructing.
        """
        from .denoiser import prior_domain

        return prior_domain(self.checkpoint)

    def to_row(self) -> dict[str, Any]:
        return {
            "solver_id": self.solver_id,
            "solver_family": self.family,
            "solver_checkpoint": self.checkpoint,
            "solver_steps": self.n_steps,
        }


def draw_seed(seed: int, index: int) -> int:
    """The seed for one posterior draw — a function of ``(seed, index)`` alone (D-044).

    Draws used to share one generator across the whole batch, so the noise a draw
    received depended on how many draws were requested alongside it. That made batching a
    property of the result, which is what stopped posterior draws from being computed in
    smaller batches to fit in memory (P-24). Seeding per draw makes a draw the same
    whether it is computed alone or with fifteen others.
    """
    digest = hashlib.blake2b(f"{int(seed)}|{int(index)}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


class PosteriorSampler(ABC):
    """Draws several plausible reconstructions for one set of measurements."""

    family: str = "unknown"
    supports_modalities: tuple[str, ...] = ()
    #: Draws per forward batch; 0 means all at once. Lower it to fit memory (P-24, D-043)
    #: without changing the result — see :func:`draw_seed`.
    max_draws_per_batch: int = 0

    def _sample_in_chunks(
        self, y: Tensor, operator: ForwardOperator, n_samples: int, seed: int
    ) -> Tensor:
        """Run :meth:`_sample_draws` over draw batches and concatenate.

        The concatenation is exactly the tensor a single batch would have produced,
        because each draw carries its own seed.
        """
        seeds = [draw_seed(seed, i) for i in range(int(n_samples))]
        size = self.max_draws_per_batch or len(seeds)
        if size < 1:
            raise ValueError(f"max_draws_per_batch must be >= 1 or 0, got {size}")
        chunks = [seeds[i:i + size] for i in range(0, len(seeds), size)]
        return torch.cat(
            [self._sample_draws(y, operator, chunk) for chunk in chunks], dim=0
        )

    def _sample_draws(
        self, y: Tensor, operator: ForwardOperator, draw_seeds: list[int]
    ) -> Tensor:
        """One batch of draws, one per entry of ``draw_seeds``.

        Implemented by the pretrained back-ends; the reference samplers override
        :meth:`sample` directly and do not use it.
        """
        raise NotImplementedError

    @abstractmethod
    def info(self) -> SolverInfo:
        """Identity and provenance of this solver."""

    @abstractmethod
    def sample(
        self,
        y: Tensor,
        operator: ForwardOperator,
        n_samples: int,
        seed: int = 0,
    ) -> Tensor:
        """Return ``n_samples`` reconstructions, shape ``(n_samples, *image_shape)``.

        Implementations must be deterministic given ``seed`` — the whole
        reproducibility package depends on it — and must never mutate
        ``operator``.
        """

    def check_supports(self, operator: ForwardOperator) -> None:
        if self.supports_modalities and operator.modality not in self.supports_modalities:
            raise ValueError(
                f"solver {self.info().solver_id!r} does not support modality "
                f"{operator.modality!r} (supports {self.supports_modalities})"
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.info().solver_id})"
