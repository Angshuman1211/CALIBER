"""Flow-matching posterior samplers — FlowDPS [26].

Week 2, task T2.2. The second solver family exists to establish
solver-agnosticism *early* rather than at the Week 10 ablation. If wiring this
requires any change to the scoring, calibration, or evaluation code, the
abstraction has leaked and the fix belongs in ``solvers/base.py``, not here.
"""

from __future__ import annotations

from torch import Tensor

from ..operators.base import ForwardOperator
from .base import PosteriorSampler, SolverInfo


class FlowDPSSampler(PosteriorSampler):
    """Flow-Driven Posterior Sampling (Kim et al., ICCV 2025) — Blueprint ref [26]."""

    family = "flow_matching"
    supports_modalities = ("mri", "ct")

    def __init__(self, checkpoint: str | None = None, n_steps: int = 100):
        self.checkpoint = checkpoint
        self.n_steps = int(n_steps)

    def info(self) -> SolverInfo:
        return SolverInfo(
            solver_id=f"flow.flowdps.{self.checkpoint or 'unset'}.t{self.n_steps}",
            family="flow_matching",
            checkpoint=self.checkpoint,
            n_steps=self.n_steps,
        )

    def sample(
        self, y: Tensor, operator: ForwardOperator, n_samples: int, seed: int = 0
    ) -> Tensor:
        raise NotImplementedError(
            "FlowDPS back-end not yet wired (Week 2, T2.2). The interface is fixed; "
            "only this method should need implementing."
        )
