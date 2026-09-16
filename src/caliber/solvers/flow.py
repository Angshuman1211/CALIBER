"""Flow-matching posterior sampler — FlowDPS [26]. Task T2.2, Week 2.

The second solver family exists to establish solver-agnosticism *early* rather than at
the Week 10 ablation. If wiring it had required changing scoring, calibration, or
evaluation, the abstraction would have leaked and the fix would belong in
``solvers/base.py``. It did not: this file touches nothing outside ``solvers/``.

**What this is, stated precisely, because the distinction is load-bearing.** There is no
public flow-matching checkpoint on this machine, and the project trains nothing
(Sec. 17, D-016). What is implemented here is FlowDPS's *sampling dynamics* -- a
deterministic rectified-flow ODE integrated from noise to data, with a data-consistency
correction applied to the clean-image estimate at every step -- driven by a velocity
field derived from the available diffusion checkpoint through the standard
probability-flow correspondence. It is **not** the published FlowDPS network.

That distinction matters for what may be claimed. It is *sufficient* for the Week 2
deliverable and for the Week 10 solver ablation, because what those need is two families
whose **sampling dynamics genuinely differ** behind one interface: stochastic ancestral
sampling with per-step noise injection (``diffusion.py``) against deterministic ODE
integration in a rectified-flow parameterisation (here). Their posterior spread arises
by different mechanisms, which is exactly the axis the ablation probes. It is *not*
sufficient to claim the project reproduced FlowDPS, and the write-up must say so.

**The correspondence.** A diffusion model with ``x_t = sqrt(a_bar) x0 + sqrt(1 - a_bar) eps``
and a rectified flow with ``x_s = (1 - s) x0 + s eps`` describe the same family of
interpolants at matched signal-to-noise ratio. Matching ``SNR`` gives, for each flow time
``s``, a diffusion index ``t`` and a scale ``c = sqrt(1 - a_bar_t) / s`` with
``x_t = c * x_s``. The flow velocity is then ``v = eps - x0``, both read off the denoiser.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from ..operators.base import ForwardOperator
from .base import PosteriorSampler, SolverInfo
from .denoiser import Denoiser, DiffusersUNetDenoiser, NoiseSchedule
from .diffusion import _as_real


class FlowDPSSampler(PosteriorSampler):
    """Flow-Driven Posterior Sampling (Kim et al., ICCV 2025) — Blueprint ref [26].

    Deterministic rectified-flow integration with per-step range replacement. Because the
    ODE is deterministic given its starting point, **all posterior spread comes from the
    initial noise**, where the diffusion samplers also inject noise at every step. That is
    a real behavioural difference between the two families, not an implementation detail:
    it is expected to show up as a different ``s_disagree`` scale, and Week 4 should
    report it rather than be surprised by it.
    """

    family = "flow_matching"
    supports_modalities = ("mri", "ct")

    def __init__(
        self,
        checkpoint: str | None = None,
        n_steps: int = 100,
        denoiser: Denoiser | None = None,
        schedule: NoiseSchedule | None = None,
        device: str = "cpu",
        max_draws_per_batch: int = 0,
    ):
        self.checkpoint = checkpoint
        self.n_steps = int(n_steps)
        self.device = device
        # See _DiffusionSampler: draws per forward batch, 0 = all (P-24, D-042).
        self.max_draws_per_batch = int(max_draws_per_batch)
        self.schedule = schedule or NoiseSchedule()
        if denoiser is not None:
            self._denoiser: Denoiser | None = denoiser
        elif checkpoint is not None:
            self._denoiser = DiffusersUNetDenoiser(checkpoint, device=device)
        else:
            self._denoiser = None
        # SNR of every training index, precomputed once: the flow-time -> diffusion-index
        # map is a lookup rather than a search per step.
        ab = self.schedule.alphas_cumprod
        self._snr = (ab / (1.0 - ab).clamp_min(1e-12)).sqrt()

    @property
    def denoiser(self) -> Denoiser:
        if self._denoiser is None:
            raise ValueError(
                "FlowDPSSampler needs a checkpoint or an explicit denoiser. Pass "
                "checkpoint='checkpoints/ddpm-celebahq-256' (D-016), or a Denoiser "
                "instance for testing."
            )
        return self._denoiser

    def info(self) -> SolverInfo:
        ckpt = self.checkpoint or (self._denoiser.checkpoint_id if self._denoiser else None)
        # Basename only: the id becomes a parquet partition key and a filename,
        # and an absolute path would make banks non-portable between machines.
        short = Path(ckpt).name if ckpt else 'unset'
        return SolverInfo(
            solver_id=f"flow.flowdps.{short}.t{self.n_steps}",
            family="flow_matching",
            checkpoint=ckpt,
            n_steps=self.n_steps,
            notes=(
                "Rectified-flow ODE with per-step range replacement, driven by a "
                "diffusion-derived velocity field (probability-flow correspondence); "
                "not the published FlowDPS checkpoint"
            ),
        )

    def _diffusion_index(self, s: float) -> int:
        """Training index whose SNR matches flow time ``s``."""
        target = (1.0 - s) / max(s, 1e-6)
        return int(torch.argmin((self._snr - target).abs()).item())

    def _velocity(self, x_s: Tensor, s: float, y: Tensor, operator: ForwardOperator):
        """``(v, x0)`` at flow time ``s``, with the range component pinned to the data."""
        t = self._diffusion_index(s)
        ab = self.schedule.alpha_bar(t)
        scale = (1.0 - ab).sqrt() / max(s, 1e-6)
        eps = self.denoiser.predict_noise(x_s * scale, t)
        x0 = self.schedule.predict_x0(x_s * scale, eps, t)
        # The data-consistency step, applied to the clean-image estimate exactly as DDNM
        # does, so the two families differ in their *dynamics* and not in how strongly
        # each is tied to the measurements. Holding that fixed is what makes a solver
        # ablation attributable to the solver (Sec. 13 controlled comparison).
        x0 = torch.stack([
            _as_real(x0[b] - operator.pinv(operator.forward(_as_real(x0[b])) - y))
            for b in range(x0.shape[0])
        ])
        return eps - x0, x0

    def sample(
        self, y: Tensor, operator: ForwardOperator, n_samples: int, seed: int = 0
    ) -> Tensor:
        self.check_supports(operator)
        return self._sample_in_chunks(y, operator, n_samples, seed)

    def _sample_draws(
        self, y: Tensor, operator: ForwardOperator, draw_seeds: list[int]
    ) -> Tensor:
        shape = tuple(operator.image_shape)
        # Per draw, so that a draw is the same alone as in a batch (D-042). FlowDPS is a
        # deterministic ODE, so this initial noise is its *only* source of spread.
        x = torch.stack([
            torch.randn(*shape, generator=torch.Generator(device="cpu").manual_seed(int(s)),
                        dtype=torch.float32)
            for s in draw_seeds
        ])

        # Integrate s: 1 (noise) -> 0 (data). Euler is deliberate: the point of this
        # back-end is a different sampling *dynamic*, and a higher-order integrator would
        # buy accuracy the natural-image prior does not justify at this stage.
        ds = 1.0 / self.n_steps
        with torch.no_grad():
            for i in range(self.n_steps):
                s = 1.0 - i * ds
                v, x0 = self._velocity(x, s, y, operator)
                x = x - ds * v
        return _as_real(x0)
