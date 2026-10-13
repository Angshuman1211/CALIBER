"""Diffusion-family posterior samplers — DPS [23] and DDNM [25]. Task T2.2, Week 2.

These wrap *public pretrained checkpoints used unmodified* (Blueprint Sec. 17); nothing
here trains a prior. Both express their measurement guidance entirely through
``operator.forward`` / ``operator.pinv``, which is the property that keeps the audit
layer solver-agnostic: adding a third family must not touch scoring or calibration.

**The two differ in a way that matters to this project specifically.** DPS steers the
sampler with the *gradient* of the measurement residual, so its data consistency is
approximate and enforced softly. DDNM replaces the *range component* of each ``x0``
estimate outright, which is the same range/null-space split the project's score is built
on -- so DDNM's internals and the audit layer speak the same language, and its data
consistency is exact up to the real-image projection below. Having both wired means the
Week 10 solver ablation can separate "the score works" from "the score works for solvers
that happen to enforce consistency the same way our score measures it".

**Real images, complex measurements.** The prior is over real images; MRI measurements
are complex, and a random 1-D Cartesian mask is not conjugate-symmetric, so ``A^+`` of a
complex residual is complex even when the truth is real. Every correction is therefore
projected back onto the real image manifold the prior lives on. That projection is *not*
free -- it is exactly the operation D-024 showed can silently break a data-consistency
claim -- so both samplers expose :meth:`data_consistency_error`, and the tests assert on
the measured value rather than assuming it is zero.

**Compute.** The torch build here is CPU-only. One UNet forward at 128^2 costs ~0.6 s,
so a 100-step draw is ~1 minute and DPS is ~3x that again because it backpropagates
through the network at every step. Sub-sample the schedule and budget the posterior
count before scaling the operator grid (Sec. 17, risk R7).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from ..operators.base import ForwardOperator
from .base import PosteriorSampler, SolverInfo
from .denoiser import Denoiser, DiffusersUNetDenoiser, NoiseSchedule


def _as_real(x: Tensor) -> Tensor:
    """Project onto the real image manifold the prior is defined over.

    Takes the real part rather than the magnitude. The real part is the orthogonal
    projection onto real-valued images; the magnitude is not a projection at all and
    would fold sign information into amplitude, which is what makes ``.abs()`` the wrong
    tool here even though both "return something real".
    """
    return x.real if x.is_complex() else x


def range_replace(x0: Tensor, y: Tensor, operator: ForwardOperator,
                  device: str | None = None) -> Tensor:
    """``x0 - A^+(A x0 - y)`` for each draw: the range component pinned to the data.

    The inverse is ``operator.decomposition_pinv``, the one the score's range/null split
    uses, not ``operator.pinv``. For MRI they are the same function, so MRI draws are
    unchanged bit for bit. For CT ``pinv`` is filtered backprojection, which is not a
    pseudo-inverse: a step built on it left a real slice *less* consistent with its data
    as views got sparser (residual 0.064 to 0.207 at 30 views, against 0.001 with the CG
    inverse, T2.6). That is D-012's defect re-entering through the solver, and it would
    confound the view-count axis with the solver's failure to fit the data.

    ``device`` runs just this step elsewhere -- the CG inverse is 4-6x faster on the GPU
    at 256^2 -- and returns the result on ``x0``'s device. ``None`` keeps it where ``x0``
    is, which is what every MRI config does and what keeps those banks reproducible.
    """
    out = []
    for b in range(x0.shape[0]):
        xb = x0[b] if device is None else x0[b].to(device)
        yb = y if device is None else y.to(device)
        fixed = _as_real(xb - operator.decomposition_pinv(operator.forward(_as_real(xb)) - yb))
        out.append(fixed if device is None else fixed.to(x0.device))
    return torch.stack(out)


class _DiffusionSampler(PosteriorSampler):
    """Shared machinery: schedule, batching, seeding, and the consistency diagnostic."""

    family = "diffusion"
    supports_modalities = ("mri", "ct")

    def __init__(
        self,
        checkpoint: str | None = None,
        n_steps: int = 100,
        denoiser: Denoiser | None = None,
        schedule: NoiseSchedule | None = None,
        eta: float = 1.0,
        device: str = "cpu",
        max_draws_per_batch: int = 0,
        consistency_device: str | None = None,
    ):
        self.checkpoint = checkpoint
        self.n_steps = int(n_steps)
        self.eta = float(eta)
        self.device = device
        # Where the data-consistency step runs (see range_replace). Opt-in, for CT, whose
        # CG inverse dominates the cost of a pair on the CPU.
        self.consistency_device = consistency_device
        # 0 = all draws in one forward batch. Lower it to fit memory (P-24); the result
        # is unchanged either way, because each draw carries its own seed (D-044).
        self.max_draws_per_batch = int(max_draws_per_batch)
        self.schedule = schedule or NoiseSchedule()
        if denoiser is not None:
            self._denoiser: Denoiser | None = denoiser
        elif checkpoint is not None:
            self._denoiser = DiffusersUNetDenoiser(checkpoint, device=device)
        else:
            self._denoiser = None

    @property
    def denoiser(self) -> Denoiser:
        if self._denoiser is None:
            raise ValueError(
                f"{type(self).__name__} needs a checkpoint or an explicit denoiser. "
                "Pass checkpoint='checkpoints/ddpm-celebahq-256' (D-016), or a "
                "Denoiser instance for testing."
            )
        return self._denoiser

    @staticmethod
    def _draw_generators(draw_seeds: list[int], offset: int = 0) -> list[torch.Generator]:
        return [
            torch.Generator(device="cpu").manual_seed(int(s) + offset) for s in draw_seeds
        ]

    def _init_latents(self, draw_seeds: list[int], shape: tuple[int, int]) -> Tensor:
        # Per draw, not per batch: a draw's starting point must not depend on how many
        # other draws were asked for at the same time (D-044).
        return torch.stack([
            torch.randn(*shape, generator=g, dtype=torch.float32)
            for g in self._draw_generators(draw_seeds)
        ])

    @staticmethod
    def _noise_like(x: Tensor, gens: list[torch.Generator]) -> Tensor:
        return torch.stack([
            torch.randn(x.shape[1:], generator=g, dtype=x.dtype) for g in gens
        ])

    @staticmethod
    def data_consistency_error(draws: Tensor, y: Tensor, operator: ForwardOperator) -> float:
        """``mean ||A x_i - y|| / ||y||`` over the draws. Measured, never assumed."""
        denom = float(y.abs().pow(2).sum().sqrt()) or 1.0
        errs = [float((operator.forward(d) - y).abs().pow(2).sum().sqrt()) / denom for d in draws]
        return sum(errs) / len(errs)


class DPSSampler(_DiffusionSampler):
    """Diffusion Posterior Sampling (Chung et al., ICLR 2023) — Blueprint ref [23].

    Ancestral DDPM sampling with a gradient correction that pulls each step toward the
    measurements: ``x_{t-1} <- ddim_step(...) - zeta * grad_{x_t} ||y - A x0_hat(x_t)||``.

    The paper writes the update as ``zeta * grad ||r||^2`` with ``zeta = zeta' / ||r||``.
    Because ``grad ||r||^2 = 2 ||r|| grad ||r||``, the ``||r||`` cancels exactly and the
    update is a *constant* multiple of the gradient of the norm -- which is what is
    implemented. Applying the ``1/||r||`` on top of the gradient of the norm (the obvious
    misreading, and the first thing tried here) makes the correction shrink as the fit
    improves, so it never converges: measured data-consistency error 0.98 at 8 steps and
    0.83 at 50, worse than returning zeros.

    Keeping the multiplier constant also keeps the guidance scale-free across operators,
    which matters more here than usual: the whole project varies the operator, and a
    guidance strength that moved with ``||y||`` would confound "the reconstruction
    changed because the acquisition changed" with "the guidance strength changed because
    the acquisition changed" -- exactly the class of confound D-028 was written about.
    """

    #: Measured on the analytic denoiser at 32^2, R=4, 50 steps: data-consistency error
    #: 0.94, 0.60, **0.053**, 0.16, 0.31, 0.76 at step sizes 0.5, 2, 5, 10, 20, 50. A
    #: clean minimum -- too small and the soft constraint never bites, too large and the
    #: correction overshoots the prior. Re-tune when the prior changes (D-016 medical
    #: weights); the value is not transferable between priors.
    #:
    #: **It must not be tuned per operator.** Every coverage curve holds the solver fixed
    #: and varies only the acquisition (Sec. 13). A guidance strength re-fitted at each
    #: acceleration would make the reconstruction quality a function of the operator
    #: through two paths at once, and the coverage gap could no longer be attributed to
    #: the acquisition -- the same class of confound as D-028.
    DEFAULT_STEP_SIZE = 5.0

    def __init__(self, checkpoint: str | None = None, n_steps: int = 100,
                 step_size: float = DEFAULT_STEP_SIZE, **kwargs):
        super().__init__(checkpoint=checkpoint, n_steps=n_steps, **kwargs)
        self.step_size = float(step_size)

    def info(self) -> SolverInfo:
        ckpt = self.checkpoint or (self._denoiser.checkpoint_id if self._denoiser else None)
        # Basename only: the id becomes a parquet partition key and a filename,
        # and an absolute path would make banks non-portable between machines.
        short = Path(ckpt).name if ckpt else 'unset'
        return SolverInfo(
            solver_id=f"diffusion.dps.{short}.t{self.n_steps}",
            family="diffusion",
            checkpoint=ckpt,
            n_steps=self.n_steps,
            notes=f"DPS, normalised guidance step_size={self.step_size:g}, eta={self.eta:g}",
        )

    def sample(
        self, y: Tensor, operator: ForwardOperator, n_samples: int, seed: int = 0
    ) -> Tensor:
        self.check_supports(operator)
        return self._sample_in_chunks(y, operator, n_samples, seed)

    def _sample_draws(
        self, y: Tensor, operator: ForwardOperator, draw_seeds: list[int]
    ) -> Tensor:
        shape = tuple(operator.image_shape)
        x = self._init_latents(draw_seeds, shape)
        gens = self._draw_generators(draw_seeds, offset=1)
        steps = self.schedule.timesteps(self.n_steps)

        for i, t in enumerate(steps):
            t_prev = steps[i + 1] if i + 1 < len(steps) else -1
            x = x.detach().requires_grad_(True)
            eps = self.denoiser.predict_noise(x, t)
            x0 = self.schedule.predict_x0(x, eps, t)

            # One residual per draw, so a batch behaves exactly like independent runs.
            residual = torch.stack([
                (operator.forward(_as_real(x0[b])) - y).abs().pow(2).sum().sqrt()
                for b in range(x0.shape[0])
            ])
            (grad,) = torch.autograd.grad(residual.sum(), x)

            with torch.no_grad():
                noise = self._noise_like(x, gens) if self.eta > 0 else None
                x_next = self.schedule.ddim_step(
                    x0.detach(), eps.detach(), t_prev, eta=self.eta, noise=noise
                )
                # The paper's update is zeta * grad ||r||^2 with zeta = zeta'/||r||.
                # Since grad ||r||^2 = 2 ||r|| grad ||r||, that is exactly a *constant*
                # multiple of the gradient of the norm -- the ||r|| cancels. Dividing by
                # ||r|| again (the obvious misreading) makes the correction vanish as
                # soon as the fit is decent, which showed up as a data-consistency error
                # of 1.04: worse than returning zeros.
                x = x_next - self.step_size * grad
        return x.detach()


class DDNMSampler(_DiffusionSampler):
    """Denoising Diffusion Null-Space Model (Wang et al., ICLR 2023) — ref [25].

    At each step the ``x0`` estimate has its range component replaced by the measured
    data, ``x0 <- x0 - A^+(A x0 - y)``, leaving the null-space component to the prior.
    That is literally the Stage-2 decomposition this project scores on, which makes DDNM
    the natural first back-end: what the sampler is free to invent and what the score
    calls the free component are the same subspace.

    It is also much cheaper than DPS on this machine, because it needs no backward pass.
    """

    def __init__(self, checkpoint: str | None = None, n_steps: int = 100, **kwargs):
        kwargs.setdefault("eta", 0.85)
        super().__init__(checkpoint=checkpoint, n_steps=n_steps, **kwargs)

    def info(self) -> SolverInfo:
        ckpt = self.checkpoint or (self._denoiser.checkpoint_id if self._denoiser else None)
        # Basename only: the id becomes a parquet partition key and a filename,
        # and an absolute path would make banks non-portable between machines.
        short = Path(ckpt).name if ckpt else 'unset'
        return SolverInfo(
            solver_id=f"diffusion.ddnm.{short}.t{self.n_steps}",
            family="diffusion",
            checkpoint=ckpt,
            n_steps=self.n_steps,
            notes=f"DDNM range-replacement, eta={self.eta:g}",
        )

    def sample(
        self, y: Tensor, operator: ForwardOperator, n_samples: int, seed: int = 0
    ) -> Tensor:
        self.check_supports(operator)
        return self._sample_in_chunks(y, operator, n_samples, seed)

    def _sample_draws(
        self, y: Tensor, operator: ForwardOperator, draw_seeds: list[int]
    ) -> Tensor:
        shape = tuple(operator.image_shape)
        x = self._init_latents(draw_seeds, shape)
        gens = self._draw_generators(draw_seeds, offset=1)
        steps = self.schedule.timesteps(self.n_steps)

        with torch.no_grad():
            for i, t in enumerate(steps):
                t_prev = steps[i + 1] if i + 1 < len(steps) else -1
                eps = self.denoiser.predict_noise(x, t)
                x0 = self.schedule.predict_x0(x, eps, t)
                x0 = range_replace(x0, y, operator, self.consistency_device)
                noise = self._noise_like(x, gens) if self.eta > 0 else None
                x = self.schedule.ddim_step(x0, eps, t_prev, eta=self.eta, noise=noise)
        return x
