"""The denoiser behind a diffusion or flow sampler, and its noise schedule (T2.2).

Why this layer exists at all. The Week 2 card asks for *two solver families behind the
same interface*, and the solvers themselves are the part of the system most likely to
drag a heavyweight dependency and a 434 MB checkpoint into every code path that touches
them. Splitting the *algorithm* (DPS, DDNM, FlowDPS -- how the guidance and the sampling
step are built) from the *network* (which weights predict the noise) keeps three things
true:

1. The sampling algorithms are unit-testable in milliseconds on CPU against
   :class:`AnalyticDenoiser`, with no checkpoint and no GPU. That matters here: the
   torch build on this machine is CPU-only, and one real 100-step draw at 128^2 costs
   about a minute.
2. Swapping the prior -- natural-image now, medical later (D-016) -- touches one adapter,
   not three samplers.
3. The determinism requirement in ``solvers/base.py`` can be *asserted* rather than
   hoped for, because a deterministic test double makes bitwise comparison meaningful.

The schedule is defined here rather than read from the checkpoint, deliberately: the
shipped ``scheduler/scheduler_config.json`` is a 15-byte file containing the literal
text ``Entry not found`` -- a failed download saved as though it had succeeded (P-15,
and the same failure mode as P-08). Writing the standard DDPM linear schedule out
explicitly means the sampler does not depend on that file and the numbers are visible
rather than buried in a config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import Tensor

#: Standard DDPM training schedule for the CelebA-HQ 256 checkpoint: 1000 steps, betas
#: linear in [1e-4, 0.02]. These are the values the model was trained against; changing
#: them silently mis-scales every x0 estimate, so they are named constants, not defaults
#: buried in a signature.
N_TRAIN_STEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02


def linear_beta_schedule(
    n_steps: int = N_TRAIN_STEPS,
    beta_start: float = BETA_START,
    beta_end: float = BETA_END,
) -> Tensor:
    """Betas for the standard DDPM linear schedule."""
    return torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float64).float()


class NoiseSchedule:
    """Precomputed ``alphas_cumprod`` and the x0 / posterior algebra built on it.

    All three samplers need the same handful of quantities, and getting any one of them
    subtly wrong produces images that still *look* plausible while being systematically
    mis-scaled -- the kind of error that survives visual inspection and shows up much
    later as a bias in the score. Computing them once, here, means the three samplers
    cannot disagree about them.
    """

    def __init__(self, betas: Tensor | None = None):
        self.betas = linear_beta_schedule() if betas is None else betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.n_steps = int(self.betas.shape[0])

    def alpha_bar(self, t: int) -> Tensor:
        """``alpha_bar_t``, with ``alpha_bar_{-1} = 1`` so the final step is well defined."""
        if t < 0:
            return torch.tensor(1.0)
        return self.alphas_cumprod[t]

    def predict_x0(self, x_t: Tensor, eps: Tensor, t: int) -> Tensor:
        """Tweedie: ``x0_hat = (x_t - sqrt(1 - a_bar) * eps) / sqrt(a_bar)``."""
        ab = self.alpha_bar(t)
        return (x_t - (1.0 - ab).sqrt() * eps) / ab.sqrt()

    def ddim_step(
        self,
        x0_hat: Tensor,
        eps: Tensor,
        t_prev: int,
        eta: float = 0.0,
        noise: Tensor | None = None,
    ) -> Tensor:
        """One DDIM step from an x0 estimate.

        ``eta=0`` is the deterministic probability-flow update and ``eta=1`` recovers
        ancestral DDPM sampling. The two solver *families* in this project differ
        precisely here, which is why the parameter is explicit rather than hard-coded:
        the diffusion samplers use the stochastic update, the flow sampler integrates
        the deterministic ODE.
        """
        ab_prev = self.alpha_bar(t_prev)
        sigma = eta * (1.0 - ab_prev).sqrt()
        direction_coeff = (1.0 - ab_prev - sigma**2).clamp_min(0.0).sqrt()
        out = ab_prev.sqrt() * x0_hat + direction_coeff * eps
        if eta > 0 and t_prev >= 0:
            if noise is None:
                raise ValueError("stochastic step (eta > 0) requires explicit noise")
            out = out + sigma * noise
        return out

    def timesteps(self, n_steps: int) -> list[int]:
        """``n_steps`` training indices, evenly spaced, descending to 0.

        Sub-sampling the 1000 training steps is what makes a draw affordable at all --
        see the compute note in ``diffusion.py``.
        """
        if not 1 <= n_steps <= self.n_steps:
            raise ValueError(f"n_steps must be in [1, {self.n_steps}], got {n_steps}")
        stride = self.n_steps / n_steps
        return [int(self.n_steps - 1 - round(i * stride)) for i in range(n_steps)]


class Denoiser(ABC):
    """Predicts the noise in a noisy image. The only thing a sampler needs from a prior."""

    #: Identifies the weights in ``SolverInfo`` so a bank row traces to what produced it.
    checkpoint_id: str = "unset"

    @abstractmethod
    def predict_noise(self, x: Tensor, t: int) -> Tensor:
        """``eps_theta(x_t, t)`` for a batch ``(B, H, W)`` of single-channel images."""


class AnalyticDenoiser(Denoiser):
    """A deterministic, closed-form stand-in used to test the samplers, not to sample.

    It pretends the prior is an isotropic Gaussian of variance ``prior_var`` centred on
    zero, for which the exact score is available in closed form, so the implied ``x0``
    estimate is analytically known and a sampler's arithmetic can be checked against it.
    It runs in microseconds and needs no checkpoint, which is what makes the determinism
    and data-consistency assertions in the test suite affordable to run every time.

    It is emphatically **not** an image prior: it produces smooth blobs. Nothing it
    generates is a reconstruction, and no bank built from it is reportable (D-018).
    """

    checkpoint_id = "analytic-gaussian"

    def __init__(self, schedule: NoiseSchedule | None = None, prior_var: float = 1.0):
        self.schedule = schedule or NoiseSchedule()
        self.prior_var = float(prior_var)

    def predict_noise(self, x: Tensor, t: int) -> Tensor:
        # For x_t ~ N(0, a_bar*prior_var + (1-a_bar)), the score is linear in x, so the
        # implied noise prediction is a fixed rescaling -- exact, and trivially
        # differentiable, which is what the DPS gradient path needs to be tested.
        ab = self.schedule.alpha_bar(t)
        var = ab * self.prior_var + (1.0 - ab)
        return x * ((1.0 - ab).sqrt() / var)


class DiffusersUNetDenoiser(Denoiser):
    """Adapter for a pretrained ``diffusers`` ``UNet2DModel`` checkpoint.

    Two adaptations are needed, and both are real narrowings that belong in the write-up
    rather than buried here:

    **Greyscale to RGB.** The available checkpoint (``ddpm-celebahq-256``, D-016) has
    ``in_channels=3``. Medical slices are single-channel, so the image is replicated
    across three channels and the predicted noise is averaged back. Averaging is the
    right reduction for the *noise* (the three predictions are estimates of the same
    quantity), but it is still an approximation of a model that was never asked to see a
    greyscale input.

    **A face prior on anatomy.** These weights were trained on CelebA-HQ. Using them on
    knee MRI is D-016's deliberate ordering -- wire the pipeline against an open natural
    image prior first, swap in medical weights later -- not a claim that the prior is
    appropriate. Reconstructions from it exercise the machinery and are not reportable
    results.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.checkpoint = str(checkpoint)
        self.checkpoint_id = Path(self.checkpoint).name
        self.device = device
        self._model = None

    @property
    def model(self):
        """Loaded lazily, so constructing a solver stays cheap in ``--dry-run``."""
        if self._model is None:
            try:
                from diffusers import UNet2DModel
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise ImportError(
                    "the diffusion back-ends need `diffusers` (pip install diffusers)"
                ) from exc
            self._model = UNet2DModel.from_pretrained(self.checkpoint).to(self.device).eval()
        return self._model

    def predict_noise(self, x: Tensor, t: int) -> Tensor:
        model = self.model
        n_ch = model.config.in_channels
        batched = x.unsqueeze(1).repeat(1, n_ch, 1, 1).to(self.device)
        timestep = torch.full((batched.shape[0],), int(t), device=self.device, dtype=torch.long)
        eps = model(batched, timestep).sample
        return eps.mean(dim=1).to(x.dtype)
