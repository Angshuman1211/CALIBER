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

import math
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

    @classmethod
    def from_alphas_cumprod(cls, alphas_cumprod: Tensor) -> "NoiseSchedule":
        """A schedule given ``alpha_bar`` directly, stored as given rather than re-multiplied.

        A VE-SDE prior defines its noise levels as sigmas, and ``alpha_bar = 1/(1+sigma^2)``
        follows exactly; recovering it through a cumulative product of betas would add
        rounding at the far end of the range, where ``alpha_bar`` is ~6e-5.
        """
        ab = alphas_cumprod.to(torch.float64)
        prev = torch.cat([torch.ones(1, dtype=torch.float64), ab[:-1]])
        sched = cls(betas=(1.0 - ab / prev).float())
        sched.alphas_cumprod = ab.float()
        return sched

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
        # Back to the caller's device as well as its dtype. Only the UNet moves to the
        # GPU; the operators, FFTs and scoring stay on CPU, and they are 0.1% of the
        # time per pair (measured), so there is nothing to gain by moving them. Before
        # this returned `.to(x.dtype)` alone, so with `device: cuda` the first solver
        # step mixed a GPU tensor into CPU arithmetic and raised.
        return eps.mean(dim=1).to(device=x.device, dtype=x.dtype)


def ve_schedule(sigma_min: float, sigma_max: float, n_steps: int = N_TRAIN_STEPS) -> NoiseSchedule:
    """The VP schedule equivalent to a VE-SDE's geometric noise levels.

    A VE sample ``x0 + sigma z`` and a VP sample ``sqrt(a) x0 + sqrt(1-a) z`` are the same
    point up to the scale ``sqrt(a)`` when ``a = 1 / (1 + sigma^2)``. Index 0 is ``sigma_min``
    and the last index ``sigma_max``, so the samplers' descending timesteps walk the VE
    prior's own log-uniform grid, the one it was trained and sampled on. Reusing the DDPM
    linear schedule instead would ask the BraTS model for sigma = 157 at the first step,
    beyond the 128 it was trained to.
    """
    sigmas = torch.exp(torch.linspace(math.log(sigma_min), math.log(sigma_max), n_steps,
                                      dtype=torch.float64))
    return NoiseSchedule.from_alphas_cumprod(1.0 / (1.0 + sigmas**2))


class SongVEScoreDenoiser(Denoiser):
    """Adapter for Song et al.'s VE-SDE NCSN++ medical priors (D-016 corrected by D79, T2.10).

    The checkpoints (ICLR 2022; BraTS 2021 brain MRI, LIDC and LDCT CT) are score models:
    the network returns ``s(x, sigma) = grad log p_sigma(x)`` for ``x = x0 + sigma z`` with
    ``x0`` in [0, 1]. The samplers here speak DDPM's VP form, ``x_t = sqrt(a) x0 +
    sqrt(1-a) eps``. The two are one change of variables apart: ``sigma = sqrt((1-a)/a)``,
    ``x_VE = x_t / sqrt(a)``, and Tweedie's formula gives ``eps = -sigma * s(x_VE, sigma)``.
    That is exact, not an approximation, so DDNM, DPS and the flow sampler run unchanged.
    The schedule is the prior's own (:func:`ve_schedule`) and travels with the denoiser.

    ``checkpoint`` is the directory holding the converted ``ncsnpp_*.pt``
    (``scripts/convert_song_checkpoint.py``), or that file itself. It is loaded with
    ``weights_only=True``: a state dict and a config of plain types, nothing executable.

    **What these priors are and are not trained on.** BraTS images are skull-stripped and
    min-max normalised to [0, 1] per volume, over T1, T1ce, T2 and FLAIR. So the data
    should be normalised per slice to [0, 1] (``normalisation: max``), and anatomy the
    prior never saw -- skull, scalp, a knee -- is outside its training distribution.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        path = Path(checkpoint)
        if path.is_dir():
            found = sorted(path.glob("ncsnpp_*.pt"))
            if not found:
                raise FileNotFoundError(
                    f"{path} holds no converted ncsnpp_*.pt; run "
                    "scripts/convert_song_checkpoint.py on its Flax checkpoint first"
                )
            path = found[0]
        self.path = path
        self.checkpoint = str(checkpoint)
        self.checkpoint_id = Path(self.checkpoint).name
        self.device = device
        payload = torch.load(path, map_location="cpu", weights_only=True)
        from .ncsnpp import NCSNppConfig

        self.config = NCSNppConfig.from_dict(payload["config"])
        self._state = payload["state_dict"]
        self.source = payload.get("source", {})
        self.default_schedule = ve_schedule(self.config.sigma_min, self.config.sigma_max)
        self._model = None

    @property
    def model(self):
        """Built lazily, so constructing a solver stays cheap in ``--dry-run``."""
        if self._model is None:
            from .ncsnpp import NCSNpp

            model = NCSNpp(self.config)
            model.load_state_dict(self._state)
            self._model = model.to(self.device).eval()
        return self._model

    def predict_noise(self, x: Tensor, t: int) -> Tensor:
        if tuple(x.shape[-2:]) != (self.config.image_size,) * 2:
            raise ValueError(
                f"{self.checkpoint_id} was trained at {self.config.image_size}^2 and its "
                f"architecture fixes that; got {tuple(x.shape[-2:])}. Set data.image_shape "
                f"to [{self.config.image_size}, {self.config.image_size}]"
            )
        ab = self.default_schedule.alpha_bar(t).to(self.device)
        sigma = ((1.0 - ab) / ab).sqrt()
        x_ve = (x.to(self.device) / ab.sqrt()).unsqueeze(1)
        score = self.model(x_ve, sigma.expand(x_ve.shape[0]).to(x_ve.dtype))
        return (-sigma * score[:, 0]).to(device=x.device, dtype=x.dtype)


def build_denoiser(checkpoint: str | Path, device: str = "cpu") -> Denoiser:
    """The adapter a checkpoint needs: Song et al.'s converted NCSN++, or a diffusers UNet."""
    path = Path(checkpoint)
    if path.suffix == ".pt" or (path.is_dir() and any(path.glob("ncsnpp_*.pt"))):
        return SongVEScoreDenoiser(path, device=device)
    return DiffusersUNetDenoiser(checkpoint, device=device)


# -- what the prior was trained on, and whether that fits the modality ---------
#
# D-018 made reportability a property of the bank rather than of whoever ran it,
# and gated it on two things: is the substrate real, and is the sampler a
# generative prior rather than a reference. Both are necessary. Neither is
# sufficient, and P-18 is the gap: a *real* prior trained on the *wrong domain*
# passes both gates. `ddpm-celebahq-256` on fastMRI knees is a face model
# reconstructing anatomy, which D-016 orders deliberately and which the T2.2
# write-up lists as a reason nothing is reportable -- but until this existed,
# nothing in the code knew that, so the first fastMRI bank came out stamped
# `reportable: true`.
#
# The mapping is keyed on the checkpoint basename because that is what travels
# in `solver_id` and in the manifest.
PRIOR_DOMAINS: dict[str, str] = {
    "ddpm-celebahq-256": "natural_image",
    "ddpm-cifar10-32": "natural_image",
    "ddpm-church-256": "natural_image",
    "ddpm-bedroom-256": "natural_image",
    # Medical priors land here when D-016's outstanding half is closed. Adding a
    # key is the *whole* change needed to make a bank reportable, which is the
    # point: it is one deliberate edit, reviewable in a diff.
    #
    # **Domains carry anatomy, not just modality (D80).** Track 2's reasoning on the BraTS
    # prior is adopted for every medical prior here: a bare "mri" would stamp a brain prior
    # reportable on fastMRI *knees*, and D67 is why that is not a cosmetic limitation --
    # the criterion's validity rests on `cos theta`, and anatomy mismatch is the mechanism
    # that drives `cos theta` to zero. A cross-anatomy medical prior reproduces the
    # face-prior pathology in the one quantity the label depends on.
    #
    # Two families, deliberately kept both: having two independent medical priors makes
    # "does the result depend on which prior?" an ablation rather than an assumption.
    #
    # Song et al. VE/NCSN++ (ICLR 2022), via scripts/convert_song_checkpoint.py:
    "brats_ncsnpp_ve": "mri_brain",
    "ct2d_320_ncsnpp_ve": "ct_chest",
    "ldct_512_ncsnpp_ve": "ct_head_chest_abdomen",
    # SegGuidedDiff **unconditional** models (Konz et al., MICCAI 2024): `diffusers`
    # `UNet2DModel`, `in_channels=1`, trained on `DDPMScheduler` defaults -- so the DDPM
    # schedule above is theirs exactly, and the greyscale replication in
    # `DiffusersUNetDenoiser` is a no-op for them rather than an approximation.
    "ddim-breast_mri-256": "mri_breast",
    "ddim-ct_organ_large-256": "ct_neck_to_pelvis",
}

#: A prior with no anatomy at all, as distinct from one whose anatomy is unrecorded.
NON_ANATOMICAL = "non_anatomical"

UNKNOWN_ANATOMY = "unknown"

#: The modalities whose domains are written ``<modality>_<anatomy>``. Everything else
#: (``natural_image``) is one token and has no anatomy.
MEDICAL_MODALITIES = ("mri", "ct")


def prior_modality(checkpoint: str | Path | None) -> str:
    """The modality half of the domain: ``mri_brain`` -> ``mri``.

    Derived from :data:`PRIOR_DOMAINS` rather than stored separately. A second map would
    be a second source of truth and two chances to drift, which is the failure
    `evaluation/bank.py` was written to avoid for the score.
    """
    domain = prior_domain(checkpoint)
    for modality in MEDICAL_MODALITIES:
        if domain == modality or domain.startswith(f"{modality}_"):
            return modality
    return domain


def prior_anatomy(checkpoint: str | Path | None) -> str:
    """The anatomy half: ``mri_brain`` -> ``brain``; ``natural_image`` -> non-anatomical.

    Reported beside the domain so a cross-anatomy run carries its limitation in the
    manifest instead of in someone's memory.
    """
    if checkpoint is None:
        return UNKNOWN_ANATOMY
    domain = prior_domain(checkpoint)
    if domain == UNKNOWN_DOMAIN:
        return UNKNOWN_ANATOMY
    for modality in MEDICAL_MODALITIES:
        if domain.startswith(f"{modality}_"):
            return domain[len(modality) + 1:]
    # A face is a body part, and naming it is what makes the celebahq mismatch legible
    # rather than merely out-of-domain.
    if Path(str(checkpoint)).name == "ddpm-celebahq-256":
        return "face"
    return NON_ANATOMICAL

# Which prior domains are admissible for which modality.
#
# **D80, Track 1's ruling, answering T2.10's question.** A prior is admissible for a
# modality when it is a *medical* prior **of that modality**, whatever anatomy it saw.
# Anatomy mismatch does not block a bank; it gates which *claims* that bank can carry.
#
# This is D67's coverage/detection split applied to the prior instead of to the label:
#
# - **Coverage claims survive cross-anatomy.** Both arms are scored against the same
#   label on the same reconstructions; a prior that suits the anatomy badly changes what
#   gets labelled, not whether a transferred threshold loses coverage relative to an
#   in-group one. That comparison is the project's central claim and it holds.
# - **Detection claims do not.** The criterion's validity rests on `cos theta`, and
#   anatomy mismatch is precisely the mechanism that drives `cos theta` to zero -- the
#   face-prior pathology reappearing in the one quantity the label depends on. A
#   cross-anatomy detection number is not reportable, and T2.9 measures it at chance.
#
# So `prior_anatomy` is recorded in the manifest and read in the analysis, rather than
# being folded in here. Blocking the bank outright would also have blocked the *coverage*
# result, which is the one the project can actually make.
#
# **Not changed by this ruling:** D-015's separate question of whether M4Raw is reportable
# as a *substrate*. That is a property of the dataset, not of the prior, and it is still
# whatever D-015 says it is. An anatomy-matched prior does not launder a substrate.
ADMISSIBLE_DOMAINS: dict[str, frozenset[str]] = {
    "mri": frozenset({"mri", "mri_brain", "mri_breast"}),
    "ct": frozenset({"ct", "ct_chest", "ct_head_chest_abdomen", "ct_neck_to_pelvis"}),
    # The Week 10 natural-image restoration appendix (Sec. 11.1) is the one
    # place a natural-image prior is the appropriate prior rather than a
    # stand-in, so there it is admissible.
    "natural_image": frozenset({"natural_image"}),
}

#: Domains registered above that no modality admits *yet*, on purpose, because the ruling
#: that would admit them is outstanding. Named so that "not admitted" is a recorded state
#: rather than a typo. **Empty since D80**, which ruled on the one entry it held
#: (``mri_brain``); kept because the next prior from an unanticipated anatomy should park
#: here rather than be admitted by whoever adds it.
AWAITING_RULING: frozenset[str] = frozenset()

UNKNOWN_DOMAIN = "unknown"


def prior_domain(checkpoint: str | Path | None) -> str:
    """What a checkpoint was trained on, from its basename.

    Unrecognised checkpoints return ``"unknown"`` and are treated as *not*
    admissible by :func:`prior_is_admissible`. Failing closed is the only safe
    default here: an unknown checkpoint could be anything, and the cost of
    wrongly stamping a bank reportable is a claim in a manuscript that the data
    does not support.
    """
    if checkpoint is None:
        return UNKNOWN_DOMAIN
    return PRIOR_DOMAINS.get(Path(str(checkpoint)).name, UNKNOWN_DOMAIN)


def prior_is_admissible(checkpoint: str | Path | None, modality: str) -> bool:
    """Is this prior an appropriate prior for this modality's images? (P-18)"""
    return prior_domain(checkpoint) in ADMISSIBLE_DOMAINS.get(modality, frozenset())
