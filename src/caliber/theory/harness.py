"""Controlled score-generating setup for T3.5 (the T3.3/T3.4 placeholder harness, generalised).

Same construction as ``scripts/t3_lipschitz_real_score_check.py`` -- real
``MRIUndersamplingOperator`` / ``CTProjectionOperator``, real ``compute_score_components``,
real ``PhantomDataset`` -- but parameterised so the checks can vary the mask family,
acceleration range, centre fraction, score weights and the *solver*:

* ``solver="noise"``  draws are ``A^+ y`` plus white noise restricted to the null space. The
  null-space content carries no image information (T3.4 sec. 2), so ``s_free`` and
  ``s_disagree`` follow a closed form in the null fraction.
* ``solver="prior"``  a toy structured prior: alternating projections between data
  consistency and a Gaussian-blur smoothness prior (a DDNM-like loop). The null-space fill
  is correlated with the image and depends on ``A`` through the iteration, which is what
  T3.3 sec. 4b item 3 needs and the noise harness sidesteps.

Scales are estimated once on the calibration operator and frozen (D27).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from ..data.synthetic import PhantomDataset
from ..operators.base import ForwardOperator
from ..operators.ct import CTProjectionOperator
from ..operators.mri import MRIUndersamplingOperator
from ..scoring.labels import RegionTruth, compute_region_truth, label_hallucinated
from ..scoring.regions import RegionGrid
from ..scoring.score import (
    ScoreComponents,
    ScoreScales,
    ScoreWeights,
    combine_components,
    compute_score_components,
)


@dataclass(frozen=True)
class HarnessConfig:
    image_shape: tuple[int, int] = (128, 128)
    center_fraction: float = 0.04
    region_size: int = 32
    n_draws: int = 8
    meas_noise: float = 0.02
    fill_noise: float = 0.4
    mask_seed: int = 1
    solver: str = "noise"  # "noise" | "prior"
    prior_iters: int = 8
    prior_sigma: float = 2.0
    phantom_seed: int = 7
    n_scale_items: int = 40
    weights: ScoreWeights = field(default_factory=ScoreWeights)

    @property
    def grid(self) -> RegionGrid:
        return RegionGrid(self.image_shape, region_size=self.region_size)


def make_mri(cfg: HarnessConfig, family: str, acceleration: float) -> MRIUndersamplingOperator:
    return MRIUndersamplingOperator(
        cfg.image_shape,
        family=family,
        acceleration=float(acceleration),
        center_fraction=cfg.center_fraction,
        mask_seed=cfg.mask_seed,
    )


def make_ct(cfg: HarnessConfig, n_views: int, angular_range_deg: float = 180.0):
    return CTProjectionOperator(
        cfg.image_shape, n_views=n_views, angular_range_deg=angular_range_deg, cg_iterations=15
    )


def _randn_like_measure(y: Tensor, gen: torch.Generator) -> Tensor:
    real = torch.randn(y.shape, generator=gen)
    if y.is_complex():
        return real + 1j * torch.randn(y.shape, generator=gen)
    return real


def _blur(x: Tensor, sigma: float) -> Tensor:
    radius = max(1, int(3 * sigma))
    t = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    k2 = (k[:, None] * k[None, :])[None, None]

    def one(p: Tensor) -> Tensor:
        p = F.pad(p[None, None], (radius,) * 4, mode="reflect")
        return F.conv2d(p, k2)[0, 0]

    if x.is_complex():
        return torch.complex(one(x.real.float()), one(x.imag.float()))
    return one(x.float())


def draws_for(
    x: Tensor, op: ForwardOperator, gen: torch.Generator, cfg: HarnessConfig
) -> tuple[Tensor, Tensor]:
    """``(n_draws, H, W)`` posterior-like samples and the observed measurement."""
    y = op.forward(x)
    y_obs = y + cfg.meas_noise * _randn_like_measure(y, gen)
    draws = []
    for _ in range(cfg.n_draws):
        y_solver = y_obs + (cfg.meas_noise / 2) * _randn_like_measure(y, gen)
        base = op.decomposition_pinv(y_solver) if op.modality == "ct" else op.pinv(y_solver)
        noise_field = cfg.fill_noise * torch.randn(*cfg.image_shape, generator=gen)
        fill = op.project_null(noise_field.to(base.dtype))
        cur = base + fill
        if cfg.solver == "prior":
            for _ in range(cfg.prior_iters):
                cur = base + op.project_null(_blur(cur, cfg.prior_sigma).to(base.dtype))
        draws.append(cur)
    return torch.stack(draws, dim=0), y_obs


def phantoms(cfg: HarnessConfig, n_items: int = 400) -> PhantomDataset:
    return PhantomDataset(n_items=n_items, image_shape=cfg.image_shape, seed=cfg.phantom_seed)


def frozen_scales(cal_op: ForwardOperator, cfg: HarnessConfig) -> ScoreScales:
    """D27: estimated on the calibration operator only, before any deployment operator."""
    data = phantoms(cfg)
    gen = torch.Generator().manual_seed(1)
    comps = []
    for i in range(cfg.n_scale_items):
        samples, y = draws_for(data[i].image, cal_op, gen, cfg)
        comps.append(compute_score_components(samples, y, cal_op, cfg.grid))
    return ScoreScales.from_components(comps)


@dataclass
class TrialSet:
    """Per-trial, per-region quantities for one operator, aligned across operators by trial."""

    components: Tensor  # (n_trials, 3, n_regions): s_free, s_disagree, s_residual
    truth: list[RegionTruth] | None = None

    def scores(self, weights: ScoreWeights, scales: ScoreScales) -> Tensor:
        """``(n_trials, n_regions)`` combined scores."""
        n, _, r = self.components.shape
        flat = ScoreComponents(
            s_free=self.components[:, 0].reshape(-1),
            s_disagree=self.components[:, 1].reshape(-1),
            s_residual=self.components[:, 2].reshape(-1),
            ref_intensity=1.0,
        )
        return combine_components(flat, weights, scales).reshape(n, r)

    def err_ratio(self) -> Tensor:
        assert self.truth is not None
        return torch.stack([t.err_free / t.free_content.clamp_min(1e-6) for t in self.truth])


def run_trials(
    op: ForwardOperator,
    cfg: HarnessConfig,
    n_trials: int,
    with_truth: bool = False,
    seed_base: int = 50_000,
) -> TrialSet:
    """Score ``n_trials`` phantoms at ``op``.

    Trial ``t`` uses the same image and the same noise seed at every operator (paired
    evaluation), so differences between operators isolate the operator.
    """
    data = phantoms(cfg)
    comps, truths = [], []
    for t in range(n_trials):
        x = data[cfg.n_scale_items + t].image
        gen = torch.Generator().manual_seed(seed_base + t)
        samples, y = draws_for(x, op, gen, cfg)
        c = compute_score_components(samples, y, op, cfg.grid)
        comps.append(c.stack())
        if with_truth:
            truths.append(
                compute_region_truth(samples, x, op, cfg.grid, ref_intensity=c.ref_intensity)
            )
    return TrialSet(torch.stack(comps), truths if with_truth else None)


def positive_mask(ts: TrialSet, tau: float) -> Tensor:
    """Relative hallucination label (T1.1/D21) on a trial set, ``(n_trials, n_regions)``."""
    assert ts.truth is not None
    return torch.stack([label_hallucinated(t, threshold=tau, relative=True) for t in ts.truth])
