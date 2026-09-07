"""Solver back-ends: schedule algebra, determinism, and data consistency (T2.2).

Everything here runs against :class:`AnalyticDenoiser`, so the whole file costs
milliseconds and needs neither the 434 MB checkpoint nor a GPU. That is the point of
splitting the denoiser out: these properties are cheap enough to assert on every run,
and a property only checked when someone remembers to run the slow path is not checked.
"""

from __future__ import annotations

import pytest
import torch

from caliber.operators import MRIUndersamplingOperator
from caliber.solvers import SOLVERS, build_solver
from caliber.solvers.denoiser import (
    N_TRAIN_STEPS,
    AnalyticDenoiser,
    NoiseSchedule,
    linear_beta_schedule,
)
from caliber.solvers.diffusion import DDNMSampler, DPSSampler
from caliber.solvers.flow import FlowDPSSampler

SHAPE = (32, 32)
STEPS = 12


@pytest.fixture
def setup():
    op = MRIUndersamplingOperator(SHAPE, family="random1d", acceleration=4)
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(SHAPE, generator=gen) + 0.5
    return op, x, op.forward(x), AnalyticDenoiser()


def samplers(denoiser):
    return {
        "dps": DPSSampler(n_steps=STEPS, denoiser=denoiser),
        "ddnm": DDNMSampler(n_steps=STEPS, denoiser=denoiser),
        "flowdps": FlowDPSSampler(n_steps=STEPS, denoiser=denoiser),
    }


# -- the noise schedule -------------------------------------------------------


def test_beta_schedule_matches_the_ddpm_training_settings():
    betas = linear_beta_schedule()
    assert betas.shape == (N_TRAIN_STEPS,)
    assert float(betas[0]) == pytest.approx(1e-4)
    assert float(betas[-1]) == pytest.approx(0.02)
    assert bool((betas.diff() > 0).all())


def test_alpha_bar_decreases_and_brackets_correctly():
    s = NoiseSchedule()
    assert float(s.alpha_bar(-1)) == 1.0
    assert float(s.alpha_bar(0)) > float(s.alpha_bar(N_TRAIN_STEPS - 1))
    assert 0.0 < float(s.alpha_bar(N_TRAIN_STEPS - 1)) < 0.01


def test_predict_x0_inverts_the_forward_noising_exactly():
    """The single most consequential piece of arithmetic in the samplers.

    A wrong scaling here still produces plausible-looking images while biasing every
    reconstruction, so it is pinned against the closed form rather than eyeballed.
    """
    s = NoiseSchedule()
    gen = torch.Generator().manual_seed(0)
    x0 = torch.rand(8, 8, generator=gen)
    eps = torch.randn(8, 8, generator=gen)
    for t in (0, 250, 999):
        ab = s.alpha_bar(t)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        assert torch.allclose(s.predict_x0(x_t, eps, t), x0, atol=1e-4)


def test_timesteps_descend_to_zero_and_respect_the_budget():
    s = NoiseSchedule()
    for n in (1, 10, 100):
        ts = s.timesteps(n)
        assert len(ts) == n
        assert ts == sorted(ts, reverse=True)
        assert max(ts) < N_TRAIN_STEPS and min(ts) >= 0


@pytest.mark.parametrize("bad", [0, -5, N_TRAIN_STEPS + 1])
def test_timesteps_rejects_an_impossible_budget(bad):
    with pytest.raises(ValueError):
        NoiseSchedule().timesteps(bad)


def test_deterministic_step_needs_no_noise_but_stochastic_one_does():
    s = NoiseSchedule()
    x0 = torch.zeros(4, 4)
    eps = torch.zeros(4, 4)
    s.ddim_step(x0, eps, 10, eta=0.0)  # must not raise
    with pytest.raises(ValueError):
        s.ddim_step(x0, eps, 10, eta=1.0)


# -- the sampler contract, for every family -----------------------------------


@pytest.mark.parametrize("name", ["dps", "ddnm", "flowdps"])
def test_draws_have_the_right_shape_and_are_real_and_finite(setup, name):
    op, _, y, den = setup
    draws = samplers(den)[name].sample(y, op, n_samples=3, seed=0)
    assert draws.shape == (3, *SHAPE)
    assert not draws.is_complex(), "the prior is over real images"
    assert bool(torch.isfinite(draws).all())


@pytest.mark.parametrize("name", ["dps", "ddnm", "flowdps"])
def test_same_seed_reproduces_bitwise(setup, name):
    """``solvers/base.py`` promises this, and the Week 10 package depends on it."""
    op, _, y, den = setup
    a = samplers(den)[name].sample(y, op, n_samples=2, seed=7)
    b = samplers(den)[name].sample(y, op, n_samples=2, seed=7)
    assert torch.equal(a, b)


@pytest.mark.parametrize("name", ["dps", "ddnm", "flowdps"])
def test_different_seeds_give_a_genuine_posterior_spread(setup, name):
    op, _, y, den = setup
    a = samplers(den)[name].sample(y, op, n_samples=2, seed=1)
    b = samplers(den)[name].sample(y, op, n_samples=2, seed=2)
    assert not torch.equal(a, b)
    within = samplers(den)[name].sample(y, op, n_samples=4, seed=3).std(dim=0).mean()
    assert float(within) > 1e-4, "a sampler with no spread cannot express disagreement"


@pytest.mark.parametrize("name", ["ddnm", "flowdps"])
def test_measurement_guidance_actually_uses_the_measurements(setup, name):
    """Draws conditioned on ``y`` must fit ``y`` better than draws conditioned on noise.

    Weak-looking, and deliberately so: it is the assertion that fails loudly if the
    guidance term is ever silently disconnected, which is the failure mode that would
    otherwise show up as a mysteriously flat coverage curve in Week 4.
    """
    op, _, y, den = setup
    real = samplers(den)[name].sample(y, op, n_samples=2, seed=0)
    wrong = samplers(den)[name].sample(y * 0 + y.flip(-1), op, n_samples=2, seed=0)
    assert DDNMSampler.data_consistency_error(real, y, op) < DDNMSampler.data_consistency_error(
        wrong, y, op
    )


def test_ddnm_range_replacement_drives_consistency_down_with_more_steps(setup):
    """DDNM pins the range component, so more steps must mean a better fit."""
    op, _, y, den = setup
    coarse = DDNMSampler(n_steps=8, denoiser=den).sample(y, op, n_samples=2, seed=0)
    fine = DDNMSampler(n_steps=48, denoiser=den).sample(y, op, n_samples=2, seed=0)
    assert DDNMSampler.data_consistency_error(fine, y, op) < DDNMSampler.data_consistency_error(
        coarse, y, op
    )


def test_dps_guidance_strength_changes_the_fit(setup):
    """Guards the D-028-class bug: a step scaling that cancels itself out."""
    op, _, y, den = setup
    weak = DPSSampler(n_steps=STEPS, step_size=0.0, denoiser=den).sample(y, op, 2, seed=0)
    tuned = DPSSampler(n_steps=STEPS, step_size=5.0, denoiser=den).sample(y, op, 2, seed=0)
    assert DDNMSampler.data_consistency_error(tuned, y, op) < DDNMSampler.data_consistency_error(
        weak, y, op
    )


def test_flow_sampler_is_deterministic_given_its_starting_noise(setup):
    """The two families differ here: FlowDPS's only randomness is its initial point."""
    op, _, y, den = setup
    flow = FlowDPSSampler(n_steps=STEPS, denoiser=den)
    a = flow.sample(y, op, n_samples=1, seed=11)
    b = flow.sample(y, op, n_samples=1, seed=11)
    assert torch.equal(a, b)


# -- provenance and wiring ----------------------------------------------------


@pytest.mark.parametrize("name", ["dps", "ddnm", "flowdps"])
def test_solver_info_records_the_family_and_the_weights(setup, name):
    _, _, _, den = setup
    info = samplers(den)[name].info()
    assert info.family in ("diffusion", "flow_matching")
    assert info.n_steps == STEPS
    # The checkpoint id has to reach the bank row, or a figure cannot be traced to the
    # weights that produced it (schema rule 3).
    assert info.checkpoint == "analytic-gaussian"
    assert "analytic-gaussian" in info.solver_id


def test_the_two_families_are_distinguishable_in_the_bank(setup):
    _, _, _, den = setup
    fams = {s.info().family for s in samplers(den).values()}
    assert fams == {"diffusion", "flow_matching"}


@pytest.mark.parametrize("name", ["dps", "ddnm", "flowdps"])
def test_build_solver_constructs_every_family_from_config(name):
    assert name in SOLVERS
    solver = build_solver({"name": name, "n_steps": 5})
    assert solver.n_steps == 5


@pytest.mark.parametrize("name", ["dps", "ddnm", "flowdps"])
def test_a_solver_without_weights_fails_with_a_usable_message(name):
    """Not ``NotImplementedError`` any more, and not an obscure attribute error."""
    solver = build_solver({"name": name, "n_steps": 2})
    op = MRIUndersamplingOperator(SHAPE, family="random1d", acceleration=4)
    with pytest.raises(ValueError, match="checkpoint"):
        solver.sample(op.forward(torch.zeros(SHAPE)), op, n_samples=1)


def test_solvers_refuse_a_modality_they_do_not_support(setup):
    op, _, y, den = setup
    solver = DDNMSampler(n_steps=2, denoiser=den)
    solver.supports_modalities = ("ct",)
    with pytest.raises(ValueError, match="does not support modality"):
        solver.sample(y, op, n_samples=1)
