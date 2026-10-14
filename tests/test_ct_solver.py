"""CT behind the shared solver interface (T2.6, Week 7): the consistency step and its device.

DDNM and FlowDPS pin each estimate's range component to the data with
``x0 - A^+(A x0 - y)``. They used to take ``A^+`` from ``operator.pinv``, which for CT is
filtered backprojection -- not a pseudo-inverse -- so the step made a sparse-view estimate
*less* consistent with its data, not more: D-012's defect, re-entering through the solver.
These tests pin the fix and the two things it must not change: MRI draws, bit for bit, and
any bank whose config does not opt in to running elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from caliber.config import load_config
from caliber.operators import MRIUndersamplingOperator
from caliber.operators.ct import CTProjectionOperator
from caliber.runner import run
from caliber.scorebank import read_bank
from caliber.solvers.denoiser import AnalyticDenoiser
from caliber.solvers.diffusion import DDNMSampler, _as_real, range_replace

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE = REPO_ROOT / "configs/experiments/w01_smoke_phantom.yaml"


def _disc(n: int) -> torch.Tensor:
    """A bright disc on a dimmer one: structure a sparse-view FBP cannot carry."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
    r2 = xx**2 + yy**2
    return (r2 < 0.6).float() * 0.5 + ((xx - 0.2) ** 2 + yy**2 < 0.05).float() * 0.5


def _residual(op, x, y) -> float:
    return float((op.forward(x) - y).norm() / y.norm())


def test_mri_consistency_step_is_unchanged_bit_for_bit():
    """For MRI decomposition_pinv *is* pinv, so the old formula and the new step agree exactly."""
    op = MRIUndersamplingOperator((32, 32), family="random1d", acceleration=4)
    gen = torch.Generator().manual_seed(0)
    x0 = torch.rand(3, 32, 32, generator=gen)
    y = op.forward(torch.rand(32, 32, generator=gen))
    old = torch.stack([_as_real(x0[b] - op.pinv(op.forward(_as_real(x0[b])) - y))
                       for b in range(3)])
    assert torch.equal(range_replace(x0, y, op), old)


@pytest.mark.parametrize("views", [8, 16, 32])
def test_ct_consistency_step_fits_the_data_where_fbp_does_not(views):
    """The CG inverse cuts the residual hard at every view count; FBP falls behind as views thin."""
    op = CTProjectionOperator((48, 48), n_views=views, cg_iterations=40)
    truth = _disc(48)
    y = op.forward(truth)
    x0 = (truth + 0.1 * torch.randn(48, 48, generator=torch.Generator().manual_seed(1)))[None]
    before = _residual(op, x0[0], y)
    with_cg = _residual(op, range_replace(x0, y, op)[0], y)
    with_fbp = _residual(op, _as_real(x0[0] - op.pinv(op.forward(x0[0]) - y)), y)
    assert with_cg < 0.1 * before, (before, with_cg)
    assert with_cg < with_fbp, (with_cg, with_fbp)


def test_running_the_step_on_a_named_device_changes_nothing_on_the_same_device():
    op = CTProjectionOperator((32, 32), n_views=12)
    x0 = torch.rand(2, 32, 32, generator=torch.Generator().manual_seed(2))
    y = op.forward(_disc(32))
    assert torch.equal(range_replace(x0, y, op, device="cpu"), range_replace(x0, y, op))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_step_on_the_gpu_agrees_with_the_cpu_and_comes_back_on_the_cpu():
    op = CTProjectionOperator((64, 64), n_views=30)
    x0 = torch.rand(2, 64, 64, generator=torch.Generator().manual_seed(3))
    y = op.forward(_disc(64))
    on_gpu = range_replace(x0, y, op, device="cuda")
    assert on_gpu.device.type == "cpu"
    cpu = range_replace(x0, y, op)
    assert float((on_gpu - cpu).abs().max() / cpu.abs().max()) < 1e-3


def test_ddnm_runs_on_a_ct_operator_and_ends_consistent_with_the_data():
    op = CTProjectionOperator((32, 32), n_views=16)
    y = op.forward(_disc(32))
    sampler = DDNMSampler(n_steps=8, denoiser=AnalyticDenoiser())
    draws = sampler.sample(y, op, n_samples=2, seed=0)
    assert draws.shape == (2, 32, 32) and bool(torch.isfinite(draws).all())
    # the final x0 is range-replaced, and the last DDIM step to t=0 returns it unchanged
    assert sampler.data_consistency_error(draws, y, op) < 0.05


def test_the_solver_takes_a_consistency_device_from_config():
    cfg_solver = {"name": "ddnm", "n_steps": 4, "consistency_device": "cpu"}
    from caliber.solvers import build_solver
    assert build_solver(cfg_solver).consistency_device == "cpu"


def _smoke(out: Path, *extra: str) -> dict:
    return load_config(str(SMOKE), [
        "run.limit=2", "experiment.n_posterior=2", "run.cache_reconstructions=0",
        "run.skip_memory_check=true", f"run.output_root={out.as_posix()}", *extra])


def test_scoring_device_is_accepted_and_does_not_change_the_bank(tmp_path):
    plain = read_bank(run(_smoke(tmp_path / "plain")))
    moved = read_bank(run(_smoke(tmp_path / "moved", "run.scoring_device=cpu")))
    for col in ("s_free", "s_disagree", "s_residual", "err_total", "err_free", "free_content"):
        assert plain[col].to_numpy().tolist() == moved[col].to_numpy().tolist(), col


def test_limited_angle_arcs_are_separate_calibration_groups():
    """The arc is limited-angle's severity; three arcs in one group pool three regimes."""
    arcs = [CTProjectionOperator((32, 32), n_views=120, angular_range_deg=a).descriptor()
            for a in (150, 120, 90)]
    assert len({d.calib_group for d in arcs}) == 3
    assert arcs[0].calib_group == "ct/limited_angle/v120/a150"


def test_the_start_angle_is_a_member_of_a_group_not_a_group():
    """Like a mask seed (D-023): a rotated angle set is the same regime."""
    a = CTProjectionOperator((32, 32), n_views=60, start_angle_deg=0).descriptor()
    b = CTProjectionOperator((32, 32), n_views=60, start_angle_deg=3).descriptor()
    assert a.calib_group == b.calib_group == "ct/sparse_view/v60/a180"
