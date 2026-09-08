"""
Re-run of T3.3 assumption (ii) against T1.1's ACTUAL score code, not a proxy.

Uses the real caliber.operators.mri.MRIUndersamplingOperator, the real
caliber.scoring.score.compute_score_components / combine_components, the real
caliber.operators.distance.null_space_projection_distance, and the real
caliber.data.synthetic.PhantomDataset (which the module docstring says exists
partly for exactly this: T3.5's numerical verification).

The top-level `caliber` and `caliber.data` packages fail to import because
`caliber/data/fastmri.py` references a `caliber.data.splits` module that does
not exist in the repo (a Track 2 bug, unrelated to Track 3, not something to
fix from here). Worked around with a sys.modules bypass that skips only the
broken __init__.py bodies -- no repository files are modified.
"""
import sys
import types

import torch

for name, path in [("caliber", "src/caliber"), ("caliber.data", "src/caliber/data")]:
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m

from caliber.data.synthetic import PhantomDataset  # noqa: E402
from caliber.operators.mri import MRIUndersamplingOperator  # noqa: E402
from caliber.operators.distance import null_space_projection_distance  # noqa: E402
from caliber.scoring.regions import RegionGrid  # noqa: E402
from caliber.scoring.score import ScoreScales, ScoreWeights, compute_score_components  # noqa: E402

IMAGE_SHAPE = (128, 128)
CENTER_FRACTION = 0.04
N_DRAWS = 8
MEAS_NOISE = 0.02
FILL_NOISE = 0.4


def make_op(acceleration: float, mask_seed: int = 1) -> MRIUndersamplingOperator:
    return MRIUndersamplingOperator(
        IMAGE_SHAPE, family="random1d", acceleration=acceleration,
        center_fraction=CENTER_FRACTION, mask_seed=mask_seed,
    )


def draws_for(x: torch.Tensor, op: MRIUndersamplingOperator, gen: torch.Generator):
    """samples: (n_draws, H, W); y: the single observed measurement for the item."""
    y = op.forward(x)
    y_obs = y + MEAS_NOISE * (torch.randn(y.shape, generator=gen) + 1j * torch.randn(y.shape, generator=gen))
    draws = []
    for _ in range(N_DRAWS):
        y_solver = y_obs + (MEAS_NOISE / 2) * (
            torch.randn(y.shape, generator=gen) + 1j * torch.randn(y.shape, generator=gen)
        )
        base = op.pinv(y_solver)
        noise_field = FILL_NOISE * torch.randn(*IMAGE_SHAPE, generator=gen)
        fill = op.project_null(noise_field.to(base.dtype))
        draws.append(base + fill)
    return torch.stack(draws, dim=0), y_obs


def main():
    grid = RegionGrid(IMAGE_SHAPE, region_size=32)  # 4x4 = 16 regions
    weights = ScoreWeights()  # ratified default: equal thirds

    cal_op = make_op(acceleration=2.0)  # matches T1.3's own calibration choice
    accel_sweep = [2, 3, 4, 6, 8, 12, 16]

    n_scale_items = 40
    phantoms = PhantomDataset(n_items=200, image_shape=IMAGE_SHAPE, seed=7)

    # --- D27: estimate scales from the calibration operator ONCE, then freeze. ---
    scale_components = []
    gen = torch.Generator().manual_seed(1)
    for i in range(n_scale_items):
        x = phantoms[i].image
        samples, y = draws_for(x, cal_op, gen)
        scale_components.append(compute_score_components(samples, y, cal_op, grid))
    scales = ScoreScales.from_components(scale_components)
    print(f"frozen scales (from A_cal only, D27): free={scales.free:.4g} "
          f"disagree={scales.disagree:.4g} residual={scales.residual:.4g}")

    n_x_trials = 120
    print(f"\n{'R':>4} {'d_op':>8} {'E|delta|':>10} {'ratio':>10}")
    results = []
    for R in accel_sweep:
        deploy_op = cal_op if R == 2 else make_op(acceleration=float(R))
        d = null_space_projection_distance(cal_op, deploy_op)

        abs_deltas = []
        for t in range(n_x_trials):
            x = phantoms[n_scale_items + t].image
            gen_cal = torch.Generator().manual_seed(50_000 + t)
            gen_dep = torch.Generator().manual_seed(50_000 + t)  # paired evaluation noise

            samples_cal, y_cal = draws_for(x, cal_op, gen_cal)
            comp_cal = compute_score_components(samples_cal, y_cal, cal_op, grid)
            score_cal = comp_cal.combine(weights, scales)  # (n_regions,)

            samples_dep, y_dep = draws_for(x, deploy_op, gen_dep)
            comp_dep = compute_score_components(samples_dep, y_dep, deploy_op, grid)
            score_dep = comp_dep.combine(weights, scales)

            abs_deltas.extend((score_dep - score_cal).abs().tolist())  # pool over regions too

        e_abs_delta = float(torch.tensor(abs_deltas).mean())
        ratio = e_abs_delta / d if d > 1e-9 else float("nan")
        print(f"{R:>4} {d:>8.4f} {e_abs_delta:>10.5f} {ratio:>10.4f}")
        results.append((R, d, e_abs_delta, ratio))
    return results


if __name__ == "__main__":
    main()
