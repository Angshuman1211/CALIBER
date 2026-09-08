"""
T3.4 (L2) numerical check, part 2: per-component breakdown of T3.3's real-score
ratio, plus a test of a specific closed-form ill-posedness prediction for the
free and disagree components.

Why per-component. T3.3 combined all three score components (D25's affine
rule) before computing E|delta score| / d_op, and explicitly handed the
per-component breakdown forward as an open item (T3.3 checklist, item 2) --
narrowed to "T3.4/T3.5's territory" by the README. This script computes it.

Why a closed-form prediction for free/disagree specifically. Reading
`compute_score_components` (caliber/scoring/score.py) against the harness in
`t3_lipschitz_real_score_check.py` shows something not mentioned in T3.3: in
that harness, `base = operator.pinv(y_solver)` reconstructs the measured band
EXACTLY (pinv is the exact adjoint for a mask operator, so `fft(base)` is
supported only inside the mask), which forces `project_null(base) == 0`
exactly. Every posterior "draw" only differs from `base` by
`fill = operator.project_null(FILL_NOISE * randn(...))` -- independent
Gaussian white noise, restricted to the null space. So in THIS harness
(a placeholder for a real posterior sampler, not one yet):

    free  = |mean_i(fill_i)|            -- mean of N_DRAWS iid null-restricted
                                            noise fields
    disagree = std_i(|fill_i|)          -- their spread

Both are driven purely by null-space DIMENSION and FILL_NOISE, not by the true
image's content -- there is no learned prior filling the null space with real
structure yet, so s_free and s_disagree should be (up to one fitted constant)
proportional to sqrt(null_dim / ambient_dim), independent of what image is
being scored. This is a directly falsifiable, closed-form prediction, checked
below against the real harness rather than assumed.

s_residual is not expected to follow this: `operator.residual` and
`operator.adjoint` both re-apply the mask, so it reflects genuine in-mask
measurement-noise consistency, not null-space dimension.

Same sys.modules bypass as `t3_lipschitz_real_score_check.py` for the
unrelated `caliber.data.splits` import bug (Track 2, not touched here).
"""
import sys
import types

import torch

for name, path in [("caliber", "src/caliber"), ("caliber.data", "src/caliber/data")]:
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m

from caliber.data.synthetic import PhantomDataset  # noqa: E402
from caliber.operators.distance import null_space_projection_distance  # noqa: E402
from caliber.operators.mri import MRIUndersamplingOperator  # noqa: E402
from caliber.scoring.regions import RegionGrid  # noqa: E402
from caliber.scoring.score import ScoreScales, ScoreWeights, compute_score_components  # noqa: E402

IMAGE_SHAPE = (128, 128)
CENTER_FRACTION = 0.04
N_DRAWS = 8
MEAS_NOISE = 0.02
FILL_NOISE = 0.4
MASK_SEED = 1


def make_op(acceleration: float) -> MRIUndersamplingOperator:
    return MRIUndersamplingOperator(
        IMAGE_SHAPE, family="random1d", acceleration=acceleration,
        center_fraction=CENTER_FRACTION, mask_seed=MASK_SEED,
    )


def draws_for(x: torch.Tensor, op: MRIUndersamplingOperator, gen: torch.Generator):
    y = op.forward(x)
    y_obs = y + MEAS_NOISE * (
        torch.randn(y.shape, generator=gen) + 1j * torch.randn(y.shape, generator=gen)
    )
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
    grid = RegionGrid(IMAGE_SHAPE, region_size=32)
    ambient = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]

    cal_op = make_op(2.0)
    accel_sweep = [2, 3, 4, 6, 8, 12, 16]

    n_scale_items = 40
    phantoms = PhantomDataset(n_items=200, image_shape=IMAGE_SHAPE, seed=7)

    scale_components = []
    gen = torch.Generator().manual_seed(1)
    for i in range(n_scale_items):
        x = phantoms[i].image
        samples, y = draws_for(x, cal_op, gen)
        scale_components.append(compute_score_components(samples, y, cal_op, grid))
    scales = ScoreScales.from_components(scale_components)
    weights = ScoreWeights()  # ratified default: equal thirds

    n_x_trials = 120
    print(f"{'R':>4} {'null_frac':>9} {'sqrt(nf)':>9} "
          f"{'raw_free':>9} {'raw_dis':>9} {'raw_res':>9} "
          f"{'d_op':>8} {'ratio_free':>10} {'ratio_dis':>10} {'ratio_res':>10} "
          f"{'ratio_sum':>10} {'ratio_comb':>10}")

    results = []
    cal_null_frac = cal_op.descriptor().null_space_dim / ambient
    raw_free_cal = raw_dis_cal = raw_res_cal = None

    for R in accel_sweep:
        deploy_op = cal_op if R == 2 else make_op(float(R))
        d = null_space_projection_distance(cal_op, deploy_op)
        null_frac = deploy_op.descriptor().null_space_dim / ambient

        free_vals, dis_vals, res_vals = [], [], []
        abs_deltas_free, abs_deltas_dis, abs_deltas_res = [], [], []
        abs_deltas_combined, sum_abs_deltas = [], []

        for t in range(n_x_trials):
            x = phantoms[n_scale_items + t].image
            gen_cal = torch.Generator().manual_seed(50_000 + t)
            gen_dep = torch.Generator().manual_seed(50_000 + t)

            samples_cal, y_cal = draws_for(x, cal_op, gen_cal)
            comp_cal = compute_score_components(samples_cal, y_cal, cal_op, grid)
            score_cal = comp_cal.combine(weights, scales)

            samples_dep, y_dep = draws_for(x, deploy_op, gen_dep)
            comp_dep = compute_score_components(samples_dep, y_dep, deploy_op, grid)
            score_dep = comp_dep.combine(weights, scales)

            free_vals.extend(comp_dep.s_free.tolist())
            dis_vals.extend(comp_dep.s_disagree.tolist())
            res_vals.extend(comp_dep.s_residual.tolist())

            # standardised (same units as the combined score) per-trial deltas
            d_free = (comp_dep.s_free - comp_cal.s_free) / scales.free
            d_dis = (comp_dep.s_disagree - comp_cal.s_disagree) / scales.disagree
            d_res = (comp_dep.s_residual - comp_cal.s_residual) / scales.residual
            abs_deltas_free.extend(d_free.abs().tolist())
            abs_deltas_dis.extend(d_dis.abs().tolist())
            abs_deltas_res.extend(d_res.abs().tolist())
            # sum-of-abs-deltas (what you'd get with NO sign cancellation) vs
            # the actual combined score's abs-of-sum (T3.3's quantity)
            sum_abs_deltas.extend(((d_free.abs() + d_dis.abs() + d_res.abs()) / 3).tolist())
            abs_deltas_combined.extend((score_dep - score_cal).abs().tolist())

        raw_free = float(torch.tensor(free_vals).mean())
        raw_dis = float(torch.tensor(dis_vals).mean())
        raw_res = float(torch.tensor(res_vals).mean())
        if R == 2:
            raw_free_cal, raw_dis_cal, raw_res_cal = raw_free, raw_dis, raw_res

        e_free = float(torch.tensor(abs_deltas_free).mean())
        e_dis = float(torch.tensor(abs_deltas_dis).mean())
        e_res = float(torch.tensor(abs_deltas_res).mean())
        e_sum = float(torch.tensor(sum_abs_deltas).mean())
        e_combined = float(torch.tensor(abs_deltas_combined).mean())

        ratio_free = e_free / d if d > 1e-9 else float("nan")
        ratio_dis = e_dis / d if d > 1e-9 else float("nan")
        ratio_res = e_res / d if d > 1e-9 else float("nan")
        ratio_sum = e_sum / d if d > 1e-9 else float("nan")
        ratio_combined = e_combined / d if d > 1e-9 else float("nan")

        print(f"{R:>4} {null_frac:>9.4f} {null_frac**0.5:>9.4f} "
              f"{raw_free:>9.5f} {raw_dis:>9.5f} {raw_res:>9.5f} "
              f"{d:>8.4f} {ratio_free:>10.4f} {ratio_dis:>10.4f} {ratio_res:>10.4f} "
              f"{ratio_sum:>10.4f} {ratio_combined:>10.4f}")
        results.append((R, null_frac, raw_free, raw_dis, raw_res, d,
                         ratio_free, ratio_dis, ratio_res, ratio_sum, ratio_combined))

    print("\n-- closed-form check: raw component / sqrt(null_frac), normalised to R=2 --")
    print(f"{'R':>4} {'free/sqrt(nf)':>14} {'disagree/sqrt(nf)':>18} {'residual/sqrt(nf)':>18}")
    for R, null_frac, raw_free, raw_dis, raw_res, *_ in results:
        pred = null_frac**0.5
        norm_free = (raw_free / pred) / (raw_free_cal / cal_null_frac**0.5)
        norm_dis = (raw_dis / pred) / (raw_dis_cal / cal_null_frac**0.5)
        norm_res = (
            (raw_res / pred) / (raw_res_cal / cal_null_frac**0.5) if pred > 0 else float("nan")
        )
        print(f"{R:>4} {norm_free:>14.3f} {norm_dis:>18.3f} {norm_res:>18.3f}")

    return results


if __name__ == "__main__":
    main()
