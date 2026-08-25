"""T1.1 WI-5 — how many posterior draws does the disagreement term actually need?

Q3 splits in two (D18). The **cost** half — seconds per draw — belongs to the
real solver and waits on T2.2. The **statistical** half is answerable now, on a
mock sampler, and that is what this script measures.

The quantity that matters is *not* how fast ``s_disagree`` converges in absolute
value. Conformal calibration places a threshold on the score, so only the
**ordering** of regions has to be stable: a score that is uniformly 5% low but
ranks regions identically produces exactly the same flags. The budget is
therefore set by rank stability against a large-n reference, not by RMS error.

CPU-only, no dataset, no checkpoint. Run:

    python scripts/draw_budget.py [--reference 256] [--trials 5]
"""

from __future__ import annotations

import argparse

import torch

from caliber.operators.registry import build_operator
from caliber.scoring.regions import RegionGrid
from caliber.scoring.score import (
    ScoreScales,
    ScoreWeights,
    combine_components,
    compute_score_components,
)

SHAPE = (128, 128)


def smooth(shape, coarseness, gen):
    small = torch.randn(
        1, 1, max(2, shape[0] // coarseness), max(2, shape[1] // coarseness), generator=gen
    )
    up = torch.nn.functional.interpolate(small, size=shape, mode="bilinear", align_corners=False)
    return up.reshape(*shape)


def draws(op, base, n, scale, gen):
    """Data-consistent draws differing only inside the null space.

    ``scale`` may be a scalar or a per-pixel map. A map is the physically
    meaningful case and the default for the budget measurement: a real prior
    disagrees with itself *more* where it is inventing and less where the data
    pins the answer down. With a spatially uniform scale, ``s_disagree`` is an
    estimate of the same constant in every region, carries no ranking signal at
    all, and rank stability then plateaus at the noise floor no matter how many
    draws are taken — which measures the mock, not the score.

    The projection is kept complex on purpose: a random 1-D mask is not
    conjugate-symmetric, so discarding the imaginary part would leak the
    perturbation back into the measured component (D29).
    """
    return torch.stack(
        [base + op.project_null(smooth(SHAPE, 4, gen) * scale) for _ in range(n)], dim=0
    )


def uncertainty_map(base_scale=0.02, hot_scale=0.20, block=(2, 5)):
    """Posterior spread that is high inside the hallucinated block, low outside."""
    m = torch.full(SHAPE, base_scale)
    r0, r1 = block[0] * 16, block[1] * 16
    m[r0:r1, r0:r1] = hot_scale
    return m


def inject(op, x, gen, amplitude=0.8, block=(2, 5)):
    """Add a data-consistent hallucination so the ranking has real signal to recover.

    Measuring rank stability on a clean image is meaningless: every region is
    then statistically identical, the "true" ordering is itself noise, and the
    correlation saturates at whatever the noise floor happens to be rather than
    telling you anything about the budget.
    """
    delta = torch.zeros(SHAPE)
    r0, r1 = block[0] * 16, block[1] * 16
    delta[r0:r1, r0:r1] = smooth((r1 - r0, r1 - r0), 2, gen) * amplitude
    return x + op.project_null(delta)


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Rank correlation — the ordering is what a conformal threshold consumes."""

    def rank(t):
        order = t.argsort()
        r = torch.empty_like(order, dtype=torch.float64)
        r[order] = torch.arange(t.numel(), dtype=torch.float64)
        return r

    ra, rb = rank(a.double()), rank(b.double())
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=int, default=256, help="draws for the reference score")
    ap.add_argument("--trials", type=int, default=5, help="independent repeats per budget")
    ap.add_argument("--acceleration", type=float, default=4)
    args = ap.parse_args()

    op = build_operator(
        {
            "modality": "mri",
            "family": "random1d",
            "acceleration": args.acceleration,
            "center_fraction": 0.08,
            "mask_seed": 0,
        },
        SHAPE,
    )
    grid = RegionGrid(image_shape=SHAPE, region_size=16)
    weights = ScoreWeights()

    gen = torch.Generator().manual_seed(0)
    x = inject(op, smooth(SHAPE, 8, gen).abs() + 1.0, gen)
    y = op.forward(x)
    spread = uncertainty_map()

    ref_comps = compute_score_components(draws(op, x, args.reference, spread, gen), y, op, grid)
    scales = ScoreScales.from_components(ref_comps)
    ref_score = combine_components(ref_comps, weights, scales)

    budgets = [2, 4, 8, 16, 32, 64]
    print(f"reference = {args.reference} draws, {args.trials} trials per budget, "
          f"R = {args.acceleration:g}, with an injected hallucination\n")
    print(f"{'draws':>6}{'rank corr vs ref':>20}{'worst trial':>14}{'rel. RMS err':>15}")
    for n in budgets:
        rhos, errs = [], []
        for t in range(args.trials):
            g = torch.Generator().manual_seed(1000 + t)
            comps = compute_score_components(draws(op, x, n, spread, g), y, op, grid)
            score = combine_components(comps, weights, scales)
            rhos.append(spearman(score, ref_score))
            denom = float(torch.linalg.vector_norm(ref_score.double()))
            errs.append(float(torch.linalg.vector_norm((score - ref_score).double()) / denom))
        rhos_t = torch.tensor(rhos)
        print(f"{n:>6}{rhos_t.mean():>20.4f}{rhos_t.min():>14.4f}"
              f"{torch.tensor(errs).mean():>15.4f}")

    print(
        "\nRead the rank-correlation column, not the RMS column. A uniform scale\n"
        "error changes no flags; a reordering does. Pick the smallest budget whose\n"
        "*worst* trial clears the tolerance, since the budget has to hold for every\n"
        "item in the bank, not on average."
    )


if __name__ == "__main__":
    main()
