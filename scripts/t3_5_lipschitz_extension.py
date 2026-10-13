"""T3.5 check 1: does T3.3 assumption (ii) / T3.4's ratio shape survive outside what was tested?

Extends ``t3_lipschitz_real_score_check.py`` along the four axes T3.3 sec. 6 named, plus the
solver axis of sec. 4b item 3:

* acceleration beyond R=16 (``center_fraction`` lowered so the mask does not saturate),
* mask families ``equispaced1d`` and ``variable_density2d`` (calibration stays ``random1d`` R=2),
* the *current* score (``s_free`` + ``s_disagree``, D42) next to the original three-component one,
* the structured toy prior (``--solver prior``) next to the white-noise placeholder.

For every deployment operator it prints ``d_op`` (D19), the signed ``null_space_growth`` (D47),
the secant ratio ``E|score(dep) - score(cal)| / d`` for each score variant, and for nested masks
the closed-form shape ``(sqrt(a + d^2) - sqrt(a)) / d`` of T3.4 sec. 2b next to the free ratio.

    python scripts/t3_5_lipschitz_extension.py [--solver noise|prior] [--trials 120]
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import torch

from caliber.operators.distance import null_space_growth, null_space_projection_distance
from caliber.scoring.score import ScoreWeights
from caliber.theory.harness import HarnessConfig, frozen_scales, make_mri, run_trials

VARIANTS = {
    "3comp": ScoreWeights(),
    "2comp": ScoreWeights(free=0.5, disagree=0.5, residual=0.0),
    "free": ScoreWeights.only("free"),
    "disagree": ScoreWeights.only("disagree"),
}


def one_seed(cfg, deployments, n_trials):
    """Ratios for every deployment operator at one mask seed (calibration shares the seed)."""
    cal = make_mri(cfg, "random1d", 2)
    scales = frozen_scales(cal, cfg)
    base = run_trials(cal, cfg, n_trials)
    base_scores = {k: base.scores(w, scales) for k, w in VARIANTS.items()}
    a = cal.descriptor().null_space_dim / (cfg.image_shape[0] * cfg.image_shape[1])
    out = []
    for family, acc in deployments:
        dep = make_mri(cfg, family, acc)
        d = null_space_projection_distance(cal, dep)
        g = null_space_growth(cal, dep)
        ts = run_trials(dep, cfg, n_trials)
        ratios = {}
        for k, w in VARIANTS.items():
            delta = (ts.scores(w, scales) - base_scores[k]).abs().mean().item()
            ratios[k] = delta / d if d > 1e-9 else float("nan")
        nf = dep.descriptor().null_space_dim / (cfg.image_shape[0] * cfg.image_shape[1])
        nested = d > 1e-9 and abs(abs(g) - d) < 1e-6  # growth == d_op only for nested masks
        closed = (nf**0.5 - a**0.5) / d if nested else float("nan")
        out.append((f"{family[:6]}.R{acc:g}", d, g, ratios, closed))
    return out


def sweep(cfg, deployments, n_trials, label, seeds=1):
    runs = [one_seed(replace(cfg, mask_seed=cfg.mask_seed + s), deployments, n_trials)
            for s in range(seeds)]
    plural = "s" if seeds > 1 else ""
    print(
        f"\n== {label}  (solver={cfg.solver}, center_fraction={cfg.center_fraction}, "
        f"{seeds} mask seed{plural}) =="
    )
    head = f"{'operator':>14} {'d_op':>6} {'growth':>7} "
    head += " ".join(f"{k:>12}" for k in VARIANTS)
    print(head + f" {'closed':>7} {'free/closed':>11}")
    for i in range(len(deployments)):
        name, d, g, _, closed = runs[0][i]
        cells, free_vals = [], []
        for k in VARIANTS:
            vals = np.array([r[i][3][k] for r in runs])
            cells.append(f"{vals.mean():>6.3f}+-{vals.std():<4.3f}" if seeds > 1
                         else f"{vals.mean():>12.3f}")
            if k == "free":
                free_vals = vals
        fr = free_vals.mean() / closed if closed == closed and closed > 0 else float("nan")
        row = f"{name:>14} {d:>6.3f} {g:>+7.3f} " + " ".join(cells)
        print(row + f" {closed:>7.3f} {fr:>11.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="noise", choices=["noise", "prior"])
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=1, help="mask seeds to average over")
    args = ap.parse_args(argv)
    torch.manual_seed(0)

    # 1. T3.3's own range, reproduced (centre 0.04, R<=16, random1d).
    cfg = HarnessConfig(solver=args.solver, center_fraction=0.04)
    rng_a = [("random1d", r) for r in (3, 4, 6, 8, 12, 16)]
    sweep(cfg, rng_a, args.trials, "T3.3 range reproduced", args.seeds)

    # 2. Beyond R=16: a thinner centre keeps the mask from saturating (n_center = 3 columns).
    cfg = HarnessConfig(solver=args.solver, center_fraction=0.02)
    rng_b = [("random1d", r) for r in (4, 8, 16, 24, 32)]
    sweep(cfg, rng_b, args.trials, "R > 16, random1d", args.seeds)

    # 3. Other mask families, same calibration operator (random1d R=2).
    deps = [(f, r) for f in ("equispaced1d", "variable_density2d") for r in (2, 4, 8, 16)]
    cfg = HarnessConfig(solver=args.solver, center_fraction=0.04)
    sweep(cfg, deps, args.trials, "cross-family", args.seeds)


if __name__ == "__main__":
    main()
