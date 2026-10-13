"""T3.5 check 2: the density-difference constant D, and the TV / Kolmogorov distinction.

T3.3 sec. 3(iii) derives ``d_TV <= sqrt(D * W1)`` from two "exact identities",
``d_TV = sup|G|`` and ``W1 = int|G|`` with ``G = F_cal - F_dep``. The second is right. The
first is the **Kolmogorov** distance, not total variation (``ks <= tv``, strict in general).
This script

A. shows on an analytic family that ``tv`` can exceed ``sqrt(D * W1)`` while ``ks`` cannot, so
   the tent argument proves a bound on ``ks`` only;
B. measures ``ks, W1, tv, D`` and ``sqrt(D * W1)`` on the harness score laws (T3.3's open
   constant D, never measured before), at several operators and KDE bandwidths.

    python scripts/t3_5_density_constant.py [--solver noise|prior] [--trials 300]
"""

from __future__ import annotations

import argparse

import numpy as np

from caliber.scoring.score import ScoreWeights
from caliber.theory import distribution_distances
from caliber.theory.harness import HarnessConfig, frozen_scales, make_mri, run_trials


def analytic_counterexample() -> None:
    """q_a = 1 and q_b = 1 + eps*sin(2 pi k x) on [0, 1]: same D, TV fixed, KS and W1 shrink."""
    print("\n== A. analytic family  q_a = 1,  q_b = 1 + eps sin(2 pi k x),  eps = 0.5 ==")
    print(
        f"{'k':>4} {'ks':>9} {'w1':>9} {'tv':>9} {'D':>7} {'sqrt(DW1)':>10} "
        f"{'ks<=':>5} {'tv<=':>5}"
    )
    x = np.linspace(0, 1, 400_001)
    dx = x[1] - x[0]
    eps = 0.5
    for k in (1, 2, 5, 10, 20, 50):
        qa = np.ones_like(x)
        qb = 1 + eps * np.sin(2 * np.pi * k * x)
        g = np.cumsum(qa - qb) * dx
        ks = np.abs(g).max()
        w1 = np.abs(g).sum() * dx
        tv = 0.5 * np.abs(qa - qb).sum() * dx
        d = np.abs(qa - qb).max()
        bound = np.sqrt(d * w1)
        print(
            f"{k:>4} {ks:>9.5f} {w1:>9.5f} {tv:>9.5f} {d:>7.3f} {bound:>10.5f} "
            f"{str(ks <= bound + 1e-9):>5} {str(tv <= bound + 1e-9):>5}"
        )


def harness_laws(solver: str, n_trials: int) -> None:
    cfg = HarnessConfig(solver=solver, center_fraction=0.04)
    cal = make_mri(cfg, "random1d", 2)
    scales = frozen_scales(cal, cfg)
    w = ScoreWeights(free=0.5, disagree=0.5, residual=0.0)  # the score after D42
    cal_scores = run_trials(cal, cfg, n_trials).scores(w, scales).numpy().ravel()
    print(f"\n== B. harness score laws, solver={solver}, 2-component score, n={cal_scores.size} ==")
    print(
        f"{'dep':>10} {'ks':>7} {'w1':>7} {'tv':>7} {'D':>7} {'sqrt(DW1)':>9} "
        f"{'ks<=':>5} {'tv<=':>5} {'2*tv':>6} {'2*sqrt':>7}"
    )
    for acc in (3, 4, 8, 16):
        dep = make_mri(cfg, "random1d", acc)
        dep_scores = run_trials(dep, cfg, n_trials).scores(w, scales).numpy().ravel()
        r = distribution_distances(cal_scores, dep_scores)
        print(
            f"{'R' + str(acc):>10} {r.ks:>7.4f} {r.w1:>7.4f} {r.tv:>7.4f} {r.d_sup:>7.3f} "
            f"{r.sqrt_dw1:>9.4f} {str(r.ks_bound_holds):>5} {str(r.tv_bound_holds):>5} "
            f"{2 * r.tv:>6.3f} {2 * r.sqrt_dw1:>7.3f}"
        )
    # D is a sup of a KDE difference, so it moves with the bandwidth; report the sensitivity.
    dep = make_mri(cfg, "random1d", 8)
    dep_scores = run_trials(dep, cfg, n_trials).scores(w, scales).numpy().ravel()
    base = distribution_distances(cal_scores, dep_scores)
    print("\n   bandwidth sensitivity at R=8 (D and the bound move; ks and w1 do not):")
    print(f"   {'bw x':>6} {'bw':>8} {'D':>8} {'sqrt(DW1)':>10} {'tv':>7}")
    for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
        r = distribution_distances(cal_scores, dep_scores, bandwidth=base.bandwidth * mult)
        print(f"   {mult:>6} {r.bandwidth:>8.4f} {r.d_sup:>8.3f} {r.sqrt_dw1:>10.4f} {r.tv:>7.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="noise", choices=["noise", "prior"])
    ap.add_argument("--trials", type=int, default=300)
    args = ap.parse_args(argv)
    analytic_counterexample()
    harness_laws(args.solver, args.trials)


if __name__ == "__main__":
    main()
