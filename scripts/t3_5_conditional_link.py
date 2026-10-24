"""T3.5 check 6: the label-conditional link between operator movement and the gap.

T3.3 sec. 3b (T3.5 sec. 9) derives, under a common coupling across the two operators,

    d_K^+  <=  ( sup_t |G_A(t) - G_B(t)|  +  |pi_A - pi_B| ) / pi_A            (exact algebra)
    sup_t |G_A - G_B|  <=  P_flip + sup_t P(t lies between S_A and S_B)
    |pi_A - pi_B|      <=  P_flip

with ``G_X(t) = P(S_X <= t, label_X = 1)``, ``pi_X = P(label_X = 1)``, ``P_flip = P(label_A !=
label_B)``, ``d_K^+`` the Kolmogorov distance of the laws of the score given ``label = 1``.
Then Markov plus a density bound turn ``P_flip`` and the crossing probability into
``O(sqrt(d_op))``. This script evaluates every term on the paired harness, so each inequality
is checked on data rather than asserted:

* the exact-algebra bound versus the measured ``d_K^+`` (must hold);
* ``P_flip`` and the crossing term, and how they scale with ``d_op`` (the sqrt claim);
* ``d_K^+ / sqrt(d_op)`` across deployments -- a roughly constant value supports the rate.

    python scripts/t3_5_conditional_link.py [--solver noise|prior] [--trials 150]
"""

from __future__ import annotations

import argparse

import numpy as np

from caliber.operators.distance import null_space_projection_distance
from caliber.scoring.score import ScoreWeights
from caliber.theory import crossing_probability as crossing_prob
from caliber.theory import joint_cdf_gap as sup_joint_gap
from caliber.theory import ks_distance
from caliber.theory.harness import (
    HarnessConfig,
    frozen_scales,
    make_mri,
    positive_mask,
    run_trials,
)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="prior", choices=["noise", "prior"])
    ap.add_argument("--trials", type=int, default=150)
    args = ap.parse_args(argv)

    cfg = HarnessConfig(solver=args.solver, center_fraction=0.04)
    w = ScoreWeights(free=0.5, disagree=0.5, residual=0.0)
    cal = make_mri(cfg, "random1d", 2)
    scales = frozen_scales(cal, cfg)
    base = run_trials(cal, cfg, args.trials, with_truth=True)
    ratio_cal = base.err_ratio().numpy()
    tau = float(np.quantile(ratio_cal, 0.6))
    sa = base.scores(w, scales).numpy().ravel()
    la = positive_mask(base, tau).numpy().ravel()
    pi_a = la.mean()
    print(f"solver={args.solver}  trials={args.trials}  tau={tau:.3f}  pi_cal={pi_a:.3f}\n")
    print(
        f"{'dep':>4} {'d_op':>6} {'pi_dep':>7} {'P_flip':>7} {'cross':>7} {'supG':>7} "
        f"{'alg bound':>9} {'d_K^+':>7} {'holds':>6} {'d_K^+/sqrt d':>13} {'P_flip/sqrt d':>14}"
    )
    for acc in (3, 4, 8, 16):
        dep = make_mri(cfg, "random1d", acc)
        ts = run_trials(dep, cfg, args.trials, with_truth=True)
        sb = ts.scores(w, scales).numpy().ravel()
        lb = positive_mask(ts, tau).numpy().ravel()
        d = null_space_projection_distance(cal, dep)
        flip = float((la != lb).mean())
        cross = crossing_prob(sa, sb)
        supg = sup_joint_gap(sa, la, sb, lb)
        alg = (supg + abs(pi_a - lb.mean())) / pi_a
        dk = ks_distance(sa[la], sb[lb])
        print(
            f"{'R' + str(acc):>4} {d:>6.3f} {lb.mean():>7.3f} {flip:>7.3f} {cross:>7.3f} "
            f"{supg:>7.3f} {alg:>9.3f} {dk:>7.3f} {str(dk <= alg + 1e-9):>6} "
            f"{dk / d**0.5:>13.3f} {flip / d**0.5:>14.3f}"
        )


if __name__ == "__main__":
    main()
