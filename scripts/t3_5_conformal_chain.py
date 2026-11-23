"""T3.5 check 3: the corrected Step 1 chain, per direction, label-conditional and clustered.

Closes the three mismatches T3.3 sec. 4b flagged, on the harness (controlled, label and
ground truth available) with the repo's own ``split_threshold`` as the calibrator.

**Chain tested.** For split conformal with independent calibration scores, the coverage of a
new score drawn from ``Q_dep`` satisfies (T3.5 sec. 2, a one-line argument, no Theorem 4)

    coverage_dep >= (1 - alpha) - ks(Q_cal, Q_dep)         [ks <= tv <= 1]

so ``gap = (1 - alpha) - coverage_dep <= ks``. The script reports ``gap`` next to ``ks`` and
``2*tv`` (T3.3's boxed bound) for each ordered pair, i.e. **per direction**.

* **A. unconditional** -- all regions; the law T3.3 bounds.
* **B. label-conditional** -- positives only (relative criterion), the law the harness
  actually controls. Shows how far ``ks`` of the conditional laws is from the unconditional.
* **C. clustered** -- calibrate on whole images (16 dependent regions each); the no-shift
  coverage should be ``1 - alpha`` and is not when few images are used.

    python scripts/t3_5_conformal_chain.py [--solver noise|prior] [--trials 300]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from caliber.calibration.conformal import split_threshold
from caliber.scoring.score import ScoreWeights
from caliber.theory import ks_distance, tv_distance
from caliber.theory.harness import (
    HarnessConfig,
    absolute_epsilon,
    frozen_scales,
    make_mri,
    positive_mask,
    positive_mask_absolute,
    run_trials,
)

ALPHA = 0.1


def coverage_mc(src, dst, src_pos, dst_pos, rng, n_cal=300, reps=1500):
    """Mean coverage on ``dst`` positives of a threshold fitted on ``n_cal`` ``src`` regions."""
    covs = []
    s_idx = np.flatnonzero(src_pos)
    d_scores = dst[dst_pos]
    for _ in range(reps):
        pick = rng.choice(s_idx, size=min(n_cal, s_idx.size), replace=False)
        res = split_threshold(src[pick], np.ones(pick.size, dtype=bool), ALPHA)
        if not np.isfinite(res.threshold):
            covs.append(1.0)
        else:
            covs.append(float((d_scores >= res.threshold).mean()))
    return float(np.mean(covs))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="noise", choices=["noise", "prior"])
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--label", default="absolute", choices=["absolute", "ratio"],
                    help="D-061 absolute label (default) or the superseded relative ratio")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(0)

    cfg = HarnessConfig(solver=args.solver, center_fraction=0.04)
    w = ScoreWeights(free=0.5, disagree=0.5, residual=0.0)
    cal = make_mri(cfg, "random1d", 2)
    scales = frozen_scales(cal, cfg)

    accs = (2, 4, 8, 16)
    sets = {
        a: run_trials(make_mri(cfg, "random1d", a), cfg, args.trials, with_truth=True)
        for a in accs
    }
    scores = {a: sets[a].scores(w, scales).numpy() for a in accs}  # (trials, regions)
    ratios = {a: sets[a].err_ratio().numpy() for a in accs}
    if args.label == "ratio":
        tau = float(np.quantile(ratios[2], 0.6))
        labels = {a: positive_mask(sets[a], tau).numpy() for a in accs}
        text = f"relative ratio, tau={tau:.3f} (0.6 quantile of the R=2 ratio; superseded)"
    else:
        pooled = type(sets[2])(
            torch.cat([sets[a].components for a in accs]),
            [t for a in accs for t in sets[a].truth],
        )
        eps = absolute_epsilon(pooled, 0.75)
        labels = {a: positive_mask_absolute(sets[a], eps).numpy() for a in accs}
        text = f"absolute err_free > {eps:.4f} (0.75 quantile pooled over operators, D-061)"
    print(f"solver={args.solver}  alpha={ALPHA}  label: {text}")
    print("positive rate by R: " + ", ".join(f"R{a}={labels[a].mean():.2f}" for a in accs))

    flat = {a: scores[a].ravel() for a in accs}
    lab = {a: labels[a].ravel() for a in accs}

    def pair_rows(title, use_labels):
        print(f"\n== {title} ==")
        print(f"{'cal->dep':>10} {'gap':>8} {'ks':>7} {'2*tv':>7} {'gap<=ks':>8} {'gap<=2tv':>9}")
        for a in accs[1:]:
            for src, dst in ((2, a), (a, 2)):
                sp = lab[src] if use_labels else np.ones_like(lab[src], dtype=bool)
                dp = lab[dst] if use_labels else np.ones_like(lab[dst], dtype=bool)
                cov = coverage_mc(flat[src], flat[dst], sp, dp, rng)
                gap = (1 - ALPHA) - cov
                ks = ks_distance(flat[src][sp], flat[dst][dp])
                tv = tv_distance(flat[src][sp], flat[dst][dp])
                name = f"R{src}->R{dst}"
                print(
                    f"{name:>10} {gap:>+8.3f} {ks:>7.3f} {2 * tv:>7.3f} "
                    f"{str(gap <= ks + 0.01):>8} {str(gap <= 2 * tv + 0.01):>9}"
                )

    pair_rows("A. unconditional (all regions) -- the law T3.3 bounds", use_labels=False)
    pair_rows("B. label-conditional (hallucinated regions) -- the risk the harness controls",
              use_labels=True)

    print("\n== B'. unconditional vs conditional ks, same pair ==")
    print(f"{'pair':>8} {'ks_uncond':>10} {'ks_cond':>9}")
    for a in accs[1:]:
        ku = ks_distance(flat[2], flat[a])
        kc = ks_distance(flat[2][lab[2]], flat[a][lab[a]])
        print(f"{'R2/R' + str(a):>8} {ku:>10.3f} {kc:>9.3f}")

    print("\n== C. clustering: no-shift coverage when calibrating on m whole images ==")
    print("   (16 dependent regions per image; test region from a fresh image; target "
          f"{1 - ALPHA:.2f})")
    print(f"{'m images':>9} {'mean cov':>9} {'sd':>7} {'5% quantile':>12}")
    s = scores[2]
    n_img = s.shape[0]
    for m in (2, 5, 10, 25, 100):
        if m + 50 > n_img:  # not enough images for disjoint calibration and test sets
            continue
        covs = []
        for _ in range(1500):
            perm = rng.permutation(n_img)
            cal_i, test_i = perm[:m], perm[m : m + 50]
            cal_scores = s[cal_i].ravel()
            res = split_threshold(cal_scores, np.ones(cal_scores.size, dtype=bool), ALPHA)
            covs.append(float((s[test_i].ravel() >= res.threshold).mean()))
        covs = np.array(covs)
        print(f"{m:>9} {covs.mean():>9.3f} {covs.std():>7.3f} {np.quantile(covs, 0.05):>12.3f}")


if __name__ == "__main__":
    main()
