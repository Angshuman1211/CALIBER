"""T3.5 check 4: a CT-like operator, small enough to run anywhere.

T3.2 sec. 6 and T3.3 sec. 6 both ask whether the MRI-derived picture survives an operator whose
range is *not* a coordinate mask. This runs the same harness on the registry's
``CTProjectionOperator`` (parallel beam, 32x32 phantoms, so CG-based ``project_null`` is cheap)
and reports, for a sparse-view sweep and a limited-angle sweep:

* ``d_op`` (D19, Hutchinson estimate -- no closed form exists for CT) and signed growth (D47);
* the **effective** null fraction ``E||P_N z||^2 / E||z||^2``. CT's ``null_space_dim`` is the
  dimension-count bound ``max(0, ambient - measurements)`` and is 0 whenever views x detectors
  exceeds the pixel count, so it is useless as the ill-posedness coordinate (D-007);
* the secant ratios of the free and 2-component scores, and ``s_free / sqrt(nf_eff)`` normalised
  to the calibration operator (T3.4 sec. 2's law, on a non-mask operator).

    python scripts/t3_5_ct_check.py [--trials 40]
"""

from __future__ import annotations

import argparse

import torch

from caliber.operators.distance import null_space_growth, null_space_projection_distance
from caliber.scoring.score import ScoreWeights
from caliber.theory.harness import HarnessConfig, frozen_scales, make_ct, run_trials


def effective_null_fraction(op, n_probes: int = 16, seed: int = 0) -> float:
    gen = torch.Generator().manual_seed(seed)
    num = den = 0.0
    for _ in range(n_probes):
        z = torch.randn(*op.image_shape, generator=gen)
        num += float(torch.linalg.vector_norm(op.project_null(z)) ** 2)
        den += float(torch.linalg.vector_norm(z) ** 2)
    return num / den


def run(cfg, cal, deps, n_trials, label):
    scales = frozen_scales(cal, cfg)
    w2 = ScoreWeights(free=0.5, disagree=0.5, residual=0.0)
    w_free = ScoreWeights.only("free")
    base = run_trials(cal, cfg, n_trials)
    b2, bf = base.scores(w2, scales), base.scores(w_free, scales)
    nf_cal = effective_null_fraction(cal)
    free_cal = base.components[:, 0].mean().item()
    print(f"\n== {label} (calibration {cal.operator_id}, nf_eff={nf_cal:.3f}) ==")
    print(
        f"{'operator':>26} {'nf_eff':>7} {'d_op':>6} {'growth':>7} {'2comp':>7} {'free':>7} "
        f"{'law':>6}"
    )
    for dep in deps:
        d = null_space_projection_distance(cal, dep, n_probes=48)
        g = null_space_growth(cal, dep, n_probes=48)
        nf = effective_null_fraction(dep)
        ts = run_trials(dep, cfg, n_trials)
        r2 = (ts.scores(w2, scales) - b2).abs().mean().item() / max(d, 1e-9)
        rf = (ts.scores(w_free, scales) - bf).abs().mean().item() / max(d, 1e-9)
        law = (ts.components[:, 0].mean().item() / nf**0.5) / (free_cal / nf_cal**0.5)
        print(
            f"{dep.operator_id:>26} {nf:>7.3f} {d:>6.3f} {g:>+7.3f} {r2:>7.3f} {rf:>7.3f} "
            f"{law:>6.3f}"
        )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args(argv)
    cfg = HarnessConfig(image_shape=(32, 32), region_size=8, n_scale_items=20)

    cal = make_ct(cfg, 60)
    run(cfg, cal, [make_ct(cfg, v) for v in (45, 30, 20, 12, 8)], args.trials, "sparse view")
    cal = make_ct(cfg, 60, 180.0)
    run(
        cfg,
        cal,
        [make_ct(cfg, 60, a) for a in (150.0, 120.0, 90.0, 60.0)],
        args.trials,
        "limited angle (60 views)",
    )


if __name__ == "__main__":
    main()
