"""T3.5 check 5: the corrected Step 1 chain on a real-prior score bank, per direction.

Same chain as ``t3_5_conformal_chain.py`` but on cached bank rows (default: the committed
``banks/w02_solver_smoke`` -- DDNM + ``ddpm-celebahq-256`` on M4Raw, R=4 and R=8, 3 items), so
the score laws come from a prior that actually decided what to put in the null space.

For each ordered pair (calibrate at ``src``, deploy at ``dst``) it fits the repo's
``split_threshold`` on a random half of ``src``'s hallucinated regions, evaluates the miss
rate on all of ``dst``'s, and reports ``gap = (1 - alpha) - coverage`` next to
``ks``, ``2*tv`` of the label-conditional score laws and, for contrast, ``ks`` of the
unconditional laws. ``evaluation.metrics.bound_agreement`` summarises the comparison.
**Not a result** (face prior, 3 items, one split side): it is the first measurement of how
tight the Step 1 chain is on real scores, with a cluster bootstrap on ``ks`` to say how
little 3 items support.

    python scripts/t3_5_bank_check.py [banks/w02_solver_smoke]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from caliber.calibration.conformal import split_threshold
from caliber.evaluation.bank import derive_score_and_label
from caliber.evaluation.metrics import bound_agreement
from caliber.scorebank import read_bank
from caliber.scoring.score import ScoreWeights
from caliber.theory import ks_distance, tv_distance

ALPHA = 0.1


def pair(df, src, dst, rng, w, n_rep=400):
    d = derive_score_and_label(df, calibration_operator=src, weights=w)
    a, b = d[d.operator_id == src], d[d.operator_id == dst]
    pa, pb = a[a.label], b[b.label]
    gaps = []
    for _ in range(n_rep):
        pick = rng.choice(len(pa), size=len(pa) // 2, replace=False)
        res = split_threshold(pa.score.to_numpy()[pick], np.ones(pick.size, dtype=bool), ALPHA)
        cov = 1.0 if not np.isfinite(res.threshold) else float((pb.score >= res.threshold).mean())
        gaps.append((1 - ALPHA) - cov)
    ks_c = ks_distance(pa.score, pb.score)
    tv_c = tv_distance(pa.score, pb.score)
    ks_u = ks_distance(a.score, b.score)
    # cluster bootstrap of the conditional ks over items (the independence unit)
    items = sorted(set(a.sample_id))
    boots = []
    for _ in range(300):
        draw = rng.choice(items, size=len(items), replace=True)
        sa = pd.concat([pa[pa.sample_id == i] for i in draw])
        sb = pd.concat([pb[pb.sample_id == i] for i in draw])
        if len(sa) > 1 and len(sb) > 1:
            boots.append(ks_distance(sa.score, sb.score))
    lo, hi = np.quantile(boots, [0.05, 0.95])
    return dict(gap=float(np.mean(gaps)), ks_c=ks_c, tv2=2 * tv_c, ks_u=ks_u, lo=lo, hi=hi,
                pos_a=len(pa), pos_b=len(pb), rate_a=float(a.label.mean()),
                rate_b=float(b.label.mean()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("bank", nargs="?", default="banks/w02_solver_smoke")
    args = ap.parse_args(argv)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    df = read_bank(Path(args.bank))
    ops = sorted(df.operator_id.unique())
    w = ScoreWeights(free=0.5, disagree=0.5, residual=0.0)
    print(f"bank {args.bank}: {len(df)} rows, operators {ops}, items {df.sample_id.nunique()}")
    print(f"{'src->dst':>9} {'gap':>7} {'ks_cond':>8} {'[5%,95%]':>14} {'2*tv_c':>7} "
          f"{'ks_unc':>7} {'pos rate':>10}")
    rows = []
    for src in ops:
        for dst in ops:
            if src == dst:
                continue
            r = pair(df, src, dst, rng, w)
            rows.append(r)
            name = f"{src.split('.')[2]}->{dst.split('.')[2]}"
            print(f"{name:>9} {r['gap']:>+7.3f} {r['ks_c']:>8.3f} "
                  f"[{r['lo']:.2f},{r['hi']:.2f}]".ljust(42)
                  + f"{r['tv2']:>7.3f} {r['ks_u']:>7.3f} {r['rate_a']:.2f}->{r['rate_b']:.2f}")
    gaps = pd.Series([r["gap"] for r in rows])
    print()
    for label, key in (("ks_cond", "ks_c"), ("2*tv_c", "tv2"), ("ks_unc", "ks_u")):
        bound = pd.Series([r[key] for r in rows])
        print(f"bound_agreement vs {label:8s}:", bound_agreement(gaps, bound))


if __name__ == "__main__":
    main()
