"""T3-C: re-check the orthogonal-error approximation on a bank, per operator.

    python scripts/t3c_alignment_check.py <bank-dir> --scale-operator <id> --out t3c.csv
    python scripts/t3c_alignment_check.py --synthetic ct --solver prior
    python scripts/t3c_alignment_check.py --model            # the (kappa, beta) map only

Per operator (``caliber.theory.alignment``): the distribution of ``cos theta``; the exact split
of the reconstruction's hidden content into the part along the truth's (``a = s cos theta``)
and the invented part (``o = s |sin theta|``), with ``err^2 = (a - c)^2 + o^2``; how wrong the
orthogonal-error approximation ``err ~ hypot(s, c)`` is; D67's identity Spearman; and the
threshold ``tau`` with the alignment of the positives it is fitted on.

Across operators: the elasticity of ``s``, ``a``, ``o`` and ``tau`` with respect to the
truth's hidden content ``c`` (the ill-posedness coordinate; defined across families, unlike
null-space dimension). ``o``'s elasticity is the model's ``beta``.

``--model`` prints the sign of ``d tau / d ill-posedness`` over a grid of recovery ``kappa``
and invention elasticity ``beta`` (``threshold_direction_model``). It is the mechanism this
re-check found: with ``beta < 1`` an orthogonal prior's threshold *falls* as the operator
hides more (the positives it is fitted on are hidden truth with little invented content, and
more of them enter) while an aligned prior's *rises* (its marginal positives score like the
recovered truth, which grows). Which regime a bank is in is read off its own columns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from caliber.evaluation.bank import add_label_args, derive_score_and_label, label_from_args
from caliber.reportability import describe
from caliber.scorebank import read_bank, read_manifest
from caliber.theory.alignment import elasticities, operator_alignment, threshold_direction_model

KAPPAS = (0.0, 0.3, 0.6, 0.8, 0.9, 1.0)
BETAS = (1.0, 0.75, 0.5, 0.25, 0.0)


def print_model() -> pd.DataFrame:
    rows = []
    print("threshold direction model: sign of d tau / d ill-posedness (+ rises, - falls)")
    print(f"{'beta':>5} " + " ".join(f"k={k:<4}" for k in KAPPAS))
    for beta in BETAS:
        cells = []
        for k in KAPPAS:
            r = threshold_direction_model(k, beta, invention_scale=max(0.1, 1.0 - 0.9 * k))
            rows.append(r)
            cells.append(f"{r['tau_direction']:+d}/{r['cos_median']:+.2f}")
        print(f"{beta:5.2f} " + " ".join(f"{c:>6}" for c in cells))
    print("(each cell: direction / median cos theta it produces)")
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bank", nargs="?")
    ap.add_argument("--synthetic", choices=["mri", "ct"])
    ap.add_argument("--solver", default="prior", choices=["noise", "prior"])
    ap.add_argument("--items", type=int, default=40)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--scale-operator", default=None,
                    help="freeze the score's scales here (default: the first operator); on "
                         "CT use ct.sparse_view.v180.a180.dfull, as T2-A3 does")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--model", action="store_true", help="print the (kappa, beta) map")
    ap.add_argument("--out", default=None)
    add_label_args(ap)
    args = ap.parse_args(argv)

    if args.model:
        print_model()
        if not (args.bank or args.synthetic):
            return 0
        print()
    if args.synthetic:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from t3_6_bound_vs_measurement import synthetic

        df, _ = synthetic(args.synthetic, args.solver, args.items, args.size)
        print(f"[t3c] synthetic {args.synthetic} harness bank ({args.solver} solver)")
    elif args.bank:
        df = read_bank(Path(args.bank))
        prov = read_manifest(Path(args.bank)).get("provenance") or {}
        print(f"[t3c] {args.bank}: {describe(prov)}")
    else:
        ap.error("give a bank directory, --synthetic or --model")
    if df["free_content"].isna().all():
        print("[t3c] this bank has no free_content column; cos theta is not recoverable")
        return 2
    label = label_from_args(df, args)
    scale = args.scale_operator or sorted(df["operator_id"].unique())[0]
    d = derive_score_and_label(df, scale, **label.kwargs)
    print(f"[t3c] label: {label.text}; scales frozen on {scale}")

    table = operator_alignment(d, alpha=args.alpha)
    fam = (df.drop_duplicates("operator_id").set_index("operator_id")["operator_family"]
           .astype(str))
    table["family"] = table["operator_id"].map(fam)
    show = ["operator_id", "cos_median", "cos_q25", "cos_q75", "undefined", "c_median",
            "aligned_median", "invented_median", "invented_energy_share",
            "orth_rel_err_median", "identity_spearman", "pi", "tau", "marginal_cos_median",
            "marginal_c_over_s"]
    print()
    print(table[[c for c in show if c in table]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))

    el = elasticities(table, groups=table["family"])
    print("\nelasticity with respect to the truth's hidden content c (log-log slope, Spearman)")
    for scope, g in el.groupby("scope", sort=False):
        cells = "  ".join(f"{r.column}: {r.elasticity:+.2f} ({r.spearman:+.2f})"
                          for r in g.itertuples(index=False))
        print(f"  {scope:>14} (n={int(g['n_operators'].iloc[0])}): {cells}")
    pooled = el[el["scope"] == "pooled"].set_index("column")["elasticity"]
    beta = float(pooled.get("invented_median", np.nan))
    tau_el = float(pooled.get("tau", np.nan))
    cos = float(np.nanmedian(table["cos_median"]))
    way = ("undetermined: fewer than three certifiable operators" if not np.isfinite(tau_el)
           else "rises with the hidden truth" if tau_el > 0 else "falls with the hidden truth")
    print(f"\n[t3c] regime: median cos theta {cos:+.2f}, invention elasticity beta {beta:+.2f}, "
          f"tau elasticity {tau_el:+.2f} ({way})")
    print(f"[t3c] orthogonal-error approximation: median |hypot(s,c) - err| / err = "
          f"{np.nanmedian(table['orth_rel_err_median']):.3f}; identity Spearman "
          f"{np.nanmin(table['identity_spearman']):.2f}-"
          f"{np.nanmax(table['identity_spearman']):.2f}")
    if args.out:
        table.to_csv(args.out, index=False)
        el.to_csv(Path(args.out).with_name(Path(args.out).stem + "_elasticity.csv"),
                  index=False)
        print(f"[t3c] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
