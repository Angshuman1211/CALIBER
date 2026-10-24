"""T3.6: measured coverage gap against the T3.3 bound, per ordered operator pair, on a bank.

    python scripts/t3_6_bound_vs_measurement.py <bank-dir> [--calibration-operator ID]
    python scripts/t3_6_bound_vs_measurement.py --synthetic mri
    python scripts/t3_6_bound_vs_measurement.py --synthetic ct

For every ordered pair ``(src, dst)`` it prints the measured under-coverage gap with a
patient-level 90% bootstrap interval, the Kolmogorov distance of the label-conditional score
laws (the Step 1 bound, T3.3 sec. 2b), the label-flip / crossing decomposition behind the
sqrt(d_op) rate (sec. 3b), the constants of that rate, and whether each inequality held.
Pairs are reported **per direction** and never symmetrised (D47): ``gap <= d_K^+`` is a
statement about under-coverage, and the over-coverage direction (a negative gap, flag area
-> 1) is an efficiency failure the bound does not describe.

The label is the project's shared one (``--label``, D-061's absolute label by default) and the
bank must carry a ``design`` side for it (D-051); the score's scales are frozen on ``src`` (D27).
Distances come from the operators in the bank's manifest (D19 ``null_space_projection`` and
D47's signed ``null_space_growth``), so no dataset and no GPU is needed.

``--synthetic`` builds a bank of the same schema from the T3.5 harness, so the whole pipeline
can be run and tested without Track 2's banks, which are local to them. Its numbers are about
the harness, not about the project.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from caliber.evaluation.bank import add_label_args, label_from_args
from caliber.evaluation.metrics import bound_agreement
from caliber.operators.distance import compute_distance
from caliber.operators.registry import build_operator
from caliber.scorebank import read_bank, read_manifest
from caliber.scoring.score import ScoreWeights
from caliber.theory.bank_bound import pair_report


def operators_from_manifest(manifest: dict) -> dict:
    ops = {}
    for row in manifest["operators"]:
        spec = {"modality": row["modality"], **row["params"]}
        ops[row["operator_id"]] = build_operator(spec, tuple(row["image_shape"]))
    return ops


def synthetic(kind: str, solver: str, n_items: int, size: int = 128):
    from caliber.theory.harness import HarnessConfig, make_ct, make_mri
    from caliber.theory.synthetic_bank import build_bank

    if kind == "mri":
        cfg = HarnessConfig(solver=solver, center_fraction=0.04, image_shape=(size, size),
                            region_size=size // 4, n_scale_items=min(40, size // 2),
                            n_draws=8 if size >= 128 else 4)
        ops = [make_mri(cfg, "random1d", a) for a in (2, 4, 8, 16)]
    else:
        cfg = HarnessConfig(image_shape=(32, 32), region_size=8, n_scale_items=20, solver=solver)
        ops = [make_ct(cfg, v) for v in (60, 30, 20, 12)]
    return build_bank(ops, cfg, n_items=n_items), {o.descriptor().operator_id: o for o in ops}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bank", nargs="?", help="score-bank directory")
    ap.add_argument("--synthetic", choices=["mri", "ct"], help="build a harness bank instead")
    ap.add_argument("--solver", default="prior", choices=["noise", "prior"])
    ap.add_argument("--items", type=int, default=60, help="items per operator (synthetic)")
    ap.add_argument("--size", type=int, default=128, help="image side, MRI synthetic")
    ap.add_argument("--calibration-operator", default=None,
                    help="restrict to pairs with this operator as src (both directions if "
                         "--both-directions)")
    ap.add_argument("--both-directions", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--normalise", default="ambient")
    ap.add_argument("--out", default=None, help="write the per-pair table here as CSV")
    add_label_args(ap)
    args = ap.parse_args(argv)

    if args.synthetic:
        df, ops = synthetic(args.synthetic, args.solver, args.items, args.size)
        source = f"synthetic {args.synthetic} harness bank ({args.solver} solver)"
    elif args.bank:
        bank = Path(args.bank)
        df, ops = read_bank(bank), operators_from_manifest(read_manifest(bank))
        prov = (read_manifest(bank).get("provenance") or {})
        source = f"{bank} (reportable={prov.get('reportable', 'unknown')})"
    else:
        ap.error("give a bank directory or --synthetic")
    try:
        label = label_from_args(df, args)
    except ValueError as exc:
        print(f"[t3.6] cannot value the label: {exc}")
        return 2
    operators = sorted(df["operator_id"].unique())
    print(f"[t3.6] {source}: {len(df):,} rows, {len(operators)} operators")
    print(f"[t3.6] label: {label.text}")

    if args.calibration_operator:
        c = args.calibration_operator
        pairs = [(c, o) for o in operators if o != c]
        if args.both_directions:
            pairs += [(o, c) for o in operators if o != c]
    else:
        pairs = list(itertools.permutations(operators, 2))

    rows = []
    for src, dst in pairs:
        d = float(compute_distance(ops[src], ops[dst], "null_space_projection",
                                   normalise=args.normalise))
        g = float(compute_distance(ops[src], ops[dst], "null_space_growth",
                                   normalise=args.normalise))
        rep = pair_report(df, src, dst, label, d, g, alpha=args.alpha,
                          weights=ScoreWeights(), n_boot=args.n_boot)
        rows.append(rep.as_row())
    table = pd.DataFrame(rows)

    show = ["src", "dst", "d_op", "signed_growth", "gap", "gap_lo", "gap_hi", "ks_cond",
            "ks_cond_lo", "ks_cond_hi", "ks_uncond", "p_flip", "algebra_bound", "rate_bound",
            "pi_src", "holds_ks", "holds_rate"]
    short = table[show].copy()
    short["src"] = short["src"].str.replace(r"^(mri|ct)\.", "", regex=True)
    short["dst"] = short["dst"].str.replace(r"^(mri|ct)\.", "", regex=True)
    print()
    print(short.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    under = table[table["gap"] > 0]
    print(f"\n[t3.6] under-coverage pairs (gap > 0): {len(under)} of {len(table)}")
    for name, col in (("d_K^+ (label-conditional)", "ks_cond"), ("d_K (unconditional)",
                      "ks_uncond"), ("sqrt(d_op) rate bound", "rate_bound")):
        print(f"[t3.6] gap vs {name:28s} {bound_agreement(table['gap'], table[col])}")
    nonneg = table[table["signed_growth"].notna()]
    if len(nonneg) > 2 and nonneg["signed_growth"].nunique() > 1 and nonneg["gap"].nunique() > 1:
        rho = nonneg["gap"].rank().corr(nonneg["signed_growth"].rank())
        print(f"[t3.6] Spearman(gap, signed growth) over all pairs = {rho:+.3f}")
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"[t3.6] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
