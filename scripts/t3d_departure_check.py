"""T3-D: how a real prior departs from T3.4's sqrt(null_frac) law, by operator and family.

    python scripts/t3d_departure_check.py <bank-dir> --reference <id> --out t3d.csv
    python scripts/t3d_departure_check.py --synthetic ct --solver prior \
        --synthetic-cache runs/t3a/banks/ct_prior.parquet

T3.4 Sec. 2 derived ``s_free ~ sqrt(null_frac)`` for white-noise fill: the null hypothesis a
real prior is tested against (Sec. 6). This measures the departure and splits it exactly
(``caliber.theory.alignment.energy_departure``): over the same regions at both operators,

    R_s - R_nf = C_ref (R_a - R_nf) + (1 - C_ref) (R_o - R_nf),

with ``R_*`` energy ratios against the reference operator, ``a`` the recovered part of the
hidden content and ``o`` the invented part (T3.6 Part II), and ``C_ref`` the reference's
aligned share. ``quotient_*`` are the amplitude forms T3.4/T3.5 quote (1.0 = the law holds):
``quotient_invented`` asks whether the prior's *invention* follows the white-noise law, and
``quotient_truth`` asks whether the truth's own hidden energy does (T3.4 Sec. 1's
energy-weighted null fraction, which the law ignores).

**Null fraction.** From the bank manifest's ``null_space_dim`` over the image size, which
on CT is the dimension-count bound (``102400 - 320 * views`` at 320^2; every limited-angle
operator shares 64,000), or ``--hutchinson N`` probes of ``P_N`` for operators small enough
to apply (the harness). On a real CT bank the bound is what is available, and its
degeneracy across limited-angle arcs is itself part of the departure being measured.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from caliber.scorebank import read_bank, read_manifest
from caliber.theory.alignment import energy_departure


def hutchinson_null_fraction(op, n_probes: int = 16, seed: int = 0) -> float:
    """``tr(P_N) / n`` by Hutchinson probes (T3.5's ``effective_null_fraction``)."""
    gen = torch.Generator().manual_seed(seed)
    num = den = 0.0
    for _ in range(n_probes):
        z = torch.randn(*op.image_shape, generator=gen)
        num += float(torch.linalg.vector_norm(op.project_null(z)) ** 2)
        den += float(torch.linalg.vector_norm(z) ** 2)
    return num / den


def manifest_null_fraction(manifest: dict) -> dict[str, float]:
    out = {}
    for row in manifest.get("operators", ()):
        dim, shape = row.get("null_space_dim"), row.get("image_shape")
        if dim is not None and shape:
            out[str(row["operator_id"])] = float(dim) / float(np.prod(shape))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bank", nargs="?")
    ap.add_argument("--synthetic", choices=["mri", "ct"])
    ap.add_argument("--solver", default="prior", choices=["noise", "prior"])
    ap.add_argument("--items", type=int, default=60)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--synthetic-cache", default=None)
    ap.add_argument("--reference", default=None,
                    help="the operator every ratio is taken against (default: the one with "
                         "the smallest null fraction)")
    ap.add_argument("--hutchinson", type=int, default=0,
                    help="estimate null fractions with this many probes instead of the "
                         "manifest (synthetic banks always do)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.synthetic:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from t3_6_bound_vs_measurement import synthetic, synthetic_operators

        cache = Path(args.synthetic_cache) if args.synthetic_cache else None
        if cache is not None and cache.exists():
            df = pd.read_parquet(cache)
            ops = synthetic_operators(args.synthetic, args.solver, args.size)[0]
        else:
            df, by_id = synthetic(args.synthetic, args.solver, args.items, args.size)
            ops = list(by_id.values())
            if cache is not None:
                df.to_parquet(cache, index=False)
        nf = {o.descriptor().operator_id: hutchinson_null_fraction(o, max(args.hutchinson, 16))
              for o in ops}
        print(f"[t3d] synthetic {args.synthetic} harness bank ({args.solver} solver); null "
              "fractions by Hutchinson")
    elif args.bank:
        df = read_bank(Path(args.bank))
        manifest = read_manifest(Path(args.bank))
        if args.hutchinson:
            from t3_6_bound_vs_measurement import operators_from_manifest

            nf = {k: hutchinson_null_fraction(o, args.hutchinson)
                  for k, o in operators_from_manifest(manifest).items()}
            print(f"[t3d] {args.bank}: null fractions by Hutchinson ({args.hutchinson} probes)")
        else:
            nf = manifest_null_fraction(manifest)
            print(f"[t3d] {args.bank}: null fractions from the manifest's null_space_dim "
                  "(a dimension-count bound on CT)")
    else:
        ap.error("give a bank directory or --synthetic")
    if df["free_content"].isna().all():
        print("[t3d] this bank has no free_content column; the split is not recoverable")
        return 2
    ref = args.reference or min(nf, key=nf.get)
    table = energy_departure(df, ref, nf)
    print(f"[t3d] reference operator: {ref} (null fraction {nf.get(ref, float('nan')):.3f})")
    cols = ["operator_id", "family", "null_fraction", "R_nf", "R_s", "R_a", "R_o", "R_c",
            "aligned_share", "quotient_s", "quotient_invented", "quotient_truth",
            "aligned_contribution", "invented_contribution", "identity_residual"]
    print()
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    worst = float(np.nanmax(np.abs(table["identity_residual"])))
    print(f"\n[t3d] identity R_s = C R_a + (1 - C) R_o: max |residual| {worst:.2e} (exact algebra)")
    for fam, g in table[table["operator_id"] != ref].groupby("family"):
        print(f"[t3d] {fam}: law quotient for s {g['quotient_s'].min():.2f}-"
              f"{g['quotient_s'].max():.2f}, for the invented part "
              f"{g['quotient_invented'].min():.2f}-{g['quotient_invented'].max():.2f}, for the "
              f"truth's hidden energy {g['quotient_truth'].min():.2f}-"
              f"{g['quotient_truth'].max():.2f}; aligned share "
              f"{g['aligned_share'].min():.2f}-{g['aligned_share'].max():.2f}")
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"[t3d] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
