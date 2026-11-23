"""T3.6 / T3-A: the six-row table per ordered operator pair, on a score bank.

    python scripts/t3_6_bound_vs_measurement.py <bank-dir> --distances <T1-0 csv> \
        --scale-operator <id> --out t36_table.csv --out-corr t36_corr.csv
    python scripts/t3_6_bound_vs_measurement.py --synthetic ct
    python scripts/t3_6_bound_vs_measurement.py --synthetic mri

For every ordered pair ``(src, dst)`` it reports the rows ``PHASE4_WORK_PLAN.md`` Sec. 5 asks
of T3-A (``caliber.theory.bank_bound`` holds the definitions):

1. the measured gap, ``matched_minus_transfer`` (Track 1's frozen definition) and
   ``nominal_minus_transfer`` beside it, with a patient-level bootstrap that resamples both
   sides and **refits both thresholds inside every replicate**;
2. ``d_K`` between the label-conditional score laws, the check ``gap <= d_K + 1/(n+1)`` and
   the tightness ratio;
3. ``W1`` between the score laws, conditional and unconditional;
4. the coupled upper bounds on ``W1`` (both theorems; a violation is a bug);
5. the operator distance, and Spearman of it against ``d_K`` and ``W1`` across pairs,
   pooled and per family cell (``--out-corr``);
6. score-level TV by histogram at several bin counts.

Pairs are reported **per direction** and never symmetrised (D47).

**Distances.** On a CT bank at 320^2 recomputing ``d_op`` costs minutes per pair, and T1-0
already computed every one of them (~11 CPU hours). ``--distances`` reads that CSV
(``ct_distances.py``'s format), filling the reverse direction by symmetry for
``null_space_projection`` and by antisymmetry for ``null_space_growth``. Without it the
distances are computed from the bank's manifest, which is only sensible for small banks.

**Score scales.** ``--scale-operator src`` (the default) freezes the component scales on each
pair's calibration operator, T3.3's literal setting. ``--scale-operator <id>`` freezes them
on one operator for the whole table, which is how Track 2's T2-A3 cell tables are built:
then the score is one function at every operator, row 5 compares like with like, and the
``cov_dst`` / ``cov_matched`` columns reproduce ``pair_coverage_registry.csv`` exactly.

``--synthetic`` builds a bank of the same schema from the T3.5 harness, so the whole pipeline
can be run and tested without Track 2's banks, which are local to them. Its numbers are about
the harness, not about the project.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from caliber.evaluation import prereg
from caliber.evaluation.bank import add_label_args, label_from_args
from caliber.operators.distance import SIGNED_MEASURES, compute_distance
from caliber.operators.registry import build_operator
from caliber.reportability import COVERAGE, describe, stamp
from caliber.scorebank import read_bank, read_manifest
from caliber.theory.bank_bound import (
    HIST_BINS,
    LEVEL,
    N_BOOT,
    distance_correlations,
    pair_report,
    score_bank,
)

METRIC = "null_space_projection"
SIGNED = "null_space_growth"


def operators_from_manifest(manifest: dict) -> dict:
    ops = {}
    for row in manifest["operators"]:
        spec = {"modality": row["modality"], **row["params"]}
        ops[row["operator_id"]] = build_operator(spec, tuple(row["image_shape"]))
    return ops


def load_distances(path: str | Path) -> dict[str, dict[tuple[str, str], float]]:
    """``{measure: {(a, b): value}}`` from a ``ct_distances.py`` CSV, both orders filled.

    One order per pair is all T1-0 computes: the metric is symmetric and the signed growth is
    exactly antisymmetric on the same seeded probes, so the reverse is filled rather than
    looked for. A value already present in both orders is kept as written.
    """
    out: dict[str, dict[tuple[str, str], float]] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                value = float(row["value"])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
            m, a, b = row["measure"], row["calibration"], row["deployment"]
            table = out.setdefault(m, {})
            table[(a, b)] = value
            table.setdefault((b, a), -value if m in SIGNED_MEASURES else value)
    return out


def synthetic_operators(kind: str, solver: str, size: int = 128):
    """The harness operators and config; cheap (no reconstruction runs)."""
    from caliber.theory.harness import HarnessConfig, make_ct, make_mri

    if kind == "mri":
        cfg = HarnessConfig(solver=solver, center_fraction=0.04, image_shape=(size, size),
                            region_size=size // 4, n_scale_items=min(40, size // 2),
                            n_draws=8 if size >= 128 else 4)
        ops = [make_mri(cfg, "random1d", a) for a in (2, 4, 8, 16)]
    else:
        cfg = HarnessConfig(image_shape=(32, 32), region_size=8, n_scale_items=20, solver=solver)
        ops = [make_ct(cfg, v) for v in (60, 30, 20, 12)]
    return ops, cfg


def synthetic(kind: str, solver: str, n_items: int, size: int = 128):
    from caliber.theory.synthetic_bank import build_bank

    ops, cfg = synthetic_operators(kind, solver, size)
    return build_bank(ops, cfg, n_items=n_items), {o.descriptor().operator_id: o for o in ops}


def short_names(operators) -> dict[str, str]:
    """Operator ids with the dot-tokens every id shares removed, for a readable table."""
    parts = {op: op.split(".") for op in operators}
    width = min(len(p) for p in parts.values())
    common = {i for i in range(width) if len({p[i] for p in parts.values()}) == 1}
    common |= {-i for i in range(1, width + 1)
               if len({p[-i] for p in parts.values()}) == 1}
    out = {}
    for op, p in parts.items():
        keep = [t for i, t in enumerate(p) if i not in common and (i - len(p)) not in common]
        tag = ".".join(keep) or op
        for long, brief in (("sparse_view.v", "sv"), ("limited_angle.v120.a", "la"),
                            ("limited_angle.", "la"), (".a180", "")):
            tag = tag.replace(long, brief)
        out[op] = tag
    return out


def _fmt(v) -> str:
    return "  nan" if not np.isfinite(v) else f"{v:+.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bank", nargs="?", help="score-bank directory")
    ap.add_argument("--synthetic", choices=["mri", "ct"], help="build a harness bank instead")
    ap.add_argument("--solver", default="prior", choices=["noise", "prior"])
    ap.add_argument("--items", type=int, default=60, help="items per operator (synthetic)")
    ap.add_argument("--size", type=int, default=128, help="image side, MRI synthetic")
    ap.add_argument("--synthetic-cache", default=None,
                    help="parquet path: load the harness bank from here if it exists, else "
                         "build it and save it here (the CT harness takes minutes to build)")
    ap.add_argument("--calibration-operator", default=None,
                    help="restrict to pairs with this operator as src (both directions if "
                         "--both-directions)")
    ap.add_argument("--both-directions", action="store_true")
    ap.add_argument("--alpha", type=float, default=prereg.ALPHA)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--level", type=float, default=LEVEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--normalise", default="ambient")
    ap.add_argument("--distances", default=None,
                    help="T1-0's ct_distances.py CSV; without it distances are computed "
                         "from the manifest (minutes per pair at 320^2)")
    ap.add_argument("--scale-operator", default="src",
                    help="'src' (per pair, T3.3) or one operator id for the whole table "
                         "(T2-A3's convention; ct.sparse_view.v180.a180.dfull on CT)")
    ap.add_argument("--cluster", default=None,
                    help="resampling unit (default patient_id if present, else sample_id)")
    ap.add_argument("--out", default=None, help="write the per-pair table here as CSV")
    ap.add_argument("--out-corr", default=None, help="write row 5's correlations here as CSV")
    add_label_args(ap)
    args = ap.parse_args(argv)

    prov: dict = {}
    if args.synthetic:
        cache = Path(args.synthetic_cache) if args.synthetic_cache else None
        if cache is not None and cache.exists():
            df = pd.read_parquet(cache)
            ops = {o.descriptor().operator_id: o
                   for o in synthetic_operators(args.synthetic, args.solver, args.size)[0]}
        else:
            df, ops = synthetic(args.synthetic, args.solver, args.items, args.size)
            if cache is not None:
                df.to_parquet(cache, index=False)
        source = f"synthetic {args.synthetic} harness bank ({args.solver} solver)"
    elif args.bank:
        bank = Path(args.bank)
        manifest = read_manifest(bank)
        prov = manifest.get("provenance") or {}
        df = read_bank(bank)
        ops = None if args.distances else operators_from_manifest(manifest)
        source = str(bank)
    else:
        ap.error("give a bank directory or --synthetic")
    try:
        label = label_from_args(df, args)
    except ValueError as exc:
        print(f"[t3.6] cannot value the label: {exc}")
        return 2
    operators = sorted(df["operator_id"].unique())
    family_of = (df.drop_duplicates("operator_id").set_index("operator_id")["operator_family"]
                 .astype(str).to_dict())
    print(f"[t3.6] {source}: {len(df):,} rows, {len(operators)} operators")
    if not args.synthetic:
        print(f"[t3.6] stamp: {describe(prov)}; these numbers are {stamp(prov, COVERAGE)} "
              "as coverage quantities")
    print(f"[t3.6] label: {label.text}")
    print(f"[t3.6] scales frozen on: {args.scale_operator}; interval: {args.level:.0%} over "
          f"{args.n_boot} patient resamples, both sides, thresholds refitted in each")

    if args.calibration_operator:
        c = args.calibration_operator
        pairs = [(c, o) for o in operators if o != c]
        if args.both_directions:
            pairs += [(o, c) for o in operators if o != c]
    else:
        pairs = list(itertools.permutations(operators, 2))

    dist = load_distances(args.distances) if args.distances else {}
    if args.distances:
        for m in (METRIC, SIGNED):
            missing = [p for p in pairs if p not in dist.get(m, {})]
            if missing:
                print(f"[t3.6] {len(missing)} pair(s) have no {m} in {args.distances} "
                      f"(e.g. {missing[0]}); they carry NaN")

    def distance(a: str, b: str, measure: str) -> float:
        if args.distances:
            return float(dist.get(measure, {}).get((a, b), float("nan")))
        return float(compute_distance(ops[a], ops[b], measure, normalise=args.normalise))

    scored_by: dict[str, pd.DataFrame] = {}

    def scored(src: str) -> pd.DataFrame:
        key = src if args.scale_operator == "src" else args.scale_operator
        if key not in scored_by:
            scored_by[key] = score_bank(df, label, key)
        return scored_by[key]

    rows, t0 = [], time.time()
    for i, (src, dst) in enumerate(pairs, 1):
        rep = pair_report(df, src, dst, label, distance(src, dst, METRIC),
                          distance(src, dst, SIGNED), alpha=args.alpha, n_boot=args.n_boot,
                          seed=args.seed, level=args.level, scored=scored(src),
                          cluster=args.cluster)
        rows.append(rep.as_row())
        if i % 9 == 0 or i == len(pairs):
            print(f"[t3.6] {i}/{len(pairs)} pairs, {time.time() - t0:.0f} s")
    table = pd.DataFrame(rows)
    name = short_names(operators)
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"[t3.6] wrote {args.out}")

    # -- rows 1-2 -------------------------------------------------------------------------
    print("\nrows 1-2: gap (matched / nominal) against d_K^+ + 1/(n+1)")
    print(f"{'src':>10} {'dst':>10} {'d_op':>6} {'growth':>7} {'gap_m':>7} {'[lo,':>7} "
          f"{'hi]':>7} {'gap_n':>7} {'d_K+':>6} {'[lo,':>6} {'hi]':>6} {'tight':>7} "
          f"{'loc':>5} {'w/fix':>5}")
    for r in table.itertuples(index=False):
        print(f"{name[r.src]:>10} {name[r.dst]:>10} {r.d_op:6.3f} {_fmt(r.signed_growth):>7} "
              f"{_fmt(r.gap_matched):>7} {_fmt(r.gap_matched_lo):>7} {_fmt(r.gap_matched_hi):>7} "
              f"{_fmt(r.gap):>7} {r.ks_cond:6.3f} {r.ks_cond_lo:6.3f} {r.ks_cond_hi:6.3f} "
              f"{_fmt(r.tightness):>7} {r.location_ratio:5.2f} {r.interval_width_ratio:5.2f}")

    # -- rows 3, 4, 6 ---------------------------------------------------------------------
    print("\nrows 3-4, 6: W1 and its coupled bounds; histogram TV of the conditional laws")
    tv_cols = [f"tv_h{b}" for b in HIST_BINS]
    print(f"{'src':>10} {'dst':>10} {'W1+':>6} {'W1':>6} {'E|ds|':>6} {'W1+bnd':>7} "
          + " ".join(f"{c:>6}" for c in tv_cols) + f" {'p_flip':>6} {'rate':>6}")
    for r in table.itertuples(index=False):
        tvs = " ".join(f"{getattr(r, c):6.3f}" for c in tv_cols)
        print(f"{name[r.src]:>10} {name[r.dst]:>10} {r.w1_cond:6.3f} {r.w1_uncond:6.3f} "
              f"{r.w1_coupled:6.3f} {r.w1_coupled_cond:7.3f} {tvs} {r.p_flip:6.3f} "
              f"{r.rate_bound:6.2f}")

    ev = table[table["evaluable"]]
    print(f"\n[t3.6] evaluable pairs (both thresholds certifiable, dst has held-out positives): "
          f"{len(ev)} of {len(table)}")
    if len(ev):
        under = ev[ev["gap_matched"] > 0]
        print(f"[t3.6] row 1: refit interval / fixed-threshold interval, nominal gap: median "
              f"{ev['interval_width_ratio'].median():.2f}x (range "
              f"{ev['interval_width_ratio'].min():.2f}-{ev['interval_width_ratio'].max():.2f})")
        print(f"[t3.6] row 2: gap_matched <= d_K^+ + 1/(n+1) at the point: "
              f"{int(ev['holds_slack'].sum())} of {len(ev)}; separated intervals against it: "
              f"{int(ev['significant_violation_matched'].sum())}")
        print(f"[t3.6] row 2: gap_nominal <= d_K^+ at the point: {int(ev['holds_ks'].sum())} "
              f"of {len(ev)}; separated intervals: {int(ev['significant_violation'].sum())}")
        if len(under):
            print(f"[t3.6] row 2: tightness over the {len(under)} under-covering pairs: median "
                  f"{under['tightness'].median():.2f} (range {under['tightness'].min():.2f}-"
                  f"{under['tightness'].max():.2f}); share of d_K reached at the threshold: "
                  f"median {under['location_ratio'].median():.2f}")
        print(f"[t3.6] row 4: W1 <= E|ds| at {int(table['holds_coupled'].sum())} of "
              f"{len(table)}; W1+ <= its coupled bound at "
              f"{int(table['holds_coupled_cond'].sum())} of {len(table)} (theorems)")
        print(f"[t3.6] row 4: coupling ratio W1 / E|ds|: median "
              f"{(table['w1_uncond'] / table['w1_coupled']).median():.2f}; "
              f"W1+ / bound: median "
              f"{(table['w1_cond'] / table['w1_coupled_cond']).median():.2f}")
        tv = table[tv_cols]
        print(f"[t3.6] row 6: histogram TV of the conditional laws spans "
              f"{tv.min(axis=1).median():.3f}-{tv.max(axis=1).median():.3f} (median pair, "
              f"{HIST_BINS[0]} to {HIST_BINS[-1]} bins); d_K^+ median "
              f"{table['ks_cond'].median():.3f}")

    # -- row 5 ----------------------------------------------------------------------------
    finite = table[np.isfinite(table["d_op"])]
    if len(finite) >= 3 and finite["d_op"].nunique() > 1:
        corr = distance_correlations(finite, family_of, n_boot=args.n_boot, level=args.level,
                                     seed=args.seed)
        if args.out_corr:
            corr.to_csv(args.out_corr, index=False)
            print(f"[t3.6] wrote {args.out_corr}")
        print("\nrow 5: Spearman of operator distance against the law distances "
              "(operator-resampled interval)")
        for (x, y), g in corr.groupby(["x", "y"], sort=False):
            cells = "  ".join(f"{s}: {_fmt(r)} (n={n})" for s, r, n in
                              zip(g["scope"], g["rho"], g["n_pairs"], strict=True))
            flag = g["cells_agree_in_sign"].iloc[0]
            agree = ("n/a" if flag is None or pd.isna(flag)
                     else "agree" if bool(flag) else "DISAGREE")
            print(f"  {x:>13} vs {y:<12} {cells}   [cell signs: {agree}]")
    else:
        print("\n[t3.6] row 5 skipped: fewer than three pairs with a finite distance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
