"""Does the range/null decomposition earn its keep on a solver that does not pre-confine it?

D68, promoted from ablation to load-bearing: **this is the one experiment that can give the
score a positive claim.** Posterior dispersion ranks hallucinations within 0.001-0.012 AUROC
of ``s_disagree`` at every operator, and the reason is mechanical rather than statistical --
DDNM replaces the range component structurally, so the posterior spread is *already* confined
to the null space and projecting it there adds nothing. A gradient-steered solver (DPS) leaves
0.817-0.852 of the error in the null space against DDNM's 0.99, so it is the only
configuration in which the decomposition has anything left to do.

Three sections, in the order they have to be read:

1. **The premise.** The null-error share per solver. If DPS's share is not materially below
   DDNM's on these banks, the ablation's whole rationale is absent and sections 2-3 are not
   worth reading.
2. **The comparison**, per solver and per operator: the score against a dispersion-like
   baseline, and both against the anatomy control that has beaten them before.
3. **The verdict**, which is a *difference of excesses* between solvers, and which is written
   so that it can come back negative.

## The share has to be computed at image level, and this is not a detail

``err_free`` and ``err_total`` are per-region RMS values, and ``P_N`` is a **non-local**
operator: projecting and then restricting to a region is not the same as restricting and then
projecting, so energy crosses region boundaries and the orthogonal decomposition does **not**
hold region by region. Measured on the M4Raw bank, ``err_free^2 / err_total^2`` exceeds 1 in
**97 of 384 regions**, reaching 1.035 -- not a bank defect, just the wrong place to take the
ratio.

So the share is taken as an area-weighted energy ratio over each volume, where it is the
global MSE ratio and is exact. The per-region out-of-range count is printed as a diagnostic
rather than hidden, because a *large* excess there would indicate a genuine normalisation
problem of the kind ``identity_residual`` exists to catch.

    python scripts/solver_ablation.py runs/<ddnm_bank> runs/<dps_bank> \
        --calibration-operator mri.random1d.R2.c0.08.s0
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from caliber.evaluation import add_label_args, derive_score_and_label, label_from_args
from caliber.scoring.attribution import (
    AUC_EXCESS_MIN,
    MIN_CLASS_FOR_AUC,
    identity_residual,
    rank_auc,
)
from caliber.scorebank import read_bank

#: Below this gap in null-error share, two solvers do not differ in the way D68 requires and
#: the ablation has no premise. DDNM 0.99 against DPS 0.817-0.852 is a gap of ~0.14-0.18, so
#: 0.05 is well inside what the argument assumed.
MIN_SHARE_GAP = 0.05


def _agnostic(scores, labels) -> float:
    """``max(auc, 1 - auc)`` -- D65's rule, applied to score and control alike."""
    auc = rank_auc(scores, labels)
    return max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan")


def null_error_share(df: pd.DataFrame, cluster_col: str = "sample_id") -> pd.DataFrame:
    """Area-weighted ``err_free^2 / err_total^2`` per volume, where the ratio is exact.

    Returns one row per (operator, volume), plus the count of regions whose per-region ratio
    exceeds 1 -- which is expected and bounded, for the reason in the module docstring.
    """
    for col in ("err_free", "err_total"):
        if col not in df.columns:
            raise KeyError(f"bank has no {col!r} column")
    area = (df["region_area"].to_numpy(dtype=float) if "region_area" in df.columns
            else np.ones(len(df)))
    frame = df.assign(
        __free=area * df["err_free"].to_numpy(dtype=float) ** 2,
        __total=area * df["err_total"].to_numpy(dtype=float) ** 2,
        __over=(df["err_free"].to_numpy(dtype=float)
                > df["err_total"].to_numpy(dtype=float) * (1.0 + 1e-4)),
    )
    grouped = frame.groupby(["operator_id", cluster_col], observed=True).agg(
        free=("__free", "sum"), total=("__total", "sum"),
        n_regions=("__free", "size"), n_over=("__over", "sum"),
    ).reset_index()
    grouped["share"] = grouped["free"] / grouped["total"].clip(lower=1e-12)
    return grouped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("banks", nargs="+", help="one bank per solver arm, or one mixed bank")
    ap.add_argument("--calibration-operator", required=True)
    add_label_args(ap)
    ap.add_argument("--baseline", default="s_disagree",
                    help="dispersion-like reference column already in the bank")
    ap.add_argument("--splits", nargs="*", default=["test"])
    ap.add_argument("--cluster", default="sample_id")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    frames = []
    for path in args.banks:
        df = read_bank(path)
        if "solver_id" not in df.columns:
            print(f"[ablation] {path} has no solver_id column; cannot tell the arms apart",
                  file=sys.stderr)
            return 1
        frames.append(df)
    bank = pd.concat(frames, ignore_index=True)
    solvers = sorted(bank["solver_id"].astype(str).unique())
    if len(solvers) < 2:
        print(f"[ablation] these banks hold one solver ({solvers[0]}); D68's ablation needs "
              "two, so that the decomposition can be compared where the range component is "
              "and is not structurally replaced", file=sys.stderr)
        return 1
    print(f"[ablation] solvers: {', '.join(solvers)}")

    residual = identity_residual(bank["s_free"], bank["free_content"], bank["err_free"])
    print(f"[ablation] identity check over {int(residual['n']):,} regions: median relative "
          f"{residual['median_relative']:.2e}, max {residual['max_relative']:.2e}")

    # -- 1. the premise ---------------------------------------------------------
    print("\n[ablation] 1. NULL-ERROR SHARE (area-weighted per volume; the ratio is only "
          "exact at image level)")
    shares = {}
    rows_share = []
    for sid, part in bank.groupby("solver_id"):
        table = null_error_share(part, cluster_col=args.cluster)
        shares[str(sid)] = float(table["share"].median())
        over = int(table["n_over"].sum())
        total = int(table["n_regions"].sum())
        print(f"  {sid}")
        print(f"    median share {table['share'].median():.4f}  "
              f"range {table['share'].min():.4f}-{table['share'].max():.4f}  "
              f"({len(table)} volume-operator cells)")
        print(f"    per-region ratio above 1 in {over}/{total} regions "
              f"({over / max(total, 1):.1%}) -- expected: P_N is non-local")
        for oid, p in table.groupby("operator_id"):
            rows_share.append({"solver_id": str(sid), "operator_id": str(oid),
                               "median_share": float(p["share"].median()),
                               "n_cells": len(p)})

    ordered = sorted(shares.items(), key=lambda kv: kv[1])
    gap = ordered[-1][1] - ordered[0][1]
    print(f"\n[ablation] lowest share {ordered[0][0]} at {ordered[0][1]:.4f}; highest "
          f"{ordered[-1][0]} at {ordered[-1][1]:.4f}; gap {gap:.4f}")
    if gap < MIN_SHARE_GAP:
        print(f"[ablation] PREMISE ABSENT: the gap is below {MIN_SHARE_GAP}. These solvers do "
              "not differ in how much error they leave in the null space, so there is no "
              "configuration here in which the decomposition has more to do. Sections 2-3 "
              "are reported but D68's rationale does not hold on these banks.",
              file=sys.stderr)
    else:
        print(f"[ablation] premise holds: {ordered[0][0]} leaves materially less of its error "
              "in the null space, so it is where the decomposition can contribute")

    # -- 2. the comparison, per solver and per operator -------------------------
    try:
        label = label_from_args(bank, args)
    except ValueError as exc:
        print(f"[ablation] {exc}", file=sys.stderr)
        return 1
    print(f"\n[ablation] label: {label.text}")
    print("\n[ablation] 2. DETECTION, per solver x operator, direction-agnostic AUC")
    print(f"{'solver':<38}{'operator':<26}{'npos':>6}{'score':>8}{'base':>7}"
          f"{'anat':>7}{'excess':>8}")
    rows = []
    for sid, part in bank.groupby("solver_id"):
        scored = derive_score_and_label(part, args.calibration_operator, **label.kwargs)
        sub = scored[scored["split"].isin(args.splits)]
        for oid, frame in sub.groupby("operator_id"):
            labels = frame["label"].to_numpy().astype(bool)
            n_pos, n_neg = int(labels.sum()), int((~labels).sum())
            row = {"solver_id": str(sid), "operator_id": str(oid), "n_positive": n_pos}
            if min(n_pos, n_neg) < MIN_CLASS_FOR_AUC:
                row["note"] = f"below the {MIN_CLASS_FOR_AUC}-per-class floor (D32)"
                rows.append(row)
                print(f"{str(sid):<38}{str(oid):<26}{n_pos:>6}   {row['note']}")
                continue
            score_auc = _agnostic(frame["score"], labels)
            base_auc = (_agnostic(frame[args.baseline], labels)
                        if args.baseline in frame.columns else float("nan"))
            anat_auc = _agnostic(frame["free_content"], labels)
            row.update({"score_auc": score_auc, "baseline_auc": base_auc,
                        "anatomy_auc": anat_auc,
                        "excess_over_baseline": score_auc - base_auc,
                        "excess_over_anatomy": score_auc - anat_auc, "note": ""})
            rows.append(row)
            print(f"{str(sid):<38}{str(oid):<26}{n_pos:>6}{score_auc:>8.4f}"
                  f"{base_auc:>7.4f}{anat_auc:>7.4f}"
                  f"{score_auc - base_auc:>+8.4f}")

    table = pd.DataFrame(rows)
    usable = table[table["note"] == ""] if "note" in table.columns else table
    if usable.empty:
        print("\n[ablation] no operator has enough of both classes to compare on",
              file=sys.stderr)
        return 1

    # -- 3. the verdict ---------------------------------------------------------
    print("\n[ablation] 3. VERDICT -- does the decomposition's advantage differ by solver?")
    print("[ablation] per operator, never pooled across them (D60, retracted)")
    per_solver = {}
    for sid, part in usable.groupby("solver_id"):
        beats_base = int((part["excess_over_baseline"] >= AUC_EXCESS_MIN).sum())
        beats_anat = int((part["excess_over_anatomy"] >= AUC_EXCESS_MIN).sum())
        per_solver[str(sid)] = (beats_base, beats_anat, len(part))
        print(f"  {sid}: beats the baseline by >= {AUC_EXCESS_MIN} at {beats_base}/"
              f"{len(part)} operators; beats the anatomy control at {beats_anat}/{len(part)}")

    low_solver = ordered[0][0]
    low = per_solver.get(low_solver)
    if low is None:
        print(f"[ablation] {low_solver} has no comparable operator, so D68 is UNDECIDED")
    elif low[0] == 0:
        print(f"[ablation] D68 ANSWERED, NEGATIVE: on {low_solver} -- the solver that leaves "
              f"most room for it -- the score beats the baseline at none of {low[2]} "
              "operators.\n           The decomposition does not earn its keep even where "
              "the mechanism allows it, and\n           the paper should state that as a "
              "finding rather than leave it open.")
    elif low[1] == 0:
        print(f"[ablation] D68 PARTIAL: on {low_solver} the score beats the baseline at "
              f"{low[0]}/{low[2]} operators but never\n           the anatomy control, so "
              "the advantage may be anatomical rather than a detection of invention.")
    else:
        print(f"[ablation] D68 ANSWERED, POSITIVE: on {low_solver} the score beats both the "
              f"baseline ({low[0]}/{low[2]})\n           and the anatomy control "
              f"({low[1]}/{low[2]}). This is the first positive claim available for the "
              "score itself.")
    print("[ablation] Detection numbers remain subject to D67: if the label is a magnitude "
          "one, read\n           label_comparison.py first -- an excess here can be the "
          "identity rather than a detector.")

    if args.out:
        pd.concat([table, pd.DataFrame(rows_share)], ignore_index=True).to_csv(
            args.out, index=False
        )
        print(f"[ablation] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
