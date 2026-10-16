"""How many patients does a coverage number need? — Week 4 grid sizing (P-23).

The Week 4 card says to expand the score bank across the MRI grid. The tempting reading
is "more operators", and P-23 is the reason that is the wrong instinct: with 12 test
volumes, a per-operator coverage estimate carried a 95% interval of [0.46, 0.92]. Spending
the patient budget across more operators buys more numbers that cannot carry a claim.

Two constraints decide the grid, and only one of them is compute:

* **Precision** — how many patients per operator a target interval needs, measured from a
  real bank rather than assumed.
* **A hard ceiling** — fastMRI ``singlecoil_val`` has **96 patients**, so the test side
  holds roughly 40. No amount of CPU time moves that.

**Two corrections from T1.4 (D45), and the reason this script no longer says the grid is
hopeless.** Its first version divided the patient budget by the number of operators and
concluded that a six-operator grid could only reach ±0.305.

1. **The design is fully crossed, so the budget is not divided.** Retrospective
   undersampling applies every mask to the *same* volume (Sec. 13), so every operator sees
   every patient — verified on both banks in the repository, and now checked here rather
   than assumed (`is_crossed`). Each operator gets all ~40 test patients whatever the grid
   size, so the per-operator half-width stays where one operator puts it.
2. **The headline estimand is the coverage *gap*, and it is paired.** The same patient
   appears at both operators, so the difference is a within-patient contrast and the
   patient's own difficulty — the dominant variance term — cancels. Sizing the study
   against a single operator's SE asks for a precision the claim never needed. The paired
   row below is the one the protocol is sized against.

This script reports all of it, for a given bank, so the choice is made with numbers. It
decides nothing: how to spend the budget is the protocol's call (T1.4, Track 1).

    python scripts/sizing.py runs/<run_id> --calibration-operator mri.random1d.R2.c0.08.s0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from caliber.calibration import apply_threshold, split_threshold
from caliber.calibration.conformal import min_positives
from caliber.evaluation import (
    add_label_args,
    cluster_bootstrap_coverage,
    clusters_for_paired_precision,
    clusters_for_precision,
    derive_score_and_label,
    is_crossed,
    label_from_args,
    paired_gap_curve,
)
from caliber.scorebank import read_bank, read_manifest

#: fastMRI singlecoil_val, measured: 199 volumes from 96 distinct patients (D-027).
FASTMRI_PATIENTS = 96
#: How many of those land on the test side at the default ``calib_fraction`` -- counted
#: on the real patient list, not assumed. ``test_split_golden.py`` pins the calibration
#: side at 52, so the test side is 44. The earlier ``96 * 0.42 = 40`` here was a guess,
#: and it under-reported the achievable half-width by about 6%.
FASTMRI_TEST_PATIENTS = 44
#: Interval half-widths worth asking about. 0.10 distinguishes "restored" from "failed";
#: 0.05 is what a published coverage claim would want.
TARGETS = (0.10, 0.05, 0.02)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Coverage precision vs patients (P-23)")
    ap.add_argument("bank")
    ap.add_argument("--calibration-operator", default=None)
    ap.add_argument("--alpha", type=float, default=0.10)
    add_label_args(ap)
    ap.add_argument("--cluster", default="sample_id",
                    help="unit of exchangeability; patient_id where a bank carries it")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    df = read_bank(args.bank)
    provenance = (read_manifest(args.bank).get("provenance") or {})
    calib_op = args.calibration_operator or sorted(df["operator_id"].unique())[0]
    try:
        label = label_from_args(df, args)
    except ValueError as exc:
        print(f"[sizing] {exc}", file=sys.stderr)
        return 1
    df = derive_score_and_label(df, calib_op, **label.kwargs)

    print(f"[sizing] bank             {args.bank}")
    print(f"[sizing] reportable       {provenance.get('reportable')}")
    print(f"[sizing] label            {label.text}")
    print(f"[sizing] cluster unit     {args.cluster}")
    if args.cluster == "sample_id":
        print(
            "[sizing] NOTE             clustering by volume, not patient. fastMRI has "
            "~2 volumes per patient (D-027), so the true intervals are wider than these.",
            file=sys.stderr,
        )

    has_split = "split" in df.columns and df["split"].nunique() > 1
    calib_pool = df[df["split"] == "calib"] if has_split else df
    evaluate = df[df["split"] == "test"] if has_split else df
    if not has_split:
        print("[sizing] WARNING          single-split bank; thresholds are in-sample "
              "(P-23)", file=sys.stderr)

    crossing = is_crossed(evaluate, cluster_col=args.cluster)
    print(f"[sizing] design           {'CROSSED' if crossing['crossed'] else 'NOT crossed'} "
          f"-- {crossing['reason']} ({crossing['n_shared']} shared clusters)")

    rows = []
    flagged_parts = []
    for oid, part in evaluate.groupby("operator_id", observed=True):
        own = calib_pool[calib_pool["operator_id"] == oid]
        result = split_threshold(own["score"], own["label"], alpha=args.alpha)
        if not result.certifiable:
            print(
                f"[sizing] NOT CERTIFIABLE  {oid} has {result.n_positives} positives, "
                f"under the {min_positives(args.alpha)} needed at alpha={args.alpha}; "
                "it flags everything and its coverage is not a guarantee (D-026)",
                file=sys.stderr,
            )
        flagged = apply_threshold(part, result.threshold, "score")
        flagged_parts.append(flagged)
        stats = cluster_bootstrap_coverage(
            flagged["flag"], flagged["label"], flagged[args.cluster]
        )
        rows.append({"operator_id": oid, **stats})

    print(f"\n{'operator':<30}{'coverage':>9}{'95% CI':>18}{'SE':>8}{'clusters':>10}")
    print("-" * 75)
    for r in rows:
        ci = f"[{r['lo']:.3f}, {r['hi']:.3f}]"
        print(f"{r['operator_id']:<30}{r['coverage']:>9.3f}{ci:>18}"
              f"{r['se']:>8.3f}{r['n_clusters']:>10}")

    usable = [r for r in rows if r["se"] > 0]
    if not usable:
        print("\n[sizing] no operator has a usable SE; nothing to extrapolate from",
              file=sys.stderr)
        return 1
    worst = max(usable, key=lambda r: r["se"])
    print(f"\nsized on the widest operator ({worst['operator_id']}, SE {worst['se']:.3f} "
          f"at {worst['n_clusters']} clusters)")
    header = (f"{'target half-width':>18}{'clusters needed':>18}"
              f"{'as % of test patients':>26}")
    print("\n" + header)
    print("-" * len(header))
    budget = FASTMRI_TEST_PATIENTS            # the test side, at this project's split
    needs = {}
    for target in TARGETS:
        n = clusters_for_precision(worst["se"], worst["n_clusters"], target)
        needs[target] = n
        print(f"{target:>18.2f}{n:>18}{n / budget:>31.0%}")

    print("\nwhat the ceiling implies for the grid:")
    print(f"{'operators':>12}{'test patients each':>22}{'achievable half-width':>24}")
    print("-" * 58)
    for k in (1, 2, 3, 6, 12):
        # In a crossed design every operator sees every test patient, so the budget is
        # NOT divided by k (D45). The earlier version of this table divided it and
        # concluded a six-operator grid was hopeless; that was the arithmetic of a
        # split design, not this one. Where the design is genuinely not crossed, the
        # division is right again -- hence the branch rather than a constant.
        per_op = budget if crossing["crossed"] else budget / k
        half = 1.96 * worst["se"] * (worst["n_clusters"] / per_op) ** 0.5
        print(f"{k:>12}{per_op:>22.0f}{half:>24.3f}")
    if crossing["crossed"]:
        print("       (crossed design: every operator sees every test patient, so adding"
              "\n        operators costs compute, not precision)")

    # -- the estimand the protocol is actually sized against (D45) ----------------
    paired = paired_gap_curve(
        pd.concat(flagged_parts, ignore_index=True), calib_op, cluster_col=args.cluster
    )
    deployments = paired[paired["deployment_operator"] != calib_op]
    print(f"\npaired coverage gap against {calib_op}:")
    print(paired.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if len(deployments):
        worst_gap = deployments.loc[deployments["se_paired"].idxmax()]
        print(
            f"\nthe headline claim is sized on the PAIRED se, not the per-operator one:\n"
            f"{'target half-width':>18}{'clusters needed (paired)':>26}"
            f"{'vs per-operator':>18}"
        )
        for target in TARGETS:
            n_paired = clusters_for_paired_precision(
                float(worst_gap["se_paired"]), int(worst_gap["n_clusters"]), target
            )
            print(f"{target:>18.2f}{n_paired:>26}{needs[target]:>18}")
        print(
            f"\npairing is {worst_gap['narrower_by']:.2f}x narrower than differencing two "
            "independent estimates here.\n"
            "That factor is a property of how much variance is between patients rather "
            "than within them,\nso it is reported per bank and never assumed."
        )

    print(
        "\nRead this as an order of magnitude, not a power calculation: the SE is "
        f"extrapolated as 1/sqrt(n) from {worst['n_clusters']} clusters, and it assumes "
        "the patients added look like the ones measured."
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"bank": args.bank, "tau": args.tau, "cluster": args.cluster,
             "per_operator": rows, "clusters_needed": needs,
             "patient_ceiling": FASTMRI_PATIENTS}, indent=2), encoding="utf-8")
        print(f"\n[sizing] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
