"""How many patients does a coverage number need? — Week 4 grid sizing (P-23).

The Week 4 card says to expand the score bank across the MRI grid. The tempting reading
is "more operators", and P-23 is the reason that is the wrong instinct: with 12 test
volumes, a per-operator coverage estimate carried a 95% interval of [0.46, 0.92]. Spending
the patient budget across more operators buys more numbers that cannot carry a claim.

Two constraints decide the grid, and only one of them is compute:

* **Precision** — how many patients per operator a target interval needs, measured from a
  real bank rather than assumed.
* **A hard ceiling** — fastMRI ``singlecoil_val`` has **96 patients**, so the test side
  holds roughly 40. No amount of CPU time moves that. Splitting it across *k* operators
  divides it.

This script reports both, for a given bank, so the choice is made with numbers. It decides
nothing: how to spend the budget is the protocol's call (T1.4, Track 1).

    python scripts/sizing.py runs/<run_id> --calibration-operator mri.random1d.R2.c0.08.s0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from caliber.calibration import apply_threshold, split_conformal_threshold
from caliber.evaluation import (
    cluster_bootstrap_coverage,
    clusters_for_precision,
    derive_score_and_label,
)
from caliber.scorebank import read_bank, read_manifest

#: fastMRI singlecoil_val, measured: 199 volumes from 96 distinct patients (D-027).
FASTMRI_PATIENTS = 96
#: Interval half-widths worth asking about. 0.10 distinguishes "restored" from "failed";
#: 0.05 is what a published coverage claim would want.
TARGETS = (0.10, 0.05, 0.02)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Coverage precision vs patients (P-23)")
    ap.add_argument("bank")
    ap.add_argument("--calibration-operator", default=None)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--tau", type=float, default=1.75, help="Track 1's ratified value")
    ap.add_argument("--cluster", default="sample_id",
                    help="unit of exchangeability; patient_id where a bank carries it")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    df = read_bank(args.bank)
    provenance = (read_manifest(args.bank).get("provenance") or {})
    calib_op = args.calibration_operator or sorted(df["operator_id"].unique())[0]
    df = derive_score_and_label(df, calib_op, threshold=args.tau)

    print(f"[sizing] bank             {args.bank}")
    print(f"[sizing] reportable       {provenance.get('reportable')}")
    print(f"[sizing] tau              {args.tau}  (Track 1, D-037/D-038)")
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

    rows = []
    for oid, part in evaluate.groupby("operator_id", observed=True):
        own = calib_pool[calib_pool["operator_id"] == oid]
        tau = split_conformal_threshold(own["score"], own["label"], alpha=args.alpha)
        flagged = apply_threshold(part, tau, "score")
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
    budget = FASTMRI_PATIENTS * 0.42          # the test side, at this project's split
    needs = {}
    for target in TARGETS:
        n = clusters_for_precision(worst["se"], worst["n_clusters"], target)
        needs[target] = n
        print(f"{target:>18.2f}{n:>18}{n / budget:>31.0%}")

    print("\nwhat the ceiling implies for the grid:")
    print(f"{'operators':>12}{'test patients each':>22}{'achievable half-width':>24}")
    print("-" * 58)
    for k in (1, 2, 3, 6, 12):
        per_op = budget / k
        # invert the same 1/sqrt(n) scaling
        half = 1.96 * worst["se"] * (worst["n_clusters"] / per_op) ** 0.5
        print(f"{k:>12}{per_op:>22.0f}{half:>24.3f}")

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
