"""CPU-only analysis of a score bank — the other half of the Sec. 17 strategy.

Demonstrates the two baselines the project is built around (Week 3 card) and
the headline relationship (Week 4):

  naive-transferred   calibrate on one operator, deploy on all others.
                      This is the failure the project exists to expose.
  matched-per-operator calibrate separately on every operator. The upper-bound
                      reference: what you would get if the acquisition never changed.
  group-conditional   one threshold per acquisition regime — the guaranteed backbone.

Reads a bank produced by ``caliber``; touches no GPU and no dataset.

    python scripts/analyse_bank.py runs/<run_id> --calibration-operator <operator_id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from caliber.calibration import apply_threshold, mondrian_thresholds, split_conformal_threshold
from caliber.evaluation import coverage_curve, evaluate_groups, monotonicity
from caliber.operators.distance import compute_distance
from caliber.operators.registry import build_operator
from caliber.scorebank import read_bank, read_manifest
from caliber.scoring.labels import RegionTruth, label_hallucinated


def rebuild_operators(manifest: dict) -> dict:
    """Reconstruct operator objects from the manifest, so distances need no data."""
    image_shape = tuple(manifest.get("operators", [{}])[0].get("image_shape", (128, 128)))
    ops = {}
    for row in manifest["operators"]:
        spec = {"modality": row["modality"], **row["params"]}
        ops[row["operator_id"]] = build_operator(spec, tuple(row["image_shape"]) or image_shape)
    return ops


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyse a CALIBER score bank (CPU only)")
    ap.add_argument("bank", help="score-bank directory produced by the runner")
    ap.add_argument("--alpha", type=float, default=0.10, help="target region miss rate")
    ap.add_argument("--label-threshold", type=float, default=0.10)
    ap.add_argument("--calibration-operator", default=None)
    ap.add_argument("--distance", default="null_space_projection")
    ap.add_argument("--normalise", default="ambient")
    ap.add_argument("--w-free", type=float, default=1.0)
    ap.add_argument("--w-disagree", type=float, default=1.0)
    ap.add_argument("--w-residual", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="write the metric tables here as CSV")
    args = ap.parse_args()

    bank = Path(args.bank)
    df = read_bank(bank)
    manifest = read_manifest(bank)
    print(f"[analyse] {len(df):,} region rows, {df['operator_id'].nunique()} operators")

    prov = manifest.get("provenance") or {}
    if prov and not prov.get("reportable", True):
        print(
            f"[analyse] NOT REPORTABLE — dataset={prov.get('dataset')}, "
            f"solver_family={prov.get('solver_family')}. Numbers below exercise the "
            "pipeline; none of them may appear in the write-up."
        )

    # Score and label are both derived here, from cached components. Changing
    # either costs nothing — no reconstruction is repeated.
    df["score"] = (
        args.w_free * df["s_free"]
        + args.w_disagree * df["s_disagree"]
        + args.w_residual * df["s_residual"]
    )
    truth = RegionTruth(
        err_total=torch.tensor(df["err_total"].to_numpy()),
        err_free=torch.tensor(df["err_free"].to_numpy()),
        ann_coverage=torch.tensor(df["ann_coverage"].to_numpy()),
    )
    df["label"] = label_hallucinated(truth, threshold=args.label_threshold).numpy()
    print(
        f"[analyse] {int(df['label'].sum()):,} hallucinated regions "
        f"({df['label'].mean():.1%} of all regions) at err_free > {args.label_threshold}"
    )
    if df["label"].sum() == 0:
        print("[analyse] no positive regions — lower --label-threshold")
        return 1

    nominal = 1.0 - args.alpha
    calib_op = args.calibration_operator or sorted(df["operator_id"].unique())[0]
    print(f"[analyse] calibration operator: {calib_op}")

    calib = df[df["operator_id"] == calib_op]

    # 1. Naive transfer — one threshold from the calibration operator, used everywhere.
    tau = split_conformal_threshold(calib["score"], calib["label"], alpha=args.alpha)
    naive = evaluate_groups(
        apply_threshold(df, tau, "score"), by=("operator_id",), nominal=nominal
    )
    naive["method"] = "naive_transferred"

    # 2. Matched per operator — the upper bound.
    matched_parts = []
    for _oid, part in df.groupby("operator_id"):
        t = split_conformal_threshold(part["score"], part["label"], alpha=args.alpha)
        matched_parts.append(apply_threshold(part, t, "score"))
    matched = evaluate_groups(
        pd.concat(matched_parts), by=("operator_id",), nominal=nominal
    )
    matched["method"] = "matched_per_operator"

    # 3. Group-conditional (Mondrian) — the guaranteed backbone.
    taus = mondrian_thresholds(df, "score", "label", alpha=args.alpha)
    mondrian = evaluate_groups(
        apply_threshold(df, taus, "score"), by=("operator_id",), nominal=nominal
    )
    mondrian["method"] = "group_conditional"

    results = pd.concat([naive, matched, mondrian], ignore_index=True)

    # Operator distance, computed from the registry alone.
    ops = rebuild_operators(manifest)
    distances = {
        oid: compute_distance(ops[calib_op], op, args.distance, normalise=args.normalise)
        for oid, op in ops.items()
    }

    print(f"\n[analyse] coverage by operator (nominal {nominal:.2f})")
    for method in ("naive_transferred", "matched_per_operator", "group_conditional"):
        curve = coverage_curve(
            results[results["method"] == method].copy(), distances, args.distance
        )
        rho = monotonicity(curve, "coverage_gap")
        print(f"\n  {method}  (gap-vs-distance monotonicity rho = {rho:+.3f})")
        print(
            curve[["operator_id", "operator_distance", "empirical_coverage",
                   "coverage_gap", "flag_area", "region_fdr"]]
            .to_string(index=False, float_format=lambda v: f"{v:.4f}")
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        results["operator_distance"] = results["operator_id"].map(distances)
        results.to_csv(out, index=False)
        (out.with_suffix(".distances.json")).write_text(
            json.dumps(distances, indent=2), encoding="utf-8"
        )
        print(f"\n[analyse] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
