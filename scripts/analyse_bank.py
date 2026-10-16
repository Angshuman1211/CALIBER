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
import sys
from pathlib import Path

import pandas as pd
import torch

from caliber.calibration import (
    apply_threshold,
    mondrian_calibration_results,
    split_threshold,
)
from caliber.calibration.conformal import min_positives
from caliber.evaluation import (
    add_label_args,
    coverage_curve,
    derive_score_and_label,
    evaluate_groups,
    label_from_args,
    monotonicity,
)
from caliber.operators.distance import compute_distance
from caliber.operators.registry import build_operator
from caliber.scorebank import read_bank, read_manifest
from caliber.scoring.score import ScoreComponents, ScoreScales, ScoreWeights


def rebuild_operators(manifest: dict) -> dict:
    """Reconstruct operator objects from the manifest, so distances need no data."""
    image_shape = tuple(manifest.get("operators", [{}])[0].get("image_shape", (128, 128)))
    ops = {}
    for row in manifest["operators"]:
        spec = {"modality": row["modality"], **row["params"]}
        ops[row["operator_id"]] = build_operator(spec, tuple(row["image_shape"]) or image_shape)
    return ops


def combination_note(weights: ScoreWeights, scales: ScoreScales) -> str:
    """One line saying how the score was combined, so a table is never orphaned."""
    return (
        f"= {weights.free:.2f}*free/{scales.free:.3g} "
        f"+ {weights.disagree:.2f}*disagree/{scales.disagree:.3g} "
        f"+ {weights.residual:.2f}*residual/{scales.residual:.3g}  "
        "(standardised affine, scales frozen on the calibration operator)"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Analyse a CALIBER score bank (CPU only)")
    ap.add_argument("bank", help="score-bank directory produced by the runner")
    ap.add_argument("--alpha", type=float, default=0.10, help="target region miss rate")
    add_label_args(ap)
    ap.add_argument("--calibration-operator", default=None)
    ap.add_argument("--distance", default="null_space_projection")
    ap.add_argument("--normalise", default="ambient")
    ap.add_argument("--distances-from", default=None,
                    help="CSV from ct_distances.py: reuse its distances for the calibration "
                         "operator instead of recomputing them (CT costs minutes per pair)")
    # Weights are non-negative and sum to 1 (D25). Equal thirds is the ratified
    # default, not a placeholder: standardisation is what makes it meaningful.
    ap.add_argument("--w-free", type=float, default=1 / 3)
    ap.add_argument("--w-disagree", type=float, default=1 / 3)
    ap.add_argument("--w-residual", type=float, default=1 / 3)
    ap.add_argument("--out", default=None, help="write the metric tables here as CSV")
    args = ap.parse_args(argv)

    bank = Path(args.bank)
    df = read_bank(bank)
    manifest = read_manifest(bank)
    print(f"[analyse] {len(df):,} region rows, {df['operator_id'].nunique()} operators")

    prov = manifest.get("provenance") or {}
    if prov and not prov.get("reportable", True):
        print(
            f"[analyse] NOT REPORTABLE - dataset={prov.get('dataset')}, "
            f"solver_family={prov.get('solver_family')}. Numbers below exercise the "
            "pipeline; none of them may appear in the write-up."
        )

    nominal = 1.0 - args.alpha
    calib_op = args.calibration_operator or sorted(df["operator_id"].unique())[0]
    print(f"[analyse] calibration operator: {calib_op}")

    # Score and label are both derived here, from cached components. Changing
    # either costs nothing — no reconstruction is repeated.
    #
    # The combination is T1.1's ratified standardised affine rule, not a raw
    # weighted sum. That distinction is not cosmetic: the three components differ
    # by an order of magnitude, so summing them at 1,1,1 silently reduces the
    # score to `s_free` (score.py, D26). Scales are estimated on the calibration
    # operator and then frozen — re-estimating per deployment operator would pull
    # the score distributions back towards each other and shrink the very
    # coverage gap this analysis exists to measure (D27).
    # One definition, in the library: `scripts/sizing.py` needs the same derivation, and
    # a second copy is how P-19 happened.
    weights = ScoreWeights(free=args.w_free, disagree=args.w_disagree, residual=args.w_residual)
    label = label_from_args(df, args)
    df = derive_score_and_label(df, calib_op, weights=weights, **label.kwargs)
    scales = ScoreScales.from_components(
        ScoreComponents(
            s_free=torch.tensor(df[df["operator_id"] == calib_op]["s_free"].to_numpy()),
            s_disagree=torch.tensor(df[df["operator_id"] == calib_op]["s_disagree"].to_numpy()),
            s_residual=torch.tensor(df[df["operator_id"] == calib_op]["s_residual"].to_numpy()),
            ref_intensity=1.0,
        )
    )
    print(f"[analyse] score {combination_note(weights, scales)}")
    print(
        f"[analyse] {int(df['label'].sum()):,} hallucinated regions "
        f"({df['label'].mean():.1%} of all regions); label: {label.text}"
    )
    if df["label"].sum() == 0:
        print("[analyse] no positive regions under this label")
        return 1
    if bool(df["label"].all()):
        print(
            "[analyse] WARNING  every region is positive, so region FDR carries no "
            "information (P-02).",
            file=sys.stderr,
        )

    # Conformal calibration must be out of sample. A bank written with
    # `split: auto` carries both sides, assigned per patient and identically in
    # every acquisition regime (D-022, D-027); calibrating and evaluating on the
    # same rows inflates coverage at the calibration operator specifically, which
    # is the one number the naive-transfer story is read against.
    #
    # This applies to *every* method, not only the naive one. The first version of
    # this fix (P-19) held out the naive threshold but still fitted the matched and
    # group-conditional thresholds on the evaluation rows themselves, which is why
    # they reported exactly 0.900 — in-sample conformal lands on nominal by
    # construction — and why the comparison against naive was held-out vs
    # in-sample (P-23). Every threshold below is fitted on `calib_pool` and scored
    # only on `evaluate`.
    has_split = "split" in df.columns and df["split"].nunique() > 1
    if has_split:
        calib_pool = df[df["split"] == "calib"]
        evaluate = df[df["split"] == "test"].copy()
        print(
            f"[analyse] held out: thresholds fitted on {len(calib_pool):,} split=calib rows, "
            f"scored on {len(evaluate):,} split=test rows (patient-disjoint)"
        )
    else:
        calib_pool = df
        evaluate = df
        print(
            "[analyse] WARNING  bank has a single split, so every threshold is fitted on "
            "the rows it is scored on and all coverage below is in-sample. Rebuild with "
            "`split: auto` for held-out numbers.",
            file=sys.stderr,
        )

    # 1. Naive transfer — one threshold from the calibration operator, used everywhere.
    calib = calib_pool[calib_pool["operator_id"] == calib_op]
    uncertifiable: list[str] = []
    naive_result = split_threshold(calib["score"], calib["label"], alpha=args.alpha)
    tau = naive_result.threshold
    if not naive_result.certifiable:
        uncertifiable.append(f"naive/{calib_op} ({naive_result.n_positives} positives)")
    naive = evaluate_groups(
        apply_threshold(evaluate, tau, "score"), by=("operator_id",), nominal=nominal
    )
    naive["method"] = "naive_transferred"

    # 2. Matched per operator — the upper bound: each operator's own calibration rows.
    matched_parts = []
    for oid, part in evaluate.groupby("operator_id"):
        own = calib_pool[calib_pool["operator_id"] == oid]
        result = split_threshold(own["score"], own["label"], alpha=args.alpha)
        if not result.certifiable:
            uncertifiable.append(f"matched/{oid} ({result.n_positives} positives)")
        matched_parts.append(apply_threshold(part, result.threshold, "score"))
    matched = evaluate_groups(
        pd.concat(matched_parts), by=("operator_id",), nominal=nominal
    )
    matched["method"] = "matched_per_operator"

    # 3. Group-conditional (Mondrian) — the guaranteed backbone.
    mondrian_results = mondrian_calibration_results(
        calib_pool, "score", "label", alpha=args.alpha
    )
    uncertifiable += [
        f"group_conditional/{g} ({r.n_positives} positives)"
        for g, r in sorted(mondrian_results.items())
        if not r.certifiable
    ]
    taus = {g: r.threshold for g, r in mondrian_results.items()}
    mondrian = evaluate_groups(
        apply_threshold(evaluate, taus, "score"), by=("operator_id",), nominal=nominal
    )
    mondrian["method"] = "group_conditional"

    results = pd.concat([naive, matched, mondrian], ignore_index=True)

    # D-026: a group too thin to certify lands at -inf, which flags every region --
    # valid, useless, and previously silent. The old reference implementation rounded
    # up instead and returned a plausible-looking threshold whose true miss rate could
    # be 1/2. Say so here rather than letting the number be read as calibrated.
    if uncertifiable:
        print(
            f"[analyse] NOT CERTIFIABLE  fewer than {min_positives(args.alpha)} "
            f"positives at alpha={args.alpha}, so these flag everything and their "
            f"coverage is not a conformal guarantee:\n           "
            + "\n           ".join(uncertifiable),
            file=sys.stderr,
        )

    # Operator distance, computed from the registry alone.
    if args.distances_from:
        cached = pd.read_csv(args.distances_from)
        cached = cached[(cached["calibration"] == calib_op) & (cached["measure"] == args.distance)]
        distances = dict(zip(cached["deployment"], cached["value"].astype(float), strict=True))
        distances[calib_op] = 0.0
        missing = sorted(set(df["operator_id"].astype(str)) - set(distances))
        if missing:
            print(f"[analyse] {args.distances_from} lacks {args.distance} from {calib_op} to "
                  f"{missing}", file=sys.stderr)
            return 1
    else:
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
