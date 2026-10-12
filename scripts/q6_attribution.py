"""Q6 — does `s_free` fire on invention, or on anatomy that was always there?

Reads a cached score bank; touches no GPU, no dataset and no solver.

    PYTHONPATH=src python3 scripts/q6_attribution.py runs/<run_id> --tau 1.75

What it reports, in the order it reports it:

1. **The identity check.** The three free-component columns are related exactly
   (see ``caliber.scoring.attribution``). If they are not, they were not written
   by one consistent normalisation and nothing below means anything, so this is
   checked before anything is computed rather than after something looks odd.
2. **The fill ratio and content alignment.** ``u = s_free / free_content`` and
   ``cos θ``. These say whether the prior's hidden content is invention or
   recovered anatomy, and they supply the precondition D37's pass/fail check
   needs: ``err_free/free_content > 1`` is *forced* whenever ``u > 2``, however
   well the prior is aligned.
3. **Per-component attribution.** For each component, AUC against the label,
   the same AUC with anatomical content held fixed, and the verdict from the
   pre-registered thresholds. The oracle ratio ``s_free / free_content`` is
   scored alongside as the ceiling: it is the statistic the relative criterion
   is actually monotone in, and it needs ground truth, so it is a diagnostic
   upper bound and **never a deployable score**.
4. **The ablation, on flagged area at fixed alpha.** Power is pinned at
   ``1 − α`` by the calibrator, so area is the axis that can move. This is where
   D2's honest-negative outcome would show up: if disagreement alone flags as
   little area as the combination, the free-component decomposition is not
   earning its place.

Track 1 (T1.1 / Q6). Written before the real bank was available on this machine,
so the thresholds in ``attribution.py`` are pre-registered rather than fitted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from caliber.scorebank import read_bank, read_manifest
from caliber.scoring.attribution import (
    CANONICAL_COMPONENTS,
    attribute_components,
    component_ablation,
    content_alignment,
    fill_ratio,
    identity_residual,
    q6_verdict,
)
from caliber.scoring.labels import RegionTruth, label_hallucinated
from caliber.scoring.score import ScoreComponents, ScoreScales, ScoreWeights, combine_components

#: Relative-criterion tau. D37's provisional value, chosen on calibration
#: arithmetic (9 positives per group at alpha=0.1) and explicitly not reportable.
DEFAULT_TAU = 1.75

#: Above this the identity is broken, not merely float-noisy. float32 columns
#: through a difference of squares lose a few digits; 1% would be a real
#: inconsistency in how the columns were normalised.
IDENTITY_TOLERANCE = 0.01


def components_of(frame: pd.DataFrame) -> ScoreComponents:
    return ScoreComponents(
        s_free=torch.tensor(frame["s_free"].to_numpy()),
        s_disagree=torch.tensor(frame["s_disagree"].to_numpy()),
        s_residual=torch.tensor(frame["s_residual"].to_numpy()),
        ref_intensity=1.0,
    )


def describe(name: str, values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return f"{name:20s} no usable regions"
    return (
        f"{name:20s} n={finite.size:5d}  min {finite.min():7.3f}  "
        f"median {np.median(finite):7.3f}  max {finite.max():7.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Q6 attribution for a CALIBER score bank")
    ap.add_argument("bank", help="score-bank directory produced by the runner")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU,
                    help=f"relative criterion err_free/free_content > tau (default {DEFAULT_TAU})")
    ap.add_argument("--alpha", type=float, default=0.10, help="target region miss rate")
    ap.add_argument("--calibration-operator", default=None,
                    help="operator the score scales are frozen on (D27); default: first")
    ap.add_argument("--strata", type=int, default=4,
                    help="quantile strata of free_content for the detail-matched AUC")
    ap.add_argument("--w-free", type=float, default=1 / 3)
    ap.add_argument("--w-disagree", type=float, default=1 / 3)
    ap.add_argument("--w-residual", type=float, default=1 / 3)
    ap.add_argument("--out", default=None, help="write the two tables here as CSV")
    args = ap.parse_args(argv)

    bank = Path(args.bank)
    df = read_bank(bank)
    manifest = read_manifest(bank)
    print(f"[q6] {len(df):,} region rows, {df['operator_id'].nunique()} operators")

    prov = manifest.get("provenance") or {}
    if prov and not prov.get("reportable", True):
        print(
            f"[q6] NOT REPORTABLE -- dataset={prov.get('dataset')}, "
            f"solver_family={prov.get('solver_family')}. Q6's answer from this bank is a "
            "development reading, not a result."
        )

    # -- 1. the identity, before anything is read off the columns ---------------
    res = identity_residual(df["s_free"], df["free_content"], df["err_free"])
    print(
        f"[q6] identity check on {int(res['n']):,} regions: median relative residual "
        f"{res['median_relative']:.2e}, max {res['max_relative']:.2e}"
    )
    if not np.isfinite(res["max_relative"]) or res["max_relative"] > IDENTITY_TOLERANCE:
        print(
            "[q6] the three free-component columns do not satisfy the exact identity. "
            "They were probably normalised by different reference intensities, in which "
            "case every ratio below compares quantities in different units. Fix the bank "
            "before reading further.",
            file=sys.stderr,
        )
        return 1

    # -- 2. what the prior put in the null space -------------------------------
    u = fill_ratio(df["s_free"], df["free_content"])
    cos = content_alignment(df["s_free"], df["free_content"], df["err_free"])
    df["fill_ratio"], df["content_alignment"] = u, cos
    print(describe("fill ratio u", u))
    print(describe("cos theta", cos))
    over_filled = float(np.nanmean(u > 2.0)) if np.isfinite(u).any() else float("nan")
    print(
        f"[q6] {over_filled:.1%} of regions have u > 2, where err_free/free_content > 1 is "
        "forced regardless of alignment -- D37's pass/fail check is only diagnostic on the rest"
    )
    aligned = float(np.nanmean(cos > 0.5)) if np.isfinite(cos).any() else float("nan")
    print(
        f"[q6] cos theta > 0.5 in {aligned:.1%} of regions: hidden content that agrees with "
        "the truth's, i.e. recovered anatomy rather than invention"
    )

    # -- 3. score and label, both derived from cache ---------------------------
    calib_op = args.calibration_operator or sorted(df["operator_id"].unique())[0]
    weights = ScoreWeights(free=args.w_free, disagree=args.w_disagree, residual=args.w_residual)
    scales = ScoreScales.from_components(components_of(df[df["operator_id"] == calib_op]))
    df["score"] = combine_components(components_of(df), weights, scales).numpy()
    if any(scales.degenerate):
        dead = [c for c, d in zip(CANONICAL_COMPONENTS, scales.degenerate, strict=True) if d]
        print(f"[q6] excluded as degenerate (D35): {', '.join(dead)}")

    truth = RegionTruth(
        err_total=torch.tensor(df["err_total"].to_numpy()),
        err_free=torch.tensor(df["err_free"].to_numpy()),
        free_content=torch.tensor(df["free_content"].to_numpy()),
        ann_coverage=torch.tensor(df["ann_coverage"].to_numpy()),
    )
    df["label"] = label_hallucinated(truth, threshold=args.tau, relative=True).numpy()
    n_pos = int(df["label"].sum())
    print(
        f"[q6] {n_pos:,} hallucinated regions ({df['label'].mean():.1%}) at "
        f"err_free/free_content > {args.tau}"
    )
    if n_pos == 0 or n_pos == len(df):
        print(
            "[q6] the label is constant, so every statistic below is undefined. Move tau "
            "into the range where the criterion separates (D-034's curve).",
            file=sys.stderr,
        )
        return 1

    # The oracle ratio: the statistic the relative criterion is monotone in.
    # It needs free_content, which needs ground truth, so it is a ceiling and a
    # diagnostic — not a candidate score. Named to make misuse obvious.
    df["ORACLE_s_free_over_content"] = df["s_free"] / df["free_content"].clip(lower=1e-3)
    candidate_cols = [*CANONICAL_COMPONENTS, "score", "ORACLE_s_free_over_content"]
    # `s_range` is the deployment-available denominator this analysis motivates
    # (the A2 ask). Scored automatically if a bank carries it.
    if "s_range" in df.columns:
        df["s_free_over_range"] = df["s_free"] / df["s_range"].clip(lower=1e-3)
        candidate_cols.append("s_free_over_range")

    # -- 4. attribution --------------------------------------------------------
    attribution = attribute_components(
        df, component_cols=tuple(candidate_cols), n_strata=args.strata
    )
    print("\n[q6] per-component attribution")
    print(attribution.drop(columns=["reason"]).to_string(index=False))

    ablation = component_ablation(
        df,
        candidates={c: c for c in candidate_cols if c in df.columns},
        alpha=args.alpha,
    )
    print(f"\n[q6] ablation at alpha={args.alpha:g} -- compare flag_area, not power")
    print(ablation.to_string(index=False))

    verdict = q6_verdict(attribution, ablation)
    print("\n" + verdict.summary())
    for _, row in attribution.iterrows():
        print(f"  {row['component']:28s} {row['verdict']:20s} {row['reason']}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        attribution.to_csv(out / "q6_attribution.csv", index=False)
        ablation.to_csv(out / "q6_ablation.csv", index=False)
        df[
            ["operator_id", "region_id", "s_free", "free_content", "err_free",
             "fill_ratio", "content_alignment", "label"]
        ].to_csv(out / "q6_per_region.csv", index=False)
        print(f"[q6] wrote three tables to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
