"""The published Hallucination Index [22] on the MRI bank: a yardstick for the label and the score.

Tivnan et al.'s index measures how far a generative model's reconstructions sit from a
zero-hallucination reference -- the truth plus noise at the same apparent level -- and reads as
the share of the error that is hallucination rather than noise. It needs the ground truth, so
it cannot flag anything at deployment (Blueprint Sec. 15.1, "non-conformal context"). What it
can do is stand beside the project's own label as an independent, published definition of
hallucination, and ask two questions of it:

1. **Does the project's label agree with it?** P-31 found the D-060 label still led by its
   denominator within operators. An index that measures error *against noise* instead of
   *against hidden content* is a check from a different direction.
2. **Which score ranks the index's hallucinations?** CALIBER's score, its components, and
   posterior dispersion, on the same regions.

Per region it uses the bank's ``err_total`` (RMS of mean reconstruction minus truth) and the
dispersion run's ``s_dispersion`` (the posterior spread), so it needs the dispersion parquet
and no GPU. A region counts as hallucinated by the index when its hallucination-to-noise ratio
exceeds 1: more of its error is bias than noise. See ``hallucination_index`` for the adaptation.

    python scripts/hallucination_index.py runs/<run_id>_by_patient \
        --dispersion runs/w05_analysis/dispersion.parquet \
        --calibration-operator mri.random1d.R2.c0.08.s0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from caliber.evaluation import CANDIDACY_QUANTILE, candidacy_gate, derive_score_and_label
from caliber.evaluation.baselines import auroc, cluster_bootstrap_metric, hallucination_index

KEYS = ["sample_id", "slice_idx", "operator_id", "region_id"]
COLUMNS = KEYS + ["split", "patient_id", "n_posterior", "s_free", "s_disagree", "s_residual",
                  "err_total", "err_free", "free_content", "ann_coverage"]
DETECTORS = {"CALIBER score": "score", "s_free": "s_free", "s_disagree": "s_disagree",
             "posterior dispersion": "s_dispersion"}


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa between two boolean labels: agreement beyond what chance gives."""
    a, b = a.astype(bool), b.astype(bool)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bank", help="a bank with patient_id (add_patient_ids.py)")
    ap.add_argument("--dispersion", required=True, help="dispersion_baseline.py generate output")
    ap.add_argument("--calibration-operator", required=True)
    ap.add_argument("--tau", type=float, default=1.75, help="D37's ratified value")
    ap.add_argument("--candidacy-quantile", type=float, default=CANDIDACY_QUANTILE)
    ap.add_argument("--hnr-threshold", type=float, default=1.0,
                    help="index label: squared bias over noise variance above this")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    root = Path(args.bank)
    raw = pd.read_parquet(root / "scores.parquet", columns=COLUMNS)
    raw["operator_id"] = raw["operator_id"].astype(str)
    region = int(json.loads((root / "manifest.json").read_text(encoding="utf-8"))
                 ["config"]["regions"]["size"])
    gate = candidacy_gate(raw, args.candidacy_quantile)
    df = derive_score_and_label(raw, args.calibration_operator, threshold=args.tau,
                                min_free_content=gate)
    disp = pd.read_parquet(args.dispersion, columns=KEYS + ["s_dispersion"])
    disp["operator_id"] = disp["operator_id"].astype(str)
    df = df[df["split"] == "test"].merge(disp, on=KEYS, how="inner")
    if df.empty:
        print("[hi] no test rows match the dispersion file", file=sys.stderr)
        return 1
    if (df["s_dispersion"] <= 0).any():
        print(f"[hi] dropping {int((df['s_dispersion'] <= 0).sum())} regions with no spread",
              file=sys.stderr)
        df = df[df["s_dispersion"] > 0]

    n_draws = int(df["n_posterior"].iloc[0])
    hi, hnr = hallucination_index(df["err_total"], df["s_dispersion"],
                                  n_pixels=region * region, n_draws=n_draws)
    df["hi"], df["hnr"] = hi, hnr
    df["hi_label"] = df["hnr"] > args.hnr_threshold
    print(f"[hi] {len(df):,} test regions, {df['patient_id'].nunique()} patients, "
          f"{region}x{region} regions, {n_draws} draws; D-060 gate free_content > {gate:.4f}")
    print(f"[hi] index saturation: {float((df['hi'] > 0.99).mean()):.1%} of regions have "
          "HI > 0.99, so everything below ranks by the hallucination-to-noise ratio")

    rows = []
    for oid, part in df.groupby("operator_id"):
        img = part.groupby(["sample_id", "slice_idx"])["hnr"].mean()
        row = {"operator_id": oid,
               "image_hnr_median": float(img.median()),
               "hi_positive_rate": float(part["hi_label"].mean()),
               "d060_positive_rate": float(part["label"].mean()),
               "kappa_hi_vs_d060": kappa(part["hi_label"].to_numpy(), part["label"].to_numpy()),
               "auroc_hnr_for_d060": auroc(part["hnr"], part["label"])}
        for name, col in DETECTORS.items():
            r = cluster_bootstrap_metric(auroc, part[col], part["hi_label"], part["patient_id"],
                                         n_boot=args.boot)
            row[f"{name} auroc"] = r["value"]
            row[f"{name} lo"], row[f"{name} hi"] = r["lo"], r["hi"]
        rows.append(row)
    t = pd.DataFrame(rows).set_index("operator_id")
    pd.set_option("display.width", 220)
    print("\n[hi] the index against the project's label (D-060), per operator")
    print(t[["image_hnr_median", "hi_positive_rate", "d060_positive_rate", "kappa_hi_vs_d060",
             "auroc_hnr_for_d060"]].round(3).to_string())
    print(f"\n[hi] AUROC for the index's label (hnr > {args.hnr_threshold:g}), 95% by patient")
    for oid, r in t.iterrows():
        cells = "  ".join(f"{name} {r[f'{name} auroc']:.3f} [{r[f'{name} lo']:.3f}, "
                          f"{r[f'{name} hi']:.3f}]" for name in DETECTORS)
        print(f"   {oid:<36} {cells}")
    if args.out:
        t.to_csv(args.out)
        print(f"[hi] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
