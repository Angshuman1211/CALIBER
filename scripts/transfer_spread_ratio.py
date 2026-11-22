"""The spread of a transfer cell against the density ratio at the threshold (D92).

    python scripts/transfer_spread_ratio.py runs/<bank> --cluster sample_id \
        --pair-dir runs/<analysis>/pairs --out runs/<analysis>/spread_ratio.csv

Track 1's D92 carries split conformal's Beta law from the matched cell to every transfer
cell, and ``prereg.TRANSFER_SPREAD_PREDICTION`` freezes what follows from it before any
number was looked at: over the patient splits,

    sd[coverage(a -> b)] / sd[coverage(a -> a)]  ==  f_b / f_a  at the threshold

for every ordered pair, with no fitted parameter.

The left side is read from ``pair_coverage_by_split.csv`` (``pair_coverage_by_split.py``).
The right side is ``reliability.transfer_beta_reference(...).slope``: a central difference
on the empirical probability-probability map of the two operators' positive scores, so no
density is estimated. The scores are the registry split's positives, calibration and test
sides pooled.

The prediction is for the ratio, and D92 says the finite test set's inflation of each
spread divides out only where the two cells hold similar numbers of positives. The table
therefore carries ``positives_ratio``, and the summary is printed for all pairs and for
the pairs whose counts are within ``--similar`` of each other.

It also answers D92's question about the bank: whether, within one split, every operator
calibrates on the same subjects.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from caliber.data.splits import split_for
from caliber.evaluation import add_label_args, derive_score_and_label, label_from_args, prereg
from caliber.evaluation.gap import rank_correlation
from caliber.evaluation.reliability import transfer_beta_reference
from caliber.reportability import COVERAGE, describe, stamp
from caliber.scorebank import read_bank, read_manifest

PROJECT_SALT = "caliber-split-v1"
PROBE_SALTS = [PROJECT_SALT] + [f"probe-{i}" for i in range(12)]
SCALE_OPERATOR = "ct.sparse_view.v180.a180.dfull"
KEYS = ["calibration", "deployment"]


def spread_ratios(by_split: pd.DataFrame, scores: dict[str, np.ndarray],
                  n_calibration: dict[str, int], alpha: float) -> pd.DataFrame:
    """One row per ordered pair: the measured spread ratio and the predicted one.

    ``by_split`` is the per-split cell table (``calibration``, ``deployment``, ``coverage``,
    ``n_positives_test``); ``scores`` the positive scores at each operator;
    ``n_calibration`` the size of each operator's calibration set of positives.
    """
    cells = by_split.groupby(KEYS)
    sd = cells["coverage"].std(ddof=1)
    counts = cells["n_positives_test"].mean()
    rows = []
    for a in sorted(scores):
        for b in sorted(scores):
            if a == b:
                continue
            ref = transfer_beta_reference(scores[a], scores[b], n_positives=n_calibration[a],
                                          alpha=alpha)
            rows.append({
                "calibration": a, "deployment": b,
                "sd_transfer": sd[(a, b)], "sd_matched": sd[(a, a)],
                "measured_ratio": sd[(a, b)] / sd[(a, a)],
                "predicted_ratio": ref.slope,
                "positives_ratio": counts[(a, b)] / counts[(a, a)],
            })
    return pd.DataFrame(rows)


def agreement(table: pd.DataFrame) -> dict[str, float]:
    """How closely the measured ratios track the predicted ones, on the log scale."""
    m = table["measured_ratio"].to_numpy(dtype=float)
    p = table["predicted_ratio"].to_numpy(dtype=float)
    ok = np.isfinite(m) & np.isfinite(p) & (m > 0) & (p > 0)
    if ok.sum() < 3:
        return {"n": float(ok.sum())}
    error = np.log(m[ok] / p[ok])
    return {
        "n": float(ok.sum()),
        "rho": rank_correlation(p[ok], m[ok]),
        "log_log_slope": float(np.polyfit(np.log(p[ok]), np.log(m[ok]), 1)[0]),
        "median_measured_over_predicted": float(np.exp(np.median(error))),
        "median_abs_log_error": float(np.median(np.abs(error))),
        "within_20_percent": float(np.mean(np.abs(error) < np.log(1.2))),
        "within_factor_2": float(np.mean(np.abs(error) < np.log(2.0))),
        "same_side_of_one": float(np.mean((m[ok] > 1.0) == (p[ok] > 1.0))),
    }


def shared_calibration_cohort(raw: pd.DataFrame, cluster: str, calib_fraction: float,
                              design_fraction: float) -> tuple[bool, int]:
    """Do all operators calibrate on the same subjects in every split? And how many."""
    subjects = sorted(raw[cluster].unique())
    same, size = True, 0
    for salt in PROBE_SALTS:
        side = {p: split_for(p, calib_fraction=calib_fraction, design_fraction=design_fraction,
                             salt=salt) for p in subjects}
        on_calib = raw[raw[cluster].map(side) == "calib"]
        sets = {frozenset(g[cluster]) for _, g in on_calib.groupby("operator_id")}
        same &= len(sets) == 1
        if salt == PROJECT_SALT:
            size = len(next(iter(sets)))
    return same, size


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bank")
    ap.add_argument("--pair-dir", required=True, help="pair_coverage_by_split.py output directory")
    ap.add_argument("--cluster", default="patient_id", help="sample_id on CT")
    ap.add_argument("--scale-operator", default=SCALE_OPERATOR)
    ap.add_argument("--calib-fraction", type=float, default=0.5)
    ap.add_argument("--design-fraction", type=float, default=0.10)
    ap.add_argument("--similar", type=float, default=1.5,
                    help="two cells have similar positive counts within this factor")
    ap.add_argument("--out", required=True)
    add_label_args(ap)
    args = ap.parse_args(argv)

    bank = Path(args.bank)
    raw = read_bank(bank)
    prov = read_manifest(bank).get("provenance") or {}
    print(f"[spread] bank    {bank.name}; prereg {prereg.fingerprint()}")
    print(f"[spread] stamp   {describe(prov)}; coverage numbers are {stamp(prov, COVERAGE)}")

    same, size = shared_calibration_cohort(raw, args.cluster, args.calib_fraction,
                                           args.design_fraction)
    print(f"[spread] cohort  every operator calibrates on the same {args.cluster}s in every "
          f"split: {same} ({size} on the registry split)")

    label = label_from_args(raw, args)
    df = derive_score_and_label(raw, args.scale_operator, **label.kwargs)
    pos = df[df["label"].astype(bool) & df["split"].isin(["calib", "test"])]
    scores = {op: g["score"].to_numpy(dtype=float) for op, g in pos.groupby("operator_id")}
    n_cal = {op: int((g["split"] == "calib").sum()) for op, g in pos.groupby("operator_id")}
    sizes = sorted(len(v) for v in scores.values())
    print(f"[spread] positives per operator, calibration and test sides: {sizes[0]} to {sizes[-1]}")

    by_split = pd.read_csv(Path(args.pair_dir) / "pair_coverage_by_split.csv")
    table = spread_ratios(by_split, scores, n_cal, prereg.ALPHA)
    table.to_csv(args.out, index=False)

    lo, hi = 1.0 / args.similar, args.similar
    similar = table[(table["positives_ratio"] > lo) & (table["positives_ratio"] < hi)]
    print(f"[spread] measured ratio {table['measured_ratio'].min():.2f} to "
          f"{table['measured_ratio'].max():.2f}; predicted {table['predicted_ratio'].min():.2f} "
          f"to {table['predicted_ratio'].max():.2f}; positive-count ratio "
          f"{table['positives_ratio'].min():.2f} to {table['positives_ratio'].max():.2f}")
    for name, part in ((f"all {len(table)} ordered pairs", table),
                       (f"positive counts within {args.similar:g}x", similar)):
        a = agreement(part)
        if len(a) == 1:
            print(f"[spread] {name}: too few pairs ({int(a['n'])})")
            continue
        print(f"[spread] {name} (n = {int(a['n'])}): rho {a['rho']:+.3f}, log-log slope "
              f"{a['log_log_slope']:.2f}, median measured / predicted "
              f"{a['median_measured_over_predicted']:.2f}, median |log error| "
              f"{a['median_abs_log_error']:.2f}, within 20% at {a['within_20_percent']:.0%}, "
              f"within 2x at {a['within_factor_2']:.0%}, same side of 1 at "
              f"{a['same_side_of_one']:.0%}")
    narrow = int((table["measured_ratio"] > 1.0).sum())
    print(f"[spread] the matched cell's spread understates the transfer cell's at {narrow} of "
          f"{len(table)} pairs (largest ratio {table['measured_ratio'].max():.2f})")
    print(f"[spread] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
