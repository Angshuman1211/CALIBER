"""Coverage of every ordered operator pair, on the registry split and over splits (T2-A3, D86).

    python scripts/pair_coverage_by_split.py runs/<bank> --cluster sample_id \
        --out-dir runs/<analysis>/pairs

For each ordered pair (a, b): the threshold is fitted on operator a's calibration patients
and coverage is measured on operator b's test patients. ``a == b`` is the matched-operator
control. Two tables come out.

``pair_coverage_registry.csv`` -- the registry split, which stays the reporting split
(D85, D86). Every cell carries the interval D73 asks of every coverage number: patients
resampled on **both** sides and the threshold refitted in each replicate
(``caliber.evaluation.reliability.clustered_coverage``).

``pair_coverage_by_split.csv`` and ``pair_coverage_split_mean.csv`` -- the same cells on
each probe split, and their mean and spread. D86 moves T1-A's unsafe label to the
across-split mean, because one draw of the calibration set displaces every pair the same
way and that displacement cannot average out over pairs. Each probe split re-draws every
patient's side and re-values the label on its own design side, exactly as
``split_noise_floor.py`` does, so a split is a complete repetition of the protocol on the
same cohort. They are re-cuts of one cohort, not independent replications.

The score's component scales are frozen on ``--scale-operator`` (D26/D27), as in
``t1a_safety_rule.py``: re-estimating them per operator would shrink the gap being measured.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from caliber.data.splits import split_for
from caliber.evaluation import add_label_args, derive_score_and_label, label_from_args, prereg
from caliber.evaluation.reliability import clustered_coverage
from caliber.reportability import COVERAGE, describe, stamp
from caliber.scorebank import read_bank, read_manifest

PROJECT_SALT = "caliber-split-v1"
PROBE_SALTS = [PROJECT_SALT] + [f"probe-{i}" for i in range(12)]
SCALE_OPERATOR = "ct.sparse_view.v180.a180.dfull"


def positives_by_cluster(df: pd.DataFrame, cluster: str) -> dict:
    """``{(operator, split): {cluster: sorted positive scores}}``."""
    pos = df[df["label"].astype(bool)]
    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for (op, split, cl), g in pos.groupby(["operator_id", "split", cluster]):
        out.setdefault((op, split), {})[cl] = np.sort(g["score"].to_numpy(dtype=float))
    return out


def threshold(scores: np.ndarray, alpha: float) -> float:
    """The project's split-conformal rule: the ``floor(alpha (n + 1))``-th smallest positive."""
    k = int(math.floor(alpha * (scores.size + 1)))
    return float("-inf") if k < 1 else float(np.sort(scores)[k - 1])


def pair_cells(by, operators, alpha: float) -> list[dict]:
    """Point coverage of every ordered pair on one split."""
    rows = []
    for a in operators:
        cal = by.get((a, "calib"), {})
        pooled = np.concatenate(list(cal.values())) if cal else np.array([])
        tau = threshold(pooled, alpha)
        for b in operators:
            test = by.get((b, "test"), {})
            scores = np.concatenate(list(test.values())) if test else np.array([])
            coverage = float((scores >= tau).mean()) if scores.size else float("nan")
            rows.append({"calibration": a, "deployment": b, "tau": tau,
                         "n_positives_calib": int(pooled.size),
                         "n_positives_test": int(scores.size), "coverage": coverage})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bank")
    ap.add_argument("--cluster", default="patient_id", help="sample_id on CT")
    ap.add_argument("--scale-operator", default=SCALE_OPERATOR)
    ap.add_argument("--calib-fraction", type=float, default=0.5)
    ap.add_argument("--design-fraction", type=float, default=0.10)
    ap.add_argument("--n-boot", type=int, default=prereg.N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    add_label_args(ap)
    args = ap.parse_args(argv)

    bank = Path(args.bank)
    raw = read_bank(bank)
    prov = read_manifest(bank).get("provenance") or {}
    if args.cluster not in raw.columns:
        print(f"[pairs] no {args.cluster!r} column", file=sys.stderr)
        return 1
    alpha = prereg.ALPHA
    operators = sorted(raw["operator_id"].unique())
    patients = sorted(raw[args.cluster].unique())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[pairs] bank        {bank.name}")
    print(f"[pairs] stamp       {describe(prov)}; coverage numbers are {stamp(prov, COVERAGE)}")
    print(f"[pairs] cells       {len(operators)} operators, {len(operators) ** 2} ordered pairs "
          f"({len(operators)} matched), {len(patients)} {args.cluster}s")

    def sides(salt: str) -> pd.Series:
        side = {p: split_for(p, calib_fraction=args.calib_fraction,
                             design_fraction=args.design_fraction, salt=salt) for p in patients}
        return raw[args.cluster].map(side)

    if not (sides(PROJECT_SALT) == raw["split"]).all():
        print("[pairs] the bank's split column does not match split_for with the project salt "
              "and these fractions; refusing to compare probe splits against it", file=sys.stderr)
        return 1

    # -- the registry split, with the both-sides interval on every cell ------------------
    label = label_from_args(raw, args)
    df = derive_score_and_label(raw, args.scale_operator, **label.kwargs)
    print(f"[pairs] label       {label.text}")
    by = positives_by_cluster(df, args.cluster)
    rows = []
    for cell in pair_cells(by, operators, alpha):
        cal = list(by.get((cell["calibration"], "calib"), {}).values())
        test = list(by.get((cell["deployment"], "test"), {}).values())
        r = clustered_coverage(cal, test, n_boot=args.n_boot, seed=args.seed)
        rows.append({**cell, "lo": r.lo, "hi": r.hi, "sd": r.sd,
                     "matched": cell["calibration"] == cell["deployment"]})
    registry = pd.DataFrame(rows)
    registry.to_csv(out / "pair_coverage_registry.csv", index=False)

    # -- every probe split: sides re-drawn, the label re-valued on its own design side ---
    rows = []
    for salt in PROBE_SALTS:
        probe = raw.copy()
        probe["split"] = sides(salt)
        try:
            lab = label_from_args(probe, args)
            d = derive_score_and_label(probe, args.scale_operator, **lab.kwargs)
        except ValueError as exc:   # a split whose design side cannot value the label
            print(f"[pairs] {salt}: skipped - {exc}", file=sys.stderr)
            continue
        rows += [{"salt": salt, **c} for c in pair_cells(positives_by_cluster(d, args.cluster),
                                                         operators, alpha)]
    splits = pd.DataFrame(rows)
    splits.to_csv(out / "pair_coverage_by_split.csv", index=False)
    n_splits = splits["salt"].nunique()
    if n_splits < prereg.UNSAFE_LABEL_MIN_SPLITS:
        print(f"[pairs] only {n_splits} usable splits; the across-split label needs "
              f"{prereg.UNSAFE_LABEL_MIN_SPLITS}", file=sys.stderr)
        return 1
    mean = (splits.groupby(["calibration", "deployment"])["coverage"]
            .agg(split_mean="mean", split_sd=lambda v: v.std(ddof=1), split_min="min",
                 split_max="max", n_splits="count").reset_index())
    mean = mean.merge(registry[["calibration", "deployment", "coverage", "lo", "hi", "matched"]]
                      .rename(columns={"coverage": "registry"}), on=["calibration", "deployment"])
    mean["registry_minus_mean"] = mean["registry"] - mean["split_mean"]
    mean.to_csv(out / "pair_coverage_split_mean.csv", index=False)

    matched = mean[mean["matched"]]
    off = mean[~mean["matched"]]
    print(f"[pairs] splits      {n_splits} (the registry split and {n_splits - 1} probes)")
    print(f"[pairs] matched     registry {matched['registry'].mean():.3f}, across-split mean "
          f"{matched['split_mean'].mean():.3f} (range {matched['split_mean'].min():.3f}-"
          f"{matched['split_mean'].max():.3f})")
    print(f"[pairs] transfer    registry minus across-split mean over the {len(off)} ordered "
          f"pairs: median {off['registry_minus_mean'].median():+.3f}, "
          f"{int((off['registry_minus_mean'] < 0).sum())} of {len(off)} below")
    below = off[off["split_mean"] < prereg.NOMINAL]
    print(f"[pairs] below nominal on the across-split mean: {len(below)} of {len(off)} pairs; "
          f"on the registry point estimate: {int((off['registry'] < prereg.NOMINAL).sum())}; "
          f"registry interval wholly below: {int((off['hi'] < prereg.NOMINAL).sum())}")
    print(f"[pairs] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
