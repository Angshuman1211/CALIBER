"""T1-B: fit the gap curve from operator distance and try to beat three baselines (D88).

Reads T2-A3's cell table (``pair_coverage_by_split.py``'s output directory), the T1-0
signed distance matrix, and the operator manifest that defines the grid. Writes the four
tables T1-B reports and prints the verdict.

**The gap is a difference, and that is why it can be read off a draw the level cannot.**
D86 moved T1-A's unsafe label to the across-split mean because one calibration draw
displaces every cell together, so a level read from one draw carries that displacement.
A gap is ``coverage(b -> b) - coverage(a -> b)``: a displacement common to both cells
cancels to first order. So the two arms should agree far better here than they do for the
level, and the sensitivity table is printed every time to show whether they actually do --
if they do not, the displacement is not common-mode and D86's own reasoning needs revisiting.

Both gap definitions and both family scopes are always printed. Neither is selectable:
:mod:`caliber.evaluation.prereg` names which is primary.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from caliber.evaluation import gap as G
from caliber.evaluation import prereg
from caliber.evaluation.reliability import across_split_columns

KEYS = ["calibration", "deployment"]


def _load_distances(path: str, measure: str) -> dict[tuple[str, str], float]:
    """``(a, b) -> signed distance``, the reverse direction filled by antisymmetry."""
    out: dict[tuple[str, str], float] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["measure"] != measure:
                continue
            try:
                value = float(row["value"])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
            a, b = row["calibration"], row["deployment"]
            out[(a, b)] = value
            out.setdefault((b, a), -value)
    if not out:
        raise SystemExit(f"[t1b] no finite {measure!r} rows in {path}")
    return out


def _cells(directory: str) -> pd.DataFrame:
    """The full cell table, diagonal included, carrying both coverage arms."""
    root = Path(directory)
    reg_path = root / "pair_coverage_registry.csv"
    split_path = root / "pair_coverage_by_split.csv"
    if not reg_path.exists():
        raise SystemExit(f"[t1b] {reg_path} is missing; run pair_coverage_by_split.py first")
    registry = pd.read_csv(reg_path)
    missing = [c for c in (*KEYS, "coverage") if c not in registry.columns]
    if missing:
        raise SystemExit(f"[t1b] {reg_path} is missing {missing}")
    cells = registry[[*KEYS, "coverage"]].copy()
    if split_path.exists():
        per_split = pd.read_csv(split_path)
        col = "salt" if "salt" in per_split.columns else "split"
        across = across_split_columns(per_split, overlap=0.0, split_col=col)
        cells = cells.merge(
            across[[*KEYS, "coverage_across", "n_splits"]], on=KEYS, how="left"
        )
    else:
        print(f"[t1b] {split_path} absent; the across-split arm cannot be formed",
              file=sys.stderr)
    return cells


def _print(frame: pd.DataFrame, cols: list[str], title: str) -> None:
    present = [c for c in cols if c in frame.columns]
    print(f"\n== {title} " + "=" * max(0, 76 - len(title)))
    if frame.empty:
        print("  (nothing to report)")
        return
    print(frame[present].to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pair_coverage_dir", help="pair_coverage_by_split.py output directory")
    ap.add_argument("--distances", required=True, help="T1-0 distance matrix CSV")
    ap.add_argument("--manifest", required=True, help="manifest.json defining the grid")
    ap.add_argument("--measure", default=prereg.GAP_PREDICTION_DISTANCE,
                    help="frozen; overriding it is a new analysis")
    ap.add_argument("--out", default=None, help="directory for the CSVs")
    ap.add_argument("--n-boot", type=int, default=prereg.N_BOOT)
    args = ap.parse_args(argv)

    print(f"[t1b] prereg {prereg.fingerprint()} frozen {prereg.FROZEN_ON} "
          f"(amended {', '.join(d for d, _ in prereg.AMENDMENTS)})")
    print(f"[t1b] distance={args.measure} form={prereg.GAP_PREDICTION_FORM} "
          f"gap={prereg.GAP_DEFINITION} metric={prereg.GAP_ERROR_METRIC} "
          f"cluster={prereg.GAP_CLUSTER_UNIT}")
    print("[t1b] what Track 1 already knew when the gap was declared (D88):")
    for item in prereg.GAP_DECLARATION_KNEW:
        print(f"         - {item}")

    meta = G.operator_metadata(args.manifest)
    cells = _cells(args.pair_coverage_dir)
    unknown = sorted(set(cells[KEYS].astype(str).to_numpy().ravel()) - set(meta))
    if unknown:
        raise SystemExit(
            f"[t1b] the cell table names operators the grid does not: {unknown}. The "
            "distance matrix and the bank must be the same nine operators."
        )
    distances = _load_distances(args.distances, args.measure)

    arms = [c for c in ("coverage_across", "coverage") if c in cells.columns]
    primary_arm = arms[0]
    results: dict[tuple[str, str], dict] = {}
    for arm in arms:
        for definition in prereg.GAP_DEFINITION_SENSITIVITY:
            frame = cells.dropna(subset=[arm]).copy()
            try:
                frame["gap"] = G.gap_column(frame, definition, coverage_col=arm)
            except KeyError as exc:
                print(f"[t1b] {definition} on {arm}: skipped - {exc}", file=sys.stderr)
                continue
            off = frame[frame["calibration"] != frame["deployment"]].reset_index(drop=True)
            off["signed_distance"] = [
                distances.get((a, b), float("nan"))
                for a, b in zip(off["calibration"], off["deployment"])
            ]
            n_missing = int(off["signed_distance"].isna().sum())
            if n_missing:
                print(f"[t1b] {n_missing} of {len(off)} pairs have no {args.measure} "
                      "distance; they cannot be predicted and are dropped from both arms",
                      file=sys.stderr)
            results[(arm, definition)] = G.gap_prediction(
                off, meta, n_boot=args.n_boot, seed=0
            )
            results[(arm, definition)]["n_pairs_in"] = len(off)

    if not results:
        print("[t1b] no arm could be formed; nothing to report")
        return 2

    key = (primary_arm, prereg.GAP_DEFINITION)
    if key not in results:
        print(f"[t1b] the frozen combination {key} could not be formed")
        return 2
    primary = results[key]
    print(f"\n[t1b] primary arm: gap={prereg.GAP_DEFINITION} on {primary_arm} over "
          f"{primary['n_pairs_in']} ordered pairs")

    _print(primary["accuracy"],
           ["protocol", "scope", "predictor", "n_pairs", "n_observations", "mae", "rmse",
            "mae_over_gap_sd", "extrapolated", "defined"],
           "held-out accuracy (mae_over_gap_sd above 1 is worse than saying nothing)")
    _print(primary["accuracy_by_fold"],
           ["protocol", "fold", "scope", "predictor", "n_pairs", "mae", "extrapolated"],
           "by fold: 'fit on one family, predict the other' in both directions")
    _print(primary["comparisons"],
           ["protocol", "scope", "baseline", "n_common", "n_dropped", "mae_distance",
            "mae_baseline", "delta_mae", "delta_lo", "delta_hi", "verdict"],
           f"distance vs each baseline, {prereg.GAP_CLUSTER_UNIT}-clustered interval")
    _print(primary["verdict"],
           ["protocol", "scope", "n_baselines_defined", "n_baselines_beaten", "unbeaten",
            "reportable"],
           "verdict (conjunctive: every defined baseline must be beaten)")

    high = primary["accuracy"]
    high = high[(high["predictor"] == "signed_distance") & (high["extrapolated"] > 0.5)]
    if len(high):
        print(f"\n[t1b] WARNING: {len(high)} reported cell(s) are over half extrapolated. "
              "Outside its training range a monotone fit is clamped, so it is a constant "
              "predictor there and must not be read as skill.")

    rows = []
    for (arm, definition), out in sorted(results.items()):
        for _, r in out["verdict"].iterrows():
            rows.append({
                "coverage_arm": arm,
                "gap_definition": definition,
                "frozen_primary": bool(
                    arm == primary_arm and definition == prereg.GAP_DEFINITION
                ),
                **r.to_dict(),
            })
    sensitivity = pd.DataFrame(rows)
    _print(sensitivity,
           ["coverage_arm", "gap_definition", "frozen_primary", "protocol", "scope",
            "n_baselines_beaten", "unbeaten", "reportable"],
           "sensitivity: both coverage arms x both gap definitions (D88)")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stamp = {"prereg_fingerprint": prereg.fingerprint(), "coverage_arm": primary_arm}
        for name in ("accuracy", "accuracy_by_fold", "comparisons", "verdict"):
            primary[name].assign(**stamp).to_csv(out / f"t1b_{name}.csv", index=False)
        sensitivity.assign(prereg_fingerprint=prereg.fingerprint()).to_csv(
            out / "t1b_sensitivity.csv", index=False
        )
        print(f"\n[t1b] wrote {out}/t1b_accuracy.csv, t1b_accuracy_by_fold.csv, "
              "t1b_comparisons.csv, t1b_verdict.csv, t1b_sensitivity.csv")

    good = primary["verdict"]["reportable"]
    print(f"\n[t1b] reportable on {int(good.sum())} of {len(good)} protocol x scope cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
