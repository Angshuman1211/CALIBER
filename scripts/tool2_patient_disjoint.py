"""Tool 2 hold-out with calibration and evaluation on different patients.

D-056 asked Track 2 to reproduce Tool 2's verdict on the nine-operator grid. Track 1's
``tool2_holdout.py`` passes ``split_col="__none__"``: every arm calibrates on all rows of the
other operators and is scored on all rows of the held-out one. The design is crossed
(D-049), so those are the **same patients** on both sides -- the leak D-027 closed for
calibration and test. It also makes the ``matched`` arm in-sample, which is why it lands on
exactly 0.900.

This is that script with one change: calibration uses ``split == "calib"`` patients only and
evaluation uses ``split == "test"`` patients only. Scoring, distance, bandwidth, the four
arms, the interior/endpoint rule and the metrics are imported from ``tool2_holdout`` and not
re-implemented, so any difference in the verdict is the patient separation and nothing else.

    python scripts/add_patient_ids.py runs/<run_id>
    python scripts/tool2_patient_disjoint.py runs/<run_id>_by_patient
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from tool2_holdout import evaluate, group_of, rebuild_operators

from caliber.calibration import (
    apply_flags,
    calibrate_mondrian,
    calibrate_naive,
    calibrate_weighted,
)
from caliber.evaluation.bank import derive_score_and_label
from caliber.operators.distance import compute_distance
from caliber.scorebank import read_bank, read_manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bank")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--tau", type=float, default=1.75, help="D37's ratified value")
    ap.add_argument("--distance", default="null_space_projection")
    ap.add_argument("--normalise", default="ambient")
    ap.add_argument("--cluster", default="patient_id")
    ap.add_argument("--bandwidth", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    df = read_bank(args.bank)
    if args.cluster not in df.columns:
        print(f"[tool2-pd] no {args.cluster!r} column; run scripts/add_patient_ids.py first",
              file=sys.stderr)
        return 1
    if set(df["split"].unique()) < {"calib", "test"}:
        print("[tool2-pd] needs a bank with calib and test sides (split: auto)", file=sys.stderr)
        return 1

    ops = rebuild_operators(read_manifest(args.bank))
    operators = sorted(df["operator_id"].unique())
    # Track 1's interior rule, unchanged: some operator measures more, some measures less.
    measured = {o: int(ops[o].descriptor().measurement_dim) for o in operators}
    interior = {
        o: any(measured[x] > measured[o] for x in operators if x != o)
        and any(measured[x] < measured[o] for x in operators if x != o)
        for o in operators
    }

    rows = []
    for held_out in operators:
        scored = derive_score_and_label(df, held_out, threshold=args.tau)
        dep = group_of(scored, held_out)
        others = [o for o in operators if o != held_out]
        distance_of = {
            group_of(scored, o): float(compute_distance(
                ops[held_out], ops[o], measure=args.distance, normalise=args.normalise))
            for o in others
        }
        nearest = min(others, key=lambda o: distance_of[group_of(scored, o)])

        calib = scored[scored["split"] == "calib"]
        pool = calib[calib["operator_id"] != held_out]
        deploy = scored[(scored["operator_id"] == held_out) & (scored["split"] == "test")].copy()

        arms = {
            "matched": calibrate_naive(calib, calibration_group=dep, alpha=args.alpha,
                                       split_col="__none__"),
            "mondrian": calibrate_mondrian(pool, alpha=args.alpha, split_col="__none__"),
            "nearest": calibrate_naive(pool, calibration_group=group_of(scored, nearest),
                                       alpha=args.alpha, split_col="__none__"),
            "weighted": calibrate_weighted(pool, deployment_group=dep, distance_of=distance_of,
                                           alpha=args.alpha, bandwidth=args.bandwidth,
                                           split_col="__none__"),
        }
        for name, report in arms.items():
            metrics = evaluate(apply_flags(deploy, report, out_col="flag"), "flag", args.cluster)
            rows.append({"held_out": held_out, "interior": interior[held_out], "arm": name,
                         **metrics, "valid": metrics["hi"] >= 1.0 - args.alpha})

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for scope, part in (("interior", table[table["interior"]]),
                        ("endpoint", table[~table["interior"]])):
        summary = part.groupby("arm").agg(
            coverage=("coverage", "mean"), flag_area=("flag_area", "mean"),
            folds_valid=("valid", "sum"), folds=("valid", "size"))
        print(f"[tool2-pd] {scope} folds (calibrate on calib patients, score on test patients)")
        print(summary.round(4).to_string(), "\n")
    print(table[table["interior"]][["held_out", "arm", "coverage", "lo", "hi", "flag_area",
                                    "valid"]].round(4).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\n[tool2-pd] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
