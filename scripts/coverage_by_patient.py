"""Naive transfer against group-conditional calibration, both directions, by patient.

The O3 headline: does a threshold calibrated at one operator keep its promise at another?
D-056 requires both directions -- calibrating at the least aggressive operator and
deploying upward is the benign one (failure shows as over-flagging), and descending is
where coverage collapses. One direction alone either understates the risk or
cherry-picks.

Thresholds are fitted on ``split == "calib"`` patients and scored on ``split == "test"``
patients, and every 95% interval is a cluster bootstrap over **patients** (D-027), which
needs the ``patient_id`` column that ``add_patient_ids.py`` writes. By D-056's rule, an arm's
flag area means something only when its interval reaches nominal: a small area bought by
missing hallucinations is not efficiency.

    python scripts/add_patient_ids.py runs/<run_id>
    python scripts/coverage_by_patient.py runs/<run_id>_by_patient \
        --low mri.random1d.R2.c0.08.s0 --high mri.random1d.R8.c0.08.s0
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from caliber.calibration import apply_threshold, split_threshold
from caliber.evaluation import (
    add_label_args,
    cluster_bootstrap_coverage,
    derive_score_and_label,
    label_from_args,
)
from caliber.scorebank import read_bank


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bank")
    ap.add_argument("--low", required=True, help="least aggressive operator (ascending)")
    ap.add_argument("--high", required=True, help="most aggressive operator (descending)")
    ap.add_argument("--alpha", type=float, default=0.10)
    add_label_args(ap)
    ap.add_argument("--cluster", default="patient_id")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    raw = read_bank(args.bank)
    if args.cluster not in raw.columns:
        print(f"[coverage] no {args.cluster!r} column; run scripts/add_patient_ids.py first",
              file=sys.stderr)
        return 1
    nominal = 1.0 - args.alpha
    label = label_from_args(raw, args)
    print(f"[coverage] label: {label.text}")

    rows = []
    for direction, calib_op in (("ascending", args.low), ("descending", args.high)):
        df = derive_score_and_label(raw, calib_op, **label.kwargs)
        cal, test = df[df["split"] == "calib"], df[df["split"] == "test"]
        own_calib = cal[cal["operator_id"] == calib_op]
        naive_tau = split_threshold(own_calib["score"], own_calib["label"], alpha=args.alpha)
        for oid, part in test.groupby("operator_id"):
            own = cal[cal["operator_id"] == oid]
            group_tau = split_threshold(own["score"], own["label"], alpha=args.alpha)
            for method, tau in (("naive", naive_tau.threshold), ("group", group_tau.threshold)):
                flagged = apply_threshold(part, tau, "score")
                s = cluster_bootstrap_coverage(flagged["flag"], flagged["label"],
                                               flagged[args.cluster])
                rows.append({"direction": direction, "calibrated_at": calib_op,
                             "deployed_at": oid, "method": method,
                             "coverage": s["coverage"], "lo": s["lo"], "hi": s["hi"],
                             "flag_area": float(flagged["flag"].mean()),
                             "reaches_nominal": s["hi"] >= nominal})

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for direction, part in table.groupby("direction", sort=False):
        wide = part.pivot(index="deployed_at", columns="method",
                          values=["coverage", "lo", "hi", "flag_area"])
        print(f"\n[coverage] {direction}: calibrated at {part['calibrated_at'].iloc[0]}")
        print(f"{'deployed at':<36}{'naive':>8}{'95% CI':>18}{'area':>7}"
              f"{'group':>9}{'95% CI':>18}{'area':>7}")
        for oid, r in wide.iterrows():
            naive_ci = f"[{r[('lo', 'naive')]:.3f}, {r[('hi', 'naive')]:.3f}]"
            group_ci = f"[{r[('lo', 'group')]:.3f}, {r[('hi', 'group')]:.3f}]"
            print(f"{oid:<36}{r[('coverage', 'naive')]:>8.3f}{naive_ci:>18}"
                  f"{r[('flag_area', 'naive')]:>7.3f}{r[('coverage', 'group')]:>9.3f}"
                  f"{group_ci:>18}{r[('flag_area', 'group')]:>7.3f}")
    failing = table[(table["method"] == "naive") & ~table["reaches_nominal"]]
    print(f"\n[coverage] naive transfer misses nominal ({nominal:.2f}) with its whole interval "
          f"in {len(failing)} of {int((table['method'] == 'naive').sum())} cells")
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"[coverage] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
