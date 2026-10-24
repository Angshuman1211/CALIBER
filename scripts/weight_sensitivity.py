"""Does the headline claim depend on the score weighting, and who flags less? (D76, D39)

Two questions that need no new bank and no GPU.

**Sensitivity (D76).** D54 found that 8 design patients cannot separate any weighting from
any other, so the project ships equal weights as a stated default. That is a gap only if
the result *moves* with the weighting. This sweeps ``w_free`` across the grid and
recomputes the paired coverage restoration at each point. A claim that holds across the
whole grid does not need a selected weight.

**Area at matched coverage (D39, D51).** Fixing the threshold and comparing coverage lets a
detector win by flagging almost nothing -- the nearest-operator arm posted the project's
smallest flag area at coverage 0.547. Fixing *coverage* and comparing area cannot be gamed
that way, and it stays reportable under D67 because it is a statement about operating
points rather than about ranking.

    python scripts/weight_sensitivity.py runs/<run_id>_by_patient \
        --low mri.random1d.R2.c0.08.s0 --high mri.random1d.R8.c0.08.s0
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from caliber.evaluation import (
    add_label_args,
    area_at_matched_coverage,
    derive_score_and_label,
    label_from_args,
    weight_sensitivity,
)
from caliber.scorebank import read_bank


def _area_table(raw: pd.DataFrame, calib_op: str, label, target: float,
                baseline: str, split: str) -> pd.DataFrame | None:
    scored = derive_score_and_label(raw, calib_op, **label.kwargs)
    frame = scored[scored["split"] == split]
    if frame.empty:
        return None
    detectors = {"CALIBER score": "score"}
    for col in (baseline, "s_free", "s_disagree"):
        if col in frame.columns:
            detectors[col] = col
    rows = []
    for oid, part in frame.groupby("operator_id"):
        table = area_at_matched_coverage(part, detectors, target)
        table.insert(0, "operator_id", str(oid))
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bank")
    ap.add_argument("--low", required=True, help="least aggressive operator (ascending)")
    ap.add_argument("--high", required=True, help="most aggressive operator (descending)")
    ap.add_argument("--alpha", type=float, default=0.10)
    add_label_args(ap)
    ap.add_argument("--cluster", default="patient_id")
    ap.add_argument("--baseline", default="s_dispersion",
                    help="extra column to include in the area comparison, if present")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--area-out", default=None)
    args = ap.parse_args(argv)

    raw = read_bank(args.bank)
    if args.cluster not in raw.columns:
        print(f"[weights] no {args.cluster!r} column; run scripts/add_patient_ids.py first",
              file=sys.stderr)
        return 1
    try:
        label = label_from_args(raw, args)
    except ValueError as exc:
        print(f"[weights] {exc}", file=sys.stderr)
        return 1
    print(f"[weights] label: {label.text}")
    nominal = 1.0 - args.alpha

    curves = []
    for direction, (calib_op, deploy_op) in (
        ("ascending", (args.low, args.high)),
        ("descending", (args.high, args.low)),
    ):
        try:
            result = weight_sensitivity(
                raw, calib_op, deploy_op, label_kwargs=label.kwargs, alpha=args.alpha,
                cluster_col=args.cluster, n_boot=args.boot,
            )
        except (ValueError, KeyError) as exc:
            print(f"[weights] {direction}: {exc}", file=sys.stderr)
            continue
        curve = result.curve.copy()
        curve.insert(0, "direction", direction)
        curves.append(curve)
        print(f"\n[weights] {direction}: calibrated at {calib_op}, deployed at {deploy_op}")
        print("[weights] gap = coverage(group-conditional) - coverage(transferred), paired "
              "within patient")
        print(f"{'w_free':>8}{'w_disag':>9}{'gap':>9}{'95% CI':>20}{'n_pos':>7}")
        for r in curve.itertuples(index=False):
            print(f"{r.w_free:>8.2f}{r.w_disagree:>9.2f}{r.gap:>+9.3f}"
                  f"{f'[{r.lo:+.3f}, {r.hi:+.3f}]':>20}{r.n_positive:>7}")
        print(f"[weights] {result.verdict()}")

    if not curves:
        print("[weights] neither direction could be swept on this bank", file=sys.stderr)
        return 1

    print(f"\n[weights] flag area at matched coverage {nominal:.2f} (D39 with D51's gate: "
          "coverage is fixed, so area cannot be bought by flagging nothing)")
    area = _area_table(raw, args.low, label, nominal, args.baseline, "test")
    if area is None:
        print("[weights] no test-split rows, so no operating-point comparison",
              file=sys.stderr)
    else:
        wide = area.pivot(index="operator_id", columns="detector", values="flag_area")
        print(wide.round(4).to_string())
        best = wide.idxmin(axis=1)
        print(f"[weights] smallest area per operator: {dict(best)}")
        print("[weights] read this per operator. Pooling averaged two opposite "
              "per-operator biases into 'exactly chance' once already (D60, retracted).")
        if args.area_out:
            area.to_csv(args.area_out, index=False)
            print(f"[weights] wrote {args.area_out}")

    if args.out:
        pd.concat(curves, ignore_index=True).to_csv(args.out, index=False)
        print(f"[weights] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
