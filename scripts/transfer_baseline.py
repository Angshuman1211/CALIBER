"""Does the baseline's threshold transfer across operators as well as CALIBER's? (D73)

The dispersion baseline ties ``s_disagree`` on AUROC at every operator (T2.6), and D67
ruled AUROC on this bank unreportable because the label is nearly a function of the score.
So the baseline comparison has only ever been run on a disqualified axis. This runs it on
the axis the project actually claims: **coverage under operator shift**.

Per deployment operator, each detector gets its own split-conformal threshold twice --
fitted at the calibration operator and fitted in place -- and the coverage lost between
them is that detector's transfer penalty. The two penalties are then differenced, paired on
one bootstrap resample of patients. See :mod:`caliber.evaluation.transfer` for why a
difference of penalties is comparable across detectors when their coverages are not.

Both directions are run (D-056): ascending from the least aggressive operator, descending
from the most aggressive.

    # with Track 2's regenerated dispersion table (the GPU step is already done)
    python scripts/transfer_baseline.py runs/<run_id>_by_patient \
        --low mri.random1d.R2.c0.08.s0 --high mri.random1d.R8.c0.08.s0 \
        --dispersion runs/<analysis>/dispersion.parquet

    # or against a component already in the bank, needing no extra file
    python scripts/transfer_baseline.py runs/<run_id>_by_patient \
        --low ... --high ... --baseline s_disagree
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from caliber.evaluation import (
    add_label_args,
    derive_score_and_label,
    label_from_args,
)
from caliber.evaluation.transfer import PENALTY_FLOOR, transfer_table, transfer_verdict
from caliber.scorebank import read_bank

#: The merge keys of ``dispersion_baseline.py``'s output, which this must match exactly.
KEYS = ["sample_id", "slice_idx", "operator_id", "region_id"]

#: Shown in the table when a detector is not comparable at an operator (D51's gate).
NOT_INTERPRETABLE = "  (in-group arm misses nominal -- penalty not comparable)"


def _load_baseline(df: pd.DataFrame, args) -> tuple[pd.DataFrame, str, str]:
    """Attach the baseline score column, from a dispersion file or from the bank."""
    if args.dispersion:
        disp = pd.read_parquet(args.dispersion)
        missing = [k for k in KEYS + ["s_dispersion"] if k not in disp.columns]
        if missing:
            raise ValueError(
                f"{args.dispersion} has no {missing} column(s); it should be the output of "
                "scripts/dispersion_baseline.py generate"
            )
        merged = df.merge(disp[KEYS + ["s_dispersion"]], on=KEYS, how="inner")
        if merged.empty:
            raise ValueError(
                "no bank row matches the dispersion file on "
                f"{KEYS} -- are they from the same run?"
            )
        kept = len(merged) / len(df) if len(df) else 0.0
        print(f"[transfer] merged dispersion: {len(merged):,} of {len(df):,} bank rows "
              f"({kept:.0%}) have a dispersion value")
        return merged, "s_dispersion", "posterior dispersion (full image)"
    if args.baseline not in df.columns:
        raise ValueError(
            f"the bank has no {args.baseline!r} column; pass --dispersion for the "
            "regenerated full-image dispersion, or name a column that is in the bank"
        )
    return df, args.baseline, args.baseline


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bank")
    ap.add_argument("--low", required=True, help="least aggressive operator (ascending)")
    ap.add_argument("--high", required=True, help="most aggressive operator (descending)")
    ap.add_argument("--alpha", type=float, default=0.10)
    add_label_args(ap)
    ap.add_argument("--dispersion", default=None,
                    help="parquet from dispersion_baseline.py generate; gives s_dispersion")
    ap.add_argument("--baseline", default="s_disagree",
                    help="bank column to use as the baseline when --dispersion is absent")
    ap.add_argument("--cluster", default="patient_id")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--floor", type=float, default=PENALTY_FLOOR,
                    help=f"smallest penalty difference worth claiming (default {PENALTY_FLOOR})")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    raw = read_bank(args.bank)
    if args.cluster not in raw.columns:
        print(f"[transfer] no {args.cluster!r} column; run scripts/add_patient_ids.py first",
              file=sys.stderr)
        return 1
    try:
        label = label_from_args(raw, args)
    except ValueError as exc:
        print(f"[transfer] {exc}", file=sys.stderr)
        return 1
    print(f"[transfer] label: {label.text}")
    print(f"[transfer] alpha {args.alpha:g}, nominal coverage {1 - args.alpha:.2f}, "
          f"penalty floor {args.floor:g}")

    all_rows, any_usable = [], False
    for direction, calib_op in (("ascending", args.low), ("descending", args.high)):
        try:
            scored = derive_score_and_label(raw, calib_op, **label.kwargs)
            frame, baseline_col, baseline_name = _load_baseline(scored, args)
            table, results = transfer_table(
                frame, baseline_col, "score", calib_op,
                cluster_col=args.cluster, alpha=args.alpha, n_boot=args.boot,
                floor=args.floor, baseline_name=baseline_name, caliber_name="CALIBER score",
            )
        except (ValueError, KeyError) as exc:
            print(f"[transfer] {direction}: {exc}", file=sys.stderr)
            continue

        verdict = transfer_verdict(results)
        any_usable = any_usable or verdict["n_interpretable"] > 0
        table.insert(0, "direction", direction)
        all_rows.append(table)

        print(f"\n[transfer] {direction}: calibrated at {calib_op}")
        print("[transfer] penalty = coverage(in-group threshold) - "
              "coverage(transferred threshold), per detector")
        print("[transfer] delta = penalty(baseline) - penalty(CALIBER); positive favours "
              "CALIBER")
        print(f"{'deployed at':<34}{'base':>8}{'CALIBER':>9}{'delta':>8}"
              f"{'95% CI':>18}  verdict")
        for r in results:
            ci = f"[{r.lo:+.3f}, {r.hi:+.3f}]"
            print(f"{r.caliber.deployment_operator:<34}{r.baseline.penalty:>8.3f}"
                  f"{r.caliber.penalty:>9.3f}{r.delta:>+8.3f}{ci:>18}  {r.verdict}")
            if not r.interpretable:
                print(NOT_INTERPRETABLE)
                for d in (r.baseline, r.caliber):
                    if not d.reaches_nominal:
                        print(f"    {d.detector}: in-group coverage {d.coverage_group:.3f} "
                              f"[{d.coverage_group_lo:.3f}, {d.coverage_group_hi:.3f}] "
                              f"against nominal {d.nominal:.2f}")
                    if d.trivial:
                        print(f"    {d.detector}: in-group arm flags "
                              f"{d.flag_area_group:.1%} of all regions")
        # paired.py's rule: the variance reduction is printed so it can come back at 1 and
        # be believed. Pairing cancels the patient's difficulty only if patients differ.
        gains = [r.se_unpaired / r.se_paired for r in results
                 if r.se_paired and r.se_paired > 0]
        if gains:
            print(f"[transfer] pairing narrowed the interval by "
                  f"{min(gains):.2f}x to {max(gains):.2f}x across these operators "
                  "(1.00x means it bought nothing, which is a fact about the patients)")
        print(f"[transfer] {verdict['headline']}")
        if verdict["undetermined"]:
            print(f"[transfer] the interval is +/-{verdict['floor_for_equivalence']:.3f} at "
                  "its widest; more calibration patients narrow it, and the penalty's noise "
                  "falls with them")

    if not all_rows:
        print("[transfer] neither direction could be measured on this bank", file=sys.stderr)
        return 1
    if not any_usable:
        print("\n[transfer] no operator had both detectors reaching nominal in-group, so no "
              "comparison here is interpretable; this is a result about the bank, not the "
              "detectors", file=sys.stderr)
    if args.out:
        pd.concat(all_rows, ignore_index=True).to_csv(args.out, index=False)
        print(f"\n[transfer] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
