"""Select the score's component weights on a bank's design split (T1.1, D28).

    PYTHONPATH=src python3 scripts/select_weights.py runs/<run_id> \
        --calibration-operator mri.random1d.R2.c0.08.s0

Closes the last unset number in the score definition. The rule is pre-registered in
`caliber.scoring.weights` and this script only applies it, so running it on a new bank
cannot change what the rule is — which is the point of having written it down first.

Needs a bank with `split=design` rows, which `configs/experiments/w04_design_split.yaml`
produces in **~1.9 h** (8 design volumes x 3 operators, at T2.3's measured 288 s per
reconstruction):

    caliber configs/experiments/w04_design_split.yaml

`design_fraction=0.10` is Track 1's ruling (D48/D-054) and carves out of the calibration
range, so the test side is untouched and coverage already measured stays comparable
(D-051). `keep_splits: design` is what keeps the run at 1.9 h rather than 47.8 — without
it the loader labels every volume and reconstructs all 199.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from caliber.evaluation.bank import derive_score_and_label
from caliber.scorebank import read_bank, read_manifest
from caliber.scoring.weights import DEFAULT_GRID, select_weights


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Select score weights on the design split")
    ap.add_argument("bank")
    ap.add_argument("--calibration-operator", default=None)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--tau", type=float, default=1.75, help="D37's ratified value")
    ap.add_argument("--cluster", default="sample_id")
    ap.add_argument("--points", type=int, default=len(DEFAULT_GRID),
                    help="sweep resolution; more points does not make a flat curve sharper")
    ap.add_argument("--out", default=None, help="write the sweep here as CSV")
    args = ap.parse_args(argv)

    df = read_bank(args.bank)
    provenance = read_manifest(args.bank).get("provenance") or {}
    calib_op = args.calibration_operator or sorted(df["operator_id"].unique())[0]
    df = derive_score_and_label(df, calib_op, threshold=args.tau)

    print(f"[weights] bank             {args.bank}")
    print(f"[weights] reportable       {provenance.get('reportable')}")
    print(f"[weights] calibration op   {calib_op}")
    print(f"[weights] tau              {args.tau}  (D37)")
    if not provenance.get("reportable", False):
        print(
            "[weights] the weights selected here are DEVELOPMENT values. The score's "
            "definition is only pinned by a selection on a reportable bank (D-016, D-035).",
            file=sys.stderr,
        )

    grid = tuple(round(i / (args.points - 1), 4) for i in range(args.points))
    try:
        selection = select_weights(
            df, calib_op, grid=grid, alpha=args.alpha, cluster_col=args.cluster
        )
    except (ValueError, KeyError) as exc:
        print(f"[weights] {exc}", file=sys.stderr)
        return 1

    print(f"\n[weights] sweep on the design split ({selection.n_design_clusters} clusters)")
    print(selection.curve.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n[weights] SELECTED  {selection.note()}")
    print(
        "[weights] the level of flag area above is optimistic -- the design rows are both "
        "calibrated and\n          evaluated on. The comparison between points is what the "
        "rule uses, and that is unbiased."
    )
    if selection.at_endpoint:
        print(
            "[weights] an endpoint means one component earned no weight. That is a finding "
            "about the\n          score's composition and belongs in the write-up, not a "
            "value to nudge off the edge."
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        selection.curve.to_csv(args.out, index=False)
        print(f"[weights] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
