"""The curve-collapse test across modalities (T1.4, Sec. 21 R5).

    PYTHONPATH=src python3 scripts/curve_collapse.py \
        --mri-coverage <mri_coverage.csv> --mri-distances <mri_distances.csv> \
        --ct-coverage  <ct_coverage.csv>  --ct-distances  <ct_distances.csv>

The cross-domain claim has a strong form -- the two modalities' coverage-gap curves collapse
onto a single curve against operator distance, so the gap is a function of how far the
operator moved and not of the modality -- and a weak one, that the same failure and the same
fix appear in both. This runs the test that separates them, because "we looked at the figure"
is not a test and a reviewer will ask for the statistic.

D70 already expects the strong form to fail: the gap is ordered by signed distance on CT
(rho = +0.976) and barely on MRI (+0.316), because MRI's worst failures sit at matched
acceleration across mask families where the scalar distance is ~0.

**The verdict rests on magnitude, not significance.** The F-test is reported and does not
decide: it asks whether the fits differ at all, and it is hypersensitive where residuals are
small -- a control with slopes 0.3025 and 0.3029 came back at p = 0.031. What decides is
whether one curve is wrong by more than the measurement noise, against D69's 0.05 coverage
floor.

Inputs are the harness's own coverage tables plus the cached distance CSVs, so no bank is
read and nothing is recomputed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from caliber.evaluation.collapse import NOISE_FLOOR, curve_collapse_test


def points_for(
    coverage_csv: str, distances_csv: str, modality: str, method: str, measure: str,
    nominal: float,
) -> pd.DataFrame:
    """One row per deployment operator: signed distance and the measured coverage gap."""
    coverage = pd.read_csv(coverage_csv)
    if "method" in coverage.columns:
        coverage = coverage[coverage["method"].astype(str) == method]
    if coverage.empty:
        raise ValueError(f"{coverage_csv} has no rows for method {method!r}")

    dist = pd.read_csv(distances_csv)
    dist = dist[dist["measure"].astype(str) == measure]
    if dist.empty:
        have = sorted(pd.read_csv(distances_csv)["measure"].unique())
        raise ValueError(f"{distances_csv} has no {measure!r} rows; it holds {have}")
    signed = {
        (str(r.calibration), str(r.deployment)): float(r.value)
        for r in dist.itertuples(index=False)
    }

    rows = []
    for r in coverage.itertuples(index=False):
        operator = str(r.operator_id)
        matches = [v for (a, b), v in signed.items() if b == operator]
        if not matches:
            continue
        rows.append(
            {
                "modality": modality,
                "operator_id": operator,
                "distance": matches[0],
                "gap": nominal - float(r.coverage),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Curve-collapse test across modalities")
    ap.add_argument("--mri-coverage", required=True)
    ap.add_argument("--mri-distances", required=True)
    ap.add_argument("--ct-coverage", required=True)
    ap.add_argument("--ct-distances", required=True)
    ap.add_argument("--method", default="naive_transferred",
                    help="the arm whose gap is tested; the fix should show no trend at all")
    ap.add_argument("--measure", default="null_space_growth")
    ap.add_argument("--nominal", type=float, default=0.90)
    ap.add_argument("--noise-floor", type=float, default=NOISE_FLOOR,
                    help="coverage discrepancy below which one curve counts as adequate (D69)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    try:
        points = pd.concat(
            [
                points_for(args.mri_coverage, args.mri_distances, "mri", args.method,
                           args.measure, args.nominal),
                points_for(args.ct_coverage, args.ct_distances, "ct", args.method,
                           args.measure, args.nominal),
            ],
            ignore_index=True,
        )
    except (ValueError, KeyError) as exc:
        print(f"[collapse] {exc}", file=sys.stderr)
        return 1

    if points.empty:
        print(
            "[collapse] no deployment operator in either coverage table has a distance in "
            "its CSV. Check that the distances were computed from the same calibration "
            "operator the coverage table was calibrated at.",
            file=sys.stderr,
        )
        return 1

    print(f"[collapse] method    {args.method}")
    print(f"[collapse] measure   {args.measure} (signed, D-053)")
    print(f"[collapse] points    {len(points)} "
          f"({', '.join(f'{m}={n}' for m, n in points.modality.value_counts().items())})")
    result = curve_collapse_test(points, noise_floor=args.noise_floor)
    print("\n[collapse] Sec. 21 R5 -- does one curve fit both modalities?")
    print(result.summary())

    if not result.collapses:
        print(
            "\n[collapse] So the manuscript claims the weak form: the same failure and the "
            "same fix in both\n           modalities, with the distance ordering "
            "demonstrated where the distance is a genuine\n           scalar. Claiming "
            "collapse would be claiming something measured to be false (D70)."
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        points.to_csv(args.out, index=False)
        print(f"[collapse] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
