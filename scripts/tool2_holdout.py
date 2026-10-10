"""Leave-one-operator-out: does Tool 2 beat its own baselines? (T1.4, answers Q4)

    PYTHONPATH=src python3 scripts/tool2_holdout.py runs/<run_id>

Tool 2 — operator-distance weighted recalibration — is the project's primary contribution
(§9) and until this script existed it appeared in **no** analysis path. It was implemented,
unit-tested on synthetic banks, and never once pointed at a bank. Q4 asked for the
experiment that shows it beating its own Mondrian baseline; this is that experiment, and it
is a re-analysis of a cached bank, so it costs no GPU.

## The design

For each operator in turn, pretend it was never calibrated on — it is the **deployment**
operator — and compare four ways of thresholding it:

| arm | where the threshold comes from | available at deployment? |
|---|---|---|
| `matched` | the held-out operator's own calibration rows | **no** — the oracle upper bound |
| `mondrian` | its own group, which does not exist | yes, and it has nothing to use |
| `nearest` | the single closest operator's group, by D19 distance | yes |
| `weighted` | every other group, weighted by D19 distance (Tool 2) | yes |

`matched` is the ceiling and is reported to show how much is left on the table.
`mondrian` is the honest floor: `apply_flags` flags an uncalibrated group **entirely**,
which is conservative — it costs efficiency, never validity — so its flagged area goes to
1.0 and it is exactly the case Tool 2 exists for. `nearest` is the baseline that matters
most, because it is what a sensible person would do without Tool 2, and if Tool 2 cannot
beat *it* then the weighting earns nothing.

## Validity first, then flagged area — and this corrects D39

D39 ruled that component and method comparisons are made on flagged area rather than power,
because a conformal threshold pins the miss rate at α. **That has a precondition this
experiment violates by design.** The miss rate is pinned only when the calibration scores
are exchangeable with the deployment scores, and operator shift is precisely the failure of
that exchangeability — the thing the project exists to study. Here coverage is *not* pinned:
running this harness, the nearest-operator arm averaged coverage 0.547 while posting the
smallest flagged area of any arm, because flagging almost nothing is cheap when you have
stopped catching anything.

So an arm competes on area **only if it is valid**. Each arm is marked valid when the upper
end of its coverage interval reaches nominal, and an invalid arm's small flagged area is
reported as the symptom it is. D39 stands where it was argued — within one calibration
regime, comparing score components — and does not transfer to across-regime comparisons.

## Interpolation and extrapolation are different questions

Distance weighting **cannot extrapolate**. It reweights the calibration operators it has; if
every one of them sits on the same side of the deployment operator, no reweighting moves the
threshold past them. Folds are therefore split into *interior* deployment operators — some
calibration operator measures more and some measures less — and *endpoint* ones. Q4's
"off-grid operators no group covers" has to mean *between* grid points for the weighting to
have anything to work with, and the endpoint folds are reported to show what happens when it
does not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from caliber.calibration import (
    apply_flags,
    calibrate_mondrian,
    calibrate_naive,
    calibrate_weighted,
    certification_note,
)
from caliber.evaluation import flag_area, miss_rate, region_fdr
from caliber.evaluation.bank import cluster_bootstrap_coverage, derive_score_and_label
from caliber.operators.distance import compute_distance
from caliber.operators.registry import build_operator
from caliber.scorebank import read_bank, read_manifest


def rebuild_operators(manifest: dict) -> dict:
    ops = {}
    for row in manifest["operators"]:
        spec = {"modality": row["modality"], **row["params"]}
        ops[row["operator_id"]] = build_operator(spec, tuple(row["image_shape"]))
    return ops


def group_of(df: pd.DataFrame, operator_id: str) -> str:
    return str(df[df["operator_id"] == operator_id]["calib_group"].iloc[0])


def evaluate(frame: pd.DataFrame, flag_col: str, cluster_col: str) -> dict[str, float]:
    labels = frame["label"].to_numpy().astype(bool)
    flags = frame[flag_col].to_numpy().astype(bool)
    stats = cluster_bootstrap_coverage(flags, labels, frame[cluster_col])
    return {
        "coverage": stats["coverage"],
        "lo": stats["lo"],
        "hi": stats["hi"],
        "flag_area": flag_area(flags),
        "fdr": region_fdr(flags, labels),
        "miss_rate": miss_rate(flags, labels),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Leave-one-operator-out Tool 2 comparison")
    ap.add_argument("bank")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--tau", type=float, default=1.75, help="D37's ratified value")
    ap.add_argument("--distance", default="null_space_projection",
                    help="D19's ratified metric; the weighting needs a metric, not D47's "
                         "signed axis")
    ap.add_argument("--normalise", default="ambient")
    ap.add_argument("--cluster", default="sample_id")
    ap.add_argument("--bandwidth", type=float, default=None,
                    help="kernel bandwidth; default is the median non-zero distance (D33)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    df = read_bank(args.bank)
    manifest = read_manifest(args.bank)
    provenance = manifest.get("provenance") or {}
    operators = sorted(df["operator_id"].unique())
    if len(operators) < 3:
        print(
            f"[tool2] leave-one-out needs at least 3 operators; this bank has "
            f"{len(operators)}. With two, holding one out leaves a single group and "
            "weighting has nothing to weight.",
            file=sys.stderr,
        )
        return 1

    ops = rebuild_operators(manifest)
    # A deployment operator is interior if some calibration operator measures more of
    # k-space and some measures less. Weighting can only interpolate between the regimes
    # it holds, so this distinction decides what the fold is even testing.
    measured = {o: int(ops[o].descriptor().measurement_dim) for o in operators}
    interior = {
        o: any(measured[x] > measured[o] for x in operators if x != o)
        and any(measured[x] < measured[o] for x in operators if x != o)
        for o in operators
    }
    print(f"[tool2] bank             {args.bank}")
    print(f"[tool2] reportable       {provenance.get('reportable')}")
    print(f"[tool2] operators        {len(operators)}")
    print(f"[tool2] distance         {args.distance}/{args.normalise} (D19)")
    if not provenance.get("reportable", False):
        print(
            "[tool2] NOT REPORTABLE — these numbers exercise the comparison, they do not "
            "settle it.",
            file=sys.stderr,
        )

    rows: list[dict[str, object]] = []
    notes: list[str] = []
    for held_out in operators:
        # Scales are frozen on the *held-out* operator's complement: at deployment the
        # held-out operator is the one you do not have calibration data for, so its own
        # scales are not available either. Freezing on the nearest available group is the
        # honest choice and matches what D27 forbids — re-estimating per deployment.
        scored = derive_score_and_label(df, held_out, threshold=args.tau)
        deployment_group = group_of(scored, held_out)
        others = [o for o in operators if o != held_out]

        distance_of = {
            group_of(scored, o): float(
                compute_distance(
                    ops[held_out], ops[o], measure=args.distance, normalise=args.normalise
                )
            )
            for o in others
        }
        nearest = min(others, key=lambda o: distance_of[group_of(scored, o)])

        # Everything outside the held-out operator is calibration data; the held-out
        # operator's rows are the evaluation set for every arm, so the arms are compared
        # on identical rows (the pairing D45 argues for).
        pool = scored[scored["operator_id"] != held_out].copy()
        deploy = scored[scored["operator_id"] == held_out].copy()

        arms = {}
        arms["matched"] = calibrate_naive(
            scored, calibration_group=deployment_group, alpha=args.alpha, split_col="__none__"
        )
        arms["mondrian"] = calibrate_mondrian(pool, alpha=args.alpha, split_col="__none__")
        arms["nearest"] = calibrate_naive(
            pool, calibration_group=group_of(scored, nearest), alpha=args.alpha,
            split_col="__none__",
        )
        arms["weighted"] = calibrate_weighted(
            pool, deployment_group=deployment_group, distance_of=distance_of,
            alpha=args.alpha, bandwidth=args.bandwidth, split_col="__none__",
        )

        for name, report in arms.items():
            flagged = apply_flags(deploy, report, out_col="flag")
            metrics = evaluate(flagged, "flag", args.cluster)
            ess = next(iter(report.results.values())).ess
            nominal = 1.0 - args.alpha
            rows.append(
                {
                    "held_out": held_out,
                    "interior": interior[held_out],
                    "arm": name,
                    **metrics,
                    # Valid if the interval reaches nominal. An arm whose coverage is
                    # below nominal is not trading area for coverage, it is simply
                    # broken, and its area must not be read as efficiency.
                    "valid": bool(metrics["hi"] >= nominal),
                    "ess": ess,
                    "certifiable": report.reportable,
                }
            )
            if not report.reportable:
                notes.append(f"{held_out} / {name}: {certification_note(report)}")

    table = pd.DataFrame(rows)
    print("\n[tool2] leave-one-operator-out — read flag_area, not coverage (D39)")
    for held_out, part in table.groupby("held_out", sort=False):
        print(f"\n  held out: {held_out}")
        show = part.drop(columns=["held_out"])
        print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    order = ["matched", "mondrian", "nearest", "weighted"]
    for label, subset in (
        ("interior deployment operators — what Q4 is about", table[table["interior"]]),
        ("endpoint deployment operators — outside what weighting can reach", table[~table["interior"]]),
    ):
        if subset.empty:
            print(f"\n[tool2] no {label.split(' —')[0]}")
            continue
        folds = subset["held_out"].nunique()
        summary = subset.groupby("arm").agg(
            coverage=("coverage", "mean"),
            flag_area=("flag_area", "mean"),
            folds_valid=("valid", "sum"),
        ).reindex(order)
        print(f"\n[tool2] {label}  ({folds} fold(s))")
        print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

    # Q4's verdict is taken on the interior folds, among arms that are valid in all of
    # them. An arm that under-covers has not earned a place in the comparison.
    inner = table[table["interior"]]
    if inner.empty:
        print(
            "\n[tool2] Q4: UNDECIDED — this bank has no interior deployment operator, so "
            "every fold asks\n        the weighting to extrapolate. Q4 needs a grid with an "
            "operator between two others."
        )
    else:
        n_folds = inner["held_out"].nunique()
        valid = {
            arm: int(part["valid"].sum()) == n_folds for arm, part in inner.groupby("arm")
        }
        areas = inner.groupby("arm")["flag_area"].mean()
        deployable = [a for a in ("mondrian", "nearest", "weighted") if valid.get(a)]
        print(f"\n[tool2] Q4, on {n_folds} interior fold(s). Valid in every fold: "
              f"{', '.join(deployable) if deployable else 'none'}")
        if "weighted" not in deployable:
            print("        Tool 2 is NOT valid on every interior fold — it has not earned "
                  "its place here.")
        else:
            beaten = [a for a in deployable if a != "weighted" and areas[a] <= areas["weighted"]]
            if beaten:
                print(
                    f"        Tool 2 is valid but flags no less area than "
                    f"{', '.join(beaten)}. A negative result, and D2's shape: the "
                    "weighting\n        buys nothing over a simpler rule, which the paper "
                    "should state rather than bury."
                )
            else:
                print(
                    f"        Tool 2 is valid and flags the least area of any deployable "
                    f"arm ({areas['weighted']:.4f}),\n        against the "
                    f"{areas['matched']:.4f} oracle. This is Q4's experiment passing."
                )
    for note in notes:
        print(f"[tool2] {note}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"[tool2] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
