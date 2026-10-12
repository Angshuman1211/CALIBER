"""Per-component Lipschitz ratios and the weight-free bound on L, from a cached bank.

    PYTHONPATH=src python3 scripts/t1_component_lipschitz.py banks/w02_solver_smoke

**Why this exists.** T3.3's assumption (ii) needs the score to be `L`-Lipschitz in the
operator distance, and §3 of that derivation reduces the claim to the three components
separately because D25 makes the combination an affine, non-negative weighted sum:
`L = Σ wᵢ Lᵢ`. Track 3 has been unable to state a value for `L` because the weights
are Track 1's and are still unset (D28) — and, until now, because every measurement of
the per-component constants had been made on the placeholder harness, whose null-space
fill is injected white noise rather than anything a prior decided (T3.4 §2, §5).

Two things unblock that, and this script computes both.

**1. The weights are not needed for a valid bound.** D25 constrains the weights to the
simplex — non-negative, summing to one — so `L = Σ wᵢ Lᵢ` is a *convex combination* of
the per-component constants and therefore

    L  ≤  max_i Lᵢ

for **every** admissible weight vector. T3.3 can state a valid bound today and tighten it
when D28 lands; only the tightest constant waits on the weights.

**Only the upper end brackets.** `L ≥ min_i Lᵢ` is *false*, and measurably so: the
components can move in opposite directions under the same operator change, and then the
combination is less operator-sensitive than any of its parts. On the real-prior bank
`s_residual` falls where `s_free` and `s_disagree` rise, and at equal thirds they cancel
to a net mean shift of −0.004 against +0.376 from the other two — T3.4 §3's
cross-component cancellation, measured on a prior rather than on the placeholder harness.
Cancellation only ever helps a coverage-gap bound, so losing the lower end costs T3.3
nothing; but it does mean a small combined secant is not evidence of a small constant.

**The frozen scales are a precondition of that reduction, not hygiene.** The components
enter the combination standardised, `cᵢ / scaleᵢ`, and the step above treats `scaleᵢ` as
a constant. D27 is what makes it one: scales are estimated on the calibration operator
and frozen. If they were re-estimated per deployment operator, `scaleᵢ` would be a
function of `A`, the standardised component would not inherit its component's Lipschitz
constant, and the per-component reduction would silently stop being valid. Anyone
"simplifying" D27 away breaks T3.3's §3, not just the statistics.

**2. `Lᵢ` measured from a bank, on whatever prior produced it.** For two operators the
local ratio is

    ℓᵢ(A→A') = | E[cᵢ(A')] − E[cᵢ(A)] | / (scaleᵢ · d_op(A, A'))

**These are lower bounds on `Lᵢ`, not estimates of it.** If `cᵢ(A, ·)` is `Lᵢ`-Lipschitz
then `|E[cᵢ(A')] − E[cᵢ(A)]| ≤ E|cᵢ(A') − cᵢ(A)| ≤ Lᵢ · d_op`, so `ℓᵢ ≤ Lᵢ` always, with
equality only if every region moves by the same amount in the same direction. What that
buys T3.3 is a floor: a component whose mean already moves this much cannot have a small
constant, which bounds how tight the final result can be. It does not license quoting
`ℓᵢ` as `Lᵢ`.

Signed shifts are printed alongside, because their **signs** are what decides whether the
components reinforce or cancel, and the magnitudes alone hide that. With more than two
operators every pair is reported, so a constant that drifts across the grid is visible
rather than averaged away.

Track 1 (T1.1), for Track 3 (T3.3, T3.4).
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd
import torch

from caliber.operators.distance import compute_distance
from caliber.operators.registry import build_operator
from caliber.scorebank import read_bank, read_manifest
from caliber.scoring.score import ScoreComponents, ScoreScales

COMPONENTS = ("s_free", "s_disagree", "s_residual")


def rebuild_operators(manifest: dict) -> dict:
    """Operators from the manifest alone — distances need no dataset and no GPU."""
    ops = {}
    for row in manifest["operators"]:
        spec = {"modality": row["modality"], **row["params"]}
        ops[row["operator_id"]] = build_operator(spec, tuple(row["image_shape"]))
    return ops


def components_of(frame: pd.DataFrame) -> ScoreComponents:
    return ScoreComponents(
        s_free=torch.tensor(frame["s_free"].to_numpy()),
        s_disagree=torch.tensor(frame["s_disagree"].to_numpy()),
        s_residual=torch.tensor(frame["s_residual"].to_numpy()),
        ref_intensity=1.0,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-component Lipschitz ratios from a bank")
    ap.add_argument("bank", help="score-bank directory")
    ap.add_argument("--calibration-operator", default=None,
                    help="operator the scales are frozen on (D27); default: first")
    ap.add_argument("--distance", default="null_space_projection")
    ap.add_argument("--normalise", default="ambient")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="components to leave out of the bracket, e.g. --exclude s_residual (D41)")
    ap.add_argument("--out", default=None, help="write the per-pair table here as CSV")
    args = ap.parse_args(argv)

    bank = Path(args.bank)
    df = read_bank(bank)
    manifest = read_manifest(bank)
    ops = rebuild_operators(manifest)
    operators = sorted(df["operator_id"].unique())
    print(f"[lipschitz] {len(df):,} rows, {len(operators)} operators")

    prov = manifest.get("provenance") or {}
    if prov and not prov.get("reportable", True):
        print(
            f"[lipschitz] NOT REPORTABLE -- dataset={prov.get('dataset')}, "
            f"solver_family={prov.get('solver_family')}. Constants below are a development "
            "reading, not a result."
        )
    if len(operators) < 2:
        print("[lipschitz] a Lipschitz ratio needs at least two operators", flush=True)
        return 1

    calib_op = args.calibration_operator or operators[0]
    scales = ScoreScales.from_components(components_of(df[df["operator_id"] == calib_op]))
    print(f"[lipschitz] scales frozen on {calib_op} (D27): "
          f"free={scales.free:.4g} disagree={scales.disagree:.4g} residual={scales.residual:.4g}")
    if any(scales.degenerate):
        dead = [c for c, d in zip(COMPONENTS, scales.degenerate, strict=True) if d]
        print(f"[lipschitz] degenerate, excluded from the score by D35: {', '.join(dead)}")

    scale_of = {"s_free": scales.free, "s_disagree": scales.disagree, "s_residual": scales.residual}
    means = df.groupby("operator_id")[list(COMPONENTS)].mean()

    rows = []
    for left, right in itertools.combinations(operators, 2):
        d = float(compute_distance(ops[left], ops[right],
                                   measure=args.distance, normalise=args.normalise))
        if d <= 0:
            print(f"[lipschitz] {left} and {right} are at distance 0 -- skipped")
            continue
        row = {"operator_a": left, "operator_b": right, "d_op": d}
        signed = {}
        for c in COMPONENTS:
            signed[c] = float(means.loc[right, c] - means.loc[left, c]) / scale_of[c]
            row[f"shift_{c}"] = signed[c]
            row[f"ell_{c}"] = abs(signed[c]) / d
        # Whether the components reinforce or cancel is a property of the signs, and it
        # is what decides if a small combined secant means a small constant (T3.4 §3).
        row["reinforcing"] = len({v > 0 for v in signed.values() if v != 0}) == 1
        rows.append(row)

    table = pd.DataFrame(rows)
    print(f"\n[lipschitz] per-pair local ratios ({args.distance}/{args.normalise}, D19)")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    live = [c for c in COMPONENTS if c not in args.exclude]
    worst = {c: float(table[f"ell_{c}"].max()) for c in live}
    hi = max(worst.values())
    print("\n[lipschitz] worst local ratio per component, over all pairs (lower bounds on L_i):")
    for c in live:
        print(f"    {c:12s} {worst[c]:.4f}")
    if args.exclude:
        print(f"    (excluded: {', '.join(args.exclude)})")
    print(
        f"\n[lipschitz] D25 makes the combination convex, so for ANY admissible weights\n"
        f"                L  <=  max_i L_i\n"
        f"            and the measured floor on that maximum is {hi:.4f}. The weights (D28)\n"
        f"            decide where L sits below it; they are not needed for the bound to hold."
    )
    if not bool(table["reinforcing"].all()):
        mixed = table.loc[~table["reinforcing"], ["operator_a", "operator_b"]]
        print(
            f"\n[lipschitz] components move in OPPOSITE directions on {len(mixed)} of "
            f"{len(table)} pair(s).\n"
            "            They partially cancel, so the combination can be less "
            "operator-sensitive\n"
            "            than any component (T3.4 Sec.3). That is why there is no lower bracket: a\n"
            "            small combined secant is not evidence of a small constant."
        )

    if args.out:
        table.to_csv(args.out, index=False)
        print(f"[lipschitz] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
