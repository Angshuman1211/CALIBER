"""Why T1-A's rule fails, cell by cell, from the distance matrix and ONE bit of the bank.

    python scripts/t1a_rule_anatomy.py --out runs/t1a_anatomy

T1-A came back `no_better_than_trivial` on the CT bank (D89). This script answers the
follow-up question -- *why those cells and not others* -- without the per-pair coverage
table, because it needs only the operator grid, which is on Track 1's disk, plus one
ordering fact taken from the bank and named as such.

**What is derived and what is taken.** D89's identity says
``sign(gap) == sign(tau_a - tau_b)`` exactly. Within a family the threshold ordering follows
from the acquisition knob and the fact that the score tracks reconstruction error: fewer
views and narrower arcs mean larger errors mean a higher threshold. Across the families the
knobs do not compare, so the relative placement of the two blocks is **one free bit**, and
it is fitted on the 18 sparse-view-to-limited-angle pairs alone -- the cell where every pair
covers. Everything printed for the other 54 pairs is then a prediction, not a fit.

**The prediction has no free magnitude.** The identity fixes the gap's *sign*, so it says
nothing about how far a cell falls -- but a pair with ``tau_a < tau_b`` has ``gap < 0``, it
covers at least as well as the deployment operator's own matched cell, and it therefore
**cannot be unsafe**. That is falsifiable with no fitted parameter, and it is the claim to
check against the per-pair table when it arrives.

Track 2's reported rates are recorded in :data:`REPORTED` with their provenance. Nothing
here recomputes them; the point is to reconcile against them, and the reconciliation is an
integer check -- the per-cell rates have to resolve to whole pair counts against the
predicted-safe denominators built here, or one of the two sides is wrong.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from caliber.evaluation import prereg
from caliber.evaluation.gap import operator_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISTANCES = REPO_ROOT / "runs" / "t1a_ct_distances" / "distances.csv"
DEFAULT_MANIFEST = REPO_ROOT / "runs" / "t1a_ct_distances" / "manifest.json"

#: Track 2's T1-A run on their tables, 21 November 2026, reported in their D87 reply.
#: ``false_safe`` is None where the cell holds no unsafe pair and the rate is undefined.
REPORTED = {
    "sv->sv": {"false_safe": 0.333, "trivial": 0.167},
    "sv->la": {"false_safe": None, "trivial": 0.0},
    "la->sv": {"false_safe": 1.000, "trivial": 0.889},
    "la->la": {"false_safe": 0.667, "trivial": 0.333},
}
REPORTED_POOLED = {"false_safe": 0.361, "trivial": 0.319, "youden_j": -0.10, "p": 0.81}

#: Coverage values Track 2 quoted in prose, 21 November 2026, on the across-split mean.
#: ``(calibration, deployment): coverage``. These are the only per-pair numbers Track 1
#: holds; the full table is on Track 2's disk.
QUOTED_COVERAGE = {
    ("sv 180", "sv 30"): 0.964,
    ("sv 180", "la 90"): 0.994,
    ("sv 30", "sv 180"): 0.731,
    ("sv 30", "sv 120"): 0.676,
    ("la 90", "sv 120"): 0.377,
}

#: The matched arm, all nine operators, same source. A range, not a point.
MATCHED_RANGE = (0.892, 0.905)

#: The one bit, and where it comes from. ``limited_angle`` sits above every sparse-view
#: operator in threshold order; read off the sv->la cell, which holds no unsafe pair.
CROSS_FAMILY_BIT = "limited_angle_above_sparse_view"
CROSS_FAMILY_SOURCE = "the 18 sv->la pairs, 0 of them unsafe"


def short_name(op: str, meta: dict) -> str:
    """``la 90`` / ``sv 120``, the names the partners use in prose."""
    m = meta[op]
    if m["family"] == "limited_angle":
        return "la " + str(int(m["arc_deg"]))
    return "sv " + str(int(m["views"]))


def threshold_rank(meta: dict) -> dict[str, int]:
    """Operators in increasing threshold order, under the stated reasoning.

    Within sparse view the knob is the view count and within limited angle it is the arc,
    both running the same way: less data, larger reconstruction error, larger score,
    higher threshold. The cross-family offset is :data:`CROSS_FAMILY_BIT`.
    """
    if CROSS_FAMILY_BIT != "limited_angle_above_sparse_view":
        raise ValueError(f"unhandled cross-family bit {CROSS_FAMILY_BIT!r}")

    def key(op: str) -> tuple[int, float]:
        m = meta[op]
        block = 1 if m["family"] == "limited_angle" else 0
        knob = -float(m["arc_deg"]) if block else -float(m["views"])
        return (block, knob)

    return {op: i for i, op in enumerate(sorted(meta, key=key))}


def load_signed(path: Path, measure: str) -> dict[tuple[str, str], float]:
    """The signed distance both ways. Antisymmetry is filled, so it is not evidence."""
    frame = pd.read_csv(path)
    rows = frame[frame["measure"] == measure]
    if rows.empty:
        raise SystemExit(f"no {measure!r} rows in {path}")
    out: dict[tuple[str, str], float] = {}
    for a, b, v in zip(rows["calibration"], rows["deployment"], rows["value"]):
        out[(str(a), str(b))] = float(v)
        out[(str(b), str(a))] = -float(v)
    return out


def anatomy(meta: dict, signed: dict[tuple[str, str], float]) -> pd.DataFrame:
    """One row per ordered pair: what the rule says, what the identity says, and the cell."""
    rank = threshold_rank(meta)
    rows = []
    for a, b in itertools.permutations(sorted(meta), 2):
        d = signed.get((a, b), float("nan"))
        rows.append({
            "calibration": a,
            "deployment": b,
            "name": short_name(a, meta) + " -> " + short_name(b, meta),
            "cell": ("sv" if meta[a]["family"] == "sparse_view" else "la") + "->"
                    + ("sv" if meta[b]["family"] == "sparse_view" else "la"),
            "signed_distance": d,
            "rule_says_safe": bool(d <= prereg.SIGN_RULE_THRESHOLD),
            "tau_calibration_rank": rank[a],
            "tau_deployment_rank": rank[b],
            # sign(gap) == sign(tau_a - tau_b); gap <= 0 means the cell cannot be unsafe
            "identity_gap_positive": bool(rank[a] > rank[b]),
            "identity_says_cannot_be_unsafe": bool(rank[a] < rank[b]),
        })
    return pd.DataFrame(rows)


def reconcile(table: pd.DataFrame) -> pd.DataFrame:
    """Do Track 2's per-cell rates resolve to whole pair counts on these denominators?"""
    rows = []
    for cell, part in table.groupby("cell", sort=True):
        n = int(len(part))
        n_safe = int(part["rule_says_safe"].sum())
        rep = REPORTED.get(cell, {})
        trivial, false_safe = rep.get("trivial"), rep.get("false_safe")
        n_unsafe = None if trivial is None else trivial * n
        n_false = None if false_safe is None else false_safe * n_safe

        def whole(x: float | None) -> bool:
            return x is not None and abs(x - round(x)) < 0.02

        rows.append({
            "cell": cell,
            "n_pairs": n,
            "rule_predicted_safe": n_safe,
            "reported_trivial": trivial,
            "implied_unsafe": None if n_unsafe is None else round(n_unsafe),
            "unsafe_is_whole": whole(n_unsafe),
            "reported_false_safe": false_safe,
            "implied_false_safe": None if n_false is None else round(n_false),
            "false_safe_is_whole": whole(n_false) if false_safe is not None else True,
            "identity_cannot_be_unsafe": int(part["identity_says_cannot_be_unsafe"].sum()),
        })
    return pd.DataFrame(rows)


def ordering_constraints(meta: dict) -> pd.DataFrame:
    """Threshold-order constraints implied by the quoted coverages, and whether they hold.

    Two kinds, and the second is the stronger:

    ``vs_matched``
        ``coverage(a->b)`` above the whole matched range forces ``tau_a < tau_b``, below it
        forces ``tau_a > tau_b``. Inside the range it decides nothing. This leans on the
        matched arm being the same quantity, which is what makes it the weaker kind.

    ``within_row``
        Two cells sharing a calibration operator share ``tau_a`` exactly, so comparing them
        cancels the threshold and compares the two deployment operators' score
        distributions directly at one point: ``coverage(a->b) < coverage(a->c)`` means more
        of ``b``'s positive scores sit below that threshold than ``c``'s. **No matched value
        is needed.** Under stochastic ordering of the two score distributions this gives
        ``tau_b < tau_c``; without it, it still shows the distributions are ordered that way
        at the point the transfer actually used.
    """
    rank = threshold_rank(meta)
    by_short = {short_name(o, meta): o for o in meta}
    lo, hi = MATCHED_RANGE
    rows = []

    for (a, b), cov in sorted(QUOTED_COVERAGE.items()):
        if cov > hi:
            implied = "tau_a < tau_b"
            holds = rank[by_short[a]] < rank[by_short[b]]
        elif cov < lo:
            implied = "tau_a > tau_b"
            holds = rank[by_short[a]] > rank[by_short[b]]
        else:
            implied, holds = "undecided (inside the matched range)", True
        rows.append({"kind": "vs_matched", "basis": f"{a} -> {b} = {cov:.3f}",
                     "implies": implied, "consistent_with_knob_order": bool(holds)})

    seen: dict[str, list[tuple[str, float]]] = {}
    for (a, b), cov in QUOTED_COVERAGE.items():
        seen.setdefault(a, []).append((b, cov))
    for a, targets in sorted(seen.items()):
        for (b, cb), (c, cc) in itertools.combinations(sorted(targets), 2):
            lower, higher = (b, c) if cb < cc else (c, b)
            rows.append({
                "kind": "within_row",
                "basis": f"from {a}: {b} = {cb:.3f} vs {c} = {cc:.3f}",
                "implies": f"tau({lower}) < tau({higher})",
                "consistent_with_knob_order":
                    bool(rank[by_short[lower]] < rank[by_short[higher]]),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", type=Path, default=DEFAULT_DISTANCES)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--measure", default=prereg.GAP_PREDICTION_DISTANCE)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    meta = operator_metadata(args.manifest)
    signed = load_signed(args.distances, args.measure)
    table = anatomy(meta, signed)
    rec = reconcile(table)

    print(f"[anatomy] prereg {prereg.fingerprint()} measure={args.measure} "
          f"rule = safe iff signed_distance <= {prereg.SIGN_RULE_THRESHOLD}")
    print(f"[anatomy] {len(meta)} operators, {len(table)} ordered pairs")
    print(f"[anatomy] threshold order taken as: within family from the knob; across "
          f"families ONE BIT ({CROSS_FAMILY_BIT}), fitted on {CROSS_FAMILY_SOURCE}")

    rank = threshold_rank(meta)
    print("\n-- implied threshold order, lowest first --")
    for op in sorted(rank, key=lambda o: rank[o]):
        print(f"   {rank[op]}  {short_name(op, meta):8s} views={meta[op]['views']:4d} "
              f"arc={int(meta[op]['arc_deg']):4d} nulldim={meta[op]['null_space_dim']}")

    print("\n-- reconciliation against Track 2's reported rates --")
    cols = ["cell", "n_pairs", "rule_predicted_safe", "reported_trivial", "implied_unsafe",
            "unsafe_is_whole", "reported_false_safe", "implied_false_safe",
            "false_safe_is_whole", "identity_cannot_be_unsafe"]
    print(rec[cols].to_string(index=False))

    # pandas stores a column of mixed None/int as float with NaN, so filter on finiteness
    # rather than on None -- "is not None" lets the NaN through and the int() then fails.
    def total(column: str) -> int:
        values = rec[column].to_numpy(dtype=float)
        return int(np.nansum(values[np.isfinite(values)]))

    unsafe = total("implied_unsafe")
    false_safe = total("implied_false_safe")
    n_safe = int(rec["rule_predicted_safe"].sum())
    n = int(rec["n_pairs"].sum())
    print(f"\n[anatomy] implied pooled trivial    {unsafe}/{n} = {unsafe / n:.4f} "
          f"(reported {REPORTED_POOLED['trivial']})")
    print(f"[anatomy] implied pooled false-safe {false_safe}/{n_safe} = "
          f"{false_safe / n_safe:.4f} (reported {REPORTED_POOLED['false_safe']})")
    whole = bool(rec["unsafe_is_whole"].all() and rec["false_safe_is_whole"].all())
    print(f"[anatomy] every per-cell rate resolves to a whole pair count: {whole}")

    # How far apart are the two 36-pair "safe" sets?
    rule_safe = set(map(tuple, table.loc[table["rule_says_safe"], ["calibration",
                                                                   "deployment"]].values))
    id_safe = set(map(tuple, table.loc[table["identity_says_cannot_be_unsafe"],
                                       ["calibration", "deployment"]].values))
    print(f"\n[anatomy] the rule calls {len(rule_safe)} pairs safe; the identity "
          f"guarantees {len(id_safe)}; they agree on {len(rule_safe & id_safe)}")
    print(f"[anatomy] so the rule calls {len(rule_safe - id_safe)} pairs safe that the "
          "identity says may be unsafe -- that set is where every false-safe must live")

    cons = ordering_constraints(meta)
    print("\n-- is the knob-based threshold order consistent with the quoted coverages? --")
    print(cons.to_string(index=False))
    broken = cons[~cons["consistent_with_knob_order"]]
    if len(broken):
        print(f"\n[anatomy] {len(broken)} of {len(cons)} constraint(s) VIOLATE the "
              "knob-based order. The knob order is a prediction, so a violation is a "
              "finding about that operator, not a defect of this script:")
        for _, r in broken.iterrows():
            print(f"   {r['kind']:11s} {r['basis']:44s} implies {r['implies']}")
    else:
        print(f"\n[anatomy] all {len(cons)} constraint(s) consistent with the knob order")

    print("\n-- the prediction with no fitted magnitude --")
    print("[anatomy] no pair with tau_a < tau_b can be unsafe. Per cell, the number of "
          "such pairs is in identity_cannot_be_unsafe above;")
    print("[anatomy] sv->la is 18 of 18, and Track 2 report 0 unsafe pairs there. The "
          "other three cells are checkable per pair when the table arrives.")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out / "t1a_pair_anatomy.csv", index=False)
        rec.to_csv(args.out / "t1a_reconciliation.csv", index=False)
        cons.to_csv(args.out / "t1a_order_constraints.csv", index=False)
        print(f"\n[anatomy] wrote {args.out}/t1a_pair_anatomy.csv, t1a_reconciliation.csv, "
              "t1a_order_constraints.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
