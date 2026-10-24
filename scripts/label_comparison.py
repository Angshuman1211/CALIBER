"""Is the score measuring hallucination, or measuring the label's own arithmetic? (D74)

D67 ruled detection numbers unreportable on this bank: by D38's identity, at ``cos θ ≈ 0``
``err_free ≈ sqrt(s_free² + free_content²)``, so a label that thresholds ``err_free`` is a
function of the score's dominant component and the 0.95-1.00 AUROCs measure the identity.
Four label forms were tried and all four thresholded a magnitude, so all four inherited it.

D74's label thresholds a **direction** instead: ``cos θ``, the alignment between the
reconstruction's hidden content and the truth's, which is scale-free. This script runs the
score against both labels side by side, so the difference is visible rather than asserted.

Each row is **one operator**: pooling averaged two opposite per-operator biases into
"exactly chance" once already (D60, retracted). Each AUC is **direction-agnostic**,
``max(auc, 1 - auc)``: a fixed sign scored an anatomy control as ``1 - truth`` and flipped
a verdict from "+0.329 beats anatomy" to "FAILS, -0.408" once corrected (D65). The anatomy
control is ``free_content``, the truth's own hidden content, which is the confound the
label has to survive; ``excess`` is the score's AUC minus its, and
:data:`AUC_EXCESS_MIN` is the margin that counts.

    python scripts/label_comparison.py runs/<run_id>_by_patient \
        --calibration-operator mri.random1d.R2.c0.08.s0
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from caliber.evaluation import derive_score_and_label, label_spec
from caliber.scoring.attribution import (
    AUC_EXCESS_MIN,
    AUC_INFORMATIVE,
    MIN_CLASS_FOR_AUC,
    identity_residual,
    rank_auc,
    spearman,
)
from caliber.scorebank import read_bank

#: Labels compared, in the order the table prints them.
COMPARED = ("absolute", "alignment")


def _agnostic(scores, labels) -> float:
    """``max(auc, 1 - auc)``: D65's rule, because the informative direction is not known.

    Applied to the score as well as to the control. Giving the score a fixed direction and
    the control an agnostic one would hand the score a handicap and call it a result.
    """
    auc = rank_auc(scores, labels)
    return max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bank")
    ap.add_argument("--calibration-operator", required=True,
                    help="the operator whose score scales are frozen (D26/D27)")
    ap.add_argument("--splits", nargs="*", default=["test"],
                    help="which splits to measure on; the design split is never included")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    raw = read_bank(args.bank)
    missing = [c for c in ("s_free", "free_content", "err_free") if c not in raw.columns]
    if missing:
        print(f"[label] this bank has no {missing}; the identity cannot be inverted",
              file=sys.stderr)
        return 1

    residual = identity_residual(raw["s_free"], raw["free_content"], raw["err_free"])
    print(f"[label] identity check on {int(residual['n']):,} regions: median relative "
          f"residual {residual['median_relative']:.2e}, max {residual['max_relative']:.2e}")
    if not np.isfinite(residual["max_relative"]) or residual["max_relative"] > 1e-3:
        print("[label] the bank does not satisfy the identity, so cos theta is not "
              "recoverable and the alignment label is meaningless here. Check that the "
              "truth and score columns share one normalisation (runner.py's ref_intensity).",
              file=sys.stderr)
        return 1

    # How much of each label is the score's own arithmetic, measured directly.
    s = raw["s_free"].to_numpy(dtype=float)
    c = raw["free_content"].to_numpy(dtype=float)
    e = raw["err_free"].to_numpy(dtype=float)
    print(f"[label] spearman(err_free, hypot(s_free, free_content)) = "
          f"{spearman(e, np.hypot(s, c)):+.4f}  <- what a magnitude label inherits")

    rows = []
    for name in COMPARED:
        try:
            spec = label_spec(raw, name)
        except ValueError as exc:
            print(f"[label] {name}: {exc}", file=sys.stderr)
            continue
        scored = derive_score_and_label(raw, args.calibration_operator, **spec.kwargs)
        if name == "alignment":
            cos = scored["cos_theta"].to_numpy(dtype=float)
            defined = np.isfinite(cos)
            print(f"[label] cos theta defined on {defined.mean():.1%} of regions, "
                  f"sd {np.std(cos[defined]):.4f}")
            print(f"[label] spearman(cos theta, s_free) = "
                  f"{spearman(cos, s):+.4f}  <- what the direction label inherits")
        print(f"[label] {name}: {spec.text}")

        part = scored[scored["split"].isin(args.splits)]
        for oid, frame in part.groupby("operator_id"):
            labels = frame["label"].to_numpy().astype(bool)
            n_pos, n_neg = int(labels.sum()), int((~labels).sum())
            row = {"label": name, "operator_id": str(oid),
                   "n_positive": n_pos, "n_negative": n_neg}
            if min(n_pos, n_neg) < MIN_CLASS_FOR_AUC:
                row["note"] = f"below the {MIN_CLASS_FOR_AUC}-per-class floor (D32)"
                rows.append(row)
                continue
            score_auc = _agnostic(frame["score"], labels)
            anatomy_auc = _agnostic(frame["free_content"], labels)
            row.update({
                "score_auc": score_auc,
                "anatomy_auc": anatomy_auc,
                "excess": score_auc - anatomy_auc,
                "beats_anatomy": bool(score_auc - anatomy_auc >= AUC_EXCESS_MIN),
                "informative": bool(score_auc >= AUC_INFORMATIVE),
                "note": "",
            })
            rows.append(row)

    if not rows:
        print("[label] no label could be valued on this bank", file=sys.stderr)
        return 1
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"\n[label] per operator, direction-agnostic AUC, on split(s) {args.splits}")
    print(f"{'label':<11}{'operator':<30}{'npos':>6}{'score':>8}{'anatomy':>9}"
          f"{'excess':>8}  verdict")
    for r in rows:
        if r.get("note"):
            print(f"{r['label']:<11}{r['operator_id']:<30}{r['n_positive']:>6}"
                  f"{'':>25}  {r['note']}")
            continue
        verdict = ("beats anatomy" if r["beats_anatomy"]
                   else "does NOT beat anatomy")
        print(f"{r['label']:<11}{r['operator_id']:<30}{r['n_positive']:>6}"
              f"{r['score_auc']:>8.4f}{r['anatomy_auc']:>9.4f}{r['excess']:>+8.4f}  {verdict}")

    usable = table[table["note"] == ""] if "note" in table else table
    print()
    for name in COMPARED:
        part = usable[usable["label"] == name] if "label" in usable else usable
        if part.empty:
            continue
        wins = int(part["beats_anatomy"].sum())
        print(f"[label] {name}: score clears anatomy by >= {AUC_EXCESS_MIN} at "
              f"{wins} of {len(part)} operators; score AUC ranges "
              f"{part['score_auc'].min():.3f} to {part['score_auc'].max():.3f}")
    print("[label] a label the score cannot fail against is not a test. Read the excess "
          "column, per operator, and not the AUC alone.")
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"[label] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
