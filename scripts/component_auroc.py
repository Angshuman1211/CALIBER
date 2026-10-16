"""Which score component ranks hallucinated regions best, per operator, under the reported label.

D-059 asked for this before equal weights are written up as a finding: the earlier ordering
(``s_disagree`` beating the combined score everywhere) was measured against the ratio-only
label, and ``s_disagree`` is the component that label's artefact most rewards. D-060 then
moved the gate, so the ordering is measured here under the candidacy gate.

Per operator, on test patients only: AUROC of each component, of the combined score (equal
weights, scales frozen at the calibration operator, D26/D27), and of ``-free_content``.
That last one is the label check, not a detector -- D-060's gate exists to bring it to
chance, and if it is far from 0.5 the label is still denominator-driven and nothing else in
the table means much.

The intervals are a **paired** bootstrap over patients (D-027): every detector is scored on
the same resampled patients in each draw, so the gap between the top two gets its own
interval. An ordering is only claimed where that interval excludes zero.

Reads only the columns it needs, so it runs in a few hundred MB beside other jobs.

    python scripts/component_auroc.py runs/<run_id>_by_patient \
        --out runs/<analysis>/component_auroc.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from caliber.evaluation import add_label_args, derive_score_and_label, label_from_args
from caliber.evaluation.baselines import auroc

COLUMNS = ["operator_id", "split", "patient_id", "s_free", "s_disagree", "s_residual",
           "err_total", "err_free", "free_content", "ann_coverage"]
DETECTORS = {"s_free": "s_free", "s_disagree": "s_disagree", "s_residual": "s_residual",
             "combined (equal)": "score"}
LABEL_CHECK = "-free_content (label check)"


def paired_bootstrap(part: pd.DataFrame, cols: dict[str, str], n_boot: int, seed: int):
    """Point AUROC per detector, and ``n_boot`` draws of all of them on shared resamples."""
    y = part["label"].to_numpy(bool)
    scores = {name: part[col].to_numpy(float) for name, col in cols.items()}
    ids, inverse = np.unique(part["patient_id"].to_numpy(), return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(ids))]
    rng = np.random.default_rng(seed)
    point = {name: auroc(s, y) for name, s in scores.items()}
    draws = {name: [] for name in scores}
    for _ in range(n_boot):
        pick = np.concatenate([members[k] for k in rng.integers(0, len(ids), len(ids))])
        for name, s in scores.items():
            draws[name].append(auroc(s[pick], y[pick]))
    return point, {name: np.asarray(d, dtype=float) for name, d in draws.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bank", help="a bank with patient_id (add_patient_ids.py)")
    ap.add_argument("--calibration-operator", default="mri.random1d.R2.c0.08.s0",
                    help="where the combined score's scales are frozen (D26/D27)")
    add_label_args(ap)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--cluster", default="patient_id",
                    help="the patient column; sample_id where one subject is one patient "
                         "(TotalSegmentator, D-013)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    root = Path(args.bank)
    path = root / "scores.parquet" if root.is_dir() else root
    try:
        cols = [args.cluster if c == "patient_id" else c for c in COLUMNS]
        raw = pd.read_parquet(path, columns=cols).rename(columns={args.cluster: "patient_id"})
    except Exception as exc:  # a missing column is the usual cause: no patient_id yet
        print(f"[components] cannot read {COLUMNS} from {path} ({exc}); "
              "run scripts/add_patient_ids.py first", file=sys.stderr)
        return 1
    raw["operator_id"] = raw["operator_id"].astype(str)

    label = label_from_args(raw, args)
    print(f"[components] label: {label.text}")
    df = derive_score_and_label(raw, args.calibration_operator, **label.kwargs)
    df = df[df["split"] == "test"].copy()
    df["neg_free_content"] = -df["free_content"]
    cols = {**DETECTORS, LABEL_CHECK: "neg_free_content"}

    rows, verdicts = [], []
    for oid, part in df.groupby("operator_id"):
        point, draws = paired_bootstrap(part, cols, args.boot, seed=0)
        for name in cols:
            lo, hi = np.nanquantile(draws[name], [0.025, 0.975])
            rows.append({"operator_id": oid, "detector": name, "auroc": point[name],
                         "lo": lo, "hi": hi, "positive_rate": float(part["label"].mean()),
                         "n_patients": int(part["patient_id"].nunique())})
        ranked = sorted(DETECTORS, key=lambda n: point[n], reverse=True)
        diff = draws[ranked[0]] - draws[ranked[1]]
        dlo, dhi = np.nanquantile(diff, [0.025, 0.975])
        best_single = max(("s_free", "s_disagree", "s_residual"), key=lambda n: point[n])
        cdiff = draws["combined (equal)"] - draws[best_single]
        clo, chi = np.nanquantile(cdiff, [0.025, 0.975])
        verdicts.append({"operator_id": oid, "first": ranked[0], "second": ranked[1],
                         "gap": point[ranked[0]] - point[ranked[1]], "gap_lo": dlo,
                         "gap_hi": dhi, "separated": bool(dlo > 0),
                         "best_single": best_single,
                         "combined_minus_best": point["combined (equal)"] - point[best_single],
                         "cmb_lo": clo, "cmb_hi": chi})

    table = pd.DataFrame(rows)
    wide = table.pivot(index="operator_id", columns="detector", values="auroc")
    pd.set_option("display.width", 200)
    print(f"\n[components] AUROC on test patients ({df['patient_id'].nunique()} patients), "
          f"combined score scaled at {args.calibration_operator}")
    print(wide[list(cols)].round(3).to_string())
    v = pd.DataFrame(verdicts).set_index("operator_id")
    print("\n[components] ordering: top two, and the paired 95% interval on their gap")
    for oid, r in v.iterrows():
        tag = "separated" if r["separated"] else "tied"
        print(f"   {oid:<36} {r['first']:<17} > {r['second']:<17} by {r['gap']:+.3f} "
              f"[{r['gap_lo']:+.3f}, {r['gap_hi']:+.3f}]  {tag}")
    print("\n[components] equal-weight combination against the best single component")
    for oid, r in v.iterrows():
        print(f"   {oid:<36} combined - {r['best_single']:<11} {r['combined_minus_best']:+.3f} "
              f"[{r['cmb_lo']:+.3f}, {r['cmb_hi']:+.3f}]")
    firsts = v["first"].value_counts()
    print(f"\n[components] first place: {dict(firsts)}; separated from second in "
          f"{int(v['separated'].sum())} of {len(v)} operators")
    check = table[table["detector"] == LABEL_CHECK]["auroc"]
    print(f"[components] label check, AUROC(-free_content): {check.min():.3f}-{check.max():.3f} "
          "(chance is 0.5)")
    if args.out:
        table.to_csv(args.out, index=False)
        v.to_csv(args.out.replace(".csv", "_ordering.csv"))
        print(f"[components] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
