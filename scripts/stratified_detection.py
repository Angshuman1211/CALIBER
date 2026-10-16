"""Detection numbers stratified by free_content tercile within operator, as D-061 requires.

D-061 moved the anatomy confound out of the label and into the analysis: the label is the
plain absolute ``err_free > eps``, and every detection number is reported **within
``free_content`` terciles, within operator**, beside an **anatomy control** -- ``free_content``
itself scored through the identical stratified path. D62's warning is why the control is not
optional: stratifying a continuous confound only attenuates it, so a pure anatomy meter still
scores ~0.87 "detail-matched". **The excess over the control is the load-bearing number**;
the stratified AUC alone would have answered Q6 wrongly.

This is Track 1's machinery (`caliber.scoring.attribution.attribute_components`) applied one
operator at a time with three strata, so a component is only compared with regions that hide
about as much as it does. For each score column it reports:

- the plain AUC;
- the detail-matched (stratified) AUC, and how many strata survived;
- the anatomy control's own detail-matched AUC;
- the excess over that control;
- the pre-registered Q6 verdict.

    python scripts/stratified_detection.py runs/<run_id>_by_patient \
        --calibration-operator mri.random1d.R2.c0.08.s0 \
        [--dispersion runs/w05_analysis/dispersion.parquet] [--require-annotation]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from caliber.evaluation import add_label_args, derive_score_and_label, label_from_args
from caliber.scoring.attribution import attribute_components, stratified_auc, within_cell_rank

KEYS = ["sample_id", "slice_idx", "operator_id", "region_id"]
COLUMNS = KEYS + ["split", "s_free", "s_disagree", "s_residual", "err_total", "err_free",
                  "free_content", "ann_coverage"]
SCORES = ("score", "s_free", "s_disagree", "s_residual")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bank")
    ap.add_argument("--calibration-operator", required=True,
                    help="where the combined score's scales are frozen (D26/D27)")
    ap.add_argument("--dispersion", default=None,
                    help="dispersion_baseline.py output, to add posterior dispersion")
    ap.add_argument("--strata", type=int, default=3, help="D-061: free_content terciles")
    add_label_args(ap)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    root = Path(args.bank)
    raw = pd.read_parquet(root / "scores.parquet" if root.is_dir() else root, columns=COLUMNS)
    raw["operator_id"] = raw["operator_id"].astype(str)
    label = label_from_args(raw, args)
    print(f"[strat] label: {label.text}")
    df = derive_score_and_label(raw, args.calibration_operator, **label.kwargs)
    df = df[df["split"] == "test"].copy()
    scores = list(SCORES)
    if args.dispersion:
        disp = pd.read_parquet(args.dispersion, columns=KEYS + ["s_dispersion"])
        disp["operator_id"] = disp["operator_id"].astype(str)
        df = df.merge(disp, on=KEYS, how="inner")
        scores.append("s_dispersion")
    df["solver_id"] = "bank"   # attribute_components ranks within (operator, solver) cells

    rows = []
    for oid, part in df.groupby("operator_id"):
        if part["label"].sum() == 0 or (~part["label"]).sum() == 0:
            continue
        att = attribute_components(part, component_cols=tuple(scores), label_col="label",
                                   n_strata=args.strata)
        anat = within_cell_rank(part, "free_content", ("operator_id", "solver_id")).to_numpy()
        control = stratified_auc(anat, part["label"].to_numpy(), anat, n_strata=args.strata)
        att.insert(0, "operator_id", oid)
        att["anatomy_control"] = control.value
        att["positive_rate"] = float(part["label"].mean())
        rows.append(att)
    t = pd.concat(rows, ignore_index=True)

    pd.set_option("display.width", 220)
    print(f"\n[strat] {args.strata} free_content strata within each operator; excess = "
          "detail-matched AUC minus the anatomy control's")
    show = t.pivot(index="operator_id", columns="component", values="auc_over_anatomy")
    print("excess over the anatomy control:")
    print(show[[c for c in scores if c in show.columns]].round(3).to_string())
    print("\ndetail-matched AUC (strata used in brackets for the combined score):")
    dm = t.pivot(index="operator_id", columns="component", values="auc_detail_matched")
    used = t[t["component"] == "score"].set_index("operator_id")["strata_used"]
    dm["strata"] = used
    dm["anatomy control"] = t.groupby("operator_id")["anatomy_control"].first()
    print(dm[[c for c in scores if c in dm.columns] + ["anatomy control", "strata"]]
          .round(3).to_string())
    print("\nQ6 verdicts:")
    print(t.pivot(index="operator_id", columns="component", values="verdict")
          [[c for c in scores if c in show.columns]].to_string())
    if args.out:
        t.to_csv(args.out, index=False)
        print(f"[strat] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
