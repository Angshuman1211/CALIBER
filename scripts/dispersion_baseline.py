"""Posterior-dispersion reference detector on a finished bank (Blueprint §15.1, T2.6).

A *non-conformal* detector needs no calibration data, only evaluation data, so this works
on the **test patients only**: it regenerates their posterior draws, measures the
full-image spread, and scores how well that spread ranks hallucinated regions.

The exception is D73, which gives the dispersion detector a conformal threshold of its own
and asks whether it transfers across operators (``transfer_baseline.py``). That threshold
is fitted on calibration patients, so ``generate --splits calib test`` adds their draws to
the same file; already-generated pairs are skipped, so this extends a test-only table.

Why regenerate: the bank stores ``s_disagree``, which is the spread in the operator's null
space (a CALIBER score component), not the full-image spread the literature's dispersion
detector uses. The draws are a pure function of the config and the pair (D-041), so the
same draws come back -- and the script proves it, by recomputing ``s_disagree`` from them
and comparing it with the bank's value before trusting anything else.

    # 1. regenerate (GPU; resumable - rerun the same command after any interruption)
    python scripts/dispersion_baseline.py generate configs/experiments/<run>.yaml \
        runs/<run_id> --out runs/<analysis>/dispersion.parquet
    # 2. score it against CALIBER on the same regions
    python scripts/dispersion_baseline.py evaluate runs/<run_id>_by_patient \
        --dispersion runs/<analysis>/dispersion.parquet \
        --calibration-operator mri.random1d.R2.c0.08.s0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from caliber.config import load_config
from caliber.data import build_dataset
from caliber.evaluation import add_label_args, derive_score_and_label, label_from_args
from caliber.evaluation.baselines import (
    auroc,
    average_precision,
    cluster_bootstrap_metric,
    posterior_dispersion,
)
from caliber.operators.registry import OperatorRegistry
from caliber.runner import NONFINITE_RETRIES, add_measurement_noise, item_seed
from caliber.scorebank import read_bank
from caliber.scoring.regions import RegionGrid
from caliber.scoring.score import compute_score_components
from caliber.solvers import build_solver

KEYS = ["sample_id", "slice_idx", "operator_id", "region_id"]


def generate(args) -> int:
    cfg = load_config(args.config)
    exp = cfg["experiment"]
    shape = tuple(cfg["data"]["image_shape"])
    registry = OperatorRegistry.from_specs(exp["operators"], shape)
    ops = [op for op in registry
           if not args.operators or op.descriptor().operator_id in args.operators]
    grid = RegionGrid(image_shape=shape, region_size=int(cfg["regions"]["size"]))
    # grid.table() is a list of dicts in region order, the order the runner writes and
    # grid.reduce returns; taking region_id from it keeps the join to the bank exact.
    region_ids = [row["region_id"] for row in grid.table()]
    dataset = build_dataset({**cfg["data"]["dataset"], "keep_splits": list(args.splits)})
    n_items = len(dataset) if not args.limit_volumes else min(args.limit_volumes, len(dataset))
    solver = build_solver(exp["solver"])
    n_post = int(exp["n_posterior"])

    bank = read_bank(args.bank)[KEYS + ["s_disagree"]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(out).to_dict("records") if out.exists() else []
    done = {(r["sample_id"], int(r["slice_idx"]), r["operator_id"]) for r in rows}
    total = n_items * len(ops)
    print(f"[dispersion] {total} pairs on {'+'.join(args.splits)}, {len(done)} already done; "
          f"solver on {getattr(solver, 'device', 'cpu')}")

    started, new = time.time(), 0
    for i in range(n_items):
        item = dataset[i]
        for op in ops:
            oid = op.descriptor().operator_id
            if (str(item.sample_id), int(item.slice_idx), oid) in done:
                continue
            # exactly the runner's seed (runner.py: int(run_cfg.get("seed", 0)))
            seed = item_seed(int(cfg["run"].get("seed", 0)), item.sample_id, item.slice_idx,
                             oid)
            sigma = float(op.descriptor().params.get("noise_sigma", 0.0) or 0.0)
            y = add_measurement_noise(op.forward(item.image), sigma, seed)
            samples = solver.sample(y, op, n_samples=n_post, seed=seed)
            for _ in range(NONFINITE_RETRIES):  # P-29, same seed
                if bool(torch.isfinite(samples).all()):
                    break
                samples = solver.sample(y, op, n_samples=n_post, seed=seed)
            if not bool(torch.isfinite(samples).all()):
                print(f"[dispersion] {item.sample_id}/{oid} still non-finite; skipped",
                      file=sys.stderr)
                continue
            comp = compute_score_components(samples, y, op, grid)
            disp = posterior_dispersion(samples, grid, comp.ref_intensity)
            for k, rid in enumerate(region_ids):
                rows.append({"sample_id": str(item.sample_id), "slice_idx": int(item.slice_idx),
                             "operator_id": oid, "region_id": int(rid),
                             "s_dispersion": float(disp[k]),
                             "s_disagree_regenerated": float(comp.s_disagree[k])})
            new += 1
            if new % args.save_every == 0:
                pd.DataFrame(rows).to_parquet(out, index=False)
                per_pair = (time.time() - started) / new
                left = (total - len(done) - new) * per_pair / 3600
                print(f"[dispersion] {len(done) + new}/{total} pairs "
                      f"({per_pair:.1f} s/pair, ~{left:.1f} h left)")
    table = pd.DataFrame(rows)
    table.to_parquet(out, index=False)

    # The draws are only the bank's draws if s_disagree comes back the same.
    check = table.merge(bank, on=KEYS, how="inner")
    if check.empty:
        print("[dispersion] none of these pairs is in the bank; cannot check", file=sys.stderr)
        return 1
    rel = (np.abs(check["s_disagree_regenerated"] - check["s_disagree"])
           / np.maximum(np.abs(check["s_disagree"]), 1e-12))
    print(f"[dispersion] wrote {out}: {len(table):,} rows")
    print(f"[dispersion] reproduction check against the bank's s_disagree over "
          f"{len(check):,} regions: median relative difference {rel.median():.1e}, "
          f"99th percentile {rel.quantile(0.99):.1e}")
    if rel.quantile(0.99) > args.tolerance:
        print("[dispersion] the regenerated draws do NOT match the bank; do not use them",
              file=sys.stderr)
        return 1
    return 0


def evaluate(args) -> int:
    bank = read_bank(args.bank)
    if "patient_id" not in bank.columns:
        print("[dispersion] no patient_id; run scripts/add_patient_ids.py first", file=sys.stderr)
        return 1
    # the reported label (D-061), valued on the design split and frozen
    label = label_from_args(bank, args)
    print(f"[dispersion] label: {label.text}")
    scored = derive_score_and_label(bank, args.calibration_operator, **label.kwargs)
    disp = pd.read_parquet(args.dispersion)[KEYS + ["s_dispersion"]]
    df = scored[scored["split"] == "test"].merge(disp, on=KEYS, how="inner")
    if df.empty:
        print("[dispersion] no test-side rows match the dispersion file", file=sys.stderr)
        return 1
    detectors = {
        "posterior dispersion (full image)": "s_dispersion",
        "s_disagree (null space only)": "s_disagree",
        "CALIBER score": "score",
    }
    rows = []
    for oid, part in df.groupby("operator_id"):
        for name, col in detectors.items():
            for metric_name, metric in (("auroc", auroc), ("ap", average_precision)):
                r = cluster_bootstrap_metric(metric, part[col], part["label"],
                                             part["patient_id"], n_boot=args.boot)
                rows.append({"operator_id": oid, "detector": name, "metric": metric_name, **r})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for metric_name in ("auroc", "ap"):
        wide = t[t["metric"] == metric_name].pivot(index="operator_id", columns="detector",
                                                  values="value")
        print(f"\n[dispersion] {metric_name.upper()} by operator "
              f"({df['patient_id'].nunique()} test patients, {len(df):,} regions)")
        print(wide[list(detectors)].round(3).to_string())
    if args.out:
        t.to_csv(args.out, index=False)
        print(f"\n[dispersion] wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("config")
    g.add_argument("bank", help="the bank of record, for the reproduction check")
    g.add_argument("--out", required=True)
    g.add_argument("--operators", nargs="*", default=None)
    g.add_argument("--splits", nargs="+", default=["test"],
                   help="split sides to regenerate; calib as well for D73's conformal threshold")
    g.add_argument("--limit-volumes", type=int, default=None, help="for a smoke run")
    g.add_argument("--save-every", type=int, default=10)
    g.add_argument("--tolerance", type=float, default=1e-2,
                   help="99th-percentile relative difference allowed (GPU rounding)")
    e = sub.add_parser("evaluate")
    e.add_argument("bank", help="a bank with patient_id (add_patient_ids.py)")
    e.add_argument("--dispersion", required=True)
    e.add_argument("--calibration-operator", required=True)
    add_label_args(e)
    e.add_argument("--boot", type=int, default=300)
    e.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    return generate(args) if args.cmd == "generate" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
