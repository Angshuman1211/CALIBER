"""The mask-seed noise floor at one operator, beside the patient-split noise floor.

Track 1 asked for it after D-057: every ablation cell in T1.4 §6 is read against the noise
in a coverage number, and an effect smaller than that noise is not an effect. T2.4 §8.4
measured one source - the patient split (sd 0.028 across 13 splits). This measures the
other - the random mask itself, same operator and acceleration, different seed - and puts
the two side by side.

Three questions, each on test patients with 95% intervals by patient (D-027):

1. **Per seed** - calibrate and score on the same seed. The spread across seeds is the
   seed noise floor.
2. **Is a seed change a shift?** - calibrate on seed 0, deploy on seeds 1 and 2; and the
   pooled group D-023 prescribes (all seeds one calibration group). If seed 0's threshold
   loses coverage on another seed, D-023's "the seed is inside the group" needs revisiting.
3. **Seed against split** - the per-seed coverage under 13 patient splits, so the two
   noise sources are measured on the same footing.

Seed 0 comes from the main bank and seeds 1-2 from the seed bank. That is sound only because
draws are a pure function of the config and pair (D-041) - and the merge is checked, not
assumed: every seed must cover the same volumes, each on the same split side.

    python scripts/seed_noise_floor.py runs/<seed_run_id> --base-bank runs/<w05_run_id> \
        --operator mri.random1d.R4.c0.08 --out runs/<analysis>/seed_noise_floor.csv
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from add_patient_ids import patient_map

from caliber.calibration import apply_threshold, split_threshold
from caliber.data.splits import split_for
from caliber.evaluation import (
    add_label_args,
    cluster_bootstrap_coverage,
    derive_score_and_label,
    label_from_args,
)
from caliber.evaluation.metrics import empirical_coverage
from caliber.scorebank import read_bank

PROBE_SALTS = ["caliber-split-v1"] + [f"probe-{i}" for i in range(12)]


def coverage_at(cal: pd.DataFrame, test: pd.DataFrame, alpha: float):
    tau = split_threshold(cal["score"], cal["label"], alpha=alpha).threshold
    flagged = apply_threshold(test, tau, "score")
    return flagged


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("seed_bank", help="bank holding the extra seeds")
    ap.add_argument("--base-bank", required=True, help="bank holding seed 0")
    ap.add_argument("--operator", required=True,
                    help="operator id without the seed suffix, e.g. mri.random1d.R4.c0.08")
    ap.add_argument("--data-root", default="data/raw/fastmri/singlecoil_val")
    ap.add_argument("--alpha", type=float, default=0.10)
    add_label_args(ap)
    ap.add_argument("--calib-fraction", type=float, default=0.5)
    ap.add_argument("--design-fraction", type=float, default=0.10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    nominal = 1.0 - args.alpha

    prefix = args.operator + ".s"
    base = read_bank(args.base_bank)
    # The gate is valued on the base bank's design split over every operator, so it is the
    # same number coverage_by_patient.py uses -- a seed run must not re-fit its own label.
    label = label_from_args(base, args)
    extra = read_bank(args.seed_bank)
    df = pd.concat([base[base["operator_id"].astype(str).str.startswith(prefix)],
                    extra[extra["operator_id"].astype(str).str.startswith(prefix)]],
                   ignore_index=True)
    seeds = sorted(df["operator_id"].astype(str).unique())
    if len(seeds) < 2:
        print(f"[seeds] found {seeds}; need at least two seeds of {args.operator}",
              file=sys.stderr)
        return 1

    # the merge is only coherent if every seed saw the same volumes on the same sides
    vols = {s: set(df.loc[df["operator_id"].astype(str) == s, "sample_id"]) for s in seeds}
    sides = df.groupby("sample_id")["split"].nunique()
    if len({frozenset(v) for v in vols.values()}) != 1 or (sides > 1).any():
        print("[seeds] the seeds do not share volumes and split sides; refusing to merge",
              file=sys.stderr)
        return 1

    df["patient_id"] = df["sample_id"].map(patient_map(args.data_root))
    df = derive_score_and_label(df, seeds[0], **label.kwargs)
    df["operator_id"] = df["operator_id"].astype(str)
    cal, test = df[df["split"] == "calib"], df[df["split"] == "test"]
    print(f"[seeds] label: {label.text}, valued on the base bank")
    print(f"[seeds] {args.operator}: seeds {[s.split('.s')[-1] for s in seeds]}, "
          f"{df['sample_id'].nunique()} volumes, "
          f"{test['patient_id'].nunique()} test patients")

    rows = []

    def record(question, calibrated_on, deployed_on, flagged):
        s = cluster_bootstrap_coverage(flagged["flag"], flagged["label"], flagged["patient_id"])
        rows.append({"question": question, "calibrated_on": calibrated_on,
                     "deployed_on": deployed_on, "coverage": s["coverage"],
                     "lo": s["lo"], "hi": s["hi"], "flag_area": float(flagged["flag"].mean())})

    # 1. per seed, calibrated and scored on itself
    for s in seeds:
        record("per_seed", s, s, coverage_at(cal[cal["operator_id"] == s],
                                             test[test["operator_id"] == s], args.alpha))
    # 2a. seed 0's threshold carried to the other seeds
    for s in seeds[1:]:
        record("seed_as_shift", seeds[0], s, coverage_at(cal[cal["operator_id"] == seeds[0]],
                                                         test[test["operator_id"] == s],
                                                         args.alpha))
    # 2b. D-023's pooled group: one threshold from every seed's calibration rows
    for s in seeds:
        record("pooled_group", "all seeds", s, coverage_at(cal, test[test["operator_id"] == s],
                                                           args.alpha))

    table = pd.DataFrame(rows)
    short = {s: "seed " + s.split(".s")[-1] for s in seeds}
    pd.set_option("display.width", 200)
    for question, part in table.groupby("question", sort=False):
        print(f"\n[seeds] {question}")
        for _, r in part.iterrows():
            print(f"   calibrated on {short.get(r['calibrated_on'], r['calibrated_on']):<10} "
                  f"deployed on {short[r['deployed_on']]:<7} coverage {r['coverage']:.3f} "
                  f"[{r['lo']:.3f}, {r['hi']:.3f}]  flag area {r['flag_area']:.3f}")
    per_seed = table[table["question"] == "per_seed"]["coverage"]
    print(f"\n[seeds] seed noise floor: per-seed coverage sd {per_seed.std(ddof=1):.3f}, "
          f"range {per_seed.max() - per_seed.min():.3f}")

    # 3. seed against split: per-seed coverage under every probe split
    patients = df["patient_id"].unique()
    grid = np.zeros((len(PROBE_SALTS), len(seeds)))
    for i, salt in enumerate(PROBE_SALTS):
        side = {p: split_for(p, calib_fraction=args.calib_fraction,
                             design_fraction=args.design_fraction, salt=salt) for p in patients}
        s_side = df["patient_id"].map(side)
        for j, s in enumerate(seeds):
            mine = df["operator_id"] == s
            flagged = coverage_at(df[mine & (s_side == "calib")], df[mine & (s_side == "test")],
                                  args.alpha)
            grid[i, j] = empirical_coverage(flagged["flag"], flagged["label"])
    sd_seed = float(np.mean(grid.std(axis=1, ddof=1)))   # across seeds, within a split
    sd_split = float(np.mean(grid.std(axis=0, ddof=1)))  # across splits, within a seed
    print(f"\n[seeds] {len(PROBE_SALTS)} splits x {len(seeds)} seeds, mean coverage "
          f"{grid.mean():.3f} (nominal {nominal:.2f})")
    print(f"[seeds] sd across seeds, within a split  {sd_seed:.3f}")
    print(f"[seeds] sd across splits, within a seed  {sd_split:.3f}")
    print("[seeds] the larger of the two is the floor an ablation effect has to clear")

    if args.out:
        table.to_csv(args.out, index=False)
        pd.DataFrame(grid, index=PROBE_SALTS, columns=[short[s] for s in seeds]).to_csv(
            args.out.replace(".csv", "_by_split.csv"))
        print(f"[seeds] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
