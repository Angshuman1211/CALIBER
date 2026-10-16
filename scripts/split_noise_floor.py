"""How much a coverage number moves when only the patient split changes.

T2.4 §8.4 found group-conditional coverage short of nominal on the project's split and
traced it to the split, not the method, by re-splitting the bank 13 ways (mean 0.895, sd
0.028). That was measured against the ratio-only label (P-30). This re-runs it under the
reported label, and extends it to naive transfer, so the headline cells get the same test:
a coverage failure that comes and goes with the split is not a property of the operator.

Each probe split re-draws every patient's side with :func:`split_for` and a different salt,
then re-runs the whole pipeline on it -- including D-060's gate, re-valued on that split's
own design side, because a different split would have produced a different gate. The
project's split (salt ``caliber-split-v1``) is checked against the bank's own ``split``
column first, so the probes are known to be the same procedure.

    python scripts/split_noise_floor.py runs/<run_id>_by_patient \
        --low mri.random1d.R2.c0.08.s0 --high mri.random1d.R8.c0.08.s0 \
        --out runs/<analysis>/split_noise_floor.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from caliber.calibration import apply_threshold, split_threshold
from caliber.data.splits import split_for
from caliber.evaluation import add_label_args, derive_score_and_label, label_from_args
from caliber.evaluation.metrics import empirical_coverage

PROJECT_SALT = "caliber-split-v1"
PROBE_SALTS = [PROJECT_SALT] + [f"probe-{i}" for i in range(12)]
COLUMNS = ["operator_id", "split", "patient_id", "s_free", "s_disagree", "s_residual",
           "err_total", "err_free", "free_content", "ann_coverage"]


def coverage_cells(df: pd.DataFrame, side: pd.Series, low: str, high: str, args,
                   alpha: float) -> tuple[list[dict], float]:
    """Naive and group-conditional coverage at every deployment operator, both directions."""
    split_df = df.assign(split=side.to_numpy())
    label = label_from_args(split_df, args)
    rows = []
    for direction, calib_op in (("ascending", low), ("descending", high)):
        scored = derive_score_and_label(split_df, calib_op, **label.kwargs)
        cal, test = scored[scored["split"] == "calib"], scored[scored["split"] == "test"]
        own = cal[cal["operator_id"] == calib_op]
        naive_tau = split_threshold(own["score"], own["label"], alpha=alpha).threshold
        for oid, part in test.groupby("operator_id"):
            mine = cal[cal["operator_id"] == oid]
            group_tau = split_threshold(mine["score"], mine["label"], alpha=alpha).threshold
            for method, t in (("naive", naive_tau), ("group", group_tau)):
                flagged = apply_threshold(part, t, "score")
                rows.append({"direction": direction, "deployed_at": oid, "method": method,
                             "coverage": empirical_coverage(flagged["flag"], flagged["label"])})
    return rows, label.kwargs.get("threshold")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bank", help="a bank with patient_id (add_patient_ids.py)")
    ap.add_argument("--low", required=True, help="least aggressive operator (ascending)")
    ap.add_argument("--high", required=True, help="most aggressive operator (descending)")
    ap.add_argument("--alpha", type=float, default=0.10)
    add_label_args(ap)
    ap.add_argument("--calib-fraction", type=float, default=0.5)
    ap.add_argument("--design-fraction", type=float, default=0.10)
    ap.add_argument("--cluster", default="patient_id",
                    help="the patient column; sample_id where one subject is one patient "
                         "(TotalSegmentator, D-013)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    nominal = 1.0 - args.alpha

    root = Path(args.bank)
    cols = [args.cluster if c == "patient_id" else c for c in COLUMNS]
    df = pd.read_parquet(root / "scores.parquet" if root.is_dir() else root,
                         columns=cols).rename(columns={args.cluster: "patient_id"})
    df["operator_id"] = df["operator_id"].astype(str)
    patients = df["patient_id"].unique()

    def sides(salt: str) -> pd.Series:
        side = {p: split_for(p, calib_fraction=args.calib_fraction,
                             design_fraction=args.design_fraction, salt=salt) for p in patients}
        return df["patient_id"].map(side)

    if not (sides(PROJECT_SALT) == df["split"]).all():
        print("[splits] the bank's split column does not match split_for with the project "
              "salt and these fractions; refusing to compare probes against it",
              file=sys.stderr)
        return 1

    rows, gates = [], {}
    for salt in PROBE_SALTS:
        cells, gates[salt] = coverage_cells(df, sides(salt), args.low, args.high, args,
                                            args.alpha)
        rows += [{"salt": salt, **c} for c in cells]
    t = pd.DataFrame(rows)
    gate_values = [g for g in gates.values() if g is not None]
    if gate_values:
        print(f"[splits] label threshold re-valued per split on its own design side: "
              f"{min(gate_values):.4f}-{max(gate_values):.4f} ({args.label})")

    print(f"\n[splits] {len(PROBE_SALTS)} patient splits; coverage per cell as mean, sd, "
          f"range, and where the project's split ranks (1 = lowest)")
    summary = []
    for (direction, oid, method), part in t.groupby(["direction", "deployed_at", "method"],
                                                     sort=False):
        cov = part.set_index("salt")["coverage"]
        rank = int((cov < cov[PROJECT_SALT]).sum()) + 1
        summary.append({"direction": direction, "deployed_at": oid, "method": method,
                        "mean": cov.mean(), "sd": cov.std(ddof=1), "min": cov.min(),
                        "max": cov.max(), "project": cov[PROJECT_SALT], "project_rank": rank,
                        "below_nominal": int((cov < nominal).sum())})
    s = pd.DataFrame(summary)
    for direction, part in s.groupby("direction", sort=False):
        print(f"\n[splits] {direction}")
        for _, r in part.iterrows():
            print(f"   {r['deployed_at']:<36} {r['method']:<6} mean {r['mean']:.3f} "
                  f"sd {r['sd']:.3f} [{r['min']:.3f}, {r['max']:.3f}]  project "
                  f"{r['project']:.3f} (rank {r['project_rank']:>2}/{len(PROBE_SALTS)}), "
                  f"below {nominal:.2f} in {r['below_nominal']:>2}")
    group = t[t["method"] == "group"].groupby("salt")["coverage"].mean()
    print(f"\n[splits] group-conditional, mean over cells per split: {group.mean():.3f} "
          f"(sd {group.std(ddof=1):.3f}, range {group.min():.3f}-{group.max():.3f}); "
          f"project split {group[PROJECT_SALT]:.3f}, rank "
          f"{int((group < group[PROJECT_SALT]).sum()) + 1} of {len(group)}")
    if args.out:
        t.to_csv(args.out, index=False)
        s.to_csv(args.out.replace(".csv", "_summary.csv"), index=False)
        print(f"[splits] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
