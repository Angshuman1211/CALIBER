"""Write an analysis copy of a fastMRI bank with a ``patient_id`` column.

Coverage intervals have to be clustered by **patient** (D-027): fastMRI's 199 validation
volumes come from 96 patients, so clustering by ``sample_id`` counts many patients twice
and reports intervals too narrow by roughly sqrt(2). The scripts take
``--cluster patient_id`` "where a bank carries it", and banks written so far do not -- the
loader knows each item's patient, but the runner never wrote it down.

Until the runner records it, this recovers it from the volumes themselves and writes a
**copy**; the bank of record is never modified. It also checks, rather than assumes, that
no patient straddles two split sides (D-022).

    python scripts/add_patient_ids.py runs/<run_id>
    python scripts/sizing.py runs/<run_id>_by_patient --cluster patient_id ...
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


def patient_map(data_root: str | Path) -> dict[str, str]:
    """``{sample_id: patient_id}`` read from the volumes' own ``patient_id`` attribute."""
    import h5py

    patient = {}
    for f in Path(data_root).rglob("*.h5"):
        with h5py.File(f, "r") as fh:
            patient[f.stem] = str(fh.attrs.get("patient_id", "")) or f.stem
    return patient


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bank")
    ap.add_argument("--data-root", default="data/raw/fastmri/singlecoil_val")
    ap.add_argument("--out", default=None, help="default: <bank>_by_patient")
    args = ap.parse_args(argv)

    src = Path(args.bank)
    out = Path(args.out) if args.out else src.with_name(src.name + "_by_patient")
    if out.exists():
        print(f"[patients] {out} already exists; refusing to overwrite it", file=sys.stderr)
        return 1

    patient = patient_map(args.data_root)
    df = pd.read_parquet(src / "scores.parquet")
    df["patient_id"] = df["sample_id"].map(patient)
    missing = df.loc[df["patient_id"].isna(), "sample_id"].unique()
    if len(missing):
        print(f"[patients] no volume found for {len(missing)} sample(s), e.g. {missing[:3]}",
              file=sys.stderr)
        return 1
    straddling = df.groupby("patient_id")["split"].nunique()
    if (straddling > 1).any():
        print(f"[patients] {int((straddling > 1).sum())} patient(s) sit on two sides (D-022)",
              file=sys.stderr)
        return 1

    out.mkdir()
    df.to_parquet(out / "scores.parquet", index=False)
    for name in ("manifest.json", "operator_registry.json"):
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    sides = df.groupby("split")["patient_id"].nunique().to_dict()
    print(f"[patients] {df['patient_id'].nunique()} patients, none straddling; per side {sides}")
    print(f"[patients] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
