"""The literature's uncertainty-validation protocol, re-run on our own banks (T2-D, D-070).

    python scripts/protocol_replication.py runs/<bank>_by_patient \
        [--dispersion runs/<analysis>/dispersion.parquet] --cluster patient_id \
        --out runs/<analysis>/protocol_<bank>.csv

**The protocol.** Papers that propose an uncertainty map for a generative reconstruction
validate it the same way: the per-pixel posterior standard deviation is correlated with the
absolute error, and an ROC curve is drawn for how well it picks out the high-error regions.
A high correlation and a high AUROC are then read as "the uncertainty is informative".

**What this script asks of it.** D67 found that on this project's banks a magnitude label
is close to a function of the score: both scale with how much content a region holds.
D-070 (point 5) carries that to the protocol above. If a detector that has **no posterior
information at all** scores as well on it as the posterior spread does, the protocol is
measuring scale, and a pass on it is not evidence about uncertainty.

So the script runs the protocol as published, and beside it -- in the same pass, on the
same rows, as D-070's pre-registration requires -- two detectors that cannot know anything
about the posterior:

* ``s_free``: ``||P_N x_hat||`` per region, the null-space content of the mean
  reconstruction. This is Track 1's scale-only control (T1-E).
* ``s_range``: ``||P_R x_hat||`` per region, the measured content. How bright the region is.

**Fixed before any number was read** (D-070's pre-registration for the circularity
analysis): orientation -- a larger detector value predicts a larger error, never the
reverse, and an AUROC below 0.5 is reported as it falls; the high-error label -- a region
is high-error when its ``err_total`` is at or above the 0.90 quantile *of its own
operator* (0.75 is reported beside it, not instead of it); the rows -- the test patients;
and the reading -- a control within 0.05 AUROC of the posterior spread is "matches".

**The posterior spread.** With ``--dispersion`` it is the full-image pixelwise standard
deviation across draws, averaged over the region (``dispersion_baseline.py generate``):
the literature's own quantity. Without it the bank's ``s_disagree`` stands in, which is
the same spread measured in the operator's null space. On a data-consistent solver the
two agree closely, because DDNM pins the range component (T2.6: within 0.012 AUROC on
the fastMRI bank); the output says which one was used.

Every number here is a detection-type reading of a magnitude label. It is a statement
about the *protocol*, not a detection result, and a bank need not carry detection claims
for it to be made (D80, D82): what is reported is that the protocol cannot tell the
detectors apart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from caliber.evaluation.baselines import auroc
from caliber.reportability import describe
from caliber.scorebank import read_bank, read_manifest

KEYS = ["sample_id", "slice_idx", "operator_id", "region_id"]
#: The high-error label's quantile, per operator. Primary first.
QUANTILES = (0.90, 0.75)
#: A control this close to the posterior spread's AUROC "matches" it.
MATCH_MARGIN = 0.05
#: Detectors with no posterior information, and what each is.
CONTROLS = {"s_free": "||P_N x_hat||, scale only", "s_range": "||P_R x_hat||, brightness"}


def spearman(a, b) -> float:
    """Rank correlation, with average ranks for ties. NaN when either side is constant."""
    ra = pd.Series(np.asarray(a, dtype=float)).rank().to_numpy()
    rb = pd.Series(np.asarray(b, dtype=float)).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def high_error(err, quantile: float) -> np.ndarray:
    """The protocol's label: the top ``1 - quantile`` of the error, within these rows."""
    err = np.asarray(err, dtype=float)
    return err >= np.quantile(err, quantile)


def protocol_row(group: pd.DataFrame, detectors: list[str]) -> dict[str, float]:
    """The protocol on one operator's rows: Spearman against ``err_total`` and the AUROC
    for the high-error label, for every detector."""
    out: dict[str, float] = {"n_regions": int(len(group))}
    err = group["err_total"].to_numpy()
    for det in detectors:
        values = group[det].to_numpy()
        out[f"rho_{det}"] = spearman(values, err)
        for q in QUANTILES:
            out[f"auroc{int(q * 100)}_{det}"] = auroc(values, high_error(err, q))
    return out


def cluster_interval(group: pd.DataFrame, cluster: str, column_a: str, column_b: str,
                     n_boot: int, seed: int) -> tuple[float, float]:
    """95% interval for AUROC(a) - AUROC(b) at the primary quantile, resampling whole
    clusters. The label is re-valued inside every resample, as the threshold it depends
    on is a quantile of the resampled rows (D-070's rule for T3-A, applied here)."""
    ids, inverse = np.unique(group[cluster].to_numpy(), return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(ids))]
    err = group["err_total"].to_numpy()
    a, b = group[column_a].to_numpy(), group[column_b].to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = np.concatenate([members[k] for k in rng.integers(0, len(ids), len(ids))])
        label = high_error(err[pick], QUANTILES[0])
        draws.append(auroc(a[pick], label) - auroc(b[pick], label))
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return float("nan"), float("nan")
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bank")
    ap.add_argument("--dispersion", default=None,
                    help="full-image posterior spread from dispersion_baseline.py generate")
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--cluster", default="patient_id",
                    help="column to resample for intervals (sample_id on CT)")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the per-operator table here as CSV")
    args = ap.parse_args(argv)

    bank = Path(args.bank)
    df = read_bank(bank)
    prov = read_manifest(bank).get("provenance") or {}
    df = df[df["split"].isin(args.splits)]
    if args.cluster not in df.columns:
        print(f"[protocol] no {args.cluster!r} column; pass --cluster sample_id, or build "
              "the _by_patient bank (add_patient_ids.py)", file=sys.stderr)
        return 1

    spread = "s_disagree"
    if args.dispersion:
        disp = pd.read_parquet(args.dispersion)[KEYS + ["s_dispersion"]]
        before = len(df)
        df = df.merge(disp, on=KEYS, how="inner")
        spread = "s_dispersion"
        print(f"[protocol] full-image spread joined: {len(df):,} of {before:,} rows have it")
    controls = [c for c in CONTROLS if df[c].notna().all()]
    detectors = [spread] + controls
    if not len(df) or not controls:
        print("[protocol] nothing to compare: no rows on these splits, or no control "
              "column populated", file=sys.stderr)
        return 1

    print(f"[protocol] bank        {bank.name}")
    print(f"[protocol] stamp       {describe(prov)}")
    print(f"[protocol] rows        {len(df):,} regions, {df['operator_id'].nunique()} "
          f"operators, {df[args.cluster].nunique()} {args.cluster}s, splits {args.splits}")
    print(f"[protocol] spread      {spread}"
          + ("  (null-space spread standing in for the full-image one)"
             if spread == "s_disagree" else "  (full-image, the literature's quantity)"))
    for c in controls:
        print(f"[protocol] control     {c}: {CONTROLS[c]}")
    print(f"[protocol] label       err_total at or above its q{int(QUANTILES[0] * 100)} "
          f"within the operator (q{int(QUANTILES[1] * 100)} beside it)")

    primary = int(QUANTILES[0] * 100)
    rows = []
    for op, group in df.groupby("operator_id", sort=True):
        row = {"operator_id": op, **protocol_row(group, detectors)}
        for c in controls:
            lo, hi = cluster_interval(group, args.cluster, c, spread, args.n_boot, args.seed)
            row[f"diff{primary}_{c}_lo"], row[f"diff{primary}_{c}_hi"] = lo, hi
        rows.append(row)
    table = pd.DataFrame(rows)

    print()
    head = f"{'operator':<34}" + "".join(f"{'rho ' + d:>18}" for d in detectors) \
        + "".join(f"{f'AUC{primary} ' + d:>20}" for d in detectors)
    print(head)
    for row in rows:
        print(f"{row['operator_id']:<34}"
              + "".join(f"{row[f'rho_{d}']:>18.3f}" for d in detectors)
              + "".join(f"{row[f'auroc{primary}_{d}']:>20.3f}" for d in detectors))

    print()
    print(f"[protocol] the control against the posterior spread, AUROC at q{primary} "
          f"(control - spread, 95% interval over {args.cluster}s):")
    summary = {}
    for c in controls:
        diff = table[f"auroc{primary}_{c}"] - table[f"auroc{primary}_{spread}"]
        matches = int((diff >= -MATCH_MARGIN).sum())
        beats = int((table[f"diff{primary}_{c}_lo"] > 0).sum())
        summary[c] = (matches, beats)
        print(f"  {c:<8} median AUROC {table[f'auroc{primary}_{c}'].median():.3f} against "
              f"{table[f'auroc{primary}_{spread}'].median():.3f}; within {MATCH_MARGIN} of the "
              f"spread or above it at {matches} of {len(table)} operators; above it with the "
              f"interval clear of zero at {beats}")
    best = max(summary.values())[0]
    print("[protocol] reading     " + (
        f"a detector with no posterior information matches the posterior spread at {best} of "
        f"{len(table)} operators: on this bank the protocol does not separate them"
        if best > len(table) / 2 else
        f"the posterior spread is ahead of every control at most operators ({best} of "
        f"{len(table)} matched): on this bank the protocol does separate them"))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.insert(0, "bank", bank.name)
        table.insert(1, "spread", spread)
        table.to_csv(out, index=False)
        print(f"[protocol] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
