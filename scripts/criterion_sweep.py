"""What tau does the relative hallucination criterion need? (O2, A1 / D21)

The Week 1 sign-off fixed the criterion's *form* — ``err_free / free_content >
tau`` — and deferred the value: "tau is set only after a real prior lands
(T2.2), and its positive rate is then reported, not targeted." A real prior has
now landed, so this is Track 2 discharging that: it supplies the curve, and
Track 1 picks the point on it (the criterion is theirs, O2).

It also puts a number on P-02. The v0 absolute criterion ``err_free > 0.10``
flags essentially every region on a real substrate, which makes region FDR and
power uncomputable. The relative form exists to fix that, and the question is
whether it actually separates anything once a real prior — rather than the
reference sampler, which invents by construction (D-018) — is producing the
reconstructions.

Reads a cached score bank. Touches no GPU, no dataset and no checkpoint, which
is the whole point of Sec. 17.

    python scripts/criterion_sweep.py runs/<run_id>
    python scripts/criterion_sweep.py runs/<run_id> --out runs/criterion
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from caliber.scorebank import read_bank, read_manifest
from caliber.scoring.labels import MIN_FREE_CONTENT, positive_rate_sweep

# Spanning the plausible range rather than a guess at the answer. Below 1 the
# criterion says "the reconstruction put anything at all in the free component";
# at 1 it says "more than the truth's own free content"; above 2 it is asking
# for gross invention.
DEFAULT_TAUS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)

# The v0 absolute criterion, for the side-by-side that makes P-02 concrete.
V0_ABSOLUTE_THRESHOLD = 0.10


def summarise(df, taus, min_free_content) -> dict:
    err = torch.tensor(df["err_free"].to_numpy(), dtype=torch.float32)
    content = torch.tensor(df["free_content"].to_numpy(), dtype=torch.float32)
    usable = content > min_free_content
    ratio = err[usable] / content[usable]
    q = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n_regions": int(len(df)),
        "n_usable": int(usable.sum()),
        "absolute_positive_rate": float((err > V0_ABSOLUTE_THRESHOLD).to(torch.float32).mean()),
        "ratio_min": float(ratio.min()),
        "ratio_max": float(ratio.max()),
        "ratio_mean": float(ratio.mean()),
        "ratio_quantiles": {
            f"q{int(p * 100):02d}": float(v)
            for p, v in zip(q.tolist(), torch.quantile(ratio, q).tolist(), strict=True)
        },
        "relative_positive_rate": dict(positive_rate_sweep(err, content, taus, min_free_content)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Report the relative criterion's tau curve")
    ap.add_argument("bank", help="score-bank directory produced by the runner")
    ap.add_argument("--taus", type=float, nargs="*", default=list(DEFAULT_TAUS))
    ap.add_argument("--min-free-content", type=float, default=MIN_FREE_CONTENT)
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args(argv)

    bank = Path(args.bank)
    df = read_bank(bank)
    manifest = read_manifest(bank)

    if "free_content" not in df.columns or df["free_content"].isna().all():
        print(
            f"[criterion] {bank} has no free_content column -- it predates A1 (D21) and "
            "the relative criterion cannot be applied to it.",
            file=sys.stderr,
        )
        return 1

    df = df[df["free_content"].notna()]
    provenance = manifest.get("provenance", {})
    reportable = provenance.get("reportable")

    print(f"[criterion] bank             {bank}")
    print(f"[criterion] solver           {sorted(df['solver_id'].unique())}")
    print(f"[criterion] substrate        {sorted(df['dataset'].unique())}  reportable={reportable}")
    print(f"[criterion] regions          {len(df)} over {df['operator_id'].nunique()} operators")
    if reportable is False:
        print("[criterion] NOT REPORTABLE   tau read off this bank is provisional")

    overall = summarise(df, args.taus, args.min_free_content)

    print()
    print(
        f"v0 absolute criterion, err_free > {V0_ABSOLUTE_THRESHOLD}: "
        f"flags {100 * overall['absolute_positive_rate']:.1f}% of regions"
    )
    qs = overall["ratio_quantiles"]
    print(
        "relative ratio err_free/free_content: "
        f"min {overall['ratio_min']:.3f}  q25 {qs['q25']:.3f}  median {qs['q50']:.3f}  "
        f"q75 {qs['q75']:.3f}  max {overall['ratio_max']:.3f}"
    )
    print()
    print(f"{'tau':>6}{'positive rate':>16}")
    print("-" * 22)
    for tau, rate in overall["relative_positive_rate"].items():
        print(f"{float(tau):>6.2f}{100 * rate:>15.1f}%")

    by_group = {}
    for key in ("operator_id", "calib_group"):
        if key not in df.columns:
            continue
        by_group[key] = {
            str(name): summarise(sub, args.taus, args.min_free_content)
            for name, sub in df.groupby(key, observed=True)
        }

    print()
    # Plain hyphen, not an em dash: the Windows console mangles non-ASCII in
    # this project's environment (P-09), and a report nobody can read is not one.
    print("median ratio by operator - the criterion must not be a proxy for the operator:")
    for name, stats in by_group.get("operator_id", {}).items():
        print(f"  {name:<34}{stats['ratio_quantiles']['q50']:.3f}")

    report = {
        "bank": str(bank),
        "solvers": sorted(df["solver_id"].unique().tolist()),
        "datasets": sorted(df["dataset"].unique().tolist()),
        "reportable": reportable,
        "criterion_version": sorted(df["criterion_version"].unique().tolist()),
        "min_free_content": args.min_free_content,
        "absolute_threshold": V0_ABSOLUTE_THRESHOLD,
        "overall": overall,
        "by": by_group,
    }
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"criterion-{bank.name}.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\n[criterion] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
