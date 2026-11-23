"""Re-stamp finished banks with the claims they carry (D80, D82; T2-B of the Phase 4 plan).

    python scripts/restamp_bank.py runs/<bank> [runs/<bank> ...]            # show
    python scripts/restamp_bank.py runs/<bank> [runs/<bank> ...] --write    # apply

A bank's manifest records whether it is reportable. Until D82 that was one boolean,
decided by the code as it stood when the bank was built. Two rulings have since scoped
it: a prior of another anatomy (D80) and a substrate without annotations (D82) each
leave a bank its coverage claims and take its detection claims. Banks built before
``reportable_claims`` existed say all-or-nothing, and for the M4Raw and medical-prior CT
banks that is now wrong in both directions.

This recomputes the stamp from what the manifest already records -- the substrate, the
sampler family and the checkpoint -- with today's rules (``caliber.reportability``). It
changes ``manifest.json`` only: no row of the bank is read or written. The previous
values are kept under ``provenance.restamped`` so the edit is visible in the file it
was made to.

It refuses a manifest whose substrate or checkpoint it cannot identify, rather than
guessing: a stamp is a claim about what may be published. A bank built before its prior
was registered ("unknown") takes the domain the registry gives that checkpoint today.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from caliber.data import DATASETS
from caliber.reportability import CLAIMS, UNDECLARED_ANATOMY, claims_for
from caliber.scorebank import read_manifest
from caliber.solvers.denoiser import UNKNOWN_DOMAIN, prior_domain

#: Which ruling a re-stamp applies, recorded beside the previous values.
RULING = "D80, D82"


def restamped(manifest: dict) -> dict:
    """The manifest's provenance under today's rules. Raises ``ValueError`` when the
    manifest does not say enough to decide."""
    prov = dict(manifest.get("provenance") or {})
    name = prov.get("dataset")
    if name not in DATASETS:
        raise ValueError(f"substrate {name!r} is not a registered dataset")
    family = prov.get("solver_family")
    if not family:
        raise ValueError("the manifest records no solver family")
    solver = ((manifest.get("config") or {}).get("experiment") or {}).get("solver") or {}
    checkpoint = solver.get("checkpoint")
    dataset = DATASETS[name]
    # A reference sampler has no checkpoint and carries nothing whatever it is; for a
    # generative one the domain must be the one the bank was built under, or the config
    # and the provenance describe different priors.
    built_as, today = prov.get("prior_domain"), prior_domain(checkpoint)
    # One mismatch is expected: a bank built before its prior was registered recorded
    # "unknown", and the registry has since been told what the checkpoint is (the LIDC
    # smoke bank, built a week before D80 admitted `ct_chest`). Any other mismatch means
    # the name now points at something else.
    if family != "reference" and built_as != today and built_as != UNKNOWN_DOMAIN:
        raise ValueError(
            f"the config's checkpoint {checkpoint!r} is {today!r} today, "
            f"but the bank was built as {built_as!r}"
        )
    claims = claims_for(dataset, family, checkpoint)
    previous = {k: prov[k] for k in ("reportable", "reportable_claims") if k in prov}
    if family != "reference" and built_as != today:
        previous["prior_domain"] = built_as
        prov["prior_domain"] = today
    prov["anatomy"] = getattr(dataset, "anatomy", UNDECLARED_ANATOMY)
    prov["reportable"] = claims == CLAIMS
    prov["reportable_claims"] = sorted(claims)
    if previous != {k: prov[k] for k in ("reportable", "reportable_claims")}:
        # keep the first record of what the bank said when it was built
        prov.setdefault("restamped", {"ruling": RULING, "previous": previous})
    return prov


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("banks", nargs="+", help="bank directories (each holds a manifest.json)")
    ap.add_argument("--write", action="store_true", help="rewrite the manifests (default: show)")
    args = ap.parse_args(argv)

    status = 0
    for bank in map(Path, args.banks):
        try:
            manifest = read_manifest(bank)
            prov = restamped(manifest)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[restamp] {bank.name}: REFUSED - {exc}", file=sys.stderr)
            status = 1
            continue
        old = manifest.get("provenance") or {}
        was = old.get("reportable_claims", "all" if old.get("reportable") else "none")
        now = prov["reportable_claims"] or "none"
        changed = old != prov
        print(f"[restamp] {bank.name}: {prov['dataset']} + {prov.get('prior_domain')}  "
              f"claims {was} -> {now}{'' if changed else '  (unchanged)'}")
        if args.write and changed:
            manifest["provenance"] = prov
            path = bank / "manifest.json"
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
            tmp.replace(path)
    if not args.write:
        print("[restamp] nothing written; pass --write to apply")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
