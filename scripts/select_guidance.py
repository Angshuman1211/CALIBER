"""Pick DPS's guidance strength from a design-split sweep, by D-031's rule (T2.10).

    python scripts/select_guidance.py runs/w11_analysis/dps_sweep --prefix w11_dps_sweep_z

Reads the ``stage2_decomposition.py`` artefacts of a sweep over ``step_size`` and prints
the chosen value. D-031: one guidance strength, fixed across every operator in a curve,
chosen on data consistency, and re-measured for each prior. D-033: a setting whose error is
not mostly in the null space is not producing hallucinations to score, so it is excluded
first. Among the rest the choice is the smallest **mean** data-consistency error over the
operators; a mean, so no single operator decides it.

Refuses (exit 1) when no swept value is admissible. A DPS bank built on an inadmissible
setting would answer D68 about an unconverged solver, which is the failure T2.9 had to
state beside every R2 number.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: D-033's admissibility line: more than half the reconstruction error in the null space.
MIN_NULL_SHARE = 0.5


def read_sweep(directory: str | Path, prefix: str) -> list[dict]:
    rows = []
    for path in sorted(Path(directory).glob("stage2-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = Path(payload["config"]).stem
        m = re.search(r"z([0-9.]+)", path.name)
        if prefix not in path.name or not m:
            continue
        ops = payload["operators"]
        rows.append({
            "step_size": float(m.group(1)),
            "null_share": sum(o["null_error_share"] for o in ops) / len(ops),
            "min_null_share": min(o["null_error_share"] for o in ops),
            "dc_error": sum(o["data_consistency_error"] for o in ops) / len(ops),
            "n_items": payload["n_items"],
            "config": name,
        })
    return sorted(rows, key=lambda r: r["step_size"])


def choose(rows: list[dict]) -> dict | None:
    """The admissible setting with the smallest mean data-consistency error."""
    ok = [r for r in rows if r["min_null_share"] > MIN_NULL_SHARE]
    return min(ok, key=lambda r: r["dc_error"]) if ok else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directory")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args(argv)
    rows = read_sweep(args.directory, args.prefix)
    if not rows:
        print(f"[guidance] no sweep artefacts in {args.directory}", file=sys.stderr)
        return 1
    print(f"{'step_size':>10}{'null share':>12}{'min null':>10}{'dc error':>10}")
    for r in rows:
        print(f"{r['step_size']:>10g}{r['null_share']:>12.4f}{r['min_null_share']:>10.4f}"
              f"{r['dc_error']:>10.4f}")
    best = choose(rows)
    if best is None:
        print(f"[guidance] no swept value keeps every operator's null-error share above "
              f"{MIN_NULL_SHARE} (D-033); no DPS bank should be built from this sweep",
              file=sys.stderr)
        return 1
    edge = best["step_size"] in (rows[0]["step_size"], rows[-1]["step_size"])
    print(f"[guidance] chosen step_size {best['step_size']:g} (mean dc error "
          f"{best['dc_error']:.4f}, mean null share {best['null_share']:.4f})"
          + ("  -- at the edge of the sweep, so the optimum is not bracketed" if edge else ""))
    print(f"STEP_SIZE={best['step_size']:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
