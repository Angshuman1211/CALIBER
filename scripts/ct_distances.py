"""Every operator distance on a handful of real CT pairs (D-061, D64; T1.3's cross-modality item).

D-052 found four of the five candidate distances coincide on MRI masks, because a mask is
diagonal in k-space and so "which frequencies it keeps" is its range. CT view subsampling is
not diagonal, which makes CT the only operator class in the project where the measures can
differ at all. D-061 therefore asked for ~k+1 pairs, not the matrix:

- the calibration operator against each deployment operator (the CT x-axis);
- one arc change at fixed views and one view change at fixed arc, the pairs that test whether
  range and null space have come apart;
- ``spectral_divergence`` on all of them, its first chance to run, as it never did on MRI.

The operators are rebuilt from the bank's own manifest, so they are exactly the ones the bank
used. Distances are operator-only; no image is involved. Each finished (pair, measure) is
appended to the CSV, so an interrupted run resumes where it stopped. ~10 min per pair on CPU
at 256^2, most of it CG solves inside the Hutchinson probes and the chordal subspace estimate.

    python scripts/ct_distances.py runs/<ct_run_id> --out runs/<analysis>/ct_distances.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from caliber.operators.distance import DISTANCES, SIGNED_MEASURES, compute_distance
from caliber.operators.registry import build_operator

CALIBRATION = "ct.sparse_view.v180.a180.dfull"
EXTRA = [
    # arc change at fixed views: 120 views over 180 degrees against 120 over 90
    ("ct.sparse_view.v120.a180.dfull", "ct.limited_angle.v120.a90.dfull"),
    # view change at fixed arc: 120 views against 60, both over 180 degrees
    ("ct.sparse_view.v120.a180.dfull", "ct.sparse_view.v60.a180.dfull"),
]
FIELDS = ["calibration", "deployment", "measure", "value", "signed", "seconds", "note"]
_MOVED = ("forward", "adjoint", "pinv", "decomposition_pinv", "project_range", "project_null")


class OnDevice:
    """An operator whose heavy calls run on ``device``; everything else is the operator's own.

    The distance measures draw their probes on the CPU, seeded, exactly as they always have.
    Only the operator applications move -- above all the CG solves inside the projections,
    4-6x faster on the GPU at 256^2 (T2.6). Results come back to the CPU, so the measures
    see the same numbers to rounding (CPU vs GPU agree to ~1e-4 on a CG solve).
    """

    def __init__(self, op, device: str):
        self._op, self._device = op, device

    def __getattr__(self, name):
        attr = getattr(self._op, name)
        if name not in _MOVED:
            return attr
        def moved(x, *a, **k):
            return attr(x.to(self._device), *a, **k).to("cpu")
        return moved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bank", help="a CT bank; its manifest supplies the operators")
    ap.add_argument("--calibration", default=CALIBRATION)
    ap.add_argument("--device", default=None, help="run the operators on this device, e.g. cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    manifest = json.loads((Path(args.bank) / "manifest.json").read_text(encoding="utf-8"))
    ops = {row["operator_id"]: build_operator({"modality": row["modality"], **row["params"]},
                                              tuple(row["image_shape"]))
           for row in manifest["operators"]}
    if args.device:
        ops = {k: OnDevice(v, args.device) for k, v in ops.items()}
    if args.calibration not in ops:
        raise SystemExit(f"{args.calibration} is not in this bank: {sorted(ops)}")
    # EXTRA names CT operators explicitly, so it is filtered to what this bank actually
    # holds. Without the filter, pointing this at an MRI bank -- which is what
    # curve_collapse.py needs for its MRI distances, and the manifest makes possible since
    # the operators are rebuilt generically -- writes the real rows and then dies on a
    # KeyError for a CT id, and the `except ValueError` below does not catch it.
    extra = [(a, b) for a, b in EXTRA if a in ops and b in ops]
    pairs = [(args.calibration, o) for o in sorted(ops) if o != args.calibration] + extra
    if len(extra) < len(EXTRA):
        print(f"[distances] {len(EXTRA) - len(extra)} of the named extra pair(s) are not in "
              "this bank and were skipped", file=sys.stderr)

    out = Path(args.out)
    done = set()
    if out.is_file():
        with out.open(encoding="utf-8") as fh:
            done = {(r["calibration"], r["deployment"], r["measure"]) for r in csv.DictReader(fh)}
    new_file = not out.is_file()
    with out.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for k, (a, b) in enumerate(pairs, 1):
            for measure in DISTANCES:
                if (a, b, measure) in done:
                    continue
                t0 = time.time()
                note, value = "", float("nan")
                try:
                    value = compute_distance(ops[a], ops[b], measure)
                except ValueError as exc:  # e.g. operator_norm across sinogram shapes
                    note = str(exc)[:160]
                writer.writerow({"calibration": a, "deployment": b, "measure": measure,
                                 "value": value, "signed": measure in SIGNED_MEASURES,
                                 "seconds": round(time.time() - t0, 1), "note": note})
                fh.flush()
                print(f"[distances] pair {k}/{len(pairs)} {b:<34} {measure:<22} "
                      f"{value:+.4f} ({time.time() - t0:.0f} s) {note[:60]}", flush=True)
    print(f"[distances] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
