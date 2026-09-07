"""Stage-2 audit — where a solver's output, error and spread actually land.

Week 2 [B], third item: *begin operator decomposition, range vs. null-space
split*. The projectors were built in Week 1; this is the first time they are
turned on the output of a real pretrained prior and asked what it does with the
part of the image the measurements do not constrain.

It answers, per operator and per solver, three questions the pipeline had so far
assumed the answers to:

  1. Does reconstruction error concentrate in the null space? If it does not,
     the whole premise of scoring the free component is wrong.
  2. Is posterior spread confined to the null space? ``s_disagree`` measures
     spread there and nowhere else (``scoring/score.py``), so a solver whose
     draws disagree in the *measured* component is not being scored on the thing
     the score claims to measure.
  3. Is the split orthogonal enough for those shares to mean anything? Exactly
     so for MRI; approximately for CT (D-006, D-012).

Runs from the same config as the runner, so the operators, image size and solver
are the ones a bank would actually be built with — no separate description of the
experiment to drift out of sync.

    python scripts/stage2_decomposition.py configs/experiments/w02_solver_smoke.yaml
    python scripts/stage2_decomposition.py configs/experiments/w01_smoke_phantom.yaml \
        --limit 4 --out runs/stage2

Writes one JSON artefact per invocation. It is deliberately *not* a score-bank
column: the bank's schema is a cross-track contract and these are diagnostics
about the pipeline, not evidence about an item.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from caliber.config import ConfigError, load_config
from caliber.data import build_dataset
from caliber.operators.decomposition import aggregate_reports, decompose_samples
from caliber.operators.registry import OperatorRegistry
from caliber.runner import add_measurement_noise
from caliber.solvers import build_solver

# Printed with the table. A share is a fraction of split energy, so 0.5 is the
# "no preference" point and anything below it for error or spread contradicts
# the premise rather than merely being weak evidence for it.
NEUTRAL_SHARE = 0.5


def format_table(rows: list[dict]) -> str:
    """Fixed-width table — the artefact is the JSON, this is for reading."""
    header = (
        f"{'operator':<34}{'null_energy':>12}{'null_err':>10}{'null_spread':>13}"
        f"{'dc_err':>9}{'orth_defect':>13}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['operator_id']:<34}"
            f"{row['null_energy_share']:>12.3f}"
            f"{row['null_error_share']:>10.3f}"
            f"{row['null_spread_share']:>13.3f}"
            f"{row['data_consistency_error']:>9.3f}"
            f"{row['orthogonality_defect']:>13.2e}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage-2 range/null-space decomposition audit of solver output"
    )
    ap.add_argument("config", help="an experiment config under configs/")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--limit", type=int, default=None, help="items to audit (default: run.limit)")
    ap.add_argument("--out", default="runs/stage2", help="directory for the JSON artefact")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config, args.set)
    except ConfigError as exc:
        print(f"[stage2] config error: {exc}", file=sys.stderr)
        return 2

    experiment = cfg["experiment"]
    image_shape = tuple(cfg["data"]["image_shape"])
    registry = OperatorRegistry.from_specs(experiment["operators"], image_shape)
    solver = build_solver(experiment["solver"])
    dataset = build_dataset(cfg["data"]["dataset"])

    n_posterior = int(experiment.get("n_posterior", 8))
    seed = int(cfg.get("run", {}).get("seed", 0))
    limit = args.limit if args.limit is not None else cfg.get("run", {}).get("limit")
    items = dataset.take(limit)

    print(f"[stage2] config           {args.config}")
    print(f"[stage2] solver           {solver.info().solver_id}  ({solver.family})")
    print(f"[stage2] operators        {len(registry)}")
    print(f"[stage2] items x draws    {len(items)} x {n_posterior}")

    # The same gate the runner applies. An amplifying or non-adjoint projector
    # makes every number below meaningless, and this script exists precisely to
    # be believed about projectors.
    health = {h["operator_id"]: h for h in registry.check_health()}
    bad = [h for h in health.values() if not (h["adjoint_ok"] and h["contraction_ok"])]
    if bad:
        for h in bad:
            print(f"[stage2] OPERATOR CHECK FAILED  {h['operator_id']}", file=sys.stderr)
        return 1

    started = time.time()
    per_operator: dict[str, list] = {op.descriptor().operator_id: [] for op in registry}
    for item in items:
        for op in registry:
            desc = op.descriptor()
            item_seed = seed + abs(hash((item.sample_id, item.slice_idx, desc.operator_id))) % 10**6
            sigma = float(desc.params.get("noise_sigma", 0.0) or 0.0)
            y = add_measurement_noise(op.forward(item.image), sigma, item_seed)
            samples = solver.sample(y, op, n_samples=n_posterior, seed=item_seed)
            per_operator[desc.operator_id].append(
                decompose_samples(samples, op, x_true=item.image, y=y)
            )

    rows = [aggregate_reports(reports) for reports in per_operator.values() if reports]
    for row in rows:
        row["adjoint_rel_error"] = health[row["operator_id"]]["adjoint_rel_error"]
        row["null_projection_gain"] = health[row["operator_id"]]["null_projection_gain"]

    print()
    print(format_table(rows))
    print()

    weak = [r for r in rows if r["null_error_share"] < NEUTRAL_SHARE]
    if weak:
        print(
            "[stage2] WARNING  error is range-dominated for "
            f"{len(weak)}/{len(rows)} operators — the free-component premise "
            "does not hold here",
            file=sys.stderr,
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{experiment['name']}-{solver.info().solver_id}".replace("/", "_")
    path = out_dir / f"stage2-{stem}.json"
    path.write_text(
        json.dumps(
            {
                "config": args.config,
                "config_hash": cfg["_meta"]["config_hash"],
                "dataset": dataset.name,
                "reportable": bool(getattr(dataset, "reportable", True))
                and solver.family != "reference",
                "solver": solver.info().to_row(),
                "n_items": len(items),
                "n_posterior": n_posterior,
                "elapsed_s": round(time.time() - started, 1),
                "operators": rows,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"[stage2] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
