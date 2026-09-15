"""Config-driven experiment runner (Blueprint Sec. 17, Week 1 deliverable).

One command produces one score bank:

    caliber configs/experiments/w01_smoke_phantom.yaml
    caliber configs/experiments/w03_mri_fixed_operator.yaml --set run.limit=4

Every run is fully determined by its config plus its seed, and the resolved
config, its hash, and the environment are written into the bank's manifest.

Structure of a run, mirroring the five stages of Sec. 10:

    for each item, for each operator in the grid:
        y      = A x + noise              measurements  (retrospective)
        x_hat  = solver.sample(y, A)      Stage 1
        P_R/P_N split                     Stage 2   (inside scoring)
        scores = free / disagree / residual   Stage 3
        truth  = per-region error magnitudes
        -> one row per region in the score bank

Calibration (Stage 4) and gap analysis (Stage 5) deliberately do *not* run
here. They read the bank afterwards, on CPU. That separation is what makes the
scope affordable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .config import ConfigError, load_config
from .data import build_dataset
from .operators.registry import OperatorRegistry
from .scorebank.io import ScoreBankWriter, save_reconstructions
from .scoring.labels import CRITERION_VERSION, compute_region_truth
from .scoring.regions import RegionGrid
from .scoring.score import SCORE_VERSION, compute_score_components
from .solvers import build_solver
from .solvers.denoiser import prior_is_admissible


def add_measurement_noise(y: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    """Additive complex or real Gaussian noise at the operator's declared level."""
    if sigma <= 0:
        return y
    gen = torch.Generator().manual_seed(seed)
    if y.is_complex():
        real = torch.randn(y.shape, generator=gen)
        imag = torch.randn(y.shape, generator=gen)
        return y + sigma * (real + 1j * imag) / (2**0.5)
    return y + sigma * torch.randn(y.shape, generator=gen)


def item_seed(base_seed: int, sample_id: str, slice_idx: int, operator_id: str) -> int:
    """The seed for one (item, operator), identical in every process on every machine.

    This used to be ``base_seed + abs(hash((sample_id, slice_idx, operator_id))) % 10**6``.
    Python salts ``hash()`` for strings per process unless ``PYTHONHASHSEED`` is set, and
    nothing set it, so the same item drew a different seed on every run — three
    invocations gave 77818, 549586 and 104573 (P-25). The module docstring's promise that
    a run is "fully determined by its config plus its seed" was false for every bank the
    project had produced.

    A keyed digest has none of that. The ``seed`` column still records the value used,
    which is why older banks remain interpretable even though they cannot be regenerated.
    """
    key = f"{sample_id}|{int(slice_idx)}|{operator_id}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int(base_seed) + int.from_bytes(digest, "big") % 10**6


def calibration_group(operator) -> str:
    """Deprecated shim -- the grouping now lives on the operator descriptor.

    Kept so that existing callers and configs do not break, but the definition
    moved into the registry (D-021) so Track 1 can re-group without a Track 2
    code change. Read ``operator.descriptor().calib_group`` instead.
    """
    return operator.descriptor().calib_group


def run(cfg: dict[str, Any], dry_run: bool = False) -> Path:
    """Execute one experiment and return the score-bank directory."""
    experiment = cfg["experiment"]
    run_cfg = cfg.get("run", {})
    image_shape = tuple(cfg["data"]["image_shape"])

    # Dataset construction is deferred until after the health check, so that
    # --dry-run validates operators on a machine where the data has not landed
    # yet. That is the common case for the CT grid before Week 7.
    registry = OperatorRegistry.from_specs(experiment["operators"], image_shape)
    solver = build_solver(experiment["solver"])
    grid = RegionGrid(image_shape=image_shape, region_size=int(cfg["regions"]["size"]))

    run_id = run_cfg.get("id") or f"{experiment['name']}-{cfg['_meta']['config_hash']}"
    out_root = Path(run_cfg.get("output_root", "runs")) / run_id

    print(f"[caliber] run_id           {run_id}")
    print(f"[caliber] config_hash      {cfg['_meta']['config_hash']}")
    print(f"[caliber] solver           {solver.info().solver_id}  ({solver.family})")
    print(f"[caliber] operators        {len(registry)}")
    print(f"[caliber] regions          {grid.n_regions} of {grid.region_size}px")
    print(f"[caliber] output           {out_root}")

    health = registry.check_health()
    bad = [
        h for h in health
        if not h["adjoint_ok"] or not h["projector_ok"] or not h["contraction_ok"]
    ]
    if bad:
        for h in bad:
            print(
                f"[caliber] OPERATOR CHECK FAILED  {h['operator_id']}  "
                f"adjoint={h['adjoint_rel_error']:.2e}  idem={h['idempotence_rel_error']:.2e}  "
                f"null_gain={h['null_projection_gain']:.3f}",
                file=sys.stderr,
            )
        raise ConfigError(
            "operator health check failed; a wrong or amplifying projector corrupts "
            "every score in the bank. See decision log D-012."
        )
    print(f"[caliber] operator check   {len(health)} operators passed")

    if dry_run:
        registry.write_manifest(out_root / "operator_registry.json")
        print(f"[caliber] dry run - wrote {out_root / 'operator_registry.json'}, stopping")
        return out_root

    dataset = build_dataset(cfg["data"]["dataset"])
    print(f"[caliber] dataset          {dataset.name}  ({len(dataset)} items available)")

    # Three independent ways a run can be development-only: a substrate that is
    # not the real thing (phantoms, the M4Raw stopgap), a sampler that is not a
    # generative prior, or a prior trained on the wrong domain. Any one of them
    # makes the resulting bank unreportable, and a bank carries no visible mark
    # of any of them, so all three are announced here and recorded in the
    # manifest.
    #
    # The third gate is P-18. D-018 originally checked only the first two, and
    # they are not sufficient: `ddpm-celebahq-256` is a real generative prior and
    # fastMRI is a real substrate, so a face model reconstructing knees passed
    # both and produced a bank stamped `reportable: true`. Whether the prior
    # suits the modality is exactly as much a property of the bank as the other
    # two, and it fails closed for unrecognised checkpoints.
    info = solver.info()
    substrate_ok = bool(getattr(dataset, "reportable", True))
    solver_ok = solver.family != "reference"
    prior_ok = solver_ok and prior_is_admissible(info.checkpoint, dataset.modality)

    reportable = substrate_ok and solver_ok and prior_ok
    if not reportable:
        why = []
        if not substrate_ok:
            why.append(f"substrate {dataset.name!r} is for development only")
        if not solver_ok:
            why.append("solver is a reference, not a generative prior")
        elif not prior_ok:
            why.append(
                f"prior domain {info.prior_domain!r} is not appropriate for "
                f"{dataset.modality!r} (D-016, P-18)"
            )
        print(f"[caliber] NOT REPORTABLE   {'; '.join(why)}")

    items = dataset.take(run_cfg.get("limit"))
    n_posterior = int(experiment.get("n_posterior", 8))
    seed = int(run_cfg.get("seed", 0))
    cache_recons = int(run_cfg.get("cache_reconstructions", 0))
    # Items between checkpoints; 0 writes nothing until the end, as before. P-24:
    # at 256² this machine can kill the solver with a silent segfault, and the
    # Week 3 bank took 7.2 hours to build, so a long run should always set this.
    checkpoint_every = int(run_cfg.get("checkpoint_every", 0) or 0)
    region_rows = grid.table()

    writer = ScoreBankWriter(
        out_root,
        run_id=run_id,
        config=cfg,
        provenance={
            "dataset": dataset.name,
            "modality": dataset.modality,
            "solver_family": solver.family,
            "prior_domain": info.prior_domain,
            "reportable": reportable,
        },
    )
    started = time.time()
    n_total = len(items) * len(registry)

    # D-040: pick up an interrupted run of this same config instead of redoing it.
    done = writer.resume_from_checkpoint()
    if done:
        print(
            f"[caliber] resuming         {len(done)}/{n_total} (item, operator) pairs "
            "recovered from the checkpoint (D-040)"
        )
    n_done = n_new = len(done)

    for item in items:
        for op in registry:
            desc = op.descriptor()
            if (str(item.sample_id), int(item.slice_idx), desc.operator_id) in done:
                continue
            seed_here = item_seed(seed, item.sample_id, item.slice_idx, desc.operator_id)

            sigma = float(desc.params.get("noise_sigma", 0.0) or 0.0)
            y = add_measurement_noise(op.forward(item.image), sigma, seed_here)

            samples = solver.sample(y, op, n_samples=n_posterior, seed=seed_here)
            components = compute_score_components(samples, y, op, grid)
            truth = compute_region_truth(
                samples, item.image, op, grid, item.annotation, components.ref_intensity
            )

            writer.add_item(
                base={
                    "config_hash": cfg["_meta"]["config_hash"],
                    "dataset": item.dataset,
                    "split": item.split,
                    "sample_id": item.sample_id,
                    "slice_idx": item.slice_idx,
                    "operator_id": desc.operator_id,
                    "modality": desc.modality,
                    "operator_family": desc.family,
                    "calib_group": desc.calib_group,
                    **solver.info().to_row(),
                    "n_posterior": n_posterior,
                    "seed": seed_here,
                    "score_version": SCORE_VERSION,
                    "criterion_version": CRITERION_VERSION,
                    "ref_intensity": components.ref_intensity,
                },
                region_rows=region_rows,
                arrays={
                    "s_free": components.s_free.tolist(),
                    "s_disagree": components.s_disagree.tolist(),
                    "s_residual": components.s_residual.tolist(),
                    "err_total": truth.err_total.tolist(),
                    "err_free": truth.err_free.tolist(),
                    "free_content": truth.free_content.tolist(),
                    "ann_coverage": truth.ann_coverage.tolist(),
                },
            )

            if n_done < cache_recons:
                save_reconstructions(
                    out_root, f"{item.sample_id}_{item.slice_idx}_{desc.operator_id}",
                    samples.numpy(), item.image.numpy(),
                )

            n_done += 1
            n_new += 1
            if checkpoint_every and n_done % checkpoint_every == 0 and n_done < n_total:
                writer.checkpoint(registry_descriptors=registry.descriptors())
            if n_done % max(1, n_total // 20) == 0 or n_done == n_total:
                # Rate over pairs computed in *this* process, not recovered ones.
                rate = (n_new - len(done)) / max(1e-9, time.time() - started)
                print(f"[caliber] {n_done}/{n_total} items  ({rate:.1f}/s)")

    registry.write_manifest(out_root / "operator_registry.json")
    writer.flush(registry_descriptors=registry.descriptors())
    print(f"[caliber] wrote {len(writer)} rows to {out_root / 'scores.parquet'}")
    print(f"[caliber] elapsed {time.time() - started:.1f}s")
    return out_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="caliber",
        description="Operator-shift-robust conformal hallucination flagging - experiment runner",
    )
    parser.add_argument("config", help="path to an experiment config under configs/")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override a config value, e.g. --set run.limit=4",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="resolve config, build operators, run health checks, then stop",
    )
    parser.add_argument(
        "--print-config", action="store_true", help="print the resolved config and exit"
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, args.set)
    except ConfigError as exc:
        print(f"[caliber] config error: {exc}", file=sys.stderr)
        return 2

    if args.print_config:
        print(json.dumps(cfg, indent=2, default=str))
        return 0

    try:
        run(cfg, dry_run=args.dry_run)
    except (ConfigError, NotImplementedError, FileNotFoundError) as exc:
        print(f"[caliber] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
