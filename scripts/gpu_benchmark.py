"""Measure this machine's GPU before trusting it with a bank.

The CPU preflight in ``caliber.resources`` is fitted to measurements, not derived, and a
GPU deserves the same treatment rather than a spec-sheet guess. This reports, on the
real checkpoint and real fastMRI data:

5. **Agreement** -- one UNet pass on GPU against the same pass on CPU. GPU kernels are
   not bit-identical to CPU ones; this shows the difference is rounding, not a bug.
2. **Memory** -- peak GPU memory, and peak host commit, at 1, 2 and 4 draws per batch.
3. **Speed** -- complete item x operator pairs end to end: solver, scoring, truth.
4. **Heat** -- whether successive pairs slow down as the card warms up.

    python scripts/gpu_benchmark.py configs/experiments/w05_mri_full_val_gpu.yaml

Nothing is written to ``runs/``; it measures and prints.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from ctypes import wintypes

import torch

from caliber.config import load_config
from caliber.data import build_dataset
from caliber.operators.registry import OperatorRegistry
from caliber.scoring.labels import compute_region_truth
from caliber.scoring.regions import RegionGrid
from caliber.scoring.score import compute_score_components
from caliber.solvers import build_solver


def peak_host_commit_gb() -> float | None:
    """This process's peak commit charge, Windows only (``PeakPagefileUsage``)."""
    if not sys.platform.startswith("win"):
        return None

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    c = Counters()
    c.cb = ctypes.sizeof(Counters)
    psapi = ctypes.WinDLL("psapi")
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD
    ]
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return None
    return c.PeakPagefileUsage / 1e9


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("--pairs", type=int, default=5, help="end-to-end pairs to time")
    ap.add_argument("--batch", type=int, default=None,
                    help="skip the memory sweep and time pairs at this many draws per pass")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("no CUDA device visible to torch -- a CPU-only build, or no GPU",
              file=sys.stderr)
        return 1
    props = torch.cuda.get_device_properties(0)
    print(f"[gpu] device           {props.name}, {props.total_memory / 1e9:.1f} GB, "
          f"torch {torch.__version__}")

    cfg = load_config(args.config)
    shape = tuple(cfg["data"]["image_shape"])
    exp = cfg["experiment"]
    n_post = int(exp["n_posterior"])
    n_fwd = int(exp["solver"]["n_steps"]) * n_post

    # -- 2. memory and per-pass speed at each batch size ----------------------------
    # Measured first: Windows reports a process's *lifetime* peak commit, and the CPU
    # comparison at the end loads a second model and runs a full CPU pass, which would
    # swamp the GPU-run figure the preflight needs.
    fits = {}
    batches = () if args.batch else (1, 2, 4)
    if batches:
        gpu_solver = build_solver({**exp["solver"], "device": "cuda", "max_draws_per_batch": 0})
        den = gpu_solver.denoiser
        print(f"\n{'draws/batch':>12}{'s per pass':>12}{'s per pair':>12}"
              f"{'GPU peak GB':>13}{'host peak GB':>14}")
    for batch in batches:
        if batch > n_post:
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        xb = torch.randn(batch, *shape)
        try:
            den.predict_noise(xb, 500)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                den.predict_noise(xb, 500)
            torch.cuda.synchronize()
            per_pass = (time.perf_counter() - t0) / 5
        except torch.cuda.OutOfMemoryError:
            print(f"{batch:>12}{'out of GPU memory':>40}")
            torch.cuda.empty_cache()
            continue
        per_pair = per_pass * n_fwd / batch
        vram = torch.cuda.max_memory_reserved() / 1e9
        host = peak_host_commit_gb()
        fits[batch] = per_pair
        host_s = f"{host:.2f}" if host is not None else "n/a"
        print(f"{batch:>12}{per_pass:>12.3f}{per_pair:>12.1f}{vram:>13.2f}{host_s:>14}")

    if args.batch:
        best = args.batch
    else:
        # The smallest batch within 5% of the fastest. On an RTX 3050 one draw and two
        # ran at 12.0 and 11.9 s/pair, but two needed 5.1 GB of the card's 6.4 against
        # 2.8; picking two for 0.1 s left no headroom, and the next model load died.
        fastest = min(fits.values())
        best = min(b for b, t in fits.items() if t <= 1.05 * fastest)
        del gpu_solver, den
    torch.cuda.empty_cache()
    print(f"\n[gpu] chosen setting   max_draws_per_batch: {best if best < n_post else 0} "
          f"({best} draw{'s' if best > 1 else ''} per pass)")

    # -- 3 and 4. real pairs, end to end, watching for slowdown ---------------------
    solver = build_solver({**exp["solver"], "device": "cuda",
                           "max_draws_per_batch": best if best < n_post else 0})
    registry = OperatorRegistry.from_specs(exp["operators"], shape)
    ops = list(registry)
    grid = RegionGrid(image_shape=shape, region_size=int(cfg["regions"]["size"]))
    dataset = build_dataset(cfg["data"]["dataset"])
    print(f"\n[gpu] {args.pairs} real pairs, end to end (solver + scoring + truth):")
    times = []
    for i in range(args.pairs):
        item = dataset[i % len(dataset)]
        op = ops[i % len(ops)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = op.forward(item.image)
        samples = solver.sample(y, op, n_samples=n_post, seed=i)
        comp = compute_score_components(samples, y, op, grid)
        compute_region_truth(samples, item.image, op, grid, item.annotation, comp.ref_intensity)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        print(f"   pair {i + 1}: {times[-1]:6.1f} s   ({op.descriptor().operator_id})"
              f"   finite={bool(torch.isfinite(samples).all())}")

    steady = times[1:] if len(times) > 1 else times  # first pair carries warm-up
    per_pair = sum(steady) / len(steady)
    limit = cfg["run"].get("limit")
    n_pairs = (len(dataset) if limit is None else int(limit)) * len(ops)
    drift = (times[-1] - steady[0]) / steady[0] if len(steady) > 1 else 0.0
    print(f"\n[gpu] measured         {per_pair:.1f} s per pair (excluding warm-up)")
    print(f"[gpu] slowdown         {drift:+.0%} from first to last steady pair "
          f"({'heat is throttling the card' if drift > 0.15 else 'no meaningful slowdown'})")
    print(f"[gpu] full run         {n_pairs} pairs -> ~{per_pair * n_pairs / 3600:.1f} hours")
    host = peak_host_commit_gb()
    if host is not None:
        print(f"[gpu] host peak        {host:.2f} GB committed for the whole GPU path "
              "(the figure the preflight should use)")

    # -- 5. agreement, last: same input, CPU vs GPU ---------------------------------
    cpu_solver = build_solver({**exp["solver"], "device": "cpu", "max_draws_per_batch": 0})
    x = torch.randn(1, *shape, generator=torch.Generator().manual_seed(0))
    eps_gpu = solver.denoiser.predict_noise(x, 500)
    eps_cpu = cpu_solver.denoiser.predict_noise(x, 500)
    diff = (eps_gpu - eps_cpu).abs()
    rel = (diff.norm() / eps_cpu.norm()).item()
    print(f"[gpu] agreement        max |GPU-CPU| = {diff.max().item():.2e}, relative {rel:.2e} "
          f"({'rounding-level' if rel < 1e-2 else 'TOO LARGE - investigate'})")
    del cpu_solver

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
