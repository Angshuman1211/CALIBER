"""Pretrained reconstruction back-ends behind one common interface (T2.2)."""

from __future__ import annotations

from typing import Any

from .base import PosteriorSampler, SolverInfo
from .denoiser import (
    AnalyticDenoiser,
    Denoiser,
    DiffusersUNetDenoiser,
    NoiseSchedule,
)
from .diffusion import DDNMSampler, DPSSampler
from .flow import FlowDPSSampler
from .reference import NullSpaceNoiseSampler, PseudoInverseSampler

SOLVERS: dict[str, type[PosteriorSampler]] = {
    "dps": DPSSampler,
    "ddnm": DDNMSampler,
    "flowdps": FlowDPSSampler,
    "reference_nullspace": NullSpaceNoiseSampler,
    "reference_pinv": PseudoInverseSampler,
}


def build_solver(spec: str | dict[str, Any]) -> PosteriorSampler:
    """Instantiate a solver from ``"dps"`` or ``{"name": "dps", "n_steps": 200}``."""
    if isinstance(spec, str):
        spec = {"name": spec}
    spec = dict(spec)
    name = spec.pop("name", None)
    if name not in SOLVERS:
        raise ValueError(f"unknown solver {name!r}; available: {sorted(SOLVERS)}")
    return SOLVERS[name](**spec)


__all__ = [
    "SOLVERS",
    "AnalyticDenoiser",
    "DDNMSampler",
    "DPSSampler",
    "Denoiser",
    "DiffusersUNetDenoiser",
    "FlowDPSSampler",
    "NoiseSchedule",
    "NullSpaceNoiseSampler",
    "PosteriorSampler",
    "PseudoInverseSampler",
    "SolverInfo",
    "build_solver",
]
