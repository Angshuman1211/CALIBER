"""Dataset access, preprocessing, and the working-resolution slice interface (T2.1)."""

from __future__ import annotations

from typing import Any

from .base import Item, SliceDataset, center_crop, normalise
from .fastmri import FastMRIDataset, M4RawDataset
from .lodopab import LoDoPaBDataset
from .synthetic import PhantomDataset

DATASETS: dict[str, type[SliceDataset]] = {
    "phantom": PhantomDataset,
    "fastmri": FastMRIDataset,
    "m4raw": M4RawDataset,
    "lodopab": LoDoPaBDataset,
}


def build_dataset(spec: str | dict[str, Any]) -> SliceDataset:
    """Instantiate a dataset from ``"phantom"`` or ``{"name": "fastmri", "root": ...}``."""
    if isinstance(spec, str):
        spec = {"name": spec}
    spec = dict(spec)
    name = spec.pop("name", None)
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; available: {sorted(DATASETS)}")
    return DATASETS[name](**spec)


__all__ = [
    "DATASETS",
    "FastMRIDataset",
    "Item",
    "LoDoPaBDataset",
    "M4RawDataset",
    "PhantomDataset",
    "SliceDataset",
    "build_dataset",
    "center_crop",
    "normalise",
]
