"""Dataset interface (task T2.1).

Everything downstream sees the same ``Item`` regardless of modality: a single
2-D slice, already cropped and normalised to the project's working resolution,
plus an optional region-level annotation mask.

Ground truth is required. Not because the method uses it — the whole point is
that the flag works without the true image (Sec. 4.1) — but because *evaluating*
whether a flag fired correctly needs it. Ground truth enters the score bank only
through the ``err_*`` and ``ann_coverage`` columns, never through the score.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class Item:
    """One 2-D slice with optional annotation."""

    dataset: str
    split: str
    sample_id: str
    slice_idx: int
    image: Tensor  # (H, W), real, non-negative, normalised
    annotation: Tensor | None = None  # (H, W) bool, clinically meaningful regions
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image.dim() != 2:
            raise ValueError(f"image must be 2-D, got shape {tuple(self.image.shape)}")
        if self.annotation is not None and self.annotation.shape != self.image.shape:
            raise ValueError("annotation shape must match image shape")


class SliceDataset(ABC):
    """A finite, ordered, deterministic collection of ``Item``."""

    name: str = "unknown"
    modality: str = "unknown"

    #: Whether results from this substrate may appear in the write-up.
    #:
    #: Phantoms and the M4Raw stopgap (D-015) exist so that the pipeline can be
    #: built before fastMRI access clears. Both produce perfectly valid score
    #: banks, which is exactly the hazard: a bank is a table of numbers and
    #: carries no visible mark of the substrate it came from. The flag travels
    #: into the manifest and the runner announces it, so a development bank
    #: cannot quietly become a headline number. It mirrors the same distinction
    #: already drawn for reference solvers.
    reportable: bool = True

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> Item: ...

    def __iter__(self) -> Iterator[Item]:
        for i in range(len(self)):
            yield self[i]

    def take(self, n: int | None) -> list[Item]:
        """First ``n`` items, or all of them. Order must be stable across runs."""
        limit = len(self) if n is None else min(n, len(self))
        return [self[i] for i in range(limit)]


def center_crop(x: Tensor, shape: tuple[int, int]) -> Tensor:
    """Centre-crop or zero-pad ``x`` to ``shape``."""
    height, width = shape
    out = x
    for dim, target in ((-2, height), (-1, width)):
        current = out.shape[dim]
        if current > target:
            start = (current - target) // 2
            out = out.narrow(dim, start, target)
        elif current < target:
            pad_total = target - current
            lo, hi = pad_total // 2, pad_total - pad_total // 2
            pad = [0, 0, 0, 0]
            pad[0 if dim == -1 else 2] = lo
            pad[1 if dim == -1 else 3] = hi
            out = torch.nn.functional.pad(out, pad)
    return out


def normalise(x: Tensor, mode: str = "p99") -> Tensor:
    """Scale intensities to a comparable range across items.

    ``p99`` divides by the 99th percentile, which is robust to the bright
    outliers common in reconstructed medical images, then clamps at zero.
    """
    if mode == "none":
        return x
    if mode == "max":
        denom = float(x.abs().max()) or 1.0
    elif mode == "p99":
        denom = float(torch.quantile(x.abs().flatten().float(), 0.99)) or 1.0
    else:
        raise ValueError(f"unknown normalisation {mode!r}")
    return (x / denom).clamp_min(0.0)
