"""Region definition — the unit at which flags are issued and risk is controlled.

Blueprint Sec. 5: a per-region flag marks each region as supported by the
measurements or possibly invented, and calibration controls the rate of missed
hallucinations across regions. The region definition is therefore part of the
controlled-risk definition itself (O2, Week 2) and must be identical in MRI and
CT, so it lives here rather than inside any modality's code.

v0 is a non-overlapping patch grid. If Week 2 replaces it with anatomically
derived regions, only this module changes; the score bank keys on
``region_id`` either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class RegionGrid:
    """Non-overlapping square patches covering the image."""

    image_shape: tuple[int, int]
    region_size: int = 16

    def __post_init__(self) -> None:
        if self.region_size < 1:
            raise ValueError("region_size must be >= 1")
        if min(self.image_shape) < self.region_size:
            raise ValueError(
                f"region_size {self.region_size} exceeds image shape {self.image_shape}"
            )

    @property
    def n_rows(self) -> int:
        return self.image_shape[0] // self.region_size

    @property
    def n_cols(self) -> int:
        return self.image_shape[1] // self.region_size

    @property
    def n_regions(self) -> int:
        return self.n_rows * self.n_cols

    def _tiles(self, x: Tensor) -> Tensor:
        """Reshape ``(..., H, W)`` into ``(..., n_regions, region_size**2)``.

        Trailing pixels that do not fill a whole patch are cropped, so every
        region has identical area and region-level rates are unweighted.
        """
        rs = self.region_size
        cropped = x[..., : self.n_rows * rs, : self.n_cols * rs]
        lead = cropped.shape[:-2]
        tiled = cropped.reshape(*lead, self.n_rows, rs, self.n_cols, rs)
        tiled = tiled.movedim(-2, -3)  # -> (..., n_rows, n_cols, rs, rs)
        return tiled.reshape(*lead, self.n_regions, rs * rs)

    def reduce(self, x: Tensor, how: str = "mean") -> Tensor:
        """Reduce a per-pixel map to one value per region."""
        tiles = self._tiles(x)
        if how == "mean":
            return tiles.mean(dim=-1)
        if how == "max":
            return tiles.max(dim=-1).values
        if how == "sum":
            return tiles.sum(dim=-1)
        if how == "rms":
            return tiles.pow(2).mean(dim=-1).sqrt()
        if how == "std":
            return tiles.std(dim=-1)
        raise ValueError(f"unknown reduction {how!r}")

    def table(self) -> list[dict[str, Any]]:
        """Static per-region geometry, joined into the score bank."""
        rs = self.region_size
        rows = []
        for rid in range(self.n_regions):
            r, c = divmod(rid, self.n_cols)
            rows.append(
                {
                    "region_id": rid,
                    "region_row": r,
                    "region_col": c,
                    "region_area": rs * rs,
                }
            )
        return rows

    def mask_coverage(self, mask: Tensor) -> Tensor:
        """Fraction of each region covered by a binary annotation mask."""
        return self.reduce(mask.to(torch.float32), how="mean")
