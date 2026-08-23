"""Deterministic synthetic phantoms — development substrate, never a result.

Two jobs:

* unblock all three tracks in Week 1, before fastMRI access clears, by making
  the pipeline end-to-end runnable and producing a real score bank Track 1 can
  calibrate against;
* provide the small synthetic linear-operator setting Track 3 needs for the
  numerical verification of L1 and L2 (task T3.5), which the Blueprint notes
  requires negligible compute.

The phantom carries a planted "lesion" ellipse used as the annotation mask, so
the annotation-restricted evaluation path is exercised from day one rather than
first meeting real data in Week 3.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .base import Item, SliceDataset


def _ellipse(shape: tuple[int, int], cy: float, cx: float, ry: float, rx: float, angle: float):
    height, width = shape
    yy = torch.linspace(-1.0, 1.0, height).view(-1, 1)
    xx = torch.linspace(-1.0, 1.0, width).view(1, -1)
    ys, xs = yy - cy, xx - cx
    cos, sin = math.cos(angle), math.sin(angle)
    yr = ys * cos + xs * sin
    xr = -ys * sin + xs * cos
    return ((yr / ry) ** 2 + (xr / rx) ** 2) <= 1.0


class PhantomDataset(SliceDataset):
    """Shepp-Logan-flavoured phantoms with a randomly placed lesion."""

    name = "synthetic_phantom"
    modality = "synthetic"
    reportable = False  # an interface smoke test, never a result

    def __init__(
        self,
        n_items: int = 32,
        image_shape: tuple[int, int] = (128, 128),
        split: str = "calib",
        seed: int = 0,
        lesion: bool = True,
    ):
        self.n_items = int(n_items)
        self.image_shape = tuple(image_shape)
        self.split = split
        self.seed = int(seed)
        self.lesion = lesion

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, index: int) -> Item:
        if not 0 <= index < self.n_items:
            raise IndexError(index)
        # Seeded per item, so an item is identical regardless of iteration order.
        gen = torch.Generator().manual_seed(self.seed * 100_003 + index)
        shape = self.image_shape
        img = torch.zeros(*shape)

        def rnd(lo: float, hi: float) -> float:
            return float(torch.rand(1, generator=gen)) * (hi - lo) + lo

        img[_ellipse(shape, 0.0, 0.0, 0.85, 0.70, 0.0)] = 0.30
        img[_ellipse(shape, 0.0, 0.0, 0.78, 0.62, 0.0)] = 0.55
        for _ in range(4):
            img[
                _ellipse(
                    shape,
                    rnd(-0.35, 0.35),
                    rnd(-0.30, 0.30),
                    rnd(0.08, 0.22),
                    rnd(0.08, 0.22),
                    rnd(0.0, math.pi),
                )
            ] = rnd(0.15, 0.95)

        annotation = None
        if self.lesion:
            mask = _ellipse(shape, rnd(-0.4, 0.4), rnd(-0.35, 0.35), rnd(0.05, 0.11),
                            rnd(0.05, 0.11), rnd(0.0, math.pi))
            img[mask] = rnd(0.75, 1.0)
            annotation = mask

        return Item(
            dataset=self.name,
            split=self.split,
            sample_id=f"phantom{index:04d}",
            slice_idx=0,
            image=img,
            annotation=annotation,
            meta={"synthetic": True},
        )


def small_linear_operator_grid(n: int = 16) -> list[Tensor]:
    """Tiny explicit matrices for Track 3's numerical verification (T3.5).

    Small enough that ``P_N`` and every candidate distance can be formed
    densely and checked against the closed-form bound with no approximation.
    """
    gen = torch.Generator().manual_seed(0)
    return [torch.randn(k, n, generator=gen) for k in (n // 4, n // 2, 3 * n // 4, n)]
