"""Accelerated-MRI acquisition operators (Front I, the empirical core).

``A = M . F`` with ``F`` the orthonormal 2-D DFT and ``M`` a binary k-space
sampling mask. Because ``M`` is an orthogonal projection and ``F`` is unitary,
``A^+ = F^H M`` exactly, so

    P_R = A^+ A = F^H M F     and     P_N = F^H (I - M) F

are exact orthogonal projections. The null space is spanned by the unsampled
k-space locations, which is what makes MRI the clean modality to establish the
phenomenon on before moving to CT.

Operators are synthesised by retrospective undersampling of fully-sampled raw
measurements (Blueprint Sec. 13: the operator is known because we choose it).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .base import ForwardOperator, OperatorDescriptor

MASK_FAMILIES = ("random1d", "equispaced1d", "variable_density2d")


def build_mask(
    shape: tuple[int, int],
    family: str,
    acceleration: float,
    center_fraction: float = 0.08,
    seed: int = 0,
) -> Tensor:
    """Return a boolean k-space sampling mask of shape ``(H, W)``.

    The fully-sampled low-frequency centre is always retained, matching fastMRI
    convention; ``acceleration`` is the target ratio of ambient to sampled
    locations, achieved to within rounding.
    """
    if family not in MASK_FAMILIES:
        raise ValueError(f"unknown mask family {family!r}; expected one of {MASK_FAMILIES}")
    height, width = shape
    gen = torch.Generator().manual_seed(seed)
    mask = torch.zeros(height, width, dtype=torch.bool)

    if family == "variable_density2d":
        n_target = max(1, int(round(height * width / acceleration)))
        ky = torch.fft.fftshift(torch.fft.fftfreq(height)).view(-1, 1)
        kx = torch.fft.fftshift(torch.fft.fftfreq(width)).view(1, -1)
        radius = torch.sqrt(ky**2 + kx**2)
        density = torch.exp(-((radius / 0.15) ** 2))
        centre = radius <= (center_fraction / 2)
        mask |= centre
        remaining = n_target - int(mask.sum())
        if remaining > 0:
            weights = density.clone()
            weights[mask] = 0.0
            flat = torch.multinomial(weights.flatten(), remaining, replacement=False, generator=gen)
            mask.view(-1)[flat] = True
        return torch.fft.ifftshift(mask)

    # 1-D phase-encode undersampling: whole columns are kept or dropped.
    n_center = max(1, int(round(width * center_fraction)))
    lo = (width - n_center) // 2
    centre_cols = torch.arange(lo, lo + n_center)
    n_target = max(n_center, int(round(width / acceleration)))
    n_extra = n_target - n_center

    outer = torch.tensor([c for c in range(width) if not (lo <= c < lo + n_center)])
    if n_extra > 0 and outer.numel() > 0:
        if family == "random1d":
            pick = torch.randperm(outer.numel(), generator=gen)[:n_extra]
            extra = outer[pick]
        else:  # equispaced1d
            step = max(1, outer.numel() // n_extra)
            extra = outer[torch.arange(0, outer.numel(), step)[:n_extra]]
    else:
        extra = torch.tensor([], dtype=torch.long)

    cols = torch.cat([centre_cols, extra.to(torch.long)])
    mask_shifted = torch.zeros(height, width, dtype=torch.bool)
    mask_shifted[:, cols] = True
    return torch.fft.ifftshift(mask_shifted, dim=-1)


class MRIUndersamplingOperator(ForwardOperator):
    """Cartesian k-space undersampling, single-coil, complex image domain."""

    modality = "mri"
    exact_pinv = True

    def __init__(
        self,
        image_shape: tuple[int, int],
        family: str = "random1d",
        acceleration: float = 4.0,
        center_fraction: float = 0.08,
        mask_seed: int = 0,
        noise_sigma: float = 0.0,
        **_: Any,
    ):
        super().__init__(
            image_shape,
            {
                "family": family,
                "acceleration": float(acceleration),
                "center_fraction": float(center_fraction),
                "mask_seed": int(mask_seed),
                "noise_sigma": float(noise_sigma),
            },
        )
        self.family = family
        self.mask = build_mask(self.image_shape, family, acceleration, center_fraction, mask_seed)

    # -- linear algebra ------------------------------------------------------

    @staticmethod
    def _fft(x: Tensor) -> Tensor:
        return torch.fft.fft2(x.to(torch.complex64), norm="ortho")

    @staticmethod
    def _ifft(k: Tensor) -> Tensor:
        return torch.fft.ifft2(k, norm="ortho")

    def forward(self, x: Tensor) -> Tensor:
        """``A x`` — zero-filled k-space; retained in ambient shape for simplicity."""
        return self._fft(x) * self.mask

    def adjoint(self, y: Tensor) -> Tensor:
        return self._ifft(y * self.mask)

    def pinv(self, y: Tensor) -> Tensor:
        """Exact Moore-Penrose inverse: ``M`` is a projection, ``F`` is unitary."""
        return self.adjoint(y)

    # -- registry ------------------------------------------------------------

    @property
    def operator_id(self) -> str:
        p = self.params
        return (
            f"mri.{p['family']}.R{p['acceleration']:g}"
            f".c{p['center_fraction']:g}.s{p['mask_seed']}"
        )

    def descriptor(self) -> OperatorDescriptor:
        ambient = int(self.image_shape[0] * self.image_shape[1])
        measured = int(self.mask.sum())
        return OperatorDescriptor(
            operator_id=self.operator_id,
            modality=self.modality,
            family=self.family,
            params=self.params,
            image_shape=self.image_shape,
            measurement_dim=measured,
            ambient_dim=ambient,
            null_space_dim=ambient - measured,
            exact_pinv=True,
            sampling_support=self.mask.clone(),
            singular_values=None,  # all singular values are 1 or 0 for a mask operator
        )
