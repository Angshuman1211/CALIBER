"""Sparse-view and limited-angle CT operators (Front II, cross-domain generalisation).

Parallel-beam Radon transform implemented as rotate-and-sum with bilinear
resampling. The acquisition is varied along the three axes named in the
Blueprint (Sec. 6, Sec. 16): view count, angular range, and dose.

Two honest differences from the MRI operator, both of which propagate into the
analysis and must not be papered over:

1. ``pinv`` is filtered backprojection, not the Moore-Penrose inverse. The
   range/null-space split is therefore *approximate* for CT, and
   ``exact_pinv`` is False. Week 7 owns tightening this.
2. ``null_space_dim`` is the trivial dimension-count bound
   ``ambient - measurements``, not a computed rank. The true value needs a
   spectral estimate; see ``operators.spectrum`` and Track 3's L2.

The adjoint is obtained as the vector-Jacobian product of the forward map.
Because the forward map is linear, the VJP *is* the exact transpose, so the
dot-product test passes to floating-point precision even though the projector
uses interpolation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import ForwardOperator, OperatorDescriptor


def make_angles(n_views: int, angular_range_deg: float, start_deg: float = 0.0) -> Tensor:
    """Evenly spaced projection angles in radians over ``[start, start + range)``."""
    if n_views < 1:
        raise ValueError("n_views must be >= 1")
    deg = torch.linspace(start_deg, start_deg + angular_range_deg, n_views + 1)[:n_views]
    return deg * math.pi / 180.0


class CTProjectionOperator(ForwardOperator):
    """Parallel-beam projection over a configurable set of angles."""

    modality = "ct"
    exact_pinv = False

    def __init__(
        self,
        image_shape: tuple[int, int],
        n_views: int = 60,
        angular_range_deg: float = 180.0,
        start_angle_deg: float = 0.0,
        photon_count: float | None = None,
        family: str | None = None,
        **_: Any,
    ):
        super().__init__(
            image_shape,
            {
                "n_views": int(n_views),
                "angular_range_deg": float(angular_range_deg),
                "start_angle_deg": float(start_angle_deg),
                "photon_count": photon_count,
                "null_space_dim_method": "dimension_count",
            },
        )
        # "sparse_view" and "limited_angle" are the two CT regimes; the family
        # is what the group-conditional calibration groups on.
        self.family = family or ("limited_angle" if angular_range_deg < 180.0 else "sparse_view")
        self.angles = make_angles(n_views, angular_range_deg, start_angle_deg)
        self.n_det = int(image_shape[-1])

    # -- linear algebra ------------------------------------------------------

    def _rotation_grid(self, angles: Tensor, device: torch.device) -> Tensor:
        """Affine sampling grid rotating the image by each angle."""
        cos, sin = torch.cos(angles), torch.sin(angles)
        theta = torch.zeros(len(angles), 2, 3, device=device, dtype=torch.float32)
        theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
        theta[:, 1, 0], theta[:, 1, 1] = sin, cos
        size = (len(angles), 1, *self.image_shape)
        return F.affine_grid(theta, size, align_corners=False)

    def forward(self, x: Tensor) -> Tensor:
        """``A x`` -> sinogram of shape ``(n_views, n_det)``."""
        x = x.to(torch.float32)
        batch = x.reshape(1, 1, *self.image_shape).expand(len(self.angles), 1, *self.image_shape)
        grid = self._rotation_grid(self.angles.to(x.device), x.device)
        rotated = F.grid_sample(batch, grid, align_corners=False, padding_mode="zeros")
        return rotated.sum(dim=2).reshape(len(self.angles), self.n_det)

    def adjoint(self, y: Tensor) -> Tensor:
        """``A^H y`` — exact transpose, obtained as the VJP of the linear forward map."""
        x = torch.zeros(self.image_shape, dtype=torch.float32, device=y.device, requires_grad=True)
        out = self.forward(x)
        (grad,) = torch.autograd.grad(out, x, grad_outputs=y.to(torch.float32))
        return grad.detach()

    def _ramp_filter(self, sino: Tensor) -> Tensor:
        n = sino.shape[-1]
        n_pad = int(2 ** math.ceil(math.log2(max(64, 2 * n))))
        pad = n_pad - n
        padded = F.pad(sino, (0, pad))
        freqs = torch.fft.rfftfreq(n_pad, device=sino.device)
        ramp = 2.0 * freqs  # Ram-Lak
        filtered = torch.fft.irfft(torch.fft.rfft(padded, dim=-1) * ramp, n=n_pad, dim=-1)
        return filtered[..., :n]

    def pinv(self, y: Tensor) -> Tensor:
        """Filtered backprojection — an *approximate* inverse (``exact_pinv`` is False)."""
        filtered = self._ramp_filter(y.to(torch.float32))
        scale = math.pi / max(1, len(self.angles))
        return self.adjoint(filtered) * scale

    # -- registry ------------------------------------------------------------

    @property
    def operator_id(self) -> str:
        p = self.params
        dose = "full" if p["photon_count"] is None else f"{p['photon_count']:g}"
        return (
            f"ct.{self.family}.v{p['n_views']}"
            f".a{p['angular_range_deg']:g}.d{dose}"
        )

    def descriptor(self) -> OperatorDescriptor:
        ambient = int(self.image_shape[0] * self.image_shape[1])
        measured = int(len(self.angles) * self.n_det)
        return OperatorDescriptor(
            operator_id=self.operator_id,
            modality=self.modality,
            family=self.family,
            params=self.params,
            image_shape=self.image_shape,
            measurement_dim=measured,
            ambient_dim=ambient,
            null_space_dim=max(0, ambient - measured),
            exact_pinv=False,
            sampling_support=self.angles.clone(),
            singular_values=None,
        )
