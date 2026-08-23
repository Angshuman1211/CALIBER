"""Forward-operator interface and the range / null-space decomposition (Stage 2).

Blueprint Sec. 2.3 and Sec. 13: the decomposition assumes a linear or affine
forward operator ``A``. Every reconstruction splits into

    x = P_R x + P_N x,     P_R = A^+ A,     P_N = I - A^+ A

where ``P_R x`` is pinned down by the measurements and ``P_N x`` is the free
component the operator cannot see, and where hallucination concentrates.

Whether ``P_R`` is an exact orthogonal projection depends on whether ``pinv``
is the true Moore-Penrose pseudo-inverse. It is exact for mask-based MRI and
only approximate for the filtered-backprojection CT inverse, so operators
declare ``exact_pinv`` and downstream code is expected to honour it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class OperatorDescriptor:
    """Data-free summary of an operator, sufficient to compute operator distance.

    Track 3 consumes descriptors rather than operator objects so that the
    distance measure and the derivations never require GPU or dataset access
    (Blueprint Sec. 19, shared operator registry).
    """

    operator_id: str
    modality: str
    family: str
    params: dict[str, Any]
    image_shape: tuple[int, ...]
    measurement_dim: int
    ambient_dim: int
    null_space_dim: int
    exact_pinv: bool
    # Optional geometry payloads used by specific distance measures.
    sampling_support: Any | None = field(default=None, repr=False)
    singular_values: Any | None = field(default=None, repr=False)

    @property
    def undersampling_ratio(self) -> float:
        return self.measurement_dim / self.ambient_dim

    def to_row(self) -> dict[str, Any]:
        """Flat, serialisable form for the operator-registry manifest."""
        return {
            "operator_id": self.operator_id,
            "modality": self.modality,
            "family": self.family,
            "image_shape": list(self.image_shape),
            "measurement_dim": self.measurement_dim,
            "ambient_dim": self.ambient_dim,
            "null_space_dim": self.null_space_dim,
            "undersampling_ratio": self.undersampling_ratio,
            "exact_pinv": self.exact_pinv,
            "params": self.params,
        }


class ForwardOperator(ABC):
    """A known, linear acquisition operator ``A`` with an (approximate) inverse."""

    modality: str = "unknown"
    family: str = "unknown"
    exact_pinv: bool = False

    def __init__(self, image_shape: tuple[int, int], params: dict[str, Any]):
        self.image_shape = tuple(image_shape)
        self.params = dict(params)

    # -- core linear algebra -------------------------------------------------

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Apply ``A``: image -> measurements."""

    @abstractmethod
    def adjoint(self, y: Tensor) -> Tensor:
        """Apply ``A^H``: measurements -> image."""

    @abstractmethod
    def pinv(self, y: Tensor) -> Tensor:
        """Apply ``A^+`` (or the best available inverse): measurements -> image."""

    @abstractmethod
    def descriptor(self) -> OperatorDescriptor:
        """Return the data-free summary registered in the operator registry."""

    # -- Stage 2: operator decomposition ------------------------------------

    def project_range(self, x: Tensor) -> Tensor:
        """Measured component ``A^+ A x`` — constrained by the data."""
        return self.pinv(self.forward(x))

    def project_null(self, x: Tensor) -> Tensor:
        """Free component ``(I - A^+ A) x`` — invisible to the measurements."""
        return x - self.project_range(x)

    def decompose(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(range_component, null_component)`` with ``x = range + null``."""
        rng = self.project_range(x)
        return rng, x - rng

    def residual(self, x: Tensor, y: Tensor) -> Tensor:
        """Measurement residual ``A x - y``, the data-consistency evidence."""
        return self.forward(x) - y

    # -- diagnostics ---------------------------------------------------------

    def adjoint_test(self, n_trials: int = 3, seed: int = 0, atol: float = 1e-4) -> float:
        """Dot-product test: ``<A x, y> == <x, A^H y>``. Returns max relative error.

        A projector whose adjoint is wrong silently corrupts every score in the
        bank, so this is a gate, not a nicety.
        """
        gen = torch.Generator().manual_seed(seed)
        worst = 0.0
        for _ in range(n_trials):
            x = torch.randn(*self.image_shape, generator=gen)
            y = self.forward(x)
            v = torch.randn_like(y.real if y.is_complex() else y)
            if y.is_complex():
                v = v + 1j * torch.randn_like(y.real)
            lhs = torch.vdot(y.flatten(), v.flatten())
            rhs = torch.vdot(
                x.flatten().to(lhs.dtype), self.adjoint(v).flatten().to(lhs.dtype)
            )
            scale = max(abs(lhs), abs(rhs), 1e-12)
            worst = max(worst, float(abs(lhs - rhs) / scale))
        return worst

    def projection_idempotence_error(self, seed: int = 0) -> float:
        """``||P_R(P_R x) - P_R x|| / ||P_R x||``; ~0 iff ``P_R`` is a true projector."""
        gen = torch.Generator().manual_seed(seed)
        x = torch.randn(*self.image_shape, generator=gen)
        p1 = self.project_range(x)
        p2 = self.project_range(p1)
        denom = float(torch.linalg.vector_norm(p1)) or 1e-12
        return float(torch.linalg.vector_norm(p2 - p1)) / denom

    def null_projection_gain(self, n_trials: int = 3, seed: int = 0) -> float:
        """Worst-case ``||P_N x|| / ||x||``. An orthogonal projection gives <= 1.

        A value above 1 means ``P_N`` *amplifies* rather than projects, which
        happens when ``pinv`` is an approximate inverse rather than the true
        pseudo-inverse. It matters directly: the free-component score and the
        ``err_free`` label both then measure reconstruction error of the
        inverse, not invented structure, and are not comparable across
        modalities. See decision log D-012.
        """
        gen = torch.Generator().manual_seed(seed)
        worst = 0.0
        for _ in range(n_trials):
            x = torch.randn(*self.image_shape, generator=gen)
            free = self.project_null(x)
            denom = float(torch.linalg.vector_norm(x)) or 1e-12
            worst = max(worst, float(torch.linalg.vector_norm(free)) / denom)
        return worst

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.descriptor().operator_id})"
