"""Forward-operator interface and the range / null-space decomposition (Stage 2).

Blueprint Sec. 2.3 and Sec. 13: the decomposition assumes a linear or affine
forward operator ``A``. Every reconstruction splits into

    x = P_R x + P_N x,     P_R = A^+ A,     P_N = I - A^+ A

where ``P_R x`` is pinned down by the measurements and ``P_N x`` is the free
component the operator cannot see, and where hallucination concentrates.

Whether ``P_R`` is an exact orthogonal projection depends on whether the
inverse used to *build* it is the true Moore-Penrose pseudo-inverse — which is
not necessarily ``pinv``. ``pinv`` is the fast, non-iterative inverse used for
display reconstruction, and ``exact_pinv`` describes only that: it is exact
for mask-based MRI, and only approximate for the filtered-backprojection CT
inverse.

The decomposition itself goes through :meth:`ForwardOperator.decomposition_pinv`,
which defaults to ``pinv`` but can be overridden with a different inverse when
``pinv`` is not good enough to build a *contraction* from (CT's CG fix,
D-012). That fix is deliberately scoped to the contraction property alone —
``null_projection_gain() <= 1`` — not to idempotence: CG on CT's normal
equations is iterative and, at any iteration budget cheap enough to run per
item, converges to project_range being *approximately* idempotent, not
exactly. Chasing tighter idempotence there is not simply a matter of running
more iterations either — see the caveat on ``CTProjectionOperator.pinv_cg``.
So the idempotence gate below stays keyed on ``exact_pinv``, unchanged, and
does not apply to CT before or after D-012; the contraction gate in
``OperatorRegistry.check_health`` (``contraction_ok``) is the one D-012 fixes.
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

    @property
    def calib_group(self) -> str:
        """The group used by group-conditional calibration (Tool 1).

        One group per acquisition regime. This lives on the descriptor rather
        than in the runner (D-021) so that Track 1 owns the grouping without a
        Track 2 code change: setting ``calib_group`` in an operator's config
        spec overrides the default, and the registry manifest records whichever
        was used.

        Granularity is ``mri/{family}/R{acceleration}`` and
        ``ct/{family}/v{n_views}`` -- one group per (family, severity). Mask
        seeds are *members* of a group, not groups of their own (D-023), and
        noise is deliberately not a group axis in v1 (D-021): it multiplies the
        group count exactly where per-regime calibration data is scarcest. If
        T1.4's protocol ever varies noise as a studied axis, it becomes a group
        axis and this must reflect that.
        """
        override = self.params.get("calib_group")
        if override:
            return str(override)
        if self.modality == "mri":
            return f"mri/{self.params['family']}/R{self.params['acceleration']:g}"
        if self.modality == "ct":
            return f"ct/{self.family}/v{self.params['n_views']}"
        return f"{self.modality}/{self.family}"

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
            # Recorded in the manifest so a bank's grouping can be audited
            # against the registry that produced it, rather than inferred
            # from the runner that happened to be current (D-021).
            "calib_group": self.calib_group,
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
        """Apply ``A^+`` (or the best available inverse): measurements -> image.

        This is the *display* reconstruction — fast and non-iterative. It is
        not necessarily what the range/null-space decomposition uses; see
        :meth:`decomposition_pinv`.
        """

    @abstractmethod
    def descriptor(self) -> OperatorDescriptor:
        """Return the data-free summary registered in the operator registry."""

    # -- Stage 2: operator decomposition ------------------------------------

    def decomposition_pinv(self, y: Tensor) -> Tensor:
        """Inverse used to build ``P_R = A^+ A`` for the decomposition.

        Defaults to :meth:`pinv`. Override this — not ``project_range``
        directly — when ``pinv`` is not a genuine pseudo-inverse and a
        better one is available for the decomposition specifically (CT's CG
        fix, D-012). Keeping ``pinv`` itself unchanged means the display
        reconstruction it returns elsewhere in the pipeline is unaffected.
        """
        return self.pinv(y)

    def project_range(self, x: Tensor) -> Tensor:
        """Measured component ``A^+ A x`` — constrained by the data."""
        return self.decomposition_pinv(self.forward(x))

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
        happens when ``decomposition_pinv`` is an approximate inverse rather
        than the true pseudo-inverse. It matters directly: the free-component
        score and the ``err_free`` label both then measure reconstruction
        error of the inverse, not invented structure, and are not comparable
        across modalities. See decision log D-012 (fixed for CT via CG).
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
