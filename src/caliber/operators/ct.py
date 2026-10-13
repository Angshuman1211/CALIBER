"""Sparse-view and limited-angle CT operators (Front II, cross-domain generalisation).

Parallel-beam Radon transform implemented as rotate-and-sum with bilinear
resampling. The acquisition is varied along the three axes named in the
Blueprint (Sec. 6, Sec. 16): view count, angular range, and dose.

Two honest differences from the MRI operator, both of which propagate into the
analysis and must not be papered over:

1. ``pinv`` is filtered backprojection, not the Moore-Penrose inverse, so
   ``exact_pinv`` is False — the *display* reconstruction stays approximate,
   and so does the idempotence of ``project_range`` (``exact_pinv`` still
   gates that check, unchanged). What ``pinv`` no longer determines is
   whether the null projector *amplifies*: the range/null-space
   *decomposition* goes through ``decomposition_pinv``, conjugate gradient on
   the normal equations, instead (D-012, resolved for the contraction
   property specifically — FBP's ``P_N`` had gain up to 1.74 at 30 views;
   CG's is a contraction like MRI's, robustly so even at a handful of
   iterations). See ``pinv_cg`` for what is and is not fixed by this.
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
        cg_iterations: int = 20,
        cg_tol: float = 1e-6,
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
                # Decomposition-only numerics (D-012), not part of the
                # operator's physical identity — excluded from operator_id,
                # same treatment as null_space_dim_method.
                "cg_iterations": int(cg_iterations),
                "cg_tol": float(cg_tol),
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
        """``A^H y`` — exact transpose, obtained as the VJP of the linear forward map.

        Autograd is switched on locally because the samplers run under ``no_grad``: without
        it the first consistency step raised, which is why no solver had ever run on a CT
        operator before T2.6. Local, so it builds no graph the caller can see.
        """
        with torch.enable_grad():
            x = torch.zeros(self.image_shape, dtype=torch.float32, device=y.device,
                            requires_grad=True)
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
        """Filtered backprojection — the fast *display* inverse (``exact_pinv`` is False).

        Not used for the range/null-space decomposition; see
        :meth:`decomposition_pinv` and D-012.
        """
        filtered = self._ramp_filter(y.to(torch.float32))
        scale = math.pi / max(1, len(self.angles))
        return self.adjoint(filtered) * scale

    def _AtA(self, x: Tensor) -> Tensor:
        """Normal-equations operator ``A^T A`` — one forward pass, one adjoint pass."""
        return self.adjoint(self.forward(x))

    def pinv_cg(self, y: Tensor, n_iter: int | None = None, tol: float | None = None) -> Tensor:
        """``A^+ y`` via conjugate gradient on the normal equations ``A^T A x = A^T y``.

        Fixes D-012's *contraction* property. FBP is not the Moore-Penrose
        inverse, so the ``P_N`` built from it amplifies rather than projects
        (measured gain up to 1.74 at 30 views on real LoDoPaB data). CG on
        the normal equations heads instead towards the true least-squares
        pseudo-inverse solution: started from ``x0 = 0``, every CG iterate
        stays in the Krylov subspace spanned by ``A^T A`` and ``b = A^T y``,
        both of which live entirely in ``row-space(A)`` — ``A^T A`` maps row
        space to itself and never touches ``null-space(A)``, and ``b``
        already lies there. So CG never leaves row space, and within it heads
        towards the unique minimum-norm least-squares solution, ``A^+ y``.

        **What this does not fix: idempotence.** Measured across the Sec. 16
        operator grid (64x64 to 256x256, 8 to 180 views, sparse-view and
        limited-angle), ``null_projection_gain()`` clears the contraction bar
        robustly — often within 5-10 iterations — but
        ``projection_idempotence_error()`` settles in the few-percent range at
        best and is *not* monotone in iteration count: applying an
        incompletely-converged CG solve twice (as idempotence testing does)
        compounds approximation error in a way more iterations of the first
        solve does not simply fix, and can occasionally make worse. CT's
        ``exact_pinv`` stays ``False`` and the registry's idempotence gate
        stays keyed on it, unchanged — D-012 was never a claim that CT's
        decomposition would become idempotent-exact, only that it would stop
        amplifying.

        **Why iterations are capped, not driven to convergence.** This
        operator's row space is poorly conditioned at aggressive
        undersampling (the same tail the spectral estimator in
        ``operators.spectrum`` resolves worst, D-011): residual reduction
        stalls well short of machine precision at any iteration budget cheap
        enough to run per item — measured on CPU, contraction gain is already
        near its floor by ~20 iterations at every grid point tried (64x64 to
        256x256, 8 to 180 views), while cost keeps climbing (multiple seconds
        per call at 256x256 with 100+ views) for idempotence that does not
        reliably improve. Worse, because ``A^T A`` is rank-deficient rather than
        merely ill-conditioned, the Krylov recurrence exhausts the
        well-determined subspace after a finite number of steps — pushing
        iterations past that point amplifies accumulated rounding error in
        near-null directions instead of resolving signal, and residual growth
        can become catastrophic rather than gentle (verified: an early,
        undamped version of this diverged by 6+ orders of magnitude beyond
        ~500 iterations on a 16-view operator). So this tracks the best
        iterate seen by residual norm and returns it rather than whatever
        ``x`` the loop ends on, stopping outright once the residual has
        clearly turned upward — this is what makes an iteration count higher
        than the default *safe* to pass, not merely unnecessary.

        ``n_iter``/``tol`` default to the operator's ``cg_iterations``/
        ``cg_tol`` params (constructor args, so configurable per experiment
        and recorded for reproducibility) when not given explicitly.
        """
        if n_iter is None:
            n_iter = int(self.params["cg_iterations"])
        if tol is None:
            tol = float(self.params["cg_tol"])
        b = self.adjoint(y.to(torch.float32))
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = torch.dot(r.flatten(), r.flatten())
        b_norm_sq = torch.dot(b.flatten(), b.flatten()).clamp_min(1e-20)
        best_x, best_rs = x, rs_old
        if rs_old / b_norm_sq <= tol**2:
            return x
        for _ in range(n_iter):
            ap = self._AtA(p)
            denom = torch.dot(p.flatten(), ap.flatten())
            if float(denom) <= 1e-20:
                break
            alpha = rs_old / denom
            x = x + alpha * p
            r = r - alpha * ap
            rs_new = torch.dot(r.flatten(), r.flatten())
            if rs_new < best_rs:
                best_x, best_rs = x.clone(), rs_new
            if rs_new / b_norm_sq <= tol**2:
                return x
            if float(rs_new) > 4.0 * float(best_rs):
                break  # diverging past the well-determined subspace; bail to best_x
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return best_x

    def decomposition_pinv(self, y: Tensor) -> Tensor:
        """The decomposition uses CG (:meth:`pinv_cg`), not FBP. See D-012."""
        return self.pinv_cg(y)

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
