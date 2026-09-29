"""Randomised spectral estimation for forward operators.

Feeds two things: the spectral-divergence operator distance (Blueprint Sec.
12.1) and Track 3's L2, which relates the coverage-gap bound to null-space
dimension and singular-value decay. Runs on matvecs only, so it costs a handful
of forward/adjoint applications and needs no explicit matrix.

**Known limitation, measured, not assumed** (decision log D-011). A randomised
range finder recovers the *top* of the spectrum, and the top of the spectrum is
nearly identical across CT view counts: on 32x32 phantoms, the fitted decay
exponent moves from 1.688 to 1.676 between 180 and 20 views at k=16, and only
reaches 1.788 vs 1.524 at k=200. The information that distinguishes these
operators lives in the decay tail and the null space, which is exactly what
this estimator resolves worst.

Consequences: ``spectral_divergence`` is weak as a primary distance measure,
and L2 should prefer ``null_space_dim`` — which is exact for MRI — over a
fitted decay exponent, or say plainly that it uses an estimate whose accuracy
in the relevant range has not been established. Track 3 to decide in Week 6.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import ForwardOperator


def estimate_spectrum(
    op: ForwardOperator,
    n_components: int = 64,
    n_oversample: int = 10,
    n_power_iters: int = 2,
    seed: int = 0,
) -> Tensor:
    """Approximate the top ``n_components`` singular values of ``A`` (descending).

    Randomised range finder (Halko et al.): sketch the range with random probes,
    orthonormalise, then take the SVD of the small projected operator.
    """
    gen = torch.Generator().manual_seed(seed)
    k = n_components + n_oversample
    probes = [torch.randn(*op.image_shape, generator=gen) for _ in range(k)]

    # Carry the sketch in the operator's *own* measurement domain, complex or not,
    # and flatten to real only for the linear algebra at the end (P-28).
    #
    # The earlier version flattened to real up front, which doubled the row count
    # for a complex operator and then tried to reshape those rows back into the
    # measurement shape on the next power iteration: at 256^2 that is 131,072
    # values into a [256, 256] slot, and it raised. Every MRI operator is complex,
    # so `spectral_divergence` could not be computed for any of them -- the one
    # candidate distance that is not degenerate on mask operators (D-052).
    ys = [op.forward(probe) for probe in probes]
    for _ in range(n_power_iters):
        # (A A^H)^q A Omega -- the same iteration as before, without a round trip
        # through a real view that does not survive being reshaped back.
        ys = [op.forward(op.adjoint(y)) for y in ys]

    sketch = torch.stack([y.flatten() for y in ys], dim=1)
    if sketch.is_complex():
        sketch = torch.view_as_real(sketch).reshape(-1, k)

    q, _ = torch.linalg.qr(sketch.to(torch.float32))
    svals = torch.linalg.svdvals(q.T @ sketch.to(torch.float32))
    return svals[:n_components]


def spectral_decay_exponent(svals: Tensor, eps: float = 1e-12) -> float:
    """Fit ``sigma_i ~ i^-p`` by least squares in log-log; returns ``p``.

    A larger exponent means a more ill-posed acquisition, which is exactly the
    quantity L2 predicts the coverage-gap bound should scale with.
    """
    s = svals[svals > eps]
    if s.numel() < 3:
        return float("nan")
    idx = torch.arange(1, s.numel() + 1, dtype=torch.float64)
    x = torch.log(idx)
    y = torch.log(s.to(torch.float64))
    x_c, y_c = x - x.mean(), y - y.mean()
    return float(-(x_c @ y_c) / (x_c @ x_c))


def effective_rank(svals: Tensor, tol: float = 1e-6) -> int:
    """Number of singular values above ``tol * sigma_max``."""
    if svals.numel() == 0:
        return 0
    return int((svals > tol * svals.max()).sum())
