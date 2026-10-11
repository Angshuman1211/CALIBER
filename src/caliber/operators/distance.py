"""Candidate operator-distance measures (Blueprint Sec. 12.1, tasks T1.3 / T3.2).

Track 3 selects the primary measure in Week 3 and Track 1 owns its definition;
this module supplies the plumbing so that all three tracks compute *the same
number* from *the same registry entry*, which is what the Week 9 consistency
audit checks.

Every measure takes two operators (or their descriptors) and returns a scalar.
Measures must be modality-agnostic by Week 7: the same function must accept an
MRI pair and a CT pair.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from .base import ForwardOperator

Normalisation = str  # "none" | "ambient" | "nullspace"


def _normalise(value: float, op_a: ForwardOperator, op_b: ForwardOperator, mode: str) -> float:
    """Normalisation applied before curves from two modalities are compared.

    Week 9 tests whether MRI and CT coverage-gap curves collapse onto a shared
    trend once distance is normalised for ill-posedness; which mode achieves
    that is an empirical question, so it stays configurable.
    """
    da, db = op_a.descriptor(), op_b.descriptor()
    if mode == "none":
        return value
    if mode == "ambient":
        return value / max(1.0, float(da.ambient_dim) ** 0.5)
    if mode == "nullspace":
        denom = max(1.0, 0.5 * (da.null_space_dim + db.null_space_dim)) ** 0.5
        return value / denom
    raise ValueError(f"unknown normalisation {mode!r}")


def _both_have_masks(op_a: ForwardOperator, op_b: ForwardOperator) -> bool:
    sa = op_a.descriptor().sampling_support
    sb = op_b.descriptor().sampling_support
    return (
        isinstance(sa, Tensor)
        and isinstance(sb, Tensor)
        and sa.dtype == torch.bool
        and sa.shape == sb.shape
    )


# -- 1. null-space projection distance ---------------------------------------


def null_space_projection_distance(
    op_a: ForwardOperator,
    op_b: ForwardOperator,
    normalise: Normalisation = "ambient",
    n_probes: int = 32,
    seed: int = 0,
) -> float:
    """``||P_N(A) - P_N(B)||_F`` — how much of the hidden part of the image moved.

    For two mask-based MRI operators this has a closed form: ``P_N = F^H(I-M)F``
    with ``F`` unitary, so the Frobenius distance is exactly the square root of
    the symmetric difference of the sampling masks. Otherwise it is estimated by
    Hutchinson's identity, ``E_z ||(P_A - P_B) z||^2 = ||P_A - P_B||_F^2`` for
    ``z ~ N(0, I)``, which needs only projector applications and so works for
    any modality.
    """
    if _both_have_masks(op_a, op_b):
        mask_a = op_a.descriptor().sampling_support
        mask_b = op_b.descriptor().sampling_support
        value = float(torch.sqrt((mask_a ^ mask_b).sum().to(torch.float64)))
        return _normalise(value, op_a, op_b, normalise)

    gen = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(n_probes):
        z = torch.randn(*op_a.image_shape, generator=gen)
        diff = op_a.project_null(z) - op_b.project_null(z)
        total += float(torch.linalg.vector_norm(diff) ** 2)
    value = (total / n_probes) ** 0.5
    return _normalise(value, op_a, op_b, normalise)


# -- 1b. signed null-space growth — directional (T1.3, Track 1) ---------------


def null_space_growth(
    op_a: ForwardOperator,
    op_b: ForwardOperator,
    normalise: Normalisation = "ambient",
    n_probes: int = 32,
    seed: int = 0,
) -> float:
    """**Signed** growth of the null space from ``op_a`` (calibration) to ``op_b``
    (deployment). Positive means the deployment operator hides more.

    **Track 1 owns this one** (T1.3). It exists because every other measure in this
    registry is a *metric* — symmetric by construction — and the coverage gap is not a
    symmetric function of the operator pair. Measured on the phantom bank, naive transfer
    between the same two operators in the two directions:

    | pair | ``null_space_projection`` | coverage gap | 95% CI |
    |---|---|---|---|
    | R2 → R8 | 0.6124 | −0.097 | [−0.134, −0.068] |
    | R8 → R2 | 0.6124 | **+0.902** | [+0.881, +0.921] |

    Same x, opposite y, intervals nowhere near each other. A coverage-gap-against-distance
    curve plotted on a symmetric axis is therefore **not a function** — it is two branches
    — and the three distance values tested each carry both.

    The mechanism is the one P-21 identified: scores rise with acceleration, so a threshold
    calibrated where scores are low and deployed where they are high flags everything
    (over-coverage, exploding flag area), while the reverse direction flags almost nothing
    (coverage collapse). Those are different failures and the axis has to tell them apart.

    **Definition.** On the energy scale, the net fraction of the ambient space newly
    hidden, then a signed square root to put it on the same amplitude scale as
    :func:`null_space_projection_distance`:

        g = (|kept by A, dropped by B| − |dropped by A, kept by B|) / ambient
        d = sign(g) · sqrt(|g|)

    Two properties worth stating because they are the point:

    * **Nested masks reproduce the symmetric measure exactly, up to sign.** Where the
      deployment operator samples a subset of the calibration operator, the second term is
      zero and ``|d|`` equals ``null_space_projection``. So this is not a different
      quantity on the graded within-family axis — it is the same quantity with its
      direction restored, which keeps D19's interpretation ("the fraction of k-space lines
      whose sampled status changed") intact where that reading applies.
    * **An equal-acceleration family change collapses toward zero.** Two masks that keep
      the same fraction of k-space in different places hide as much as they reveal, so
      ``g ≈ 0`` where the symmetric measure saturates at ~0.65 (D-052). That is a
      **falsifiable prediction**: a family change at matched acceleration should show a
      small coverage gap. The Week 4 grid can test it.

    **This is not a metric and must not be used as one.** It is signed, asymmetric, and
    violates the triangle inequality, so it is unavailable to T3.3's Lipschitz assumption,
    which needs a genuine distance. That is not a conflict: ``|d| ≤
    null_space_projection`` always, so a bound proved against the symmetric measure stays
    valid here and is merely loose in the benign direction. The split is exactly the
    "plotted quantity need not be the proved quantity" caution T1.3 raised and T3.3 §5
    answered — returning in a new form, with the same resolution.
    """
    if _both_have_masks(op_a, op_b):
        mask_a = op_a.descriptor().sampling_support
        mask_b = op_b.descriptor().sampling_support
        newly_hidden = float((mask_a & ~mask_b).sum())
        newly_revealed = float((~mask_a & mask_b).sum())
    else:
        # Hutchinson again, but on each one-sided projector rather than the difference:
        # P_N(B) restricted to what A measured is the content B newly hides.
        gen = torch.Generator().manual_seed(seed)
        hidden = revealed = 0.0
        for _ in range(n_probes):
            z = torch.randn(*op_a.image_shape, generator=gen)
            hidden += float(torch.linalg.vector_norm(op_b.project_null(op_a.project_range(z))) ** 2)
            revealed += float(
                torch.linalg.vector_norm(op_a.project_null(op_b.project_range(z))) ** 2
            )
        newly_hidden, newly_revealed = hidden / n_probes, revealed / n_probes

    net = newly_hidden - newly_revealed
    value = abs(net) ** 0.5
    signed = _normalise(value, op_a, op_b, normalise)
    return signed if net >= 0 else -signed


#: Measures that are **not** metrics — signed and asymmetric. A pairwise matrix of these
#: is antisymmetric rather than symmetric, so :func:`distance_matrix` refuses them rather
#: than silently returning half of one.
SIGNED_MEASURES = frozenset({"null_space_growth"})


# -- 2. sampling-support difference -------------------------------------------


def sampling_support_difference(
    op_a: ForwardOperator,
    op_b: ForwardOperator,
    normalise: Normalisation = "ambient",
    **_,
) -> float:
    """Measure of the symmetric difference between what the two acquisitions sample."""
    da, db = op_a.descriptor(), op_b.descriptor()
    sa, sb = da.sampling_support, db.sampling_support
    if _both_have_masks(op_a, op_b):
        value = float((sa ^ sb).sum())
    elif isinstance(sa, Tensor) and isinstance(sb, Tensor):
        # CT: angle sets. Symmetric difference of the sampled directions.
        set_a = {round(float(v), 6) for v in sa}
        set_b = {round(float(v), 6) for v in sb}
        value = float(len(set_a ^ set_b)) * da.image_shape[-1]
    else:
        raise ValueError("operators expose no comparable sampling support")
    return _normalise(value**0.5, op_a, op_b, normalise)


# -- 3. principal-angle / chordal subspace distance ---------------------------


def chordal_subspace_distance(
    op_a: ForwardOperator,
    op_b: ForwardOperator,
    normalise: Normalisation = "ambient",
    **_,
) -> float:
    """Chordal distance between the two unmeasured subspaces.

    ``d_c^2 = sum_i sin^2(theta_i) = (1/2)||P_A - P_B||_F^2`` for orthogonal
    projectors, so it is a rescaling of the null-space projection distance for
    exact projectors and a documented approximation otherwise.
    """
    raw = null_space_projection_distance(op_a, op_b, normalise="none")
    return _normalise(raw / (2.0**0.5), op_a, op_b, normalise)


# -- 4. operator / pseudo-inverse norms ---------------------------------------


def operator_norm_distance(
    op_a: ForwardOperator,
    op_b: ForwardOperator,
    normalise: Normalisation = "ambient",
    n_probes: int = 32,
    seed: int = 0,
) -> float:
    """``||A - B||_F`` estimated by probing, valid only when shapes agree."""
    gen = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(n_probes):
        z = torch.randn(*op_a.image_shape, generator=gen)
        ya, yb = op_a.forward(z), op_b.forward(z)
        if ya.shape != yb.shape:
            raise ValueError("operator_norm_distance needs matching measurement shapes")
        total += float(torch.linalg.vector_norm(ya - yb) ** 2)
    return _normalise((total / n_probes) ** 0.5, op_a, op_b, normalise)


# -- 5. spectral divergence ----------------------------------------------------


def spectral_divergence(
    op_a: ForwardOperator,
    op_b: ForwardOperator,
    normalise: Normalisation = "none",
    n_components: int = 64,
    seed: int = 0,
) -> float:
    """Difference in singular-value decay: severity, not direction, of ill-posedness."""
    from .spectrum import estimate_spectrum

    sa = estimate_spectrum(op_a, n_components=n_components, seed=seed)
    sb = estimate_spectrum(op_b, n_components=n_components, seed=seed)
    n = min(sa.numel(), sb.numel())
    if n == 0:
        return 0.0
    la = torch.log(sa[:n].clamp_min(1e-12))
    lb = torch.log(sb[:n].clamp_min(1e-12))
    return _normalise(float(torch.linalg.vector_norm(la - lb) / n**0.5), op_a, op_b, normalise)


DISTANCES: dict[str, Callable[..., float]] = {
    "null_space_projection": null_space_projection_distance,
    "null_space_growth": null_space_growth,
    "sampling_support": sampling_support_difference,
    "chordal_subspace": chordal_subspace_distance,
    "operator_norm": operator_norm_distance,
    "spectral_divergence": spectral_divergence,
}


def compute_distance(op_a: ForwardOperator, op_b: ForwardOperator, measure: str, **kwargs) -> float:
    """Dispatch by name; the name is what gets recorded in the score bank."""
    if measure not in DISTANCES:
        raise ValueError(f"unknown distance {measure!r}; available: {sorted(DISTANCES)}")
    return DISTANCES[measure](op_a, op_b, **kwargs)


def distance_matrix(ops: list[ForwardOperator], measure: str, **kwargs) -> Tensor:
    """Symmetric pairwise distance matrix over a set of registered operators.

    Refuses a signed measure rather than filling in half a matrix and mirroring it: for
    ``null_space_growth`` the ``(i, j)`` and ``(j, i)`` entries have opposite signs, and
    mirroring would record the deployment direction as the calibration one for half the
    pairs. Call :func:`compute_distance` per ordered pair instead.
    """
    if measure in SIGNED_MEASURES:
        raise ValueError(
            f"{measure!r} is signed and directional, so a symmetric matrix would "
            "misreport half its entries — compute it per ordered (calibration, "
            "deployment) pair instead"
        )
    n = len(ops)
    out = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = compute_distance(ops[i], ops[j], measure, **kwargs)
            out[i, j] = out[j, i] = d
    return out
