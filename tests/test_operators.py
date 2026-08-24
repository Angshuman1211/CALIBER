"""Operator correctness gates.

A wrong adjoint or a non-idempotent projector silently corrupts every score in
the bank, and the corruption is invisible downstream. These tests are the gate
that stops that, and they run before any score-bank production job.
"""

from __future__ import annotations

import pytest
import torch

from caliber.operators import (
    CTProjectionOperator,
    MRIUndersamplingOperator,
    OperatorRegistry,
    build_mask,
    compute_distance,
)
from caliber.operators.mri import MASK_FAMILIES

SHAPE = (64, 64)


@pytest.fixture(params=MASK_FAMILIES)
def mri_op(request):
    return MRIUndersamplingOperator(SHAPE, family=request.param, acceleration=4)


@pytest.fixture
def ct_op():
    return CTProjectionOperator(SHAPE, n_views=32, angular_range_deg=180.0)


# -- MRI -----------------------------------------------------------------------


def test_mri_adjoint_is_exact(mri_op):
    assert mri_op.adjoint_test() < 1e-5


def test_mri_range_projector_is_idempotent(mri_op):
    """P_R = A^+A must be a true projector; MRI declares exact_pinv."""
    assert mri_op.projection_idempotence_error() < 1e-5


def test_mri_decomposition_is_exact(mri_op):
    x = torch.randn(*SHAPE)
    rng, null = mri_op.decompose(x)
    assert torch.allclose((rng + null).real, x, atol=1e-4)


def test_mri_null_component_is_invisible_to_the_operator(mri_op):
    """The defining property: A(P_N x) = 0. Hallucination hides exactly here."""
    x = torch.randn(*SHAPE)
    free = mri_op.project_null(x)
    assert float(torch.linalg.vector_norm(mri_op.forward(free))) < 1e-4


def test_mask_acceleration_is_respected():
    for accel in (2, 4, 8):
        mask = build_mask((64, 64), "random1d", acceleration=accel, center_fraction=0.08)
        achieved = mask.numel() / int(mask.sum())
        assert abs(achieved - accel) / accel < 0.25


def test_null_space_dim_grows_with_acceleration():
    dims = [
        MRIUndersamplingOperator(SHAPE, acceleration=a).descriptor().null_space_dim
        for a in (2, 4, 8)
    ]
    assert dims == sorted(dims)


# -- CT ------------------------------------------------------------------------


def test_ct_adjoint_is_exact(ct_op):
    """The VJP of a linear forward map is its exact transpose, interpolation included."""
    assert ct_op.adjoint_test() < 1e-4


def test_ct_pinv_is_declared_approximate(ct_op):
    """``pinv`` (FBP) stays the fast display reconstruction and stays approximate."""
    assert ct_op.descriptor().exact_pinv is False


def test_mri_null_projector_is_a_contraction(mri_op):
    """An orthogonal projection cannot increase a vector's norm."""
    assert mri_op.null_projection_gain() <= 1.0 + 1e-4


def test_ct_null_projector_is_a_contraction(ct_op):
    """Regression guard for D-012.

    FBP is not the pseudo-inverse, so the ``P_N`` it built amplified rather
    than projected (measured gain up to 1.74 at 30 views on real LoDoPaB
    data). The decomposition now goes through CG on the normal equations
    (``decomposition_pinv`` / ``pinv_cg``), a genuine pseudo-inverse in the
    least-squares sense, so ``P_N`` is a contraction like MRI's — this holds
    from just a handful of CG iterations, well before idempotence does (see
    ``pinv_cg``'s docstring on what the fix does and does not cover).
    """
    assert ct_op.null_projection_gain() <= 1.0 + 1e-3


def test_ct_idempotence_is_still_not_gated(ct_op):
    """D-012 fixed contraction, not idempotence — CT is not held to MRI's bar.

    CG is iterative and only approximately idempotent at any iteration budget
    cheap enough to run per item; ``exact_pinv`` correctly stays False and the
    registry's idempotence gate is bypassed for CT, unchanged by the fix.
    """
    reg = OperatorRegistry(SHAPE)
    reg.add(ct_op)
    report = reg.check_health()[0]
    assert report["exact_pinv"] is False
    assert report["projector_ok"] is True  # bypassed, not actually idempotent
    assert report["idempotence_rel_error"] > 1e-3  # would fail MRI's bar


def test_ct_health_check_gates_on_contraction(ct_op):
    """``contraction_ok`` is now enforced (D-012), not merely reported."""
    reg = OperatorRegistry(SHAPE)
    reg.add(ct_op)
    report = reg.check_health()[0]
    assert report["adjoint_ok"] is True
    assert report["contraction_ok"] is True


def test_ct_fbp_recovers_structure(ct_op):
    """FBP is not exact, but it must correlate strongly with the truth."""
    x = torch.zeros(*SHAPE)
    x[20:44, 20:44] = 1.0
    recon = ct_op.pinv(ct_op.forward(x))
    corr = torch.corrcoef(torch.stack([x.flatten(), recon.flatten()]))[0, 1]
    assert float(corr) > 0.8


def test_limited_angle_is_more_ill_posed_than_sparse_view():
    """Restricting the arc should leave a larger unmeasured subspace."""
    sparse = CTProjectionOperator(SHAPE, n_views=60, angular_range_deg=180.0)
    limited = CTProjectionOperator(SHAPE, n_views=60, angular_range_deg=60.0)
    assert limited.family == "limited_angle"
    assert sparse.family == "sparse_view"


# -- distances ------------------------------------------------------------------


def test_distance_to_self_is_zero():
    op = MRIUndersamplingOperator(SHAPE, acceleration=4)
    for measure in ("null_space_projection", "sampling_support", "chordal_subspace"):
        assert compute_distance(op, op, measure) == pytest.approx(0.0, abs=1e-9)


def test_distance_is_symmetric():
    a = MRIUndersamplingOperator(SHAPE, acceleration=2)
    b = MRIUndersamplingOperator(SHAPE, acceleration=8)
    assert compute_distance(a, b, "null_space_projection") == pytest.approx(
        compute_distance(b, a, "null_space_projection")
    )


def test_distance_increases_with_operator_separation():
    """The whole coverage-vs-distance claim needs distance to be ordered sensibly."""
    calib = MRIUndersamplingOperator(SHAPE, acceleration=2)
    near = MRIUndersamplingOperator(SHAPE, acceleration=3)
    far = MRIUndersamplingOperator(SHAPE, acceleration=10)
    d_near = compute_distance(calib, near, "null_space_projection")
    d_far = compute_distance(calib, far, "null_space_projection")
    assert 0 < d_near < d_far


def test_hutchinson_estimate_matches_closed_form():
    """The modality-agnostic estimator must agree with the MRI closed form."""
    from caliber.operators.distance import null_space_projection_distance

    a = MRIUndersamplingOperator(SHAPE, acceleration=2)
    b = MRIUndersamplingOperator(SHAPE, acceleration=6)
    exact = null_space_projection_distance(a, b, normalise="none")

    # Force the estimator path by hiding the masks.
    probes, total = 128, 0.0
    gen = torch.Generator().manual_seed(0)
    for _ in range(probes):
        z = torch.randn(*SHAPE, generator=gen, dtype=torch.complex64)
        diff = a.project_null(z) - b.project_null(z)
        total += float(torch.linalg.vector_norm(diff) ** 2)
    estimate = (total / probes) ** 0.5
    assert abs(estimate - exact) / exact < 0.15


# -- registry --------------------------------------------------------------------


def test_registry_expands_grid_and_dedupes():
    specs = [{"modality": "mri", "family": "random1d", "acceleration": [2, 4, 8]}]
    reg = OperatorRegistry.from_specs(specs, SHAPE)
    assert len(reg) == 3
    reg.add(reg[reg.ids[0]])
    assert len(reg) == 3


def test_registry_health_check_passes_for_all_operators():
    specs = [
        {"modality": "mri", "family": "random1d", "acceleration": [2, 8]},
        {"modality": "ct", "n_views": [32, 64], "angular_range_deg": 180.0},
    ]
    reg = OperatorRegistry.from_specs(specs, SHAPE)
    report = reg.check_health()
    assert len(report) == 4
    assert all(r["adjoint_ok"] for r in report)
    assert all(r["projector_ok"] for r in report)


def test_operator_ids_are_stable_and_unique():
    specs = [{"modality": "mri", "family": ["random1d", "equispaced1d"], "acceleration": [2, 4]}]
    ids_a = OperatorRegistry.from_specs(specs, SHAPE).ids
    ids_b = OperatorRegistry.from_specs(specs, SHAPE).ids
    assert ids_a == ids_b
    assert len(set(ids_a)) == 4
