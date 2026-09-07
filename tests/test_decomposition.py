"""Stage-2 decomposition of solver output (Week 2 [B], third item).

The projector algebra itself is tested in ``test_operators.py``. What is asserted
here is the layer built on top of it: that the split of a *reconstruction* into
range and null parts is measured correctly, that its shares are normalised the
way the orthogonality caveat requires, and that the two premises the scoring
layer silently depends on — error is null-dominated, spread is null-dominated —
are checked rather than assumed.

Everything runs against the analytic denoiser and small operators, so the file
costs milliseconds and needs no checkpoint.
"""

from __future__ import annotations

import math

import pytest
import torch

from caliber.operators import CTProjectionOperator, MRIUndersamplingOperator
from caliber.operators.decomposition import (
    DecompositionReport,
    aggregate_reports,
    decompose_samples,
    orthogonality_defect,
)
from caliber.solvers import DDNMSampler, DPSSampler
from caliber.solvers.denoiser import AnalyticDenoiser

SHAPE = (32, 32)


@pytest.fixture
def mri_op():
    return MRIUndersamplingOperator(SHAPE, family="random1d", acceleration=4, mask_seed=0)


@pytest.fixture
def item():
    gen = torch.Generator().manual_seed(0)
    return torch.rand(SHAPE, generator=gen) + 0.5


def draws(op, x, n=4, seed=0):
    sampler = DDNMSampler(n_steps=8, denoiser=AnalyticDenoiser())
    return sampler.sample(op.forward(x), op, n_samples=n, seed=seed)


# -- the split is a split -----------------------------------------------------


def test_energy_shares_sum_to_one_and_match_the_projector(mri_op, item):
    samples = draws(mri_op, item)
    rep = decompose_samples(samples, mri_op)

    mean = samples.mean(dim=0)
    rng, null = mri_op.decompose(mean)
    scale = float(torch.linalg.vector_norm(mean))

    assert rep.range_energy == pytest.approx(float(torch.linalg.vector_norm(rng)) / scale)
    assert rep.null_energy == pytest.approx(float(torch.linalg.vector_norm(null)) / scale)
    # The null share and its complement are a partition of the split energy.
    assert 0.0 <= rep.null_energy_share <= 1.0
    expected = rep.null_energy**2 / (rep.null_energy**2 + rep.range_energy**2)
    assert rep.null_energy_share == pytest.approx(expected, rel=1e-6)


def test_mri_split_is_exactly_orthogonal(mri_op, item):
    """The Pythagorean identity is what makes an energy *share* meaningful."""
    assert orthogonality_defect(mri_op, item) < 1e-6


def test_ct_orthogonality_defect_is_reported_not_assumed_away():
    """CT's decomposition is CG at a finite budget (D-012), so the defect is non-zero.

    It must still be small enough that shares mean something — this is the number
    the write-up has to quote when it says CT's decomposition is approximate.
    """
    op = CTProjectionOperator(SHAPE, n_views=32, angular_range_deg=180.0)
    gen = torch.Generator().manual_seed(0)
    defect = orthogonality_defect(op, torch.rand(SHAPE, generator=gen))
    assert 0.0 <= defect < 0.5
    assert op.null_projection_gain() <= 1.0 + 1e-3


# -- the premises the scoring layer depends on --------------------------------


def test_reconstruction_error_is_null_dominated(mri_op, item):
    """The premise of scoring the free component at all.

    DDNM replaces the range component of every estimate with the measured one, so
    what it gets wrong is what it invented — and that has to show up here, or
    ``s_free`` is measuring the wrong thing.
    """
    rep = decompose_samples(draws(mri_op, item), mri_op, x_true=item)
    assert rep.null_error_share > 0.5
    assert rep.null_error > rep.range_error


def test_posterior_spread_is_null_dominated(mri_op, item):
    """The premise ``s_disagree`` rests on — spread is measured in the free part only."""
    rep = decompose_samples(draws(mri_op, item), mri_op, x_true=item)
    assert rep.null_spread_share > 0.5


def test_an_underconverged_sampler_shows_range_dominated_error(mri_op, item):
    """The null-error share doubles as a convergence diagnostic — the Week 2 finding.

    DPS *steers* towards data consistency rather than enforcing it, so on a small
    step budget its error is still sitting in the measured component: the solver
    has not fitted the data yet, and scoring the free component of that output
    would be scoring an artefact of the step count. DDNM, which replaces the
    range component outright, has no such regime — which is exactly why the two
    back-ends need different step budgets before their banks are comparable.
    """
    y = mri_op.forward(item)

    def share(n_steps):
        sampler = DPSSampler(n_steps=n_steps, denoiser=AnalyticDenoiser())
        samples = sampler.sample(y, mri_op, n_samples=4, seed=0)
        return decompose_samples(samples, mri_op, x_true=item).null_error_share

    assert share(8) < 0.5 < share(48)


def test_a_range_wandering_sampler_is_caught(mri_op, item):
    """The check has to be able to fail, or it is not a check.

    Perturbing draws in the *measured* component is exactly the failure mode the
    spread share exists to detect: a sampler doing this is not exploring a
    posterior, it is breaking data consistency.
    """
    gen = torch.Generator().manual_seed(1)
    base = mri_op.pinv(mri_op.forward(item))
    noisy = torch.stack(
        [base + 5.0 * mri_op.project_range(torch.randn(SHAPE, generator=gen)) for _ in range(4)]
    )
    rep = decompose_samples(noisy, mri_op)
    assert rep.null_spread_share < 0.5


# -- data consistency and the optional arguments ------------------------------


def test_data_consistency_error_matches_the_residual(mri_op, item):
    y = mri_op.forward(item)
    samples = draws(mri_op, item)
    rep = decompose_samples(samples, mri_op, y=y)
    mean = samples.mean(dim=0)
    expected = float(torch.linalg.vector_norm(mri_op.residual(mean, y))) / float(
        torch.linalg.vector_norm(y)
    )
    assert rep.data_consistency_error == pytest.approx(expected, rel=1e-6)


def test_ground_truth_and_measurements_are_optional(mri_op, item):
    """Both are absent at deployment; the energy half must still compute."""
    rep = decompose_samples(draws(mri_op, item), mri_op)
    assert math.isfinite(rep.null_energy_share)
    assert math.isnan(rep.null_error_share)
    assert math.isnan(rep.data_consistency_error)


def test_single_draw_has_no_spread_rather_than_zero_spread(mri_op, item):
    """One draw cannot disagree with itself; that is missing, not zero."""
    rep = decompose_samples(draws(mri_op, item, n=1), mri_op, x_true=item)
    assert rep.n_samples == 1
    assert math.isnan(rep.null_spread_share)
    assert math.isfinite(rep.null_error_share)


def test_wrong_sample_shape_is_rejected(mri_op, item):
    with pytest.raises(ValueError, match="n_posterior"):
        decompose_samples(item, mri_op)


def test_report_is_deterministic(mri_op, item):
    a = decompose_samples(draws(mri_op, item, seed=3), mri_op, x_true=item)
    b = decompose_samples(draws(mri_op, item, seed=3), mri_op, x_true=item)
    assert a.to_row() == b.to_row()


# -- aggregation --------------------------------------------------------------


def test_aggregate_averages_and_skips_missing_values(mri_op, item):
    with_truth = decompose_samples(draws(mri_op, item), mri_op, x_true=item)
    without = decompose_samples(draws(mri_op, item, seed=1), mri_op)
    agg = aggregate_reports([with_truth, without])

    assert agg["n_items"] == 2
    assert agg["operator_id"] == mri_op.descriptor().operator_id
    assert agg["null_energy_share"] == pytest.approx(
        (with_truth.null_energy_share + without.null_energy_share) / 2
    )
    # One report has no error split; the mean is over the finite one, not NaN.
    assert agg["null_error_share"] == pytest.approx(with_truth.null_error_share)


def test_aggregate_over_mixed_operators_refuses_to_claim_one(item):
    op_a = MRIUndersamplingOperator(SHAPE, family="random1d", acceleration=4)
    op_b = MRIUndersamplingOperator(SHAPE, family="random1d", acceleration=8)
    agg = aggregate_reports(
        [
            decompose_samples(draws(op_a, item), op_a),
            decompose_samples(draws(op_b, item), op_b),
        ]
    )
    assert agg["operator_id"] == "<mixed>"


def test_aggregate_of_nothing_is_an_error():
    with pytest.raises(ValueError, match="nothing to aggregate"):
        aggregate_reports([])


def test_all_nan_field_stays_nan_rather_than_becoming_zero(mri_op, item):
    """A missing measurement must not average into a confident-looking 0.0."""
    reports = [decompose_samples(draws(mri_op, item, seed=s), mri_op) for s in (0, 1)]
    agg = aggregate_reports(reports)
    assert math.isnan(agg["null_error_share"])


# -- the harder operator regime -----------------------------------------------


def test_more_undersampling_puts_more_energy_in_the_null_space(item):
    """The null space grows with acceleration, and the split has to see that.

    This is the operator dependence the whole project turns on: the same image
    and the same solver, split by two different acquisitions.
    """
    shares = []
    for acceleration in (2, 8):
        op = MRIUndersamplingOperator(
            SHAPE, family="random1d", acceleration=acceleration, mask_seed=0
        )
        shares.append(decompose_samples(draws(op, item), op, x_true=item).null_energy_share)
    assert shares[1] > shares[0]


def test_report_row_is_flat_and_serialisable(mri_op, item):
    rep = decompose_samples(draws(mri_op, item), mri_op, x_true=item, y=mri_op.forward(item))
    row = rep.to_row()
    assert isinstance(row, dict)
    assert row["operator_id"] == mri_op.descriptor().operator_id
    assert all(isinstance(v, (str, int, float)) for v in row.values())
    assert set(DecompositionReport.__dataclass_fields__) - {"extra"} <= set(row)
