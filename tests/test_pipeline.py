"""End-to-end and cross-cutting tests: config, scoring, metrics, and the runner.

The load-bearing test here is
``test_naive_transfer_loses_coverage_under_operator_shift``. It asserts the
project's central empirical claim on synthetic data: a threshold calibrated at
one acquisition operator under-covers when deployed at a distant one, and
group-conditional calibration recovers it. Real data replaces the phantoms in
Week 3-5; if this test ever stops holding, an interface has broken.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from caliber.calibration import apply_threshold, mondrian_thresholds, split_conformal_threshold
from caliber.config import ConfigError, apply_overrides, deep_merge, load_config
from caliber.data import build_dataset
from caliber.evaluation import (
    effective_sample_size,
    empirical_coverage,
    evaluate_groups,
    flag_area,
    region_fdr,
)
from caliber.operators import MRIUndersamplingOperator
from caliber.scoring import RegionGrid, compute_region_truth, compute_score_components
from caliber.solvers import build_solver

SHAPE = (64, 64)


# -- config --------------------------------------------------------------------


def test_deep_merge_is_recursive():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    assert deep_merge(base, {"a": {"b": 9}}) == {"a": {"b": 9, "c": 2}, "d": 3}


def test_overrides_parse_yaml_scalars():
    cfg = apply_overrides({"run": {"limit": None}}, ["run.limit=8", "run.seed=true"])
    assert cfg["run"]["limit"] == 8
    assert cfg["run"]["seed"] is True


def test_smoke_config_resolves(repo_root):
    cfg = load_config(repo_root / "configs" / "experiments" / "w01_smoke_phantom.yaml")
    assert cfg["experiment"]["name"] == "w01_smoke_phantom"
    assert cfg["regions"]["size"] == 16  # inherited from base.yaml
    assert cfg["_meta"]["config_hash"]


def test_config_rejects_a_solver_sweep(repo_root):
    """Sec. 13: the operator is the only quantity that may vary in a curve."""
    with pytest.raises(ConfigError, match="only quantity permitted to vary"):
        load_config(
            repo_root / "configs" / "experiments" / "w01_smoke_phantom.yaml",
            ["experiment.solver=[dps, flowdps]"],
        )


# -- regions and scoring ---------------------------------------------------------


def test_region_grid_partitions_the_image():
    grid = RegionGrid(SHAPE, region_size=16)
    assert grid.n_regions == 16
    ones = torch.ones(*SHAPE)
    assert torch.allclose(grid.reduce(ones, "mean"), torch.ones(16))
    assert torch.allclose(grid.reduce(ones, "sum"), torch.full((16,), 256.0))


def test_region_reduce_maps_regions_to_the_right_place():
    grid = RegionGrid(SHAPE, region_size=16)
    x = torch.zeros(*SHAPE)
    x[0:16, 16:32] = 1.0  # row 0, col 1 -> region_id 1
    reduced = grid.reduce(x, "mean")
    assert int(reduced.argmax()) == 1


def test_score_components_are_finite_and_per_region():
    op = MRIUndersamplingOperator(SHAPE, acceleration=4)
    grid = RegionGrid(SHAPE, region_size=16)
    item = build_dataset({"name": "phantom", "n_items": 1, "image_shape": SHAPE})[0]
    solver = build_solver({"name": "reference_nullspace"})
    y = op.forward(item.image)
    samples = solver.sample(y, op, n_samples=4, seed=0)

    comps = compute_score_components(samples, y, op, grid)
    for arr in (comps.s_free, comps.s_disagree, comps.s_residual):
        assert arr.shape == (grid.n_regions,)
        assert torch.isfinite(arr).all()
    assert comps.s_disagree.sum() > 0  # the reference sampler does disagree


def test_pseudo_inverse_sampler_shows_no_disagreement():
    op = MRIUndersamplingOperator(SHAPE, acceleration=4)
    grid = RegionGrid(SHAPE, region_size=16)
    item = build_dataset({"name": "phantom", "n_items": 1, "image_shape": SHAPE})[0]
    y = op.forward(item.image)
    samples = build_solver("reference_pinv").sample(y, op, n_samples=4, seed=0)
    comps = compute_score_components(samples, y, op, grid)
    assert float(comps.s_disagree.abs().max()) == pytest.approx(0.0, abs=1e-6)


def test_solver_sampling_is_deterministic_given_a_seed():
    """The reproducibility package (Week 10) depends on this."""
    op = MRIUndersamplingOperator(SHAPE, acceleration=4)
    item = build_dataset({"name": "phantom", "n_items": 1, "image_shape": SHAPE})[0]
    y = op.forward(item.image)
    solver = build_solver("reference_nullspace")
    a = solver.sample(y, op, n_samples=3, seed=7)
    b = solver.sample(y, op, n_samples=3, seed=7)
    assert torch.equal(a, b)


def test_phantom_dataset_is_deterministic():
    a = build_dataset({"name": "phantom", "n_items": 4, "image_shape": SHAPE})
    b = build_dataset({"name": "phantom", "n_items": 4, "image_shape": SHAPE})
    assert torch.equal(a[2].image, b[2].image)
    assert a[2].annotation is not None


def test_unwired_solvers_fail_loudly_not_silently():
    op = MRIUndersamplingOperator(SHAPE, acceleration=4)
    y = op.forward(torch.rand(*SHAPE))
    for name in ("dps", "ddnm", "flowdps"):
        with pytest.raises(NotImplementedError, match="Week 2"):
            build_solver(name).sample(y, op, n_samples=1)


# -- metrics ----------------------------------------------------------------------


def test_metric_definitions():
    labels = [True, True, True, True, False, False]
    flags = [True, True, True, False, True, False]
    assert empirical_coverage(flags, labels) == pytest.approx(0.75)  # 3 of 4 caught
    assert region_fdr(flags, labels) == pytest.approx(0.25)  # 1 of 4 flagged is clean
    assert flag_area(flags) == pytest.approx(4 / 6)


def test_effective_sample_size_penalises_uneven_weights():
    assert effective_sample_size([1.0] * 100) == pytest.approx(100.0)
    assert effective_sample_size([1.0] + [0.0] * 99) == pytest.approx(1.0)


def test_coverage_is_nan_when_there_is_nothing_to_catch():
    import math

    assert math.isnan(empirical_coverage([False, False], [False, False]))


# -- the central claim, on synthetic data ------------------------------------------


def _bank_for(operators, n_items=12, region_size=16, seed=0):
    """Build a small in-memory score bank across a set of operators."""
    grid = RegionGrid(SHAPE, region_size=region_size)
    dataset = build_dataset({"name": "phantom", "n_items": n_items, "image_shape": SHAPE})
    solver = build_solver({"name": "reference_nullspace", "hallucination_scale": 0.25})
    rows = []
    for item in dataset:
        for op in operators:
            y = op.forward(item.image)
            samples = solver.sample(y, op, n_samples=6, seed=seed)
            comps = compute_score_components(samples, y, op, grid)
            truth = compute_region_truth(
                samples, item.image, op, grid, item.annotation, comps.ref_intensity
            )
            score = comps.s_free + comps.s_disagree + comps.s_residual
            for rid in range(grid.n_regions):
                rows.append(
                    {
                        "operator_id": op.descriptor().operator_id,
                        "calib_group": op.descriptor().operator_id,
                        "sample_id": item.sample_id,
                        "region_id": rid,
                        "score": float(score[rid]),
                        "err_free": float(truth.err_free[rid]),
                    }
                )
    df = pd.DataFrame(rows)
    df["label"] = df["err_free"] > df["err_free"].quantile(0.80)
    return df


def test_naive_transfer_loses_coverage_under_operator_shift():
    """The empirical core (O3), in miniature.

    Calibrate a threshold at R=2 and deploy it at R=10. The far operator hides
    more of the image, so its scores live on a different scale and the
    transferred threshold is miscalibrated. Group-conditional calibration,
    which thresholds within each regime, must recover the nominal level.

    **The direction is not asserted, deliberately.** Until D-024 this test
    required *under*-coverage at the far operator. That direction was an
    artefact of the `.real` cast in `NullSpaceNoiseSampler`: it put 4.1% of
    each perturbation back into the measured component, making `s_residual`
    non-zero (0.082 at R=2 vs 0.032 at R=10) and dragging the combined score
    *down* with acceleration. With the projection kept complex, `s_residual`
    is exactly 0 and `s_free` rises with acceleration (0.0326 -> 0.0377), which
    is the physically correct direction: a larger null space holds more free
    content. Naive transfer now *over*-covers, and it does so to saturation.

    Over-coverage here is not the safe side. Coverage 1.0 means every region is
    flagged, so the detector carries no information at all -- the same vacuity
    D-020 records for the absolute label threshold, reached by a different
    route. What the guarantee promises is the *nominal* level, not "at least".
    """
    calib_op = MRIUndersamplingOperator(SHAPE, family="random1d", acceleration=2)
    far_op = MRIUndersamplingOperator(SHAPE, family="random1d", acceleration=10)
    df = _bank_for([calib_op, far_op])

    alpha, nominal = 0.10, 0.90
    calib_id = calib_op.descriptor().operator_id
    far_id = far_op.descriptor().operator_id
    calib = df[df["operator_id"] == calib_id]

    tau = split_conformal_threshold(calib["score"], calib["label"], alpha=alpha)
    naive = evaluate_groups(
        apply_threshold(df, tau, "score"), by=("operator_id",), nominal=nominal
    ).set_index("operator_id")

    # Calibration operator: coverage is honoured where the threshold was set.
    assert naive.loc[calib_id, "empirical_coverage"] >= nominal - 0.05

    # Deployment operator: the guarantee is not honoured. This is the failure
    # the whole project exists to expose and repair. Magnitude, not sign --
    # see the docstring and D-024.
    assert abs(naive.loc[far_id, "coverage_gap"]) > 0.05

    # ...and it fails by saturating: the transferred threshold flags every
    # region, so the flag is vacuous rather than merely imprecise.
    assert naive.loc[far_id, "empirical_coverage"] == pytest.approx(1.0)

    # The guaranteed backbone restores it.
    taus = mondrian_thresholds(df, "score", "label", alpha=alpha)
    mondrian = evaluate_groups(
        apply_threshold(df, taus, "score"), by=("operator_id",), nominal=nominal
    ).set_index("operator_id")
    assert mondrian.loc[far_id, "empirical_coverage"] >= nominal - 0.05
    # Closer to nominal, not merely higher: naive over-covers here, so "higher"
    # would be satisfied by getting the calibration *more* wrong (D-024).
    assert abs(mondrian.loc[far_id, "coverage_gap"]) < abs(naive.loc[far_id, "coverage_gap"])
