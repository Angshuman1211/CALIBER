"""T1.1 correctness gates for the operator-aware score and its combination rule.

The central test here is the **injection test**. For a linear operator a
hallucination can be *constructed* rather than hoped for: take a clean image
``x`` and add ``P_N delta``. Because the added structure lies entirely in the
null space, ``A(x + P_N delta) == A x`` exactly — the measurements are
unchanged — while the image now contains structure the data does not support.
That is this project's definition of hallucination (Sec. 4), made synthetically
and exactly, and it gives T1.1 a verdict on the score with no solver, no
dataset, and no ground-truth ambiguity.

A score that cannot rank injected regions above clean ones is broken, and no
amount of downstream conformal calibration can rescue it: calibration is
score-agnostic and would happily certify a useless score.
"""

from __future__ import annotations

import pytest
import torch

from caliber.operators.registry import build_operator
from caliber.scoring.regions import RegionGrid
from caliber.scoring.score import (
    ScoreScales,
    ScoreWeights,
    combine_components,
    compute_score_components,
)

SHAPE = (128, 128)
REGION = 16


def _smooth(shape, coarseness, gen):
    """Low-frequency field — stands in for anatomy without needing a dataset."""
    small = torch.randn(
        1, 1, max(2, shape[0] // coarseness), max(2, shape[1] // coarseness), generator=gen
    )
    up = torch.nn.functional.interpolate(small, size=shape, mode="bilinear", align_corners=False)
    return up.reshape(*shape)


@pytest.fixture
def setup():
    gen = torch.Generator().manual_seed(0)
    op = build_operator(
        {
            "modality": "mri",
            "family": "random1d",
            "acceleration": 4,
            "center_fraction": 0.08,
            "mask_seed": 0,
        },
        SHAPE,
    )
    grid = RegionGrid(image_shape=SHAPE, region_size=REGION)
    x_true = _smooth(SHAPE, 8, gen).abs() + 1.0
    return op, grid, x_true, gen


def _inject(op, x_true, rows, cols, amplitude, gen):
    """Add ``P_N delta`` supported on a block of regions. Returns (image, mask)."""
    delta = torch.zeros(SHAPE)
    r0, r1 = rows[0] * REGION, rows[1] * REGION
    c0, c1 = cols[0] * REGION, cols[1] * REGION
    delta[r0:r1, c0:c1] = _smooth((r1 - r0, c1 - c0), 2, gen) * amplitude
    # Keep the projection complex. A random 1-D mask is not conjugate-symmetric,
    # so P_N of a real image genuinely has an imaginary part (33% of its norm
    # here); discarding it puts 4.1% of the injection back into the measured
    # component and the construction stops being a hallucination at all.
    free = op.project_null(delta)
    region_mask = torch.zeros(SHAPE, dtype=torch.bool)
    region_mask[r0:r1, c0:c1] = True
    return x_true + free, region_mask


def _draws(op, base, n, scale, gen):
    """Posterior draws that differ only inside the null space."""
    out = []
    for _ in range(n):
        out.append(base + op.project_null(_smooth(SHAPE, 4, gen) * scale))
    return torch.stack(out, dim=0)


def test_injection_is_exactly_data_consistent(setup):
    """The test's own premise: injected structure is invisible to the operator.

    Guards against the ``.real`` cast that silently breaks it — see D29 and the
    finding referred to Track 2 as D-024.
    """
    op, _, x_true, gen = setup
    x_hall, _ = _inject(op, x_true, (2, 5), (2, 5), amplitude=0.8, gen=gen)
    y_clean, y_hall = op.forward(x_true), op.forward(x_hall)
    assert torch.allclose(y_clean, y_hall, atol=1e-5), (
        "P_N delta must leave the measurements unchanged, or the injection is not a "
        "hallucination by this project's definition"
    )


def test_injected_regions_score_above_clean_regions(setup):
    """The T1.1 sanity gate: does the score rank constructed hallucination first?"""
    op, grid, x_true, gen = setup
    x_hall, mask = _inject(op, x_true, (2, 5), (2, 5), amplitude=0.8, gen=gen)
    y = op.forward(x_hall)
    samples = _draws(op, x_hall, n=8, scale=0.05, gen=gen)

    comps = compute_score_components(samples, y, op, grid)
    scales = ScoreScales.from_components(comps)
    score = combine_components(comps, ScoreWeights(), scales)

    injected = grid.reduce(mask.to(torch.float32), how="mean") > 0.5
    assert injected.any() and (~injected).any()
    assert score[injected].mean() > score[~injected].mean(), (
        "injected hallucination must score above clean tissue on average"
    )


def test_free_component_alone_detects_the_injection(setup):
    """Isolate the operator decomposition — the ablation T1.4 will run properly."""
    op, grid, x_true, gen = setup
    x_hall, mask = _inject(op, x_true, (2, 5), (2, 5), amplitude=0.8, gen=gen)
    y = op.forward(x_hall)
    samples = _draws(op, x_hall, n=8, scale=0.05, gen=gen)

    comps = compute_score_components(samples, y, op, grid)
    scales = ScoreScales.from_components(comps)
    score = combine_components(comps, ScoreWeights.only("free"), scales)

    injected = grid.reduce(mask.to(torch.float32), how="mean") > 0.5
    assert score[injected].mean() > score[~injected].mean()


def test_scales_are_frozen_not_re_estimated_per_operator(setup):
    """D27: re-standardising per deployment operator hides the coverage gap.

    Not a style preference — this test pins the behaviour the headline result
    depends on. Scores under a shifted operator must be allowed to drift away
    from the calibration distribution; re-estimating scales pulls them back.
    """
    _, grid, x_true, gen = setup
    ops = {
        r: build_operator(
            {
                "modality": "mri",
                "family": "random1d",
                "acceleration": r,
                "center_fraction": 0.08,
                "mask_seed": 0,
            },
            SHAPE,
        )
        for r in (2, 8)
    }
    comps = {}
    for r, op in ops.items():
        y = op.forward(x_true)
        comps[r] = compute_score_components(_draws(op, x_true, 6, 0.05, gen), y, op, grid)

    frozen = ScoreScales.from_components(comps[2])  # calibration operator only
    w = ScoreWeights()
    drift_frozen = abs(
        float(combine_components(comps[8], w, frozen).mean())
        - float(combine_components(comps[2], w, frozen).mean())
    )
    drift_refit = abs(
        float(combine_components(comps[8], w, ScoreScales.from_components(comps[8])).mean())
        - float(combine_components(comps[2], w, frozen).mean())
    )
    assert drift_frozen > drift_refit, (
        "re-estimating scales at the deployment operator shrank the score shift; "
        "that is exactly the artefact D27 forbids"
    )


def test_combination_is_affine_in_the_components(setup):
    """T3.3 §3 assumption (ii) leans on this: Lipschitz components -> Lipschitz score."""
    op, grid, x_true, gen = setup
    y = op.forward(x_true)
    comps = compute_score_components(_draws(op, x_true, 6, 0.05, gen), y, op, grid)
    scales, w = ScoreScales.from_components(comps), ScoreWeights()

    base = combine_components(comps, w, scales)
    bumped = combine_components(
        type(comps)(
            s_free=comps.s_free + 1.0,
            s_disagree=comps.s_disagree,
            s_residual=comps.s_residual,
            ref_intensity=comps.ref_intensity,
        ),
        w,
        scales,
    )
    # Weights are renormalised over the live components (D35), so the effective
    # free weight is w.free / sum(live weights), not the nominal w.free.
    live = scales.live_mask()
    effective_free = float((w.as_tensor() * live)[0] / (w.as_tensor() * live).sum())
    expected = effective_free / scales.free
    assert torch.allclose(bumped - base, torch.full_like(base, expected), atol=1e-4), (
        "score must stay affine in each live component — T3.3 §3 depends on it"
    )


def test_score_never_consults_ground_truth(setup):
    """Structural guard: a score that can see x_true is useless at deployment."""
    import inspect

    params = set(inspect.signature(compute_score_components).parameters)
    assert params == {"samples", "y", "operator", "grid"}
    assert not {"x_true", "truth", "target", "reference"} & params


@pytest.mark.parametrize("bad", [{"free": -0.5, "disagree": 1.0, "residual": 0.5}, {}])
def test_weights_are_validated(bad):
    if bad:
        with pytest.raises(ValueError):
            ScoreWeights(**bad)
    else:
        with pytest.raises(ValueError):
            ScoreWeights(free=1.0, disagree=1.0, residual=1.0)  # does not sum to 1


def test_degenerate_component_is_excluded_not_amplified(setup):
    """D35: a structurally-zero component must not be rescaled into noise.

    Standardisation maps every component to O(1), so a component that is exactly
    zero has its float noise amplified to the magnitude of real signal and then
    spends a full share of the weight budget on nothing. Measured before the
    guard: `s_residual` scale 9.1e-8 against `s_free` 3.5e-2, standardised to
    mean 0.95 — indistinguishable from a live component.
    """
    op, grid, x_true, gen = setup
    y = op.forward(x_true)
    samples = _draws(op, x_true, n=6, scale=0.05, gen=gen)
    comps = compute_score_components(samples, y, op, grid)
    scales = ScoreScales.from_components(comps)

    # The reference draws are data-consistent by construction, so the residual
    # is structurally zero — the DDNM case (D35).
    assert "residual" in scales.degenerate_names

    auto = combine_components(comps, ScoreWeights(), scales)
    manual = combine_components(
        comps, ScoreWeights(free=0.5, disagree=0.5, residual=0.0), scales
    )
    assert torch.allclose(auto, manual, atol=1e-6), (
        "dropping the degenerate component automatically must equal dropping it by hand"
    )


def test_all_components_degenerate_raises_rather_than_returning_zeros(setup):
    op, grid, x_true, gen = setup
    y = op.forward(x_true)
    comps = compute_score_components(_draws(op, x_true, 6, 0.05, gen), y, op, grid)
    dead = ScoreScales(
        free=1e-9, disagree=1e-9, residual=1e-9, degenerate=(True, True, True)
    )
    with pytest.raises(ValueError):
        combine_components(comps, ScoreWeights(), dead)
