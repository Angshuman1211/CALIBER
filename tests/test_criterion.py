"""The relative ground-truth criterion, err_free / free_content > tau (A1 / D21).

The absolute criterion is tested alongside the rest of the labelling in
``test_splits.py``. What is asserted here is the relative form agreed at the
Week 1 sign-off and implemented once a real prior existed: that it is genuinely
relative, that it degrades safely where the denominator vanishes, and that the
tau curve reported to Track 1 is the curve the labelling function actually
applies.
"""

from __future__ import annotations

import pytest
import torch

from caliber.scoring.labels import (
    MIN_FREE_CONTENT,
    RegionTruth,
    label_hallucinated,
    positive_rate_sweep,
)


def truth(err_free, free_content, ann=None):
    err_free = torch.tensor(err_free, dtype=torch.float32)
    free_content = torch.tensor(free_content, dtype=torch.float32)
    return RegionTruth(
        err_total=err_free.clone(),
        err_free=err_free,
        free_content=free_content,
        ann_coverage=torch.zeros_like(err_free) if ann is None else torch.tensor(ann),
    )


# -- the criterion is relative ------------------------------------------------


def test_the_same_error_labels_differently_by_how_much_content_was_there():
    """The whole reason for the denominator (D-020, D21).

    Two regions with identical ``err_free``. One is dense anatomy that carries a
    lot the operator cannot see anyway; the other is nearly featureless in the
    free component, so the same error is a much larger share of invention. The
    absolute criterion cannot tell them apart; the relative one must.
    """
    t = truth(err_free=[0.4, 0.4], free_content=[0.8, 0.1])

    absolute = label_hallucinated(t, threshold=0.10)
    assert absolute.tolist() == [True, True]

    relative = label_hallucinated(t, threshold=1.0, relative=True)
    assert relative.tolist() == [False, True]


def test_ratio_threshold_is_applied_where_it_is_stated():
    t = truth(err_free=[0.9, 1.0, 1.1], free_content=[1.0, 1.0, 1.0])
    assert label_hallucinated(t, threshold=1.0, relative=True).tolist() == [False, False, True]


def test_absolute_mode_is_unchanged_and_is_still_the_default():
    """Existing callers and every bank written so far must keep their meaning."""
    t = truth(err_free=[0.05, 0.2], free_content=[1.0, 1.0])
    assert label_hallucinated(t, threshold=0.10).tolist() == [False, True]
    assert label_hallucinated(t, threshold=0.10, relative=False).tolist() == [False, True]


# -- the degenerate denominator -----------------------------------------------


def test_empty_regions_are_excluded_rather_than_flagged():
    """A background patch has nothing in the free component to invent over.

    Without the guard this is the worst possible failure: 0/0 becomes a huge
    ratio and every empty region flags, which is P-02 again in a new costume.
    """
    t = truth(err_free=[1e-9, 0.5], free_content=[0.0, 0.5])
    labelled = label_hallucinated(t, threshold=0.5, relative=True)
    assert labelled.tolist() == [False, True]


def test_a_region_just_above_the_floor_is_still_evaluated():
    """The floor excludes nothing real; it must not quietly eat live data."""
    t = truth(err_free=[10 * MIN_FREE_CONTENT], free_content=[2 * MIN_FREE_CONTENT])
    assert label_hallucinated(t, threshold=1.0, relative=True).tolist() == [True]


def test_the_floor_is_far_below_real_anatomy():
    """Measured on the M4Raw real-prior bank the smallest free_content is 0.162."""
    assert MIN_FREE_CONTENT < 0.162 / 100


def test_the_guard_never_produces_nan_and_leaves_usable_regions_alone():
    """Zero-content regions are dropped; a usable region beside them is untouched.

    The third region has ratio 0.3/0.4 = 0.75, which clears tau=0.5 — the guard
    must not swallow it just because its neighbours were degenerate.
    """
    t = truth(err_free=[0.0, 1.0, 0.3], free_content=[0.0, 0.0, 0.4])
    labelled = label_hallucinated(t, threshold=0.5, relative=True)
    assert labelled.dtype == torch.bool
    assert labelled.tolist() == [False, False, True]


# -- annotation gating composes with both modes -------------------------------


def test_annotation_gate_still_applies_in_relative_mode():
    t = truth(err_free=[0.9, 0.9], free_content=[0.3, 0.3], ann=[0.5, 0.0])
    gated = label_hallucinated(
        t, threshold=1.0, relative=True, require_annotation=True, annotation_min_coverage=0.10
    )
    assert gated.tolist() == [True, False]


# -- the tau curve reported to Track 1 ----------------------------------------


def test_sweep_matches_the_labelling_function_at_every_tau():
    """The reported curve and the applied criterion must not drift apart."""
    gen = torch.Generator().manual_seed(0)
    err = torch.rand(200, generator=gen) * 2
    content = torch.rand(200, generator=gen) + 0.5
    t = truth(err.tolist(), content.tolist())

    taus = [0.5, 1.0, 1.5, 2.0]
    for tau, rate in positive_rate_sweep(err, content, taus):
        expected = float(label_hallucinated(t, threshold=tau, relative=True).float().mean())
        assert rate == pytest.approx(expected, abs=1e-6)


def test_sweep_is_monotone_decreasing_in_tau():
    gen = torch.Generator().manual_seed(1)
    err = torch.rand(300, generator=gen) * 3
    content = torch.rand(300, generator=gen) + 0.2
    rates = [r for _, r in positive_rate_sweep(err, content, [0.5, 1.0, 1.5, 2.0, 3.0])]
    assert rates == sorted(rates, reverse=True)


def test_sweep_ignores_regions_below_the_floor():
    err = torch.tensor([5.0, 1.0])
    content = torch.tensor([0.0, 1.0])
    # Only the second region is usable, and it does not clear tau=2.
    assert positive_rate_sweep(err, content, [2.0]) == [(2.0, 0.0)]


def test_sweep_refuses_a_bank_with_no_usable_region():
    with pytest.raises(ValueError, match="no region has enough free content"):
        positive_rate_sweep(torch.tensor([1.0]), torch.tensor([0.0]), [1.0])
