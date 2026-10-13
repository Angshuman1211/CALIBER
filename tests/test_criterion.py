"""The relative ground-truth criterion, err_free / free_content > tau (A1 / D21).

The absolute criterion is tested alongside the rest of the labelling in
``test_splits.py``. What is asserted here is the relative form agreed at the
Week 1 sign-off and implemented once a real prior existed: that it is genuinely
relative, that it degrades safely where the denominator vanishes, and that the
tau curve reported to Track 1 is the curve the labelling function actually
applies.
"""

from __future__ import annotations

import numpy as np
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


# -- D57: the relative criterion needs a conjunctive absolute gate -------------


def _truth(err_free, free_content):
    n = len(err_free)
    return RegionTruth(
        err_total=torch.tensor(err_free, dtype=torch.float32),
        err_free=torch.tensor(err_free, dtype=torch.float32),
        free_content=torch.tensor(free_content, dtype=torch.float32),
        ann_coverage=torch.zeros(n),
    )


def test_the_ratio_alone_labels_a_smooth_region_from_a_tiny_error():
    """The defect D57 fixes, in two regions.

    A region with almost nothing hidden in it yields a large ratio from an error far too
    small to matter, so the relative criterion calls it hallucinated. That is why
    `-free_content` out-ranks every score in the project against this label (AUROC 0.826
    on the real-prior bank, 0.89-0.95 on Track 2's test patients).
    """
    # smooth region: a tiny error over a tiny denominator -> ratio 4.0
    # dense region: a large error over a large denominator -> ratio 1.0
    truth = _truth(err_free=[0.02, 0.50], free_content=[0.005, 0.50])
    ratio_only = label_hallucinated(truth, threshold=1.75, relative=True).numpy()
    assert ratio_only[0] and not ratio_only[1], "the smooth region is the one flagged"


def test_the_absolute_gate_removes_it_and_is_conjunctive():
    truth = _truth(err_free=[0.02, 0.50], free_content=[0.005, 0.50])
    gated = label_hallucinated(
        truth, threshold=1.75, relative=True, absolute_floor=0.10
    ).numpy()
    assert not gated.any(), "the smooth region fails the floor, the dense one fails the ratio"


def test_the_gate_can_only_remove_positives_never_add_them():
    rng = np.random.default_rng(0)
    truth = _truth(
        err_free=rng.uniform(0.01, 1.0, 400), free_content=rng.uniform(0.01, 1.0, 400)
    )
    base = label_hallucinated(truth, threshold=1.75, relative=True).numpy()
    for floor in (0.0, 0.1, 0.3, 0.6):
        gated = label_hallucinated(
            truth, threshold=1.75, relative=True, absolute_floor=floor
        ).numpy()
        assert not (gated & ~base).any(), f"floor {floor} invented a positive"


def test_a_floor_on_the_absolute_criterion_is_refused():
    """The absolute form already thresholds err_free; a second floor would be two
    thresholds on one quantity, which is a configuration error rather than a stricter rule."""
    truth = _truth(err_free=[0.5], free_content=[0.5])
    with pytest.raises(ValueError, match="relative"):
        label_hallucinated(truth, threshold=0.10, absolute_floor=0.2)


def test_the_candidacy_gate_is_the_one_that_removes_both_artefacts():
    """D60. `min_free_content` at a design-split quantile, not its numerical default.

    A region with nothing hidden in it cannot host a null-space hallucination, so it is not
    a candidate. Gating candidacy removes the denominator artefact (the ratio inflating on
    a small denominator) *and* the numerator circularity (`err_free -> s_free` as
    `free_content` falls, by D38's identity), where gating on `err_free` removes only the
    first and amplifies the second.
    """
    # Two regions with identical ratios, one of which has nothing hidden to invent into.
    truth = _truth(err_free=[0.04, 0.80], free_content=[0.01, 0.20])
    both = label_hallucinated(truth, threshold=1.75, relative=True).numpy()
    assert both.all(), "both regions clear the ratio test, which is the problem"

    gated = label_hallucinated(
        truth, threshold=1.75, relative=True, min_free_content=0.05
    ).numpy()
    assert not gated[0], "the empty region is not a candidate"
    assert gated[1], "the region with real hidden content still counts"


def test_the_default_min_free_content_never_fires_on_live_data():
    """Why the default is a guard and not the gate: the smallest live `free_content` on the
    real-prior bank is 0.162, which the 1e-3 default clears by two orders of magnitude."""
    assert MIN_FREE_CONTENT < 0.162 / 100
