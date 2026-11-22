"""T1-A's anatomy: the reconciliation, the inversion, and the one ordering violation.

This script makes claims about data Track 1 does not hold -- the per-pair coverage table is
on Track 2's disk -- so everything here is either a property of the operator grid, which is
local and real, or a reconciliation against rates recorded verbatim in
:data:`t1a_rule_anatomy.REPORTED`. The reconciliation is an integer check: Track 2's
per-cell rates have to resolve to whole pair counts against denominators computed here from
the distance matrix, and if they do not, one of the two sides is wrong.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import t1a_rule_anatomy as A  # noqa: E402

GRID = REPO_ROOT / "runs" / "t1a_ct_distances"
needs_grid = pytest.mark.skipif(
    not (GRID / "manifest.json").exists() or not (GRID / "distances.csv").exists(),
    reason="the real CT operator grid is not in this checkout",
)


def _loaded():
    meta = A.operator_metadata(GRID / "manifest.json")
    signed = A.load_signed(GRID / "distances.csv", "null_space_growth")
    return meta, signed, A.anatomy(meta, signed)


# -- the declared threshold order -------------------------------------------------------


def test_the_threshold_order_follows_the_knob_inside_each_family():
    meta = {
        "sv.hi": {"family": "sparse_view", "views": 180, "arc_deg": 180},
        "sv.lo": {"family": "sparse_view", "views": 30, "arc_deg": 180},
        "la.wide": {"family": "limited_angle", "views": 120, "arc_deg": 150},
        "la.narrow": {"family": "limited_angle", "views": 120, "arc_deg": 90},
    }
    rank = A.threshold_rank(meta)
    # less data -> larger error -> larger score -> higher threshold
    assert rank["sv.hi"] < rank["sv.lo"]
    assert rank["la.wide"] < rank["la.narrow"]
    # and the one declared bit: the limited-angle block sits above the whole sv block
    assert max(rank["sv.hi"], rank["sv.lo"]) < min(rank["la.wide"], rank["la.narrow"])


def test_an_unhandled_cross_family_bit_raises_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(A, "CROSS_FAMILY_BIT", "something_else")
    with pytest.raises(ValueError, match="unhandled cross-family bit"):
        A.threshold_rank({"a": {"family": "sparse_view", "views": 1, "arc_deg": 180}})


# -- the grid itself --------------------------------------------------------------------


@needs_grid
def test_the_signed_distance_is_filled_antisymmetrically():
    meta, signed, _ = _loaded()
    for a, b in itertools.permutations(sorted(meta), 2):
        assert signed[(a, b)] == pytest.approx(-signed[(b, a)], abs=1e-12)


@needs_grid
def test_every_ordered_pair_of_the_nine_operators_is_present():
    meta, signed, table = _loaded()
    assert len(meta) == 9
    assert len(table) == 72
    assert table["signed_distance"].notna().all()


# -- the reconciliation -----------------------------------------------------------------


@needs_grid
def test_every_reported_cell_rate_resolves_to_a_whole_pair_count():
    _, _, table = _loaded()
    rec = A.reconcile(table)
    assert rec["unsafe_is_whole"].all(), rec
    assert rec["false_safe_is_whole"].all(), rec


@needs_grid
def test_the_pooled_rates_reconstruct_the_reported_ones():
    _, _, table = _loaded()
    rec = A.reconcile(table)

    def total(col):
        v = rec[col].to_numpy(dtype=float)
        return int(np.nansum(v[np.isfinite(v)]))

    unsafe, false_safe = total("implied_unsafe"), total("implied_false_safe")
    n, n_safe = int(rec["n_pairs"].sum()), int(rec["rule_predicted_safe"].sum())
    assert (unsafe, n) == (23, 72)
    assert (false_safe, n_safe) == (13, 36)
    assert unsafe / n == pytest.approx(A.REPORTED_POOLED["trivial"], abs=0.001)
    assert false_safe / n_safe == pytest.approx(A.REPORTED_POOLED["false_safe"], abs=0.001)


# -- the inversion ----------------------------------------------------------------------


@needs_grid
def test_inside_sparse_view_the_rule_is_exactly_inverted_not_merely_uninformative():
    """The rule's safe set is the identity's may-be-unsafe set, on all 30 pairs."""
    _, _, table = _loaded()
    sv = table[table["cell"] == "sv->sv"]
    assert len(sv) == 30
    assert (sv["rule_says_safe"] != sv["identity_says_cannot_be_unsafe"]).all()


@needs_grid
def test_the_rules_safe_set_is_anti_correlated_with_the_identitys():
    _, _, table = _loaded()
    rule = table["rule_says_safe"].to_numpy(bool)
    safe = table["identity_says_cannot_be_unsafe"].to_numpy(bool)
    n, a, b = len(table), int(rule.sum()), int(safe.sum())
    observed = int((rule & safe).sum())
    expected = a * b / n
    assert (a, b, observed) == (36, 36, 12)
    assert observed < expected, "a rule no better than chance would overlap about 18"


@needs_grid
def test_the_six_false_safe_la_to_sv_pairs_are_the_two_lowest_threshold_targets():
    """Track 2 report false-safe 1.000 in la->sv. These are the six pairs it refers to."""
    meta, _, table = _loaded()
    sel = table[(table["cell"] == "la->sv") & table["rule_says_safe"]]
    assert len(sel) == 6
    targets = {A.short_name(o, meta) for o in sel["deployment"]}
    assert targets == {"sv 120", "sv 180"}
    assert (sel["signed_distance"] < 0).all()


@needs_grid
def test_no_sparse_view_to_limited_angle_pair_can_be_unsafe():
    """The identity's one prediction with no fitted magnitude; Track 2 report 0 unsafe."""
    _, _, table = _loaded()
    cell = table[table["cell"] == "sv->la"]
    assert len(cell) == 18
    assert cell["identity_says_cannot_be_unsafe"].all()
    assert A.REPORTED["sv->la"]["trivial"] == 0.0


# -- the one ordering violation ---------------------------------------------------------


@needs_grid
def test_the_quoted_coverages_break_the_knob_order_at_exactly_one_place():
    meta, _, _ = _loaded()
    cons = A.ordering_constraints(meta)
    broken = cons[~cons["consistent_with_knob_order"]]
    assert len(broken) == 1, cons
    row = broken.iloc[0]
    # the stronger kind: it shares tau_a between the two cells, so tau cancels
    assert row["kind"] == "within_row"
    assert "sv 120" in row["implies"] and "sv 180" in row["implies"]
    assert row["implies"] == "tau(sv 120) < tau(sv 180)"


@needs_grid
def test_the_violation_is_the_operator_track_2_called_out_of_order():
    """sv 120 is hardest to cover from almost every row, which needs a low threshold."""
    meta, _, _ = _loaded()
    rank = A.threshold_rank(meta)
    by_short = {A.short_name(o, meta): o for o in meta}
    # the knob order puts sv 180 lowest and sv 120 second; the data swaps them
    assert rank[by_short["sv 180"]] == 0
    assert rank[by_short["sv 120"]] == 1


@needs_grid
def test_within_row_constraints_need_no_matched_value():
    """Changing the matched range cannot move a within_row verdict; it can move vs_matched."""
    meta, _, _ = _loaded()
    before = A.ordering_constraints(meta)
    original = A.MATCHED_RANGE
    try:
        A.MATCHED_RANGE = (0.0, 1.0)      # decides nothing against the matched arm
        after = A.ordering_constraints(meta)
    finally:
        A.MATCHED_RANGE = original
    for kind in ("within_row",):
        b = before[before["kind"] == kind].reset_index(drop=True)
        a = after[after["kind"] == kind].reset_index(drop=True)
        assert b["implies"].tolist() == a["implies"].tolist()
        assert b["consistent_with_knob_order"].tolist() == \
            a["consistent_with_knob_order"].tolist()
    assert (after[after["kind"] == "vs_matched"]["implies"]
            .str.startswith("undecided").all())
