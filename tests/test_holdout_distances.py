"""Gates for running the leave-one-out harness on CT (T1.2/T1.4, D71).

Two things stopped `tool2_holdout.py` working on CT, and neither would have announced itself:

* it recomputed every ordered pair's distance inline, which on CT costs minutes per pair and
  turns a CPU re-analysis into an overnight job;
* it decided which deployment operators are *interior* from `measurement_dim`, which is
  blind to an angular-arc change at a fixed view count -- so every arc fold would have been
  classified as an endpoint and discarded, and the arc folds are the ones CT is for.

The second test below demonstrates that blind spot on real CT operators rather than asserting
it, and the measured size of it is the argument: narrowing the arc hides *more* than halving
the view count.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from caliber.operators.ct import CTProjectionOperator
from caliber.operators.distance import compute_distance
from caliber.operators.mri import MRIUndersamplingOperator

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ct_distances  # noqa: E402
import tool2_holdout  # noqa: E402

SHAPE = (32, 32)


def growth(a, b) -> float:
    return float(compute_distance(a, b, measure="null_space_growth", normalise="ambient"))


# -- the cached distances ------------------------------------------------------


def _csv(tmp_path: Path) -> Path:
    path = tmp_path / "ct_distances.csv"
    pd.DataFrame(
        [
            {"calibration": "ct.v180", "deployment": "ct.v60",
             "measure": "null_space_growth", "value": 0.42, "signed": True,
             "seconds": 61.0, "note": ""},
            {"calibration": "ct.v180", "deployment": "ct.v60",
             "measure": "null_space_projection", "value": 0.42, "signed": False,
             "seconds": 60.0, "note": ""},
        ]
    ).to_csv(path, index=False)
    return path


def test_cached_distances_are_read_and_filtered_by_measure(tmp_path):
    loaded = tool2_holdout.load_distances(str(_csv(tmp_path)), "null_space_growth")
    assert loaded[("ct.v180", "ct.v60")] == pytest.approx(0.42)
    assert len(loaded) == 1, "the other measure's row must not leak in"


def test_a_measure_absent_from_the_cache_is_refused_and_says_what_is_there(tmp_path):
    with pytest.raises(ValueError, match="spectral_divergence"):
        tool2_holdout.load_distances(str(_csv(tmp_path)), "spectral_divergence")


# -- the measurement_dim blind spot, on real CT operators ----------------------


def test_an_arc_change_is_invisible_to_measurement_dim_but_not_to_the_operator():
    """The bug, demonstrated. Equal measurement count, different null space.

    Narrowing the angular range at a fixed view count removes directions the operator can
    see without changing how many measurements it takes, so `measurement_dim` cannot order
    these two operators at all and the old interiority test scored the change as zero.
    """
    wide = CTProjectionOperator(SHAPE, n_views=60, angular_range_deg=180.0)
    narrow = CTProjectionOperator(SHAPE, n_views=60, angular_range_deg=90.0)

    assert (
        wide.descriptor().measurement_dim == narrow.descriptor().measurement_dim
    ), "this control needs the measurement counts to be equal"
    assert abs(growth(wide, narrow)) > 0.1, "but the operator genuinely hides more"


def test_narrowing_the_arc_hides_more_than_halving_the_views():
    """Why discarding the arc folds would have thrown away the bigger effect.

    Measured at 32^2: the arc change is the larger operator shift of the two, and it is the
    one `measurement_dim` reports as no change.
    """
    wide = CTProjectionOperator(SHAPE, n_views=60, angular_range_deg=180.0)
    narrow = CTProjectionOperator(SHAPE, n_views=60, angular_range_deg=90.0)
    fewer = CTProjectionOperator(SHAPE, n_views=30, angular_range_deg=180.0)
    assert growth(wide, narrow) > growth(wide, fewer) > 0


def test_the_signed_axis_orders_ct_operators_that_measurement_dim_ties():
    """Three CT operators, two of which `measurement_dim` cannot distinguish."""
    ops = {
        "wide": CTProjectionOperator(SHAPE, n_views=60, angular_range_deg=180.0),
        "narrow": CTProjectionOperator(SHAPE, n_views=60, angular_range_deg=90.0),
        "fewer": CTProjectionOperator(SHAPE, n_views=30, angular_range_deg=180.0),
    }
    dims = {k: v.descriptor().measurement_dim for k, v in ops.items()}
    assert dims["wide"] == dims["narrow"], "tied on the old test"
    signed = {k: growth(ops["wide"], v) for k, v in ops.items() if k != "wide"}
    assert len(set(round(v, 4) for v in signed.values())) == 2, "separated on the new one"


# -- the MRI classification is unchanged --------------------------------------


def test_the_signed_test_reproduces_the_mri_interiority_it_replaces():
    """A generalisation must not move the case it generalises from.

    On nested MRI masks the signed growth has the same sign structure as the measurement
    count, so R4 is interior between R2 and R8 and the endpoints stay endpoints -- which is
    what the harness reported before the change.
    """
    ops = {
        r: MRIUndersamplingOperator(
            (128, 128), family="random1d", acceleration=r, center_fraction=0.08, mask_seed=0
        )
        for r in (2, 4, 8)
    }
    interior = {}
    for held in ops:
        others = [x for x in ops if x != held]
        g = [growth(ops[held], ops[x]) for x in others]
        interior[held] = any(v > 0 for v in g) and any(v < 0 for v in g)
    assert interior == {2: False, 4: True, 8: False}


# -- forcing the sparse-grid case (D75) ---------------------------------------


def _mri_grid(rates=(2, 3, 4, 6, 8)) -> dict:
    return {
        r: MRIUndersamplingOperator(
            (128, 128), family="random1d", acceleration=r, center_fraction=0.08, mask_seed=0
        )
        for r in rates
    }


def test_excluding_the_nearest_neighbour_can_turn_an_interior_fold_into_an_endpoint():
    """**The trap `--exclude-nearest` has to avoid.**

    Interiority is a property of the operators the fold actually retains. R4 is bracketed
    by R3 and R6 on the full grid, but drop its two nearest and every survivor can sit on
    one side -- the weighting is then extrapolating, and counting the fold as interior
    would credit it with an interpolation it cannot perform. The harness must recompute
    interiority after the exclusion, which is why `is_interior` takes the retained list.
    """
    ops = _mri_grid()
    held = 4
    others = sorted((x for x in ops if x != held), key=lambda x: abs(
        growth(ops[held], ops[x])
    ))

    def interior(retained) -> bool:
        g = [growth(ops[held], ops[x]) for x in retained]
        return any(v > 0 for v in g) and any(v < 0 for v in g)

    assert interior(others), "R4 is bracketed on the full grid"
    one_sided = [x for x in others if growth(ops[held], ops[x]) > 0]
    assert not interior(one_sided), (
        "a retained set on one side of R4 is an endpoint fold, whatever the full grid says"
    )


def test_the_nearest_arm_is_recomputed_over_what_remains():
    """Both arms must face the same handicap. If the nearest arm kept borrowing from an
    operator the fold excluded, Tool 2 would be compared against a threshold that is not
    available to it either -- a rigged comparison in Tool 2's favour."""
    ops = _mri_grid()
    held = 4
    ranked = sorted((x for x in ops if x != held),
                    key=lambda x: abs(growth(ops[held], ops[x])))
    for n_excluded in (0, 1, 2):
        retained = ranked[n_excluded:]
        nearest = min(retained, key=lambda x: abs(growth(ops[held], ops[x])))
        assert nearest not in ranked[:n_excluded], (
            "the nearest arm picked an operator the fold excluded"
        )
        assert nearest == retained[0]


def test_excluding_more_than_the_grid_holds_leaves_too_few_groups_to_weight():
    """With fewer than two retained groups there is nothing to weight, and the fold has to
    be skipped rather than silently degenerating into the nearest arm."""
    ops = _mri_grid(rates=(2, 4, 8))
    held = 4
    others = [x for x in ops if x != held]
    assert len(others[2:]) < 2, "excluding two of two leaves nothing to weight"


def test_the_exclusion_flag_is_rejected_when_negative(capsys):
    code = tool2_holdout.main(["nonexistent-bank", "--exclude-nearest", "-1"])
    assert code == 1
    assert "zero or more" in capsys.readouterr().err


# -- the distance sweep is usable on a non-CT bank (curve_collapse needs MRI rows) ----


def test_the_named_extra_pairs_are_filtered_to_what_the_bank_holds():
    """`ct_distances.py` hard-codes two CT pairs to probe arc-against-views. Pointed at an
    MRI bank -- which is what `curve_collapse.py` needs for its MRI distances, and which the
    manifest makes possible since operators are rebuilt generically -- those ids are absent,
    and the unfiltered list raises KeyError *after* writing the real rows. The script's
    `except ValueError` does not catch it, so the run dies with a half-written CSV.
    """
    mri_ops = {"mri.random1d.R2.c0.08.s0", "mri.random1d.R8.c0.08.s0"}
    kept = [(a, b) for a, b in ct_distances.EXTRA if a in mri_ops and b in mri_ops]
    assert kept == [], "no CT pair should survive the filter on an MRI bank"

    ct_ops = {"ct.sparse_view.v120.a180.dfull", "ct.limited_angle.v120.a90.dfull",
              "ct.sparse_view.v60.a180.dfull"}
    kept_ct = [(a, b) for a, b in ct_distances.EXTRA if a in ct_ops and b in ct_ops]
    assert len(kept_ct) == len(ct_distances.EXTRA), "every CT pair must survive on a CT bank"


# -- --all-pairs: one order per pair is enough (tool2 reads either order) ------------


def test_the_signed_measure_is_exactly_antisymmetric_on_ct_operators():
    """What lets `--all-pairs` compute one order only. Swapping the operators swaps "newly
    hidden" and "newly revealed" on the same seeded probes, so the reverse is the negative
    to rounding, not merely in expectation -- on CT, where the Hutchinson path is taken."""
    a = CTProjectionOperator(SHAPE, n_views=60, angular_range_deg=180.0)
    b = CTProjectionOperator(SHAPE, n_views=60, angular_range_deg=90.0)
    assert growth(a, b) == pytest.approx(-growth(b, a), abs=1e-6)


def test_all_pairs_fills_the_matrix_once_and_skips_a_pair_cached_in_reverse(tmp_path):
    """Every unordered pair is computed exactly once; a pair already in the cache in the
    other order is not recomputed, because that is the minutes-per-pair cost on CT."""
    import json

    ids = [f"mri.random1d.R{r}.c0.08.s0" for r in (2, 4, 8)]
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "manifest.json").write_text(json.dumps({"operators": [
        {"operator_id": oid, "modality": "mri", "image_shape": list(SHAPE),
         "params": {"family": "random1d", "acceleration": float(r), "center_fraction": 0.08,
                    "mask_seed": 0, "noise_sigma": 0.0}}
        for oid, r in zip(ids, (2, 4, 8))
    ]}), encoding="utf-8")
    out = tmp_path / "distances.csv"
    pd.DataFrame([{"calibration": ids[2], "deployment": ids[0],
                   "measure": "null_space_growth", "value": 0.5, "signed": True,
                   "seconds": 0.0, "note": "seeded"}]).to_csv(out, index=False)

    assert ct_distances.main([str(bank), "--calibration", ids[0], "--all-pairs",
                              "--measures", "null_space_growth", "--out", str(out)]) == 0

    rows = pd.read_csv(out)
    pairs = {frozenset((r.calibration, r.deployment)) for r in rows.itertuples()}
    assert pairs == {frozenset(p) for p in [(ids[0], ids[1]), (ids[0], ids[2]),
                                            (ids[1], ids[2])]}
    assert len(rows) == 3, "the pair cached in reverse must not be computed again"
    assert set(rows["measure"]) == {"null_space_growth"}


def test_all_pairs_does_not_need_the_default_calibration_operator(tmp_path):
    """The default --calibration is a CT id. --all-pairs uses none, so on an MRI bank it must
    not stop on that id -- the first run on the M4Raw bank did exactly that (T2.10)."""
    import json

    ids = [f"mri.random1d.R{r}.c0.08.s0" for r in (2, 8)]
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "manifest.json").write_text(json.dumps({"operators": [
        {"operator_id": oid, "modality": "mri", "image_shape": list(SHAPE),
         "params": {"family": "random1d", "acceleration": float(r), "center_fraction": 0.08,
                    "mask_seed": 0, "noise_sigma": 0.0}}
        for oid, r in zip(ids, (2, 8))
    ]}), encoding="utf-8")
    out = tmp_path / "distances.csv"
    assert ct_distances.main([str(bank), "--all-pairs", "--measures", "null_space_growth",
                              "--out", str(out)]) == 0
    assert len(pd.read_csv(out)) == 1
