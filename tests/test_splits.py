"""Split registry, calibration grouping, and the nullable bank column.

Covers the Track 2 half of Track 1's Week 1 sign-off: D-021 (grouping moves
into the registry), D-022 (patient-level, operator-independent split), A1
(``free_content``), and D-027 (volume is not patient on fastMRI).
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from caliber.data.splits import (
    DEFAULT_SALT,
    assign_splits,
    split_for,
    split_summary,
    verify_no_patient_straddles,
)
from caliber.operators import MRIUndersamplingOperator
from caliber.operators.registry import OperatorRegistry, build_operator
from caliber.scorebank.schema import (
    COLUMNS,
    NULLABLE_COLUMNS,
    SCHEMA_VERSION,
    coerce,
    validate,
)
from caliber.scoring import RegionGrid, compute_region_truth
from caliber.solvers.reference import NullSpaceNoiseSampler, PseudoInverseSampler

SHAPE = (64, 64)


# -- D-022: the split is a pure function of the patient -----------------------


def test_split_is_deterministic_across_calls_and_processes():
    keys = [f"patient-{i}" for i in range(50)]
    assert assign_splits(keys) == assign_splits(keys)
    # Order of presentation must not matter; a dict-order dependence here would
    # make two machines disagree about the same patient.
    assert assign_splits(keys) == assign_splits(list(reversed(keys)))


def test_split_does_not_depend_on_the_operator_or_the_run():
    """The binding requirement in D-022: same patient, same side, every regime."""
    patient = "e24dd3dc05"
    sides = {
        split_for(patient)
        for _ in range(5)  # stands in for five runs
    }
    assert len(sides) == 1, "split must not vary between runs"
    # There is no operator parameter to vary: the signature makes the
    # confound impossible rather than merely discouraged.
    assert "operator" not in split_for.__code__.co_varnames


def test_adding_patients_does_not_move_existing_ones():
    """Assignment is per key, not by shuffling a list, so banks stay comparable."""
    first = assign_splits([f"p{i}" for i in range(20)])
    grown = assign_splits([f"p{i}" for i in range(200)])
    for key, side in first.items():
        assert grown[key] == side


def test_calib_fraction_controls_the_balance():
    keys = [f"p{i}" for i in range(2000)]
    frac = sum(v == "calib" for v in assign_splits(keys, calib_fraction=0.25).values()) / len(keys)
    assert 0.22 < frac < 0.28


def test_changing_the_salt_redraws_the_split():
    keys = [f"p{i}" for i in range(200)]
    base = assign_splits(keys, salt=DEFAULT_SALT)
    other = assign_splits(keys, salt="caliber-split-v2")
    assert base != other


@pytest.mark.parametrize("bad", ["", None])
def test_empty_patient_key_is_rejected(bad):
    with pytest.raises(ValueError):
        split_for(bad)


def test_calib_fraction_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        split_for("p0", calib_fraction=1.5)


# -- D-027: volume is not patient ---------------------------------------------


def test_straddle_check_catches_a_patient_split_across_sides():
    """The check that would have caught D-027 before it reached a coverage number."""
    sample_to_patient = {"vol_a": "P1", "vol_b": "P1", "vol_c": "P2"}
    leaky = {"vol_a": "calib", "vol_b": "test", "vol_c": "test"}
    assert verify_no_patient_straddles(sample_to_patient, leaky) == ["P1"]

    clean = {"vol_a": "calib", "vol_b": "calib", "vol_c": "test"}
    assert verify_no_patient_straddles(sample_to_patient, clean) == []


def test_patient_level_assignment_never_straddles():
    """Two volumes per patient, the fastMRI val shape (199 volumes, 96 patients)."""
    sample_to_patient = {f"file{i}": f"P{i // 2}" for i in range(200)}
    summary = split_summary(sample_to_patient)
    assert summary["n_samples"] == 200
    assert summary["n_patients"] == 100
    assert summary["straddling_patients"] == []


def test_splitting_by_volume_leaks_patients():
    """Names the hazard explicitly, so nobody reintroduces the by-volume rule.

    P-06 recorded the rule as "by patient for LoDoPaB, by volume for fastMRI".
    That is wrong wherever a patient owns more than one volume, which on
    fastMRI singlecoil_val is *every* patient -- see D-027.
    """
    sample_to_patient = {f"file{i}": f"P{i // 2}" for i in range(200)}
    by_volume = assign_splits(sample_to_patient.keys())
    leaked = verify_no_patient_straddles(
        sample_to_patient, {s: by_volume[s] for s in sample_to_patient}
    )
    assert leaked, "splitting by volume must be shown to leak, not assumed safe"


# -- D-021: the grouping lives on the descriptor ------------------------------


def test_calib_group_default_is_family_and_severity():
    op = MRIUndersamplingOperator(SHAPE, family="random1d", acceleration=4)
    assert op.descriptor().calib_group == "mri/random1d/R4"


def test_mask_seeds_are_members_of_one_group_not_groups():
    """D-023: seed variation is within-group, and is the noise floor."""
    groups = {
        MRIUndersamplingOperator(
            SHAPE, family="random1d", acceleration=4, mask_seed=s
        ).descriptor().calib_group
        for s in range(4)
    }
    assert groups == {"mri/random1d/R4"}


def test_acceleration_and_family_are_between_group_moves():
    base = MRIUndersamplingOperator(SHAPE, family="random1d", acceleration=4)
    other_r = MRIUndersamplingOperator(SHAPE, family="random1d", acceleration=8)
    assert base.descriptor().calib_group != other_r.descriptor().calib_group


def test_config_can_override_the_group_without_a_code_change():
    """The point of D-021: Track 1 re-groups from YAML, not by editing Track 2."""
    op = build_operator(
        {"modality": "mri", "family": "random1d", "acceleration": 4,
         "calib_group": "mri/pooled"},
        SHAPE,
    )
    assert op.descriptor().calib_group == "mri/pooled"
    # ...and the override is recorded in the manifest, so a bank's grouping can
    # be audited against the registry that produced it.
    assert op.descriptor().to_row()["calib_group"] == "mri/pooled"


def test_override_does_not_change_the_operator_identity():
    """Grouping is a decision *about* the acquisition, not part of it."""
    plain = build_operator({"modality": "mri", "family": "random1d", "acceleration": 4}, SHAPE)
    grouped = build_operator(
        {"modality": "mri", "family": "random1d", "acceleration": 4, "calib_group": "x"}, SHAPE
    )
    assert plain.descriptor().operator_id == grouped.descriptor().operator_id
    registry = OperatorRegistry(SHAPE)
    registry.add(plain)
    registry.add(grouped)
    assert len(registry) == 1


# -- D-024: draws stay data-consistent ----------------------------------------


@pytest.mark.parametrize("sampler", [NullSpaceNoiseSampler(), PseudoInverseSampler()])
def test_reference_draws_are_exactly_data_consistent(sampler):
    """No `.real` cast anywhere on the path: A x_i must reproduce y exactly.

    This is the precondition the whole reference sampler exists to satisfy --
    draws differ *only* inside the null space. Before D-024 the cast put 4.1%
    of each perturbation back into the measured component.
    """
    op = MRIUndersamplingOperator(SHAPE, family="random1d", acceleration=4)
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(SHAPE, generator=gen) + 0.5
    y = op.forward(x)
    draws = sampler.sample(y, op, n_samples=3, seed=0)
    for draw in draws:
        assert torch.allclose(op.forward(draw), y, atol=1e-5)


def test_null_projection_keeps_its_imaginary_part():
    """A random 1-D mask is not conjugate-symmetric, so P_N x is genuinely complex."""
    op = MRIUndersamplingOperator(SHAPE, family="random1d", acceleration=4)
    gen = torch.Generator().manual_seed(0)
    free = op.project_null(torch.rand(SHAPE, generator=gen))
    assert free.is_complex()
    ratio = float(free.imag.norm() / free.norm())
    assert ratio > 0.1, f"imaginary part is load-bearing, measured {ratio:.3f} of the norm"


# -- A1: free_content ----------------------------------------------------------


def test_free_content_is_recorded_and_is_the_truths_own_null_space_content():
    op = MRIUndersamplingOperator(SHAPE, family="random1d", acceleration=4)
    grid = RegionGrid(image_shape=SHAPE, region_size=16)
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(SHAPE, generator=gen) + 0.5
    y = op.forward(x)
    draws = NullSpaceNoiseSampler().sample(y, op, n_samples=2, seed=0)
    truth = compute_region_truth(draws, x, op, grid)

    assert truth.free_content.shape == truth.err_free.shape
    assert torch.isfinite(truth.free_content).all()
    # It describes the ground truth alone, so it does not depend on the solver.
    other = compute_region_truth(
        PseudoInverseSampler().sample(y, op, n_samples=2, seed=1), x, op, grid
    )
    assert torch.allclose(truth.free_content, other.free_content)


def test_free_content_is_nullable_so_older_banks_still_validate():
    """A1 is additive and nullable: no SCHEMA_VERSION bump, no bank migration."""
    assert "free_content" in NULLABLE_COLUMNS
    row = {c: 0 for c in COLUMNS if c not in NULLABLE_COLUMNS}
    row.update(
        {c: "x" for c, d in COLUMNS.items() if d == "string" and c not in NULLABLE_COLUMNS}
    )
    row["schema_version"] = SCHEMA_VERSION
    row["n_posterior"] = 1
    old_bank = pd.DataFrame([row])
    assert "free_content" not in old_bank.columns

    conformed = coerce(old_bank)
    assert "free_content" in conformed.columns
    assert conformed["free_content"].isna().all()
    validate(conformed, strict=True)
