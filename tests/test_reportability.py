"""Reportability is a property of the bank, and the prior is part of it (D-018, P-18).

D-018 established that a bank must carry its own mark of being development-only,
because a bank is a table of floats and the caveat otherwise survives only in
whoever remembers which config was run. It gated on two things: a real substrate
and a generative (non-reference) sampler.

Those two are not sufficient. `ddpm-celebahq-256` is a real generative prior and
fastMRI is a real substrate, so a *face model reconstructing knees* cleared both
gates and produced a bank stamped `reportable: true` — while the T2.2 write-up
listed that same prior as a reason nothing was reportable. This file pins the
third gate shut.
"""

from __future__ import annotations

import pytest

from caliber.solvers import build_solver
from caliber.solvers.denoiser import (
    ADMISSIBLE_DOMAINS,
    PRIOR_DOMAINS,
    UNKNOWN_DOMAIN,
    prior_domain,
    prior_is_admissible,
)

CELEBA = "checkpoints/ddpm-celebahq-256"


# -- what a checkpoint was trained on -----------------------------------------


def test_the_celebahq_checkpoint_is_known_to_be_a_natural_image_prior():
    assert prior_domain(CELEBA) == "natural_image"


def test_domain_is_read_from_the_basename_so_the_path_does_not_matter():
    """`solver_id` and the manifest carry the basename, not an absolute path."""
    for path in (CELEBA, "ddpm-celebahq-256", "/tmp/models/ddpm-celebahq-256"):
        assert prior_domain(path) == "natural_image"


def test_an_unrecognised_checkpoint_is_unknown_and_not_admissible():
    """Failing closed is the point: an unknown checkpoint could be anything."""
    assert prior_domain("checkpoints/something-nobody-registered") == UNKNOWN_DOMAIN
    assert not prior_is_admissible("checkpoints/something-nobody-registered", "mri")
    assert not prior_is_admissible("checkpoints/something-nobody-registered", "ct")


def test_a_missing_checkpoint_is_unknown_rather_than_an_error():
    assert prior_domain(None) == UNKNOWN_DOMAIN
    assert not prior_is_admissible(None, "mri")


# -- the gate itself ----------------------------------------------------------


def test_a_face_prior_is_not_admissible_for_medical_modalities():
    """The whole of P-18 in one assertion."""
    assert not prior_is_admissible(CELEBA, "mri")
    assert not prior_is_admissible(CELEBA, "ct")


def test_a_face_prior_is_admissible_for_the_natural_image_appendix():
    """Sec. 11.1's Week 10 appendix is where this prior is the *right* prior."""
    assert prior_is_admissible(CELEBA, "natural_image")


def test_an_unknown_modality_admits_nothing():
    assert not prior_is_admissible(CELEBA, "ultrasound")


def test_medical_modalities_admit_only_their_own_domain():
    """No cross-admission: a CT prior is not an MRI prior."""
    assert ADMISSIBLE_DOMAINS["mri"] == frozenset({"mri"})
    assert ADMISSIBLE_DOMAINS["ct"] == frozenset({"ct"})


def test_every_registered_domain_is_one_the_admissibility_table_understands():
    """A prior domain nothing can ever admit is a typo, not a policy."""
    admissible = set().union(*ADMISSIBLE_DOMAINS.values())
    unreachable = set(PRIOR_DOMAINS.values()) - admissible
    assert not unreachable, f"registered domains no modality admits: {unreachable}"


# -- it reaches the solver, and therefore the manifest ------------------------


def test_solver_info_exposes_the_prior_domain():
    solver = build_solver({"name": "ddnm", "checkpoint": CELEBA, "n_steps": 4})
    assert solver.info().prior_domain == "natural_image"


def test_reference_samplers_have_no_prior_domain_to_speak_of():
    """They carry no checkpoint; the family gate already excludes them."""
    solver = build_solver({"name": "reference_nullspace"})
    assert solver.info().prior_domain == UNKNOWN_DOMAIN


@pytest.mark.parametrize("name", ["dps", "ddnm", "flowdps"])
def test_every_pretrained_back_end_reports_its_prior_domain(name):
    solver = build_solver({"name": name, "checkpoint": CELEBA, "n_steps": 4})
    assert solver.info().prior_domain == "natural_image"


# -- the regression this file exists for --------------------------------------


def test_fastmri_plus_a_face_prior_is_not_reportable():
    """The exact combination that produced a wrongly-stamped bank.

    A real substrate and a real generative prior, both individually fine, which
    together are still not a reportable result.
    """
    from caliber.data import DATASETS

    assert DATASETS["fastmri"].reportable is True  # substrate gate passes
    solver = build_solver({"name": "ddnm", "checkpoint": CELEBA, "n_steps": 4})
    assert solver.family != "reference"  # sampler gate passes
    assert not prior_is_admissible(solver.info().checkpoint, "mri")  # prior gate fails
