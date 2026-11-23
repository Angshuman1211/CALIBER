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
    UNKNOWN_ANATOMY,
    prior_anatomy,
    prior_modality,
    ADMISSIBLE_DOMAINS,
    AWAITING_RULING,
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


def test_medical_modalities_admit_only_their_own_modality():
    """No cross-admission: a CT prior is not an MRI prior.

    D80 widened these sets to carry anatomy (``mri_brain``, ``ct_neck_to_pelvis``), so the
    invariant is no longer a literal set -- it is that every admitted domain belongs to
    the admitting modality. Pinning the literal set would have to be edited every time a
    prior is added, which is how a guard stops guarding.
    """
    for modality in ("mri", "ct"):
        for domain in ADMISSIBLE_DOMAINS[modality]:
            assert domain == modality or domain.startswith(f"{modality}_"), (
                f"{modality!r} admits {domain!r}, which is another modality's prior"
            )
    assert not ADMISSIBLE_DOMAINS["mri"] & ADMISSIBLE_DOMAINS["ct"]


def test_every_registered_domain_is_one_the_admissibility_table_understands():
    """A prior domain nothing can ever admit is a typo, not a policy -- unless it is
    recorded as waiting on a ruling, which is a policy (``mri_brain``, T2.10)."""
    admissible = set().union(*ADMISSIBLE_DOMAINS.values())
    unreachable = set(PRIOR_DOMAINS.values()) - admissible - AWAITING_RULING
    assert not unreachable, f"registered domains no modality admits: {unreachable}"


def test_a_domain_waiting_on_a_ruling_is_admitted_nowhere_until_it_is_ruled():
    """The two lists must not overlap: a domain is either admitted or still waiting."""
    admissible = set().union(*ADMISSIBLE_DOMAINS.values())
    assert not AWAITING_RULING & admissible
    assert AWAITING_RULING <= set(PRIOR_DOMAINS.values()), "waiting on a domain never registered"


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


# -- the medical priors, and the anatomy gap they open (D79) -------------------

MRI_PRIOR = "checkpoints/segguided_diffusion/ddim-breast_mri-256"
CT_PRIOR = "checkpoints/segguided_diffusion/ddim-ct_organ_large-256"


def test_the_medical_priors_are_registered_for_their_own_modality():
    """D-016's bar: a medical prior rather than a natural-image one. Registering the key
    is the whole change that makes a bank reportable, so it has to be the right key."""
    assert prior_domain(MRI_PRIOR) == "mri_breast"
    assert prior_domain(CT_PRIOR) == "ct_neck_to_pelvis"
    assert prior_modality(MRI_PRIOR) == "mri"
    assert prior_modality(CT_PRIOR) == "ct"
    assert prior_is_admissible(MRI_PRIOR, "mri")
    assert prior_is_admissible(CT_PRIOR, "ct")


def test_a_medical_prior_is_still_not_admissible_across_modalities():
    """The CT prior is not an MRI prior, whatever else it is."""
    assert not prior_is_admissible(CT_PRIOR, "mri")
    assert not prior_is_admissible(MRI_PRIOR, "ct")
    assert not prior_is_admissible(MRI_PRIOR, "natural_image")


def test_anatomy_is_recorded_separately_from_the_domain():
    """**P-18 one level finer.** SegGuidedDiff's MRI prior is *breast* MRI, and this
    project's MRI banks are knee (fastMRI) and brain (M4Raw). Modality-level
    admissibility cannot see that, so the anatomy is carried alongside rather than folded
    in -- otherwise a breast prior on knee data is silently clean, which is precisely how
    the first fastMRI bank came out stamped reportable."""
    assert prior_anatomy(MRI_PRIOR) == "breast"
    assert prior_anatomy(CT_PRIOR) == "neck_to_pelvis"
    assert prior_anatomy("checkpoints/brats_ncsnpp_ve") == "brain"
    assert prior_anatomy(CELEBA) == "face"
    assert prior_anatomy(None) == UNKNOWN_ANATOMY
    assert prior_anatomy("checkpoints/who-knows") == UNKNOWN_ANATOMY


def test_anatomy_does_not_gate_admissibility_and_that_is_deliberate():
    """A breast-MRI prior on knee MRI is a publishable configuration with a stated
    limitation; a face prior on knee MRI is not a medical prior at all. Collapsing the two
    would either block the first wrongly or excuse the second."""
    assert prior_is_admissible(MRI_PRIOR, "mri"), "cross-anatomy stays reportable"
    assert not prior_is_admissible(CELEBA, "mri"), "cross-domain does not"


def test_every_medical_prior_declares_an_anatomy_in_its_domain(): 
    """D80 puts anatomy in the domain string, so a medical prior whose domain is a bare
    modality has an unstated limitation -- which is the whole thing the split prevents."""
    bare = sorted(k for k, v in PRIOR_DOMAINS.items() if v in {"mri", "ct"})
    assert not bare, f"medical priors with no anatomy in their domain: {bare}"


def test_cross_anatomy_stays_admissible_because_the_coverage_claim_survives_it():
    """**D80, answering T2.10.** Anatomy mismatch gates *claims*, not banks. Blocking the
    bank would also block the coverage result, which is the one the project can make; the
    detection result is already gone for the same underlying reason (D67)."""
    assert prior_is_admissible(MRI_PRIOR, "mri"), "breast prior stays admissible for MRI"
    assert prior_is_admissible("checkpoints/brats_ncsnpp_ve", "mri")
    assert not prior_is_admissible(CELEBA, "mri"), "a face prior is not a medical prior"


def test_nothing_is_parked_awaiting_a_ruling_that_has_been_made():
    """`AWAITING_RULING` held `mri_brain` pending D80. Leaving it there after the ruling
    would silently keep the BraTS banks unreportable."""
    from caliber.solvers.denoiser import AWAITING_RULING

    admissible = set().union(*ADMISSIBLE_DOMAINS.values())
    assert not (AWAITING_RULING & admissible), "a domain cannot be both parked and admitted"
    assert "mri_brain" not in AWAITING_RULING, "D80 ruled on it"
