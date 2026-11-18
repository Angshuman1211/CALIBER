"""A bank carries claims, not a boolean (D80, D82; T2-B of the Phase 4 plan).

D-018 put reportability in the manifest as one flag. D80 and D82 each found a defect
that reaches detection and not coverage -- a prior of another anatomy, a substrate with
no annotations -- and one flag cannot say "this bank's coverage numbers, and not its
detection numbers". Flipping the flag to ``True`` for M4Raw would have stamped a
detection number from it as reportable: P-18 again, the gate unable to see the thing
that mattered. These tests pin the scoped stamp, and the refusal D82 asks for.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from caliber.data import DATASETS
from caliber.reportability import (
    CLAIMS,
    COVERAGE,
    DETECTION,
    ClaimNotCarried,
    bank_claims,
    carries,
    claims_for,
    claims_of,
    describe,
    prior_claims,
    require_claim,
    stamp,
    substrate_claims,
)
from caliber.solvers import build_solver

CELEBA = "checkpoints/ddpm-celebahq-256"
BRATS = "checkpoints/song_inverse_problems/brats_ncsnpp_ve"
LIDC = "checkpoints/song_inverse_problems/ct2d_320_ncsnpp_ve"

M4RAW = DATASETS["m4raw"]
FASTMRI = DATASETS["fastmri"]
TOTALSEG = DATASETS["totalsegmentator"]
PHANTOM = DATASETS["phantom"]


# -- the substrate (D82) -------------------------------------------------------


def test_m4raw_carries_coverage_and_not_detection():
    """D82's ruling, in the place it lives."""
    assert substrate_claims(M4RAW) == {COVERAGE}


def test_m4raw_is_still_not_reportable_without_qualification():
    """The ruling scopes D-015, it does not reverse it. A reader of the old flag alone
    must still see a development substrate."""
    assert M4RAW.reportable is False


def test_a_substrate_that_declares_nothing_is_all_or_none_by_its_flag():
    assert substrate_claims(FASTMRI) == CLAIMS
    assert substrate_claims(TOTALSEG) == CLAIMS
    assert substrate_claims(PHANTOM) == frozenset()


def test_a_substrate_cannot_declare_a_claim_nobody_defined():
    class Typo:
        name = "typo"
        reportable_claims = frozenset({"coverege"})

    with pytest.raises(ValueError, match="unknown claims"):
        substrate_claims(Typo)


# -- the prior (D80) -----------------------------------------------------------


def test_a_prior_of_the_substrates_anatomy_carries_everything():
    assert prior_claims(BRATS, "mri", "brain") == CLAIMS


def test_a_medical_prior_of_another_anatomy_carries_coverage_only():
    """Anatomy gates claims, not banks. A brain prior on knees, a chest prior on
    whole-body slices: the coverage comparison stands, the detection one does not."""
    assert prior_claims(BRATS, "mri", "knee") == {COVERAGE}
    assert prior_claims(LIDC, "ct", "whole_body") == {COVERAGE}


def test_a_prior_that_is_not_admissible_carries_nothing():
    """P-18: a face prior is not a cross-anatomy medical prior, it is not a medical one."""
    assert prior_claims(CELEBA, "mri", "knee") == frozenset()
    assert prior_claims(LIDC, "mri", "brain") == frozenset()
    assert prior_claims("checkpoints/nobody-registered-this", "ct", "chest") == frozenset()
    assert prior_claims(None, "mri", "brain") == frozenset()


def test_an_undeclared_anatomy_matches_no_prior():
    """Fails closed: a new dataset that forgets to say what it shows gets coverage at
    most, never detection by accident."""
    assert prior_claims(BRATS, "mri", "undeclared") == {COVERAGE}


def test_every_substrate_that_carries_a_claim_declares_its_anatomy():
    for name, cls in DATASETS.items():
        if substrate_claims(cls):
            assert cls.anatomy != "undeclared", f"{name} carries claims and states no anatomy"


# -- the bank: both gates and the sampler, intersected -------------------------


@pytest.mark.parametrize(
    "dataset, checkpoint, expected",
    [
        (M4RAW, BRATS, {COVERAGE}),        # anatomy matches; the substrate is the limit
        (FASTMRI, BRATS, {COVERAGE}),      # the substrate is clean; the anatomy is the limit
        (TOTALSEG, LIDC, {COVERAGE}),      # T2-A's bank: chest prior, whole-body slices
        (FASTMRI, CELEBA, frozenset()),    # P-18's bank
        (M4RAW, CELEBA, frozenset()),
        (TOTALSEG, CELEBA, frozenset()),   # the CT bank of record
    ],
)
def test_the_banks_this_project_has_built(dataset, checkpoint, expected):
    assert claims_for(dataset, "diffusion", checkpoint) == expected


def test_a_reference_sampler_carries_nothing_on_any_substrate():
    assert claims_for(FASTMRI, "reference", None) == frozenset()
    solver = build_solver({"name": "reference_nullspace"})
    assert bank_claims(FASTMRI, solver) == frozenset()


def test_a_live_solver_gives_the_same_answer_as_its_checkpoint():
    solver = build_solver({"name": "ddnm", "checkpoint": CELEBA, "n_steps": 4})
    assert bank_claims(FASTMRI, solver) == claims_for(FASTMRI, solver.family, CELEBA)


# -- reading a manifest --------------------------------------------------------

COVERAGE_ONLY = {"dataset": "m4raw", "prior_domain": "mri_brain", "reportable": False,
                 "reportable_claims": ["coverage"]}


def test_a_detection_number_from_a_coverage_only_bank_is_refused():
    """**The test D82 asks for.** The number can be computed; it cannot be stamped."""
    require_claim(COVERAGE_ONLY, COVERAGE)
    with pytest.raises(ClaimNotCarried, match="does not carry detection"):
        require_claim(COVERAGE_ONLY, DETECTION)
    assert stamp(COVERAGE_ONLY, COVERAGE) == "reportable"
    assert stamp(COVERAGE_ONLY, DETECTION) == "DEVELOPMENT ONLY"


def test_the_boolean_alone_never_grants_a_claim_the_set_withholds():
    """A manifest edited by hand to ``reportable: true`` with a coverage-only set still
    refuses detection: the set is what is read."""
    forged = dict(COVERAGE_ONLY, reportable=True)
    assert not carries(forged, DETECTION)


def test_a_manifest_from_before_the_field_is_read_by_its_boolean():
    """Nothing already on disk changes what it says until it is re-stamped."""
    assert claims_of({"reportable": True}) == CLAIMS
    assert claims_of({"reportable": False}) == frozenset()
    assert claims_of({"dataset": "fastmri"}) == frozenset()


def test_a_bank_with_no_provenance_carries_nothing():
    """Before D-018 a bank marked nothing. Silence is not a claim."""
    assert claims_of(None) == frozenset()
    assert claims_of({}) == frozenset()
    with pytest.raises(ClaimNotCarried):
        require_claim({}, COVERAGE)


def test_an_unknown_claim_is_a_typo_and_not_a_refusal():
    with pytest.raises(ValueError, match="unknown claim"):
        carries(COVERAGE_ONLY, "coverege")
    with pytest.raises(ValueError, match="unknown claim"):
        require_claim(COVERAGE_ONLY, "auroc")


def test_the_header_line_names_what_is_refused():
    assert describe({"reportable": True}) == "reportable"
    assert "NOT REPORTABLE" in describe({"reportable": False})
    line = describe(COVERAGE_ONLY)
    assert "coverage ONLY" in line and "detection" in line


# -- re-stamping a finished bank -----------------------------------------------


def _restamp():
    path = Path(__file__).resolve().parents[1] / "scripts" / "restamp_bank.py"
    spec = importlib.util.spec_from_file_location("restamp_bank", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bank(tmp_path, name, dataset, family, domain, checkpoint, reportable):
    root = tmp_path / name
    root.mkdir()
    manifest = {
        "run_id": name,
        "provenance": {"dataset": dataset, "modality": DATASETS[dataset].modality,
                       "solver_family": family, "prior_domain": domain,
                       "reportable": reportable},
        "config": {"experiment": {"solver": {"name": "ddnm", "checkpoint": checkpoint}}},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _provenance(bank):
    return json.loads((bank / "manifest.json").read_text(encoding="utf-8"))["provenance"]


def test_restamping_the_m4raw_bank_gives_it_coverage_and_keeps_what_it_said(tmp_path):
    restamp = _restamp()
    bank = _bank(tmp_path, "m4raw_brats", "m4raw", "diffusion", "mri_brain", BRATS, False)
    assert restamp.main([str(bank), "--write"]) == 0
    prov = _provenance(bank)
    assert prov["reportable_claims"] == ["coverage"]
    assert prov["reportable"] is False
    assert prov["anatomy"] == "brain"
    assert prov["restamped"]["previous"] == {"reportable": False}
    # a second pass changes nothing, and does not overwrite the record of the first
    assert restamp.main([str(bank), "--write"]) == 0
    assert _provenance(bank) == prov


def test_restamping_narrows_a_cross_anatomy_bank_that_was_stamped_whole(tmp_path):
    """The LIDC bank is built by code that knew only the boolean, which says ``true``:
    admissible prior, clean substrate. Under D80 it carries coverage only."""
    restamp = _restamp()
    bank = _bank(tmp_path, "ct_lidc", "totalsegmentator", "diffusion", "ct_chest", LIDC, True)
    assert restamp.main([str(bank), "--write"]) == 0
    prov = _provenance(bank)
    assert prov["reportable"] is False and prov["reportable_claims"] == ["coverage"]
    assert prov["restamped"]["previous"] == {"reportable": True}


def test_restamping_shows_and_does_not_write_by_default(tmp_path):
    restamp = _restamp()
    bank = _bank(tmp_path, "m4raw_brats", "m4raw", "diffusion", "mri_brain", BRATS, False)
    before = (bank / "manifest.json").read_text(encoding="utf-8")
    assert restamp.main([str(bank)]) == 0
    assert (bank / "manifest.json").read_text(encoding="utf-8") == before


def test_restamping_refuses_a_manifest_it_cannot_identify(tmp_path):
    """A stamp is a claim about what may be published; guessing one is not an option."""
    restamp = _restamp()
    renamed = _bank(tmp_path, "renamed", "m4raw", "diffusion", "mri_brain",
                    "checkpoints/some-other-prior", False)
    unknown = _bank(tmp_path, "unknown", "m4raw", "diffusion", "mri_brain", BRATS, False)
    manifest = json.loads((unknown / "manifest.json").read_text(encoding="utf-8"))
    manifest["provenance"]["dataset"] = "not_a_dataset"
    (unknown / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    before = (renamed / "manifest.json").read_text(encoding="utf-8")
    assert restamp.main([str(renamed), str(unknown), "--write"]) == 1
    assert (renamed / "manifest.json").read_text(encoding="utf-8") == before


def test_restamping_a_bank_built_before_its_prior_was_registered(tmp_path):
    """The LIDC smoke bank recorded ``unknown``: the checkpoint was registered a week
    later, by D80. The registry's answer today is the one to stamp, and the manifest keeps
    what it said."""
    restamp = _restamp()
    bank = _bank(tmp_path, "smoke", "totalsegmentator", "diffusion", "unknown", LIDC, False)
    assert restamp.main([str(bank), "--write"]) == 0
    prov = _provenance(bank)
    assert prov["prior_domain"] == "ct_chest"
    assert prov["reportable_claims"] == ["coverage"]
    assert prov["restamped"]["previous"] == {"reportable": False, "prior_domain": "unknown"}

