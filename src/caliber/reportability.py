"""Which claims a bank can carry (D-018, D80, D82).

D-018 made reportability a property of the bank: it travels in the manifest, because a
bank is a table of floats and the caveat otherwise survives only in whoever remembers
which config was run. It was one boolean. Two rulings since have shown that one boolean
is the wrong shape:

* **D80 (the prior).** A medical prior of the right modality and the wrong anatomy does
  not block the bank. It gates which *claims* the bank carries: coverage under operator
  shift survives, because both arms are scored against one label on the same
  reconstructions; detection does not, because anatomy mismatch is what drives
  ``cos theta`` to zero (D67).
* **D82 (the substrate).** M4Raw has no pathology annotations, which is the stated harm
  in D-015. Coverage consumes no annotations, so M4Raw carries coverage and not
  detection.

One principle, two instances: *a defect gates the claims it reaches, not the bank*. A
boolean cannot say that. Flipping M4Raw to ``reportable = True`` would stamp a detection
number from it as reportable, which is P-18 over again -- the gate unable to see the
thing that mattered. So the manifest carries the set of claims, and a number is stamped
by asking for the claim it is.

``reportable`` stays in the manifest and keeps its meaning: the *unqualified* claim,
true only when the bank carries every claim. Banks written before this field existed
are read by that boolean -- all claims or none -- so nothing already on disk changes
what it says until it is re-stamped (``scripts/restamp_bank.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .solvers.denoiser import prior_anatomy, prior_is_admissible

#: The claims a bank can be asked to carry. Coverage: a calibrated threshold's coverage,
#: and how it moves under operator shift. Detection: anything that reads the score as a
#: detector of the label -- AUROC, FDR and power against annotations, selected weights.
COVERAGE = "coverage"
DETECTION = "detection"
CLAIMS: frozenset[str] = frozenset({COVERAGE, DETECTION})

#: A substrate whose anatomy nobody declared. It matches no prior, so an undeclared
#: substrate can carry coverage and never detection: the gate fails closed.
UNDECLARED_ANATOMY = "undeclared"


class ClaimNotCarried(ValueError):
    """A number was asked to be stamped with a claim its bank does not carry."""


def _check(claim: str) -> None:
    if claim not in CLAIMS:
        raise ValueError(f"unknown claim {claim!r}; the claims are {sorted(CLAIMS)}")


def substrate_claims(dataset: Any) -> frozenset[str]:
    """What the substrate carries: its own ``reportable_claims`` when it declares them,
    otherwise everything or nothing by its ``reportable`` flag (D-018's two cases)."""
    declared = getattr(dataset, "reportable_claims", None)
    if declared is not None:
        unknown = frozenset(declared) - CLAIMS
        if unknown:
            raise ValueError(
                f"{getattr(dataset, 'name', dataset)!r} declares unknown claims {sorted(unknown)}"
            )
        return frozenset(declared)
    return CLAIMS if bool(getattr(dataset, "reportable", True)) else frozenset()


def prior_claims(checkpoint: str | Path | None, modality: str, anatomy: str) -> frozenset[str]:
    """What the prior lets a bank on this modality and anatomy carry (D80).

    Not admissible for the modality: nothing (P-18, and an unknown checkpoint lands
    here). Admissible and of the substrate's anatomy: everything. Admissible and of
    another anatomy: coverage only.
    """
    if not prior_is_admissible(checkpoint, modality):
        return frozenset()
    if anatomy != UNDECLARED_ANATOMY and prior_anatomy(checkpoint) == anatomy:
        return CLAIMS
    return frozenset({COVERAGE})


def claims_for(dataset: Any, solver_family: str, checkpoint: str | Path | None) -> frozenset[str]:
    """The claims a bank carries: what the substrate, the sampler and the prior each
    allow, intersected. ``dataset`` may be the class -- only class attributes are read --
    which is what lets a finished bank be re-stamped from its manifest alone.

    A reference sampler is not a generative prior, so it carries nothing (D-018).
    """
    if solver_family == "reference":
        return frozenset()
    anatomy = getattr(dataset, "anatomy", UNDECLARED_ANATOMY)
    return substrate_claims(dataset) & prior_claims(checkpoint, dataset.modality, anatomy)


def bank_claims(dataset: Any, solver: Any) -> frozenset[str]:
    """:func:`claims_for`, for a live dataset and solver."""
    return claims_for(dataset, solver.family, solver.info().checkpoint)


def claims_of(provenance: Mapping[str, Any] | None) -> frozenset[str]:
    """The claims a manifest's provenance records.

    A manifest without the field predates D82 and is read by its boolean. No provenance
    at all is a bank from before D-018, which marked nothing: it carries no claim.
    """
    if not provenance:
        return frozenset()
    if "reportable_claims" in provenance:
        return frozenset(provenance["reportable_claims"]) & CLAIMS
    return CLAIMS if provenance.get("reportable", False) else frozenset()


def carries(provenance: Mapping[str, Any] | None, claim: str) -> bool:
    _check(claim)
    return claim in claims_of(provenance)


def require_claim(provenance: Mapping[str, Any] | None, claim: str) -> None:
    """Refuse to stamp a ``claim`` number from a bank that does not carry it.

    This is the refusal D82 asks for "by construction rather than by memory": the code
    that writes a number as reportable calls this with the claim the number is, and a
    detection number from a coverage-only bank stops here.
    """
    _check(claim)
    carried = claims_of(provenance)
    if claim not in carried:
        prov = provenance or {}
        raise ClaimNotCarried(
            f"this bank does not carry {claim} claims (dataset={prov.get('dataset')}, "
            f"prior_domain={prov.get('prior_domain')}; it carries "
            f"{sorted(carried) or 'none'}). The number can be computed; it cannot be "
            "reported (D80, D82)."
        )


def stamp(provenance: Mapping[str, Any] | None, claim: str) -> str:
    """The words that go beside a ``claim`` number from this bank."""
    return "reportable" if carries(provenance, claim) else "DEVELOPMENT ONLY"


def describe(provenance: Mapping[str, Any] | None) -> str:
    """One line for a script's header: what this bank can and cannot be quoted for."""
    carried = claims_of(provenance)
    if carried == CLAIMS:
        return "reportable"
    if not carried:
        return "NOT REPORTABLE - development only, no number below may appear in the write-up"
    refused = ", ".join(sorted(CLAIMS - carried))
    return (
        f"reportable for {', '.join(sorted(carried))} ONLY - {refused} numbers below "
        "are development values (D80, D82)"
    )
