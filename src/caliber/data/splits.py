"""Calibration/test split assignment — patient-level, deterministic, operator-free.

**Provenance note.** This module is imported by ``data/__init__.py``,
``data/fastmri.py`` and ``tests/test_splits.py``, all of which are committed,
but the module itself was never committed — the package could not be imported at
all. This is a Track 1 stand-in written to the contract those committed tests
already pin, so that work could continue. **Track 2 owns the real file**; when it
lands, take theirs, since the tests fix the observable behaviour either way.

The rules it implements are D23/D-022, ruled at the Week 1 sign-off:

* **Patient-level.** A patient owns every slice and often several volumes. Split
  by slice or by volume and the same anatomy lands on both sides, which inflates
  apparent coverage and quietly violates the exchangeability the conformal
  guarantee rests on. D-027 measured the damage this would have done: fastMRI
  ``singlecoil_val`` is 199 volumes from **96 patients**, so a by-volume split
  leaks essentially every patient.
* **Operator-independent.** The same patient must fall on the same side under
  *every* acquisition regime. Otherwise group-conditional calibration compares
  different patient populations across regimes, and a coverage difference
  between R=4 and R=8 is confounded with anatomy rather than attributable to
  acquisition — §13's controlled-comparison assumption, applied to the split.
  There is deliberately **no operator parameter** in this module's signatures:
  the confound is made unrepresentable rather than merely discouraged.
* **Stateless per key.** Assignment hashes the patient key rather than shuffling
  or ranking a list, so adding patients never moves the ones already assigned
  and two banks built weeks apart stay mergeable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "DEFAULT_SALT",
    "assign_splits",
    "split_for",
    "split_summary",
    "verify_no_patient_straddles",
]

DEFAULT_SALT = "caliber-split-v1"
"""Changing the salt redraws every split. Treat it as a schema version: a new
salt invalidates comparability with every bank built under the old one."""

CALIB = "calib"
TEST = "test"


def _unit_interval(key: str, salt: str) -> float:
    """Map a key to [0, 1) via BLAKE2b.

    A cryptographic digest rather than :func:`hash`, because Python's built-in
    string hash is randomised per process — two machines, or two runs, would
    disagree about the same patient and the banks would not merge.
    """
    digest = hashlib.blake2b(f"{salt}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _validate(key: Any, calib_fraction: float) -> str:
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"patient key must be a non-empty string, got {key!r}")
    if not 0.0 <= calib_fraction <= 1.0:
        raise ValueError(f"calib_fraction must be in [0, 1], got {calib_fraction}")
    return key


def split_for(
    key: Any,
    calib_fraction: float = 0.5,
    salt: str = DEFAULT_SALT,
) -> str:
    """Which side of the split a patient belongs to. Pure function of the key.

    Takes no operator, no run id and no dataset: those are exactly the
    dependencies that would break D-022's operator-independence requirement, so
    they are absent from the signature by design.
    """
    key = _validate(key, calib_fraction)
    return CALIB if _unit_interval(key, salt) < calib_fraction else TEST


def assign_splits(
    keys: Iterable[Any],
    calib_fraction: float = 0.5,
    salt: str = DEFAULT_SALT,
) -> dict[str, str]:
    """Assign every key independently. Order of presentation is irrelevant."""
    return {
        key: split_for(key, calib_fraction, salt)
        for key in dict.fromkeys(keys)  # de-duplicate, preserve first appearance
    }


def verify_no_patient_straddles(
    sample_to_patient: Mapping[str, str],
    assignment: Mapping[str, str],
) -> list[str]:
    """Patients appearing on both sides of the split, sorted. Empty is the goal.

    Run this over any assignment before a coverage number is quoted from it.
    A straddling patient is not a data-quality nuisance — it breaks the
    exchangeability the guarantee is stated under, so a coverage figure computed
    across one is not merely optimistic but undefined.
    """
    sides: dict[str, set[str]] = {}
    for sample, patient in sample_to_patient.items():
        side = assignment.get(sample)
        if side is not None:
            sides.setdefault(patient, set()).add(side)
    return sorted(p for p, s in sides.items() if len(s) > 1)


def split_summary(
    sample_to_patient: Mapping[str, str],
    calib_fraction: float = 0.5,
    salt: str = DEFAULT_SALT,
) -> dict[str, Any]:
    """Assign at patient level, then report what the assignment actually did.

    Returns the per-sample assignment alongside the counts needed to check it:
    ``straddling_patients`` must be empty, and ``n_patients`` well below
    ``n_samples`` is the normal case rather than a warning — on fastMRI
    ``singlecoil_val`` it is 96 against 199 (D-027).
    """
    patients = sorted(set(sample_to_patient.values()))
    by_patient = assign_splits(patients, calib_fraction, salt)
    assignment = {s: by_patient[p] for s, p in sample_to_patient.items()}
    return {
        "assignment": assignment,
        "n_samples": len(sample_to_patient),
        "n_patients": len(patients),
        "n_calib_patients": sum(v == CALIB for v in by_patient.values()),
        "n_test_patients": sum(v == TEST for v in by_patient.values()),
        "straddling_patients": verify_no_patient_straddles(sample_to_patient, assignment),
        "salt": salt,
        "calib_fraction": calib_fraction,
    }
