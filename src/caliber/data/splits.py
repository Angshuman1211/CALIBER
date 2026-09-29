"""The calibration/test split — one definition, owned here (D-022).

Blueprint Sec. 19 makes the split a shared interface, not a per-run detail. Track 1's
Week 1 sign-off (B3) fixed three requirements, and each one exists because breaking it
corrupts a result silently rather than loudly:

**Per patient, never per slice or per volume (D-014).** Adjacent slices are
near-duplicates. A patient on both sides of the split makes the calibration set
predictive of the test set, which inflates apparent coverage — in the direction that
flatters our own numbers, which is the worst way for an error to run.

**Operator-independent.** The same patient sits on the same side in *every* acquisition
regime. Otherwise group-conditional calibration compares different patient populations
across regimes, and a coverage difference between R=4 and R=8 is confounded with
anatomy rather than attributable to acquisition. That is the Sec. 13 controlled-
comparison assumption applied to the split; violating it would undermine the headline
claim, not merely inconvenience the Week 10 merge.

**Identical across every run.** Assignment is a pure function of ``(salt, patient_key)``.
Nothing is stored, so nothing can drift between two machines or two runs, and two banks
produced months apart remain mergeable.

A consequence worth having deliberately: because assignment is per patient rather than
by shuffling a list, **adding data does not move anybody**. Extending the substrate from
199 volumes to the full split leaves every existing assignment where it was, so a bank
built today stays comparable with one built after the data grows.

**A third side, for choices made by looking (D-051).** Track 1's D28 (score weights)
and D37 (the criterion threshold) are both *selected* by looking at scores. Selecting on
calibration data biases the threshold; selecting on test data biases the coverage number.
Neither is visible after the fact, so the patients used for those choices are held out as
a separate ``"design"`` side rather than borrowed from either.

It is carved out of the **calibration** range, and that is deliberate:

* **The test side never moves.** For any ``design_fraction``, a patient lands on ``test``
  exactly when it would have with no design side at all. A coverage number computed
  before the design split existed stays valid, and banks built either way stay mergeable.
* **The cost lands where it hurts least.** P-23 measured the binding constraint to be
  test patients, not calibration patients; the headline paired gap is sized on the test
  side.
* ``design_fraction=0.0`` is the default and reproduces the two-way split exactly, so no
  existing bank, pin, or salt is disturbed by this file gaining the capability.

.. warning::

   ``patient_key`` must be a *patient*, not a volume. On fastMRI ``singlecoil_val``
   these are not the same thing: 199 volumes carry only **96 distinct ``patient_id``
   values**, and every one of those patients has more than one volume (D-027). Splitting
   by filename there leaks essentially every patient across the split. Use
   :func:`verify_no_patient_straddles` to check, rather than assuming.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping

#: Changing this re-draws every split, invalidating comparability with existing
#: banks. It is versioned rather than free-form for exactly that reason: a bank
#: records the salt it was built under, so a re-draw is visible instead of silent.
DEFAULT_SALT = "caliber-split-v1"

#: ``"design"`` is empty unless a positive ``design_fraction`` is asked for (D-051).
SPLITS = ("design", "calib", "test")


def split_for(
    patient_key: str,
    *,
    calib_fraction: float = 0.5,
    design_fraction: float = 0.0,
    salt: str = DEFAULT_SALT,
) -> str:
    """Return ``"design"``, ``"calib"`` or ``"test"`` for one patient.

    Deterministic, operator-independent, and stable as the dataset grows. The
    hash is keyed only on the patient, so no acquisition parameter, run id, or
    dataset ordering can influence which side a patient lands on.

    ``design_fraction`` carves a held-out side for choices made by looking at scores
    (D-051). It comes out of the calibration range, so the ``test`` side is unchanged for
    every value of it, and the default ``0.0`` reproduces the two-way split exactly.
    """
    if not patient_key:
        raise ValueError("patient_key must be a non-empty identifier")
    if not 0.0 <= calib_fraction <= 1.0:
        raise ValueError(f"calib_fraction must be in [0, 1], got {calib_fraction}")
    if not 0.0 <= design_fraction <= 1.0:
        raise ValueError(f"design_fraction must be in [0, 1], got {design_fraction}")
    if design_fraction > calib_fraction:
        # Spilling past calib_fraction would start moving patients off the test side,
        # which is the one property this arrangement guarantees. Refuse and say which
        # knob to turn, rather than silently shrinking test.
        raise ValueError(
            f"design_fraction ({design_fraction}) must not exceed calib_fraction "
            f"({calib_fraction}): the design side is carved out of the calibration "
            "range so the test side never moves. Raise calib_fraction to make room."
        )
    digest = hashlib.blake2b(f"{salt}|{patient_key}".encode(), digest_size=8).digest()
    unit = int.from_bytes(digest, "big") / 2**64
    if unit < design_fraction:
        return "design"
    return "calib" if unit < calib_fraction else "test"


def assign_splits(
    patient_keys: Iterable[str],
    *,
    calib_fraction: float = 0.5,
    design_fraction: float = 0.0,
    salt: str = DEFAULT_SALT,
) -> dict[str, str]:
    """Split assignment for many patients, as ``{patient_key: split}``."""
    return {
        key: split_for(
            key,
            calib_fraction=calib_fraction,
            design_fraction=design_fraction,
            salt=salt,
        )
        for key in dict.fromkeys(patient_keys)
    }


def verify_no_patient_straddles(
    sample_to_patient: Mapping[str, str],
    sample_to_split: Mapping[str, str],
) -> list[str]:
    """Patients appearing on both sides of the split. Empty list means clean.

    The check that would have caught D-027 before it reached a coverage number:
    pass ``{sample_id: patient_id}`` alongside the split actually used, and it
    reports every patient whose samples were not kept together.
    """
    sides: dict[str, set[str]] = defaultdict(set)
    for sample_id, patient in sample_to_patient.items():
        if sample_id in sample_to_split:
            sides[patient].add(sample_to_split[sample_id])
    return sorted(patient for patient, s in sides.items() if len(s) > 1)


def split_summary(
    sample_to_patient: Mapping[str, str],
    *,
    calib_fraction: float = 0.5,
    design_fraction: float = 0.0,
    salt: str = DEFAULT_SALT,
) -> dict[str, object]:
    """Counts for a proposed split, for logging before a bank is produced."""
    assignment = assign_splits(
        sample_to_patient.values(),
        calib_fraction=calib_fraction,
        design_fraction=design_fraction,
        salt=salt,
    )
    per_sample = {s: assignment[p] for s, p in sample_to_patient.items()}
    return {
        "salt": salt,
        "calib_fraction": calib_fraction,
        "design_fraction": design_fraction,
        "n_samples": len(sample_to_patient),
        "n_patients": len(assignment),
        "patients_design": sum(1 for v in assignment.values() if v == "design"),
        "patients_calib": sum(1 for v in assignment.values() if v == "calib"),
        "patients_test": sum(1 for v in assignment.values() if v == "test"),
        "samples_design": sum(1 for v in per_sample.values() if v == "design"),
        "samples_calib": sum(1 for v in per_sample.values() if v == "calib"),
        "samples_test": sum(1 for v in per_sample.values() if v == "test"),
        "straddling_patients": verify_no_patient_straddles(sample_to_patient, per_sample),
    }
