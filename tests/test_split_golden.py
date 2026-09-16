"""The split *mapping* is pinned, not just its properties (P-22).

`test_splits.py` checks what a correct split must satisfy — deterministic,
operator-independent, stable as patients are added, balanced. It never checks
*which* side any given patient lands on. That gap is not theoretical: while
`splits.py` was missing from `main` (P-20), Track 1 wrote a stand-in to the same
contract. It passed all 21 split tests and the full suite, used the same salt —
and put **48 of the 96 real fastMRI patients on the opposite side** from the
version that built the Week 3 bank. Two banks built from the two versions would
have been silently non-comparable, which D-022 exists to forbid.

These tests make the mapping itself the contract. Any change to the hashing —
a different digest, a different key format, a different threshold rule — fails
here, even when every property test still passes. Changing the split on purpose
is still possible: it means bumping `DEFAULT_SALT` (which D-022 already requires
be recorded) and regenerating these values in the same commit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from caliber.data.splits import DEFAULT_SALT, split_for

KEYS = [f"patient-{i:03d}" for i in range(40)]

# C = calib, T = test, one character per key in KEYS order. Generated from the
# splits.py that produced runs/w03_mri_fixed_operator-fc534c8f4d898e1b.
PINNED = {
    0.5: "TCTCTCTCCTTTCTCTCTTCTTTTTCTCCCTTCTCTCTCC",
    0.3: "TCTCTCTTCTTTCTCTTTTTTTTTTCTCCCTTTTTTCTTC",
}

# SHA-256 of "patient_id:side" lines, sorted, over the real fastMRI singlecoil_val
# patients at the default fraction. A digest, so no patient identifier is committed —
# the NYU agreement forbids redistributing the data, and this reveals none of it.
REAL_ASSIGNMENT_DIGEST = "2a80bbcd46cdc4b5ac1a1c02b7e1e1b0012c7def8e39d6a9d72b7c0faab03d5c"
REAL_N_PATIENTS = 96
REAL_N_CALIB = 52

FASTMRI_ROOT = Path(__file__).resolve().parents[1] / "data/raw/fastmri/singlecoil_val"


def _encode(fraction: float) -> str:
    return "".join(
        "C" if split_for(k, calib_fraction=fraction) == "calib" else "T" for k in KEYS
    )


def test_salt_is_the_one_the_pinned_values_were_generated_under():
    """Bumping the salt is the sanctioned way to redraw — and it must fail loudly here."""
    assert DEFAULT_SALT == "caliber-split-v1"


@pytest.mark.parametrize("fraction", sorted(PINNED))
def test_assignment_of_fixed_keys_is_pinned(fraction):
    got = _encode(fraction)
    differing = sum(a != b for a, b in zip(got, PINNED[fraction], strict=True))
    assert got == PINNED[fraction], (
        f"split mapping changed for {differing}/{len(KEYS)} fixed keys at "
        f"calib_fraction={fraction}. If this is deliberate, bump DEFAULT_SALT, "
        "regenerate PINNED, and record it (D-022). If not, a bank built now is no "
        "longer comparable with the existing ones."
    )


def test_pinned_values_are_not_degenerate():
    """A pinned string of all one side would pass for a broken split."""
    for fraction, pinned in PINNED.items():
        assert "C" in pinned and "T" in pinned, fraction


@pytest.mark.skipif(not FASTMRI_ROOT.is_dir(), reason="fastMRI not present on this machine")
def test_real_fastmri_assignment_matches_the_week3_bank():
    """The split the Week 3 bank was actually built on, checked against the real data.

    Skips cleanly where the data is absent, so the repository stays testable on a
    machine without the NYU agreement.
    """
    h5py = pytest.importorskip("h5py")
    patients = set()
    for f in FASTMRI_ROOT.rglob("*.h5"):
        with h5py.File(f, "r") as fh:
            patients.add(str(fh.attrs.get("patient_id", "")) or f.stem)

    assert len(patients) == REAL_N_PATIENTS
    assert sum(split_for(p) == "calib" for p in patients) == REAL_N_CALIB
    lines = "\n".join(f"{p}:{split_for(p)}" for p in sorted(patients))
    assert hashlib.sha256(lines.encode()).hexdigest() == REAL_ASSIGNMENT_DIGEST
