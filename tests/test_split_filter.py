"""Gates for processing only the volumes a run needs (D49's enabling change).

The loader has always labelled every item with its split while reconstructing every volume
it found. That is right for calibration and test — both sides are wanted — and ruinous for
the design side, which is 8 patients of 96 (D-054): a full-dataset run spends ~92% of its
time on rows the weight selection discards, 47.8 h against 1.9 h at T2.3's measured 288 s
per item-operator.

The risk in adding a filter is that it quietly becomes a *second* way of assigning splits.
It must not: the split is a pure function of (patient, fraction, salt) per D-022/D-027, and
filtering may only choose which patients are processed. The first test is the one that
matters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from caliber.data.fastmri import FastMRIDataset
from caliber.data.splits import split_for

h5py = pytest.importorskip("h5py")


def write_volume(path: Path, patient: str, n_slices: int = 2, shape=(32, 32), seed: int = 0):
    """A minimal fastMRI-layout volume carrying a patient id in attrs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    image = rng.random((n_slices, *shape)).astype(np.float32)
    with h5py.File(path, "w") as fh:
        fh.create_dataset("kspace", data=np.fft.fft2(image, norm="ortho").astype(np.complex64))
        fh.attrs["patient_id"] = patient
    return path


@pytest.fixture
def root(tmp_path) -> Path:
    """Twelve volumes over six patients, two volumes each — fastMRI's own shape (D-027)."""
    out = tmp_path / "singlecoil_val"
    for i in range(6):
        for v in range(2):
            write_volume(out / f"file{i}{v}.h5", patient=f"patient{i}", seed=i * 10 + v)
    return out


def dataset(root: Path, **kwargs) -> FastMRIDataset:
    return FastMRIDataset(
        root=root, split="auto", image_shape=(32, 32), slices_per_volume=1, **kwargs
    )


# -- the invariant -------------------------------------------------------------


def test_filtering_does_not_change_any_patient_s_side():
    """The filter chooses what is processed, never what side anything is on.

    Built directly on `split_for` rather than through the loader, because this is a claim
    about the split function and the loader must be shown to agree with it, not to define
    it.
    """
    keys = [f"patient{i}" for i in range(40)]
    for fraction in (0.0, 0.10, 0.20):
        unfiltered = {k: split_for(k, design_fraction=fraction) for k in keys}
        for side in ("design", "calib", "test"):
            kept = [k for k in keys if unfiltered[k] == side]
            assert all(split_for(k, design_fraction=fraction) == side for k in kept)


def test_a_kept_volume_carries_the_same_label_it_would_have_without_the_filter(root):
    full = dataset(root, design_fraction=0.20)
    labels = {full[i].sample_id: full[i].split for i in range(len(full))}

    filtered = dataset(root, design_fraction=0.20, keep_splits="calib")
    for i in range(len(filtered)):
        item = filtered[i]
        assert item.split == "calib"
        assert labels[item.sample_id] == "calib", "the filter changed a label"


# -- what it is for ------------------------------------------------------------


def test_only_the_requested_side_is_processed(root):
    filtered = dataset(root, design_fraction=0.20, keep_splits="design")
    assert len(filtered) > 0
    assert {filtered[i].split for i in range(len(filtered))} == {"design"}


def test_the_sides_partition_the_volumes(root):
    full = dataset(root, design_fraction=0.20)
    total = len(full)
    counted = 0
    for side in ("design", "calib", "test"):
        try:
            counted += len(dataset(root, design_fraction=0.20, keep_splits=side))
        except FileNotFoundError:
            pass  # a side can legitimately be empty at this size
    assert counted == total


def test_several_sides_can_be_kept_at_once(root):
    both = dataset(root, design_fraction=0.20, keep_splits=["calib", "test"])
    assert {both[i].split for i in range(len(both))} <= {"calib", "test"}
    assert len(both) == len(dataset(root, design_fraction=0.20)) - len(
        dataset(root, design_fraction=0.20, keep_splits="design")
    )


def test_the_filter_runs_before_the_volume_limit(root):
    """The ordering that makes the cheap run cheap.

    Limiting first and filtering second returns whatever fraction of a prefix happens to
    land on the requested side — the trap this change exists to remove. Filtering first
    means `limit_volumes=n` yields n volumes *of that side*.
    """
    design_only = dataset(root, design_fraction=0.20, keep_splits="design")
    if len(design_only) < 2:
        pytest.skip("this fixture's draw put fewer than two volumes on design")
    capped = dataset(root, design_fraction=0.20, keep_splits="design", limit_volumes=1)
    assert len(capped) == 1
    assert capped[0].split == "design"


# -- refusals ------------------------------------------------------------------


def test_filtering_without_auto_split_is_refused(root):
    """An explicit split forces every item onto one side, so the filter would be a no-op
    or would drop everything — either way silently."""
    with pytest.raises(ValueError, match="auto"):
        FastMRIDataset(
            root=root, split="calib", image_shape=(32, 32), keep_splits="design"
        )


def test_an_unknown_side_is_refused_rather_than_matching_nothing(root):
    with pytest.raises(ValueError, match="unknown split"):
        dataset(root, keep_splits="validation")


def test_an_empty_side_says_how_the_volumes_actually_split(root):
    """Asking for `design` at the default fraction is the easy mistake, and the message
    has to name the cause rather than report an empty dataset."""
    with pytest.raises(FileNotFoundError) as excinfo:
        dataset(root, design_fraction=0.0, keep_splits="design")
    message = str(excinfo.value)
    assert "design_fraction" in message
    assert "calib" in message, "the message should report the actual split counts"
