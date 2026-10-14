"""The TotalSegmentator CT loader (D-013): split by subject, organ slices, masks on the grid.

Built on tiny synthetic NIfTI volumes written to a temporary directory, so the tests run
anywhere -- the real dataset is 2.6 GB and lives only on the data machine. Each test pins
one property that, if it broke, would produce a valid-looking bank with wrong numbers in it:
a mask off by a flip, a slice with no organ, a subject on two sides, a window that moved.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

nib = pytest.importorskip("nibabel")

from caliber.data import build_dataset  # noqa: E402
from caliber.data.splits import split_for  # noqa: E402
from caliber.data.totalsegmentator import (  # noqa: E402
    TotalSegmentatorDataset,
    choose_slices,
    to_display,
)

AFFINE = np.diag([1.5, 1.5, 1.5, 1.0])  # RAS, 1.5 mm, as every real subject is


def _write(path, array, affine=AFFINE, dtype=np.int16):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array.astype(dtype), affine), str(path))


def _subject(root, name, shape=(40, 30, 20), organ_z=(5, 6, 7, 14, 15), hu_body=40):
    """Air everywhere, a body block of soft tissue, and a 'liver' in the given z-slices."""
    ct = np.full(shape, -1000, dtype=np.int16)
    ct[5:35, 5:25, :] = hu_body
    liver = np.zeros(shape, dtype=np.uint8)
    for z in organ_z:
        liver[10:20, 8:14, z] = 1
    _write(root / name / "ct.nii.gz", ct)
    _write(root / name / "segmentations" / "liver.nii.gz", liver, dtype=np.uint8)
    return ct, liver


@pytest.fixture
def root(tmp_path):
    for i in range(12):
        _subject(tmp_path, f"s{i:04d}")
    return tmp_path


def test_choose_slices_never_lands_in_a_gap_between_organs():
    """Organs are not contiguous in z; the middle of the first-to-last range can be empty."""
    occupancy = np.zeros(40, dtype=bool)
    occupancy[[2, 3, 4, 35, 36, 37]] = True      # e.g. bladder low, lungs high, gap between
    for n in (1, 2, 3):
        chosen = choose_slices(occupancy, n)
        assert len(chosen) == n
        assert all(occupancy[z] for z in chosen), chosen


def test_choose_slices_returns_nothing_for_an_empty_profile():
    assert choose_slices(np.zeros(10, dtype=bool), 2) == []


def test_display_puts_anterior_up_and_patient_right_on_the_left():
    """RAS: x grows to the patient's right, y to anterior. Radiological view flips both."""
    slab = np.zeros((4, 3))
    slab[3, 2] = 1.0              # the most rightward, most anterior voxel
    img = to_display(slab)
    assert img.shape == (3, 4)    # rows are y, columns are x
    assert img[0, 0] == 1.0       # top (anterior), left (patient's right)


def test_items_take_a_slice_with_an_organ_and_the_mask_lands_on_it(root):
    ds = TotalSegmentatorDataset(root, image_shape=(32, 48))
    item = ds[0]
    assert item.slice_idx in (5, 6, 7, 14, 15)
    assert item.meta["slice_rule"] == "organs"
    assert item.annotation is not None and item.annotation.shape == (32, 48)
    # the organ is soft tissue (40 HU) inside the body: under the mask the image must be
    # exactly that value, which fails if image and mask take different flips or crops
    level = (40 + 1000) / 2000
    assert torch.allclose(item.image[item.annotation], torch.full((1,), level), atol=1e-6)


def test_the_hounsfield_window_maps_air_to_zero_and_is_fixed(root):
    ds = TotalSegmentatorDataset(root, image_shape=(48, 48), hu_window=(-1000, 1000))
    img = ds[0].image
    assert float(img.min()) == 0.0                       # air and padding
    assert float(img.max()) == pytest.approx(0.52)       # 40 HU, the same in every item


def test_auto_split_is_by_subject_and_matches_the_registry(root):
    ds = TotalSegmentatorDataset(root, split="auto", calib_fraction=0.5, design_fraction=0.2,
                                 slices_per_volume=2)
    for i in range(len(ds)):
        item = ds[i]
        assert item.split == split_for(item.sample_id, calib_fraction=0.5, design_fraction=0.2)
    # both slices of a subject sit on the same side: a subject cannot straddle
    sides = {}
    for i in range(len(ds)):
        item = ds[i]
        sides.setdefault(item.sample_id, set()).add(item.split)
    assert all(len(s) == 1 for s in sides.values())


def test_keep_splits_keeps_assignments_and_needs_auto(root):
    full = TotalSegmentatorDataset(root, split="auto", design_fraction=0.2)
    kept = TotalSegmentatorDataset(root, split="auto", design_fraction=0.2, keep_splits="test")
    assert kept.subjects == [s for s in full.subjects if full.side_of(s) == "test"]
    with pytest.raises(ValueError, match="split='auto'"):
        TotalSegmentatorDataset(root, split="calib", keep_splits="test")


def test_a_mask_off_the_ct_grid_is_refused(tmp_path):
    _subject(tmp_path, "s0000")
    shifted = AFFINE.copy()
    shifted[0, 3] = 3.0                                  # two voxels off in x
    _write(tmp_path / "s0000" / "segmentations" / "heart.nii.gz",
           np.ones((40, 30, 20)), affine=shifted, dtype=np.uint8)
    ds = TotalSegmentatorDataset(tmp_path, structures=("liver", "heart"))
    with pytest.raises(ValueError, match="not on the CT grid"):
        ds[0]


def test_a_subject_without_organs_falls_back_to_the_body(tmp_path):
    _subject(tmp_path, "s0000", organ_z=())
    item = TotalSegmentatorDataset(tmp_path)[0]
    assert item.meta["slice_rule"] == "body"
    assert item.annotation is None


def test_the_patient_is_the_subject_and_the_registry_builds_it(root):
    ds = build_dataset({"name": "totalsegmentator", "root": str(root)})
    assert ds.modality == "ct"
    assert ds.patient_index() == {s: s for s in ds.subjects}
