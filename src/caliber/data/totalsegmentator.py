"""TotalSegmentator CT loader -- the CT substrate with annotations (Week 7, D-013).

**Why this dataset and not LoDoPaB.** LoDoPaB carries no region annotations, and the
TotalSegmentator masks cannot be laid over LoDoPaB images: they are different cohorts
(LIDC-IDRI against Basel), so no pairing exists (D-013, P-03). D-013 resolved it as
Option A -- TotalSegmentator is the CT substrate, LoDoPaB at most an unannotated
coverage-curve source. Like LoDoPaB, the images are ground truth that the project
re-projects with its own ``CTProjectionOperator`` (D-005); the sinograms are ours.

**One subject is one patient.** ``meta.csv`` has no patient id, so the image id
(``s0000``) is the clustering unit and the split key. That assumes no patient appears
twice, which the dataset does not guarantee and this loader cannot check. It is the
same as fastMRI's 199 volumes from 96 patients going the other way: if TotalSegmentator
repeats patients, intervals clustered by subject are slightly too narrow (D-027).

**Fixed physical resolution, not a fixed field of view.** Every subject is 1.5 mm
isotropic, but in-plane extents run from 59 to 499 voxels. Slices are centre-cropped or
zero-padded to ``image_shape`` at that native spacing (256 px = 384 mm), never resized,
so a pixel means the same physical size in every item -- the same choice the fastMRI
loader makes with its centre crop.

**A fixed Hounsfield window, not a per-image percentile.** Intensities are clipped to
``hu_window`` and mapped to [0, 1], so air is 0 (and padding is air) and a given tissue
has the same value in every item. A per-image p99 would make the intensity scale a
function of what else is in the slice.

**Annotations are organs, as an overlay.** ``ann_coverage`` comes from the union of
``structures`` -- solid and hollow organs by default; bones, muscles and vessels are
left out because they cover so much of the body that coverage would be near 1
everywhere. TotalSegmentator has no lesion masks (bar kidney cysts), so unlike
fastMRI+ these mark anatomy rather than pathology. Track 1's D20 makes annotations an
overlay on the label, not part of it, so the choice does not move any coverage number.

**Which slices.** ``slices_per_volume`` axial slices spread through the middle half of
the slices that contain an annotated organ -- ranked among those slices, so every chosen
one has an organ in it -- and the body's middle when a subject has none of them (limbs,
some spines).
The rule reads only ground truth, never a reconstruction or a score, so it cannot
select for how well anything is flagged.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .base import Item, SliceDataset, center_crop
from .splits import DEFAULT_SALT, SPLITS, split_for

#: Organs whose union is the annotation. Bones, muscles and vessels are excluded on
#: purpose (see the module docstring); pass ``structures=`` to choose otherwise.
DEFAULT_STRUCTURES: tuple[str, ...] = (
    "brain", "thyroid_gland", "trachea", "esophagus", "heart",
    "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
    "lung_middle_lobe_right", "lung_lower_lobe_right",
    "liver", "spleen", "kidney_left", "kidney_right", "pancreas", "gallbladder",
    "stomach", "duodenum", "small_bowel", "colon",
    "adrenal_gland_left", "adrenal_gland_right", "urinary_bladder", "prostate",
)
HU_WINDOW = (-1000.0, 1000.0)
#: Below this many HU a voxel is air, for the body-centre fallback.
BODY_HU = -500.0
#: Masks must sit on the CT grid; TotalSegmentator's agree to ~2e-4 mm (rounding).
AFFINE_TOLERANCE_MM = 1e-2


def _require_nibabel():
    try:
        import nibabel
    except ImportError as exc:  # pragma: no cover
        raise ImportError("TotalSegmentator loading requires nibabel") from exc
    return nibabel


def to_display(slab: np.ndarray) -> np.ndarray:
    """An RAS axial slice ``(x, y)`` as an image: anterior up, patient's right on the left.

    The radiological convention. Every subject is brought to RAS first, so one flip
    serves all of them; the orientation does not affect any number, only figures.
    """
    return np.ascontiguousarray(slab[::-1, ::-1].T)


def choose_slices(occupancy: np.ndarray, n: int) -> list[int]:
    """``n`` z-indices spread through the middle half of the slices where ``occupancy`` holds.

    ``occupancy`` is a per-z boolean profile. The spread is by **rank among the occupied
    slices**, not by position in the range from first to last: organs are not contiguous
    in z (brain and thyroid in a neck scan, lungs and bladder in a whole-body one), and
    the middle of the range can fall in a gap with no organ at all. Ranking guarantees
    every chosen slice is occupied. Returns ``[]`` when nothing is, so the caller can fall
    back.
    """
    zs = np.flatnonzero(occupancy)
    if zs.size == 0:
        return []
    quantiles = [0.5] if n == 1 else [0.25 + 0.5 * i / (n - 1) for i in range(n)]
    return [int(zs[int(round(q * (zs.size - 1)))]) for q in quantiles]


class TotalSegmentatorDataset(SliceDataset):
    """Axial CT slices with organ annotations, split by subject (D-013, D-022)."""

    name = "totalsegmentator"
    modality = "ct"

    def __init__(
        self,
        root: str | Path,
        split: str = "calib",
        image_shape: tuple[int, int] = (256, 256),
        slices_per_volume: int = 1,
        structures: Sequence[str] = DEFAULT_STRUCTURES,
        hu_window: tuple[float, float] = HU_WINDOW,
        limit_volumes: int | None = None,
        calib_fraction: float = 0.5,
        design_fraction: float = 0.0,
        split_salt: str = DEFAULT_SALT,
        keep_splits: str | Sequence[str] | None = None,
    ):
        self.root = Path(root)
        # As in the fastMRI loader: "auto" decides each subject's side with the split
        # registry, identically in every run and operator (D-022, D-051); an explicit
        # side forces every item onto it, which only smoke configs should do.
        self.split = split
        self.auto_split = split == "auto"
        self.calib_fraction = float(calib_fraction)
        self.design_fraction = float(design_fraction)
        self.split_salt = split_salt
        self.image_shape = tuple(image_shape)
        self.slices_per_volume = int(slices_per_volume)
        self.structures = tuple(structures)
        self.hu_window = (float(hu_window[0]), float(hu_window[1]))
        if self.hu_window[1] <= self.hu_window[0]:
            raise ValueError(f"hu_window must be (low, high) with low < high, got {hu_window}")
        if self.slices_per_volume < 1:
            raise ValueError("slices_per_volume must be at least 1")
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"TotalSegmentator directory not found: {self.root}. See T2.1 Appendix A; "
                "the dataset is on Zenodo and only part of it is extracted (P-10)."
            )
        self.subjects = sorted(p.name for p in self.root.iterdir()
                               if (p / "ct.nii.gz").is_file())
        if not self.subjects:
            raise FileNotFoundError(f"no <subject>/ct.nii.gz under {self.root}")
        # Filter before limiting, for the reason the fastMRI loader gives: the split is a
        # property of the subject, so limiting first would spend the budget on subjects
        # the filter then drops. No file is opened: the subject id is the patient.
        if keep_splits is not None:
            self.subjects = self._filter_by_split(keep_splits)
        if limit_volumes:
            self.subjects = self.subjects[:limit_volumes]
        self._index = [(s, k) for s in self.subjects for k in range(self.slices_per_volume)]
        self._slices: dict[str, tuple[list[int], str]] = {}

    def __len__(self) -> int:
        return len(self._index)

    def side_of(self, subject: str) -> str:
        if not self.auto_split:
            return self.split
        return split_for(subject, calib_fraction=self.calib_fraction,
                         design_fraction=self.design_fraction, salt=self.split_salt)

    def patient_index(self) -> dict[str, str]:
        """``{sample_id: patient_id}``; here the identity, as one subject is one patient."""
        return {s: s for s in self.subjects}

    def _load(self, subject: str):
        """The CT volume (HU, RAS) and the union of the annotated organs on its grid."""
        nib = _require_nibabel()
        ct = nib.as_closest_canonical(nib.load(self.root / subject / "ct.nii.gz"))
        volume = np.asarray(ct.dataobj, dtype=np.float32)
        union = np.zeros(volume.shape, dtype=bool)
        seg_dir = self.root / subject / "segmentations"
        for name in self.structures:
            path = seg_dir / f"{name}.nii.gz"
            if not path.is_file():
                continue
            mask = nib.as_closest_canonical(nib.load(path))
            if mask.shape != ct.shape or not np.allclose(mask.affine, ct.affine,
                                                         atol=AFFINE_TOLERANCE_MM):
                raise ValueError(
                    f"{subject}/{name}: mask is not on the CT grid (shape {mask.shape} "
                    f"against {ct.shape}); a misplaced mask mislabels every region silently"
                )
            union |= np.asarray(mask.dataobj) > 0
        return volume, union

    def _slice_plan(self, subject: str, volume: np.ndarray, union: np.ndarray):
        if subject not in self._slices:
            chosen = choose_slices(union.any(axis=(0, 1)), self.slices_per_volume)
            rule = "organs"
            if not chosen:
                chosen = choose_slices((volume > BODY_HU).any(axis=(0, 1)),
                                       self.slices_per_volume)
                rule = "body"
            if not chosen:  # an all-air volume: nothing to choose by, take the middle
                chosen = choose_slices(np.ones(volume.shape[2], dtype=bool),
                                       self.slices_per_volume)
                rule = "middle"
            self._slices[subject] = (chosen, rule)
        return self._slices[subject]

    def __getitem__(self, index: int) -> Item:
        subject, k = self._index[index]
        volume, union = self._load(subject)
        chosen, rule = self._slice_plan(subject, volume, union)
        z = chosen[k]

        low, high = self.hu_window
        hu = np.clip(to_display(volume[:, :, z]), low, high)
        image = torch.from_numpy((hu - low) / (high - low))
        raw_shape = tuple(image.shape)
        image = center_crop(image, self.image_shape).to(torch.float32)

        # Same transform as the image, then the same crop, so the two grids agree by
        # construction rather than by coincidence (the fastMRI loader's rule).
        mask = torch.from_numpy(to_display(union[:, :, z]))
        mask = center_crop(mask.to(torch.uint8), self.image_shape).bool()
        annotation = mask if bool(mask.any()) else None

        return Item(
            dataset=self.name,
            split=self.side_of(subject),
            sample_id=subject,
            slice_idx=int(z),
            image=image,
            annotation=annotation,
            meta={"source": str(self.root / subject / "ct.nii.gz"), "patient_id": subject,
                  "slice_rule": rule, "raw_shape": raw_shape},
        )

    def _filter_by_split(self, keep_splits: str | Sequence[str]) -> list[str]:
        if not self.auto_split:
            raise ValueError(
                f"keep_splits={keep_splits!r} needs split='auto'. With an explicit "
                f"split={self.split!r} every item is forced onto that side, so filtering "
                "by the computed split would silently keep or drop everything."
            )
        wanted = {keep_splits} if isinstance(keep_splits, str) else set(keep_splits)
        unknown = wanted - set(SPLITS)
        if unknown:
            raise ValueError(f"unknown split(s) {sorted(unknown)}; the sides are {list(SPLITS)}")
        kept = [s for s in self.subjects if self.side_of(s) in wanted]
        if not kept:
            raise FileNotFoundError(
                f"no subject lands on {sorted(wanted)} among {len(self.subjects)} under "
                f"{self.root} (design_fraction={self.design_fraction:g})"
            )
        return kept
