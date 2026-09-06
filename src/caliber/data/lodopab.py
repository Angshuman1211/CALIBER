"""LoDoPaB-CT [30] loader — the cross-domain substrate (Week 7, T2.1/T2.5).

LoDoPaB stores ground-truth reconstructions and low-dose projections in HDF5
shards. This project uses the *ground truth* images and re-projects them with
its own registered ``CTProjectionOperator``, rather than consuming LoDoPaB's
supplied sinograms.

That is a deliberate choice and worth defending in the write-up: the whole
study depends on knowing the operator exactly and varying it in a controlled
way (Sec. 13, Sec. 16). Using the shipped sinograms would fix the acquisition
at LoDoPaB's geometry and make the operator grid impossible. The cost is that
the CT forward model is ours, not the scanner's, which belongs in Assumptions
and Honest Scope.

Anatomical annotations for region-level evaluation come from TotalSegmentator
[31] or AbdomenCT-1K [32]; see ``load_segmentation_mask``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .base import Item, SliceDataset, center_crop, normalise


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("LoDoPaB loading requires h5py") from exc
    return h5py


class LoDoPaBDataset(SliceDataset):
    """Ground-truth CT slices, re-projected by the project's own operator."""

    name = "lodopab"
    modality = "ct"

    def __init__(
        self,
        root: str | Path,
        split: str = "calib",
        image_shape: tuple[int, int] = (256, 256),
        normalisation: str = "p99",
        limit_files: int | None = None,
        slices_per_file: int = 4,
        segmentation_root: str | Path | None = None,
        pattern: str | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.image_shape = tuple(image_shape)
        self.normalisation = normalisation
        self.slices_per_file = int(slices_per_file)
        self.segmentation_root = Path(segmentation_root) if segmentation_root else None
        # LoDoPaB shards embed the split in the filename
        # (``ground_truth_validation_000.hdf5``), so the default pattern is
        # split-aware. Extracting train, validation and test into one directory
        # would otherwise silently mix them.
        self.pattern = pattern or f"ground_truth_{split}_*.hdf5"
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"LoDoPaB directory not found: {self.root}. See T2.1 Appendix A - "
                "the dataset is openly available on Zenodo and can be fetched before Week 7."
            )
        self.files = sorted(self.root.glob(self.pattern))
        if limit_files:
            self.files = self.files[:limit_files]
        if not self.files:
            raise FileNotFoundError(
                f"no shards matching {self.pattern!r} under {self.root}"
            )
        self._index = [(f, s) for f in self.files for s in range(self.slices_per_file)]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> Item:
        h5py = _require_h5py()
        path, offset = self._index[index]
        with h5py.File(path, "r") as fh:
            data = fh["data"]
            slice_idx = min(offset * max(1, data.shape[0] // self.slices_per_file),
                            data.shape[0] - 1)
            image = torch.as_tensor(data[slice_idx][()]).to(torch.float32)

        image = normalise(center_crop(image, self.image_shape), self.normalisation)
        annotation = self.load_segmentation_mask(path.stem, slice_idx)

        return Item(
            dataset=self.name,
            split=self.split,
            sample_id=f"{path.stem}_{slice_idx:04d}",
            slice_idx=int(slice_idx),
            image=image,
            annotation=annotation,
            meta={"source": str(path)},
        )

    def load_segmentation_mask(self, stem: str, slice_idx: int):
        """Region-level anatomical ground truth for CT flag evaluation.

        Week 7 deliverable. TotalSegmentator / AbdomenCT-1K labels are supplied
        per volume and must be resampled onto the same centre-cropped grid as
        the image before use; an off-by-one here silently mislabels every CT
        region, so it needs its own check when wired.
        """
        if self.segmentation_root is None:
            return None
        raise NotImplementedError(
            "CT anatomical annotations not yet wired (Week 7, T2.1). "
            "Leave segmentation_root unset until then."
        )
