"""fastMRI [28] + fastMRI+ [29] loader — the primary empirical substrate (T2.1).

Why raw measurements matter here: the acquisition operator is the variable
under study, so it must be *chosen*, not inherited. Fully-sampled k-space lets
us synthesise a whole grid of operators retrospectively, which is also what
makes the operator known at both calibration and deployment (Sec. 13).

Pipeline, single-coil, per slice:

    k-space (H, W complex) -> ifft2 -> centre crop -> magnitude -> normalise

Multi-coil volumes are reduced by root-sum-of-squares. That is a deliberate
simplification consistent with the Sec. 17 compute profile; it means the
operator studied is single-coil Cartesian undersampling, which should be stated
in the Experimental Setup rather than left implicit.

Data access is credentialed and has lead time — see docs/data_access.md. This
module reads whatever is present under ``data/raw/fastmri`` and raises a clear
error otherwise, so the pipeline can be developed against phantoms meanwhile.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor

from .base import Item, SliceDataset, center_crop, normalise


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError("fastMRI loading requires h5py") from exc
    return h5py


def find_volumes(root: str | Path) -> list[Path]:
    """All ``.h5`` volumes under ``root``, at any depth, in a stable order.

    The search is recursive because archives rarely unpack flat: fastMRI ships
    ``singlecoil_val.tar.gz`` that expands to a ``singlecoil_val/`` directory,
    Kaggle re-wraps that in another level, and M4Raw's zip nests under
    ``multicoil_val/``. Pointing a config at the directory the user actually
    downloaded, rather than at whichever inner directory happens to hold the
    files, removes a class of "no .h5 volumes found" confusion that costs an
    afternoon each time it happens.
    """
    return sorted(Path(root).rglob("*.h5"))


def load_annotations(csv_path: str | Path) -> dict[str, list[dict]]:
    """Parse fastMRI+ pathology annotations into ``{file_stem: [box, ...]}``.

    fastMRI+ ships bounding boxes per slice with columns
    ``file, slice, study_level, x, y, width, height, label``. Boxes are kept in
    raw pixel coordinates; conversion to the working resolution happens at load
    time so that a change of resolution does not require re-parsing.
    """
    by_file: dict[str, list[dict]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row.get("x"):
                continue  # study-level rows carry no box
            try:
                by_file[row["file"]].append(
                    {
                        "slice": int(row["slice"]),
                        "x": int(float(row["x"])),
                        "y": int(float(row["y"])),
                        "width": int(float(row["width"])),
                        "height": int(float(row["height"])),
                        # Several fastMRI+ labels carry trailing whitespace
                        # ("Joint Effusion "), which silently splits a label
                        # into two categories in any per-label breakdown.
                        "label": row.get("label", "").strip(),
                    }
                )
            except (ValueError, KeyError):
                continue
    return dict(by_file)


def boxes_to_mask(
    boxes: list[dict],
    raw_shape: tuple[int, int],
    out_shape: tuple[int, int],
) -> Tensor:
    """Rasterise bounding boxes, then centre-crop to the working resolution."""
    mask = torch.zeros(*raw_shape, dtype=torch.bool)
    for box in boxes:
        y0 = max(0, box["y"])
        x0 = max(0, box["x"])
        y1 = min(raw_shape[0], y0 + box["height"])
        x1 = min(raw_shape[1], x0 + box["width"])
        if y1 > y0 and x1 > x0:
            mask[y0:y1, x0:x1] = True
    return center_crop(mask.float(), out_shape) > 0.5


class FastMRIDataset(SliceDataset):
    """Single-slice fastMRI reader producing fully-sampled reference images."""

    name = "fastmri"
    modality = "mri"

    def __init__(
        self,
        root: str | Path,
        split: str = "calib",
        image_shape: tuple[int, int] = (256, 256),
        slices_per_volume: int = 1,
        annotations_csv: str | Path | None = None,
        normalisation: str = "p99",
        limit_volumes: int | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.image_shape = tuple(image_shape)
        self.slices_per_volume = int(slices_per_volume)
        self.normalisation = normalisation
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"fastMRI directory not found: {self.root}. See docs/data_access.md - "
                "access is credentialed and must be requested before Week 3."
            )
        self.files = find_volumes(self.root)
        if limit_volumes:
            self.files = self.files[:limit_volumes]
        if not self.files:
            raise FileNotFoundError(f"no .h5 volumes under {self.root} (searched recursively)")
        self.annotations = load_annotations(annotations_csv) if annotations_csv else {}
        self._index = [
            (f, s) for f in self.files for s in range(self.slices_per_volume)
        ]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> Item:
        h5py = _require_h5py()
        path, slice_offset = self._index[index]
        with h5py.File(path, "r") as fh:
            if "kspace" not in fh:
                # Some redistributions ship only ``reconstruction_*`` images.
                # Those cannot support this project at all: the acquisition
                # operator is the variable under study, so it has to be applied
                # to fully-sampled k-space, not inferred from a finished image.
                raise KeyError(
                    f"{path.name} has no 'kspace' dataset (found: {sorted(fh.keys())}). "
                    "Raw k-space is required — see docs/data_access.md §1."
                )
            kspace = fh["kspace"]
            n_slices = kspace.shape[0]
            # Central slices carry the most anatomy; single-slice 2-D is the
            # compute profile in Sec. 17.
            mid = n_slices // 2
            slice_idx = min(max(0, mid + slice_offset - self.slices_per_volume // 2), n_slices - 1)
            raw = torch.as_tensor(kspace[slice_idx])

        if raw.dim() == 3:  # multi-coil (coils, H, W) -> root sum of squares
            coil_images = torch.fft.ifft2(raw, norm="ortho")
            image = coil_images.abs().pow(2).sum(dim=0).sqrt()
        else:
            image = torch.fft.ifft2(raw, norm="ortho").abs()

        raw_shape = tuple(image.shape[-2:])
        image = normalise(center_crop(image, self.image_shape), self.normalisation)

        annotation = None
        boxes = [
            b for b in self.annotations.get(path.stem, []) if b["slice"] == slice_idx
        ]
        if boxes:
            annotation = boxes_to_mask(boxes, raw_shape, self.image_shape)

        return Item(
            dataset=self.name,
            split=self.split,
            sample_id=path.stem,
            slice_idx=int(slice_idx),
            image=image.to(torch.float32),
            annotation=annotation,
            meta={"source": str(path), "raw_shape": raw_shape},
        )


class M4RawDataset(FastMRIDataset):
    """Open low-field brain k-space — the development stopgap of D-015.

    M4Raw ships ISMRMRD-converted ``.h5`` with the same ``kspace`` layout
    fastMRI uses, so the reader is inherited unchanged; that compatibility is
    the entire reason it was chosen. Retrospective undersampling needs only
    fully-sampled k-space, which M4Raw has.

    It is a *separate class* rather than a ``root`` pointed at different files
    for two reasons, both of which matter more than the code they cost:

    * ``dataset`` is a score-bank column, so M4Raw rows are distinguishable
      from fastMRI rows forever, in any bank, by anyone — not only by whoever
      remembers which config was run.
    * ``reportable = False``. M4Raw is 0.3 T brain against fastMRI's 3 T knee
      and carries **no pathology annotations**, so it cannot close the
      annotation-linked evaluation (Sec. 14). It carries pipeline development,
      solver wiring, and the coverage-curve machinery — nothing that is
      written up.

    Volumes are multi-channel and reduced by root-sum-of-squares, consistent
    with D-008.
    """

    name = "m4raw"
    modality = "mri"
    reportable = False
