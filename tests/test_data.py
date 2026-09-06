"""Dataset-layer tests (T2.1).

These exist because the data layer is the one place where a mistake is both
easy to make and invisible afterwards. A misparsed annotation, a mismatched
crop, or a substrate silently swapped for a stopgap all produce a score bank
that looks entirely normal. The bank is a table of floats; it does not know
where it came from.

Fixtures synthesise fastMRI-layout ``.h5`` files rather than reading real
volumes, so the suite still runs before credentialed access clears.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py")

from caliber.data import DATASETS, build_dataset  # noqa: E402
from caliber.data.fastmri import (  # noqa: E402
    ANNOTATION_GRID,
    FastMRIDataset,
    M4RawDataset,
    boxes_to_mask,
    find_volumes,
    load_annotations,
)


def write_volume(path, n_slices=4, shape=(64, 64), coils=None, seed=0):
    """A minimal fastMRI-layout volume: fully-sampled k-space of a smooth image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    image_shape = (n_slices, *shape) if coils is None else (n_slices, coils, *shape)
    image = rng.random(image_shape).astype(np.float32)
    kspace = np.fft.fft2(image, norm="ortho").astype(np.complex64)
    with h5py.File(path, "w") as fh:
        fh.create_dataset("kspace", data=kspace)
    return path


# -- volume discovery ------------------------------------------------------------


def test_find_volumes_searches_recursively(tmp_path):
    """Archives unpack nested; a config should not have to name the inner directory."""
    write_volume(tmp_path / "singlecoil_val" / "file1000001.h5")
    write_volume(tmp_path / "singlecoil_val" / "inner" / "file1000002.h5")
    found = find_volumes(tmp_path)
    assert [p.name for p in found] == ["file1000001.h5", "file1000002.h5"]


def test_missing_volumes_says_it_searched_recursively(tmp_path):
    with pytest.raises(FileNotFoundError, match="searched recursively"):
        FastMRIDataset(root=tmp_path)


# -- reading ---------------------------------------------------------------------


def test_reads_singlecoil_volume_at_working_resolution(tmp_path):
    write_volume(tmp_path / "file1000001.h5", n_slices=4, shape=(64, 64))
    ds = FastMRIDataset(root=tmp_path, image_shape=(32, 32))
    item = ds[0]
    assert item.image.shape == (32, 32)
    assert item.image.dtype == torch.float32
    assert torch.isfinite(item.image).all()
    assert (item.image >= 0).all()  # magnitude, clamped by normalise()
    assert item.sample_id == "file1000001"


def test_multicoil_volume_is_reduced_by_root_sum_of_squares(tmp_path):
    """D-008: the studied operator is single-coil, so coils are collapsed on read."""
    write_volume(tmp_path / "file1000002.h5", n_slices=2, coils=4, shape=(64, 64))
    item = FastMRIDataset(root=tmp_path, image_shape=(64, 64))[0]
    assert item.image.dim() == 2  # coils gone, not carried downstream


def test_a_volume_without_kspace_fails_loudly(tmp_path):
    """A dataset of finished images cannot support this project.

    See Documentation/T2.1_data_and_operator_pipeline.md Appendix A.1.
    """
    path = tmp_path / "images_only.h5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("reconstruction_rss", data=np.zeros((2, 32, 32), dtype=np.float32))
    with pytest.raises(KeyError, match="Raw k-space is required"):
        FastMRIDataset(root=tmp_path, image_shape=(32, 32))[0]


def test_item_order_is_stable_across_instances(tmp_path):
    for i in range(3):
        write_volume(tmp_path / f"file100000{i}.h5", seed=i)
    first = [it.sample_id for it in FastMRIDataset(root=tmp_path, image_shape=(32, 32))]
    second = [it.sample_id for it in FastMRIDataset(root=tmp_path, image_shape=(32, 32))]
    assert first == second == sorted(first)


# -- annotations -----------------------------------------------------------------


def test_annotations_parse_and_strip_label_whitespace(tmp_path):
    csv_path = tmp_path / "knee.csv"
    csv_path.write_text(
        "file,slice,study_level,x,y,width,height,label\n"
        "file1000001,3,No,10,20,5,6,Joint Effusion \n"       # trailing space, on purpose
        "file1000001,4,Yes,,,,,\n"                            # study-level row, no box
        "file1000002,1,No,0,0,4,4,Meniscus Tear\n",
        encoding="utf-8",
    )
    ann = load_annotations(csv_path)
    assert set(ann) == {"file1000001", "file1000002"}
    assert len(ann["file1000001"]) == 1  # the study-level row is dropped
    assert ann["file1000001"][0]["label"] == "Joint Effusion"


def test_boxes_are_rasterised_then_cropped_with_the_image(tmp_path):
    """The crop must match the image crop exactly or every region is misplaced."""
    mask = boxes_to_mask(
        [{"x": 30, "y": 30, "width": 4, "height": 4}], grid_shape=(64, 64), out_shape=(32, 32)
    )
    assert mask.shape == (32, 32)
    assert mask.sum() == 16  # box is central, so it survives the crop intact
    assert mask[14:18, 14:18].all()


def test_annotation_reaches_the_item_on_the_image_grid(tmp_path):
    write_volume(tmp_path / "file1000001.h5", n_slices=4, shape=(64, 64))
    csv_path = tmp_path / "ann.csv"
    csv_path.write_text(
        "file,slice,study_level,x,y,width,height,label\n"
        "file1000001,2,No,28,28,8,8,Meniscus Tear\n",
        encoding="utf-8",
    )
    item = FastMRIDataset(
        root=tmp_path,
        image_shape=(32, 32),
        annotations_csv=csv_path,
        annotation_grid=(64, 64),
    )[0]
    assert item.slice_idx == 2  # centre slice of a 4-slice volume
    assert item.annotation is not None
    assert item.annotation.shape == item.image.shape
    assert item.annotation.any()


def test_boxes_are_placed_on_the_annotation_grid_not_the_kspace_grid(tmp_path):
    """The bug this guards against misplaces every fastMRI+ box, silently.

    Knee k-space inverts to 640x368; the fastMRI+ boxes are drawn on the
    320x320 ``reconstruction_esc`` reference. Rasterising on the k-space grid
    produced a valid-looking mask in the wrong place — and for a real box near
    the top of the annotation grid, an *empty* mask after the centre crop.
    """
    write_volume(tmp_path / "file1000000.h5", n_slices=20, shape=(640, 368))
    csv_path = tmp_path / "ann.csv"
    # A real row from knee.csv, on the 320x320 grid.
    csv_path.write_text(
        "file,slice,study_level,x,y,width,height,label\n"
        "file1000000,10,No,99,64,57,81,Meniscus Tear\n",
        encoding="utf-8",
    )
    item = FastMRIDataset(
        root=tmp_path, image_shape=(256, 256), annotations_csv=csv_path
    )[0]
    assert item.annotation is not None
    # On the 320x320 grid the box is well inside a 256 centre crop; on the
    # k-space grid it falls entirely outside it and the mask comes out empty.
    assert item.annotation.sum() == 57 * 81


def test_real_fastmri_plus_boxes_all_fit_the_declared_annotation_grid(repo_root):
    """Evidence for ANNOTATION_GRID, checked against the shipped file if present.

    Skips until fastMRI+ is fetched, so it is a check rather than a blocker.
    """
    csv_path = repo_root / "data" / "annotations" / "fastmri_plus" / "knee.csv"
    if not csv_path.exists():
        pytest.skip("fastMRI+ annotations not fetched")
    boxes = [b for v in load_annotations(csv_path).values() for b in v]
    assert boxes, "annotation file parsed to nothing"
    height, width = ANNOTATION_GRID
    assert max(b["x"] + b["width"] for b in boxes) <= width
    assert max(b["y"] + b["height"] for b in boxes) <= height
    # And they genuinely fill it — otherwise a larger grid would fit too.
    assert max(b["y"] + b["height"] for b in boxes) > height - 8


# -- the M4Raw stopgap (D-015) ---------------------------------------------------


def test_m4raw_is_registered_and_reads_fastmri_layout(tmp_path):
    """The whole argument for M4Raw is that the fastMRI reader works unchanged."""
    write_volume(tmp_path / "2022061203_FLAIR01.h5", n_slices=4, coils=4, shape=(64, 64))
    ds = build_dataset({"name": "m4raw", "root": str(tmp_path), "image_shape": (32, 32)})
    assert isinstance(ds, M4RawDataset)
    assert ds[0].image.shape == (32, 32)


def test_m4raw_rows_are_distinguishable_from_fastmri_rows(tmp_path):
    """``dataset`` is a score-bank column, so the substrate is recorded per row."""
    write_volume(tmp_path / "vol.h5", shape=(64, 64))
    fast = FastMRIDataset(root=tmp_path, image_shape=(32, 32))[0]
    m4 = M4RawDataset(root=tmp_path, image_shape=(32, 32))[0]
    assert fast.dataset == "fastmri"
    assert m4.dataset == "m4raw"


def test_development_substrates_are_marked_unreportable():
    """D-015: M4Raw carries pipeline work, never a reported result."""
    assert DATASETS["m4raw"].reportable is False
    assert DATASETS["phantom"].reportable is False
    assert DATASETS["fastmri"].reportable is True
    assert DATASETS["lodopab"].reportable is True


# -- configs against the fetcher -------------------------------------------------


def _load_fetch_module(repo_root):
    spec = importlib.util.spec_from_file_location(
        "fetch_data", repo_root / "scripts" / "fetch_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Registered before exec because the module defines a dataclass, and
    # dataclasses resolve annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_configs_reference_annotation_files_the_fetcher_actually_produces(repo_root):
    """Guards a mismatch that only surfaces the day the data lands.

    The configs named ``knee_singlecoil_annotations.csv`` while the fetcher
    wrote ``knee.csv``. Nothing failed, because neither file could exist until
    fastMRI access cleared — the error was scheduled to appear in Week 3, at
    the worst possible moment.
    """
    import yaml

    fetched = {
        f"{res.dest}/{rel}" for res in _load_fetch_module(repo_root).RESOURCES
        for _url, rel in res.files
    }
    for path in sorted((repo_root / "configs" / "experiments").glob("*.yaml")):
        csv_path = (yaml.safe_load(path.read_text(encoding="utf-8"))
                    .get("data", {}).get("dataset", {}).get("annotations_csv"))
        if csv_path:
            assert csv_path in fetched, (
                f"{path.name} wants {csv_path}, which fetch_data.py never downloads"
            )


def test_every_experiment_config_names_a_registered_dataset(repo_root):
    import yaml

    for path in sorted((repo_root / "configs" / "experiments").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        name = cfg.get("data", {}).get("dataset", {}).get("name")
        if name:
            assert name in DATASETS, f"{path.name} names unknown dataset {name!r}"
