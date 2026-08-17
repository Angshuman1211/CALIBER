"""Score-bank schema gates.

The bank is the one interface all three tracks share, so its invariants are
tested rather than trusted: no duplicate keys, no mixed schema versions, and no
run that mixes solvers (which would break the controlled comparison of Sec. 13).
"""

from __future__ import annotations

import pandas as pd
import pytest

from caliber.scorebank import (
    COLUMNS,
    KEY_COLUMNS,
    SCHEMA_VERSION,
    SchemaError,
    ScoreBankWriter,
    coerce,
    empty_frame,
    read_bank,
    read_manifest,
    validate,
)


def make_row(**overrides):
    row = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "test-run",
        "config_hash": "deadbeef",
        "dataset": "phantom",
        "split": "calib",
        "sample_id": "phantom0000",
        "slice_idx": 0,
        "operator_id": "mri.random1d.R4.c0.08.s0",
        "modality": "mri",
        "operator_family": "random1d",
        "calib_group": "mri/random1d/R4",
        "solver_id": "reference.pinv",
        "solver_family": "reference",
        "solver_checkpoint": None,
        "n_posterior": 4,
        "seed": 0,
        "region_id": 0,
        "region_row": 0,
        "region_col": 0,
        "region_area": 256,
        "score_version": "v0-placeholder",
        "s_free": 0.1,
        "s_disagree": 0.2,
        "s_residual": 0.3,
        "criterion_version": "v0-error-threshold",
        "err_total": 0.05,
        "err_free": 0.04,
        "ann_coverage": 0.0,
        "ref_intensity": 1.0,
    }
    row.update(overrides)
    return row


def test_empty_frame_matches_declared_schema():
    df = empty_frame()
    assert list(df.columns) == list(COLUMNS)


def test_coerce_rejects_missing_columns():
    df = pd.DataFrame([{"dataset": "phantom"}])
    with pytest.raises(SchemaError, match="missing required columns"):
        coerce(df)


def test_validate_accepts_a_clean_bank():
    df = coerce(pd.DataFrame([make_row(region_id=i) for i in range(4)]))
    assert validate(df, strict=False) == []


def test_validate_flags_duplicate_keys():
    df = coerce(pd.DataFrame([make_row(), make_row()]))
    problems = validate(df, strict=False)
    assert any("duplicate rows" in p for p in problems)
    assert set(KEY_COLUMNS).issubset(COLUMNS)


def test_validate_flags_mixed_solvers_in_one_run():
    """A coverage curve must vary the operator alone (Blueprint Sec. 13)."""
    df = coerce(
        pd.DataFrame(
            [
                make_row(region_id=0, solver_id="diffusion.dps"),
                make_row(region_id=1, solver_id="flow.flowdps"),
            ]
        )
    )
    problems = validate(df, strict=False)
    assert any("mixing multiple solvers" in p for p in problems)


def test_validate_strict_raises():
    df = coerce(pd.DataFrame([make_row(), make_row()]))
    with pytest.raises(SchemaError):
        validate(df, strict=True)


def test_writer_roundtrip_and_manifest(tmp_path):
    writer = ScoreBankWriter(tmp_path / "bank", run_id="test-run", config={"a": 1})
    base = {k: v for k, v in make_row().items()
            if k not in ("region_id", "region_row", "region_col", "region_area",
                         "s_free", "s_disagree", "s_residual",
                         "err_total", "err_free", "ann_coverage",
                         "schema_version", "run_id")}
    regions = [
        {"region_id": i, "region_row": 0, "region_col": i, "region_area": 256}
        for i in range(3)
    ]
    writer.add_item(
        base,
        regions,
        {
            "s_free": [0.1, 0.2, 0.3],
            "s_disagree": [0.0, 0.1, 0.2],
            "s_residual": [0.0, 0.0, 0.1],
            "err_total": [0.01, 0.02, 0.30],
            "err_free": [0.01, 0.02, 0.28],
            "ann_coverage": [0.0, 0.0, 0.5],
        },
    )
    root = writer.flush(registry_descriptors=[{"operator_id": "x"}])

    df = read_bank(root)
    assert len(df) == 3
    assert validate(df, strict=False) == []

    manifest = read_manifest(root)
    assert manifest["n_rows"] == 3
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["environment"]["python"]


def test_writer_rejects_mismatched_array_length(tmp_path):
    writer = ScoreBankWriter(tmp_path / "bank", run_id="r")
    with pytest.raises(ValueError, match="expected 2"):
        writer.add_item({}, [{"region_id": 0}, {"region_id": 1}], {"s_free": [0.1]})
