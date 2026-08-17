"""The score bank — cached reconstructions and non-conformity scores (T2.3)."""

from .io import (
    ScoreBankWriter,
    environment_snapshot,
    merge_banks,
    read_bank,
    read_manifest,
    save_reconstructions,
)
from .schema import (
    COLUMNS,
    KEY_COLUMNS,
    SCHEMA_VERSION,
    SchemaError,
    coerce,
    empty_frame,
    validate,
)

__all__ = [
    "COLUMNS",
    "KEY_COLUMNS",
    "SCHEMA_VERSION",
    "ScoreBankWriter",
    "SchemaError",
    "coerce",
    "empty_frame",
    "environment_snapshot",
    "merge_banks",
    "read_bank",
    "read_manifest",
    "save_reconstructions",
    "validate",
]
