"""Config loading for the version-controlled, config-driven runner (Blueprint Sec. 17).

Deliberately dependency-light: YAML plus ``include:`` composition plus ``--set``
overrides. Every resolved config carries a content hash so a score bank can be
traced back to the exact configuration that produced it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when a config cannot be resolved or violates a project invariant."""


def find_config_root(start: Path) -> Path:
    """Locate the ``configs/`` directory that ``include:`` paths resolve against."""
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if candidate.name == "configs":
            return candidate
        if (candidate / "configs").is_dir():
            return candidate / "configs"
    raise ConfigError(f"no configs/ directory found above {start}")


def deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge ``over`` into ``base``; scalars and lists are replaced."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_raw(path: Path, root: Path, seen: tuple[Path, ...]) -> dict:
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(p.name for p in (*seen, path))
        raise ConfigError(f"circular include: {chain}")
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")

    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        raise ConfigError(f"config must be a mapping at top level: {path}")

    includes = doc.pop("include", [])
    if isinstance(includes, str):
        includes = [includes]

    merged: dict[str, Any] = {}
    for inc in includes:
        inc_path = (root / inc) if not Path(inc).is_absolute() else Path(inc)
        if inc_path.suffix == "":
            inc_path = inc_path.with_suffix(".yaml")
        merged = deep_merge(merged, _load_raw(inc_path, root, (*seen, path)))
    return deep_merge(merged, doc)


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    """Assign ``cfg[a][b][c] = value`` for a dotted key ``a.b.c``."""
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``KEY=VALUE`` strings; values are parsed as YAML scalars."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override must be KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        set_by_path(out, key.strip(), yaml.safe_load(raw))
    return out


def config_hash(cfg: dict) -> str:
    """Stable 16-hex-char content hash used for run identity and provenance."""
    blob = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def validate_controlled_comparison(cfg: dict) -> None:
    """Enforce Blueprint Sec. 13: the solver is held fixed within a coverage curve.

    A run may sweep operators or it may compare solvers, never both at once.
    Solver-choice ablation (Week 10) runs as one experiment per solver so that
    every coverage curve still varies the operator alone.
    """
    experiment = cfg.get("experiment", {})
    solver = experiment.get("solver")
    if isinstance(solver, list):
        raise ConfigError(
            "experiment.solver must be a single solver. The acquisition operator is the "
            "only quantity permitted to vary within a coverage curve (Blueprint Sec. 13). "
            "For the solver ablation, run one experiment per solver."
        )
    if solver is None:
        raise ConfigError("experiment.solver is required")
    operators = experiment.get("operators")
    if not isinstance(operators, list) or not operators:
        raise ConfigError("experiment.operators must be a non-empty list of operator specs")


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Resolve a config file into a merged, override-applied, validated dict."""
    path = Path(path)
    root = find_config_root(path)
    cfg = _load_raw(path, root, ())
    cfg = apply_overrides(cfg, overrides or [])
    validate_controlled_comparison(cfg)
    cfg["_meta"] = {
        "config_path": str(path.resolve()),
        "config_hash": config_hash({k: v for k, v in cfg.items() if k != "_meta"}),
    }
    return cfg
