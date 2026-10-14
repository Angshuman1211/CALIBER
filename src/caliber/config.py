"""Config loading for the version-controlled, config-driven runner (Blueprint Sec. 17).

Deliberately dependency-light: YAML plus ``include:`` composition plus ``--set``
overrides. Every resolved config carries a content hash so a score bank can be
traced back to the exact configuration that produced it.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import inspect
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


#: Keys the runner reads that are deliberately *absent* from ``base.yaml``. Giving them a
#: default there would change the config hash, and so the run id, of every bank that
#: already exists (D-040), so they are declared here instead. Each one is read by
#: ``runner.py``; a key added to the runner without being added here will be refused as an
#: override, which is the direction to fail.
OPTIONAL_KEYS = frozenset({
    "run.checkpoint_every",
    "run.id",
    "run.scoring_device",
    "run.skip_memory_check",
    "experiment.n_posterior",
})


def _path_exists(cfg: dict, dotted: str) -> bool:
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


def _leaf_paths(cfg: dict, prefix: str = "") -> list[str]:
    paths = []
    for key, value in cfg.items():
        path = f"{prefix}{key}"
        paths.append(path)
        if isinstance(value, dict):
            paths.extend(_leaf_paths(value, f"{path}."))
    return paths


def _constructor_params(cls: type) -> set[str]:
    """Named keyword parameters a class accepts, walking ``**kwargs`` up the MRO.

    ``DPSSampler(**kwargs)`` forwards to ``_DiffusionSampler``, so its own signature
    understates what it takes. A catch-all such as ``**_`` is deliberately **not** treated
    as "accepts anything": that is exactly how a typo would slip through.
    """
    accepted: set[str] = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        for name, param in inspect.signature(init).parameters.items():
            if name != "self" and param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY
            ):
                accepted.add(name)
    return accepted


def _component_params(section: str, cfg: dict) -> set[str] | None:
    """Keys the named solver or dataset accepts, or ``None`` if it cannot be resolved."""
    if section == "experiment.solver":
        from .solvers import SOLVERS as registry

        spec = cfg.get("experiment", {}).get("solver")
    else:
        from .data import DATASETS as registry

        spec = cfg.get("data", {}).get("dataset")
    name = spec if isinstance(spec, str) else (spec or {}).get("name")
    if name not in registry:
        return None
    return _constructor_params(registry[name]) | {"name"}


def validate_override_keys(resolved: dict, final: dict, overrides: list[str]) -> None:
    """Refuse an override whose key does not name something the pipeline reads (D-047).

    A key is accepted if it already exists in the resolved config, if it is one of the
    runner's declared optional keys, or if it is a parameter the configured solver or
    dataset actually takes — checked against the class, after every override has been
    applied, so ``--set experiment.solver.name=dps --set experiment.solver.step_size=20``
    validates ``step_size`` against DPS rather than whatever was configured before.

    Everything else raises, with the nearest real keys as suggestions. The failure this
    prevents is silent: an override that is ignored produces a run whose config hash,
    manifest and provenance all agree with each other, because they record what ran.
    """
    known = set(_leaf_paths(resolved)) | OPTIONAL_KEYS
    for item in overrides:
        key = item.partition("=")[0].strip()
        if "[" in key or "]" in key:
            raise ConfigError(
                f"override {key!r} indexes into a list, which is not supported. Override "
                "the whole list instead, e.g. --set 'experiment.operators=[{...}]'."
            )
        if key in known or _path_exists(resolved, key):
            continue
        section, _, leaf = key.rpartition(".")
        if section in ("experiment.solver", "data.dataset"):
            accepted = _component_params(section, final)
            if accepted is not None and leaf in accepted:
                continue
            if accepted is not None:
                hint = difflib.get_close_matches(leaf, sorted(accepted), n=3)
                raise ConfigError(
                    f"override {key!r}: {leaf!r} is not a parameter of the configured "
                    f"{section.split('.')[-1]}. "
                    + (f"Did you mean {', '.join(section + '.' + h for h in hint)}? "
                       if hint else "")
                    + f"Accepted: {', '.join(sorted(accepted))}."
                )
        hint = difflib.get_close_matches(key, sorted(known), n=3, cutoff=0.6)
        raise ConfigError(
            f"override {key!r} does not name any key this config or the runner uses, so it "
            "would have been silently ignored (D-047). "
            + (f"Did you mean {', '.join(hint)}?" if hint else "Check the spelling.")
        )


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
    resolved = _load_raw(path, root, ())
    cfg = apply_overrides(resolved, overrides or [])
    validate_override_keys(resolved, cfg, overrides or [])
    validate_controlled_comparison(cfg)
    cfg["_meta"] = {
        "config_path": str(path.resolve()),
        "config_hash": config_hash({k: v for k, v in cfg.items() if k != "_meta"}),
    }
    return cfg
