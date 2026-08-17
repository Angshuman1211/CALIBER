"""The operator registry — one definition of each acquisition operator.

Blueprint Sec. 19 names this as a shared interface: one definition of each
operator and of the distance between operators, used identically in MRI, CT,
and in Track 3's derivations. The Week 9 audit (T1.3 / T3.2) checks that the
same registered representation is used everywhere, so *all* operator
construction goes through this module.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from .base import ForwardOperator
from .ct import CTProjectionOperator
from .mri import MRIUndersamplingOperator

OPERATOR_TYPES: dict[str, type[ForwardOperator]] = {
    "mri": MRIUndersamplingOperator,
    "ct": CTProjectionOperator,
}

# Parameters that define a grid axis when given as a list. Anything else that
# happens to be a list is passed through to the operator untouched.
GRID_AXES = (
    "acceleration",
    "center_fraction",
    "mask_seed",
    "family",
    "n_views",
    "angular_range_deg",
    "start_angle_deg",
    "photon_count",
)


def expand_grid(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand list-valued grid axes into the Cartesian product of operator specs.

    ``{"modality": "mri", "acceleration": [2, 4]}`` becomes two specs. This is
    how "traverse the operator space" (Sec. 16) is expressed in config.
    """
    axes = {k: v for k, v in spec.items() if k in GRID_AXES and isinstance(v, list)}
    fixed = {k: v for k, v in spec.items() if k not in axes}
    if not axes:
        return [dict(fixed)]
    keys = list(axes)
    return [dict(fixed, **dict(zip(keys, combo, strict=True)))
            for combo in itertools.product(*(axes[k] for k in keys))]


def build_operator(spec: dict[str, Any], image_shape: tuple[int, int]) -> ForwardOperator:
    """Instantiate a single operator from a fully-resolved spec."""
    spec = dict(spec)
    modality = spec.pop("modality", None)
    if modality not in OPERATOR_TYPES:
        raise ValueError(f"unknown modality {modality!r}; expected one of {sorted(OPERATOR_TYPES)}")
    return OPERATOR_TYPES[modality](image_shape=image_shape, **spec)


class OperatorRegistry:
    """Ordered, de-duplicated collection of operators keyed by ``operator_id``."""

    def __init__(self, image_shape: tuple[int, int]):
        self.image_shape = tuple(image_shape)
        self._ops: dict[str, ForwardOperator] = {}

    @classmethod
    def from_specs(
        cls, specs: list[dict[str, Any]], image_shape: tuple[int, int]
    ) -> OperatorRegistry:
        registry = cls(image_shape)
        for spec in specs:
            for resolved in expand_grid(spec):
                registry.add(build_operator(resolved, image_shape))
        return registry

    def add(self, op: ForwardOperator) -> ForwardOperator:
        """Register ``op``, returning the existing instance if the id is already known."""
        oid = op.descriptor().operator_id
        if oid in self._ops:
            return self._ops[oid]
        self._ops[oid] = op
        return op

    def __getitem__(self, operator_id: str) -> ForwardOperator:
        if operator_id not in self._ops:
            raise KeyError(f"operator {operator_id!r} not registered; have {sorted(self._ops)}")
        return self._ops[operator_id]

    def __len__(self) -> int:
        return len(self._ops)

    def __iter__(self):
        return iter(self._ops.values())

    @property
    def ids(self) -> list[str]:
        return list(self._ops)

    def descriptors(self) -> list[dict[str, Any]]:
        return [op.descriptor().to_row() for op in self._ops.values()]

    def check_health(self, adjoint_atol: float = 1e-3) -> list[dict[str, Any]]:
        """Run the adjoint and idempotence diagnostics over every registered operator.

        A silently wrong projector corrupts every score in the bank, so this
        runs as a gate before any score-bank production job.
        """
        report = []
        for op in self._ops.values():
            desc = op.descriptor()
            adj_err = op.adjoint_test()
            idem_err = op.projection_idempotence_error()
            gain = op.null_projection_gain()
            report.append(
                {
                    "operator_id": desc.operator_id,
                    "adjoint_rel_error": adj_err,
                    "idempotence_rel_error": idem_err,
                    "null_projection_gain": gain,
                    "exact_pinv": desc.exact_pinv,
                    "adjoint_ok": adj_err < adjoint_atol,
                    # Only exact-pinv operators are required to be idempotent
                    # projectors; FBP-based CT is expected to miss.
                    "projector_ok": (not desc.exact_pinv) or idem_err < adjoint_atol,
                    # Reported, not gated. An amplifying P_N is a known defect
                    # of the FBP-based CT inverse (D-012) that Week 7 must fix;
                    # failing the run today would block CT development.
                    "contraction_ok": gain <= 1.0 + 1e-3,
                }
            )
        return report

    def write_manifest(self, path: str | Path) -> Path:
        """Serialise the registry so score banks and derivations share one source."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "image_shape": list(self.image_shape),
            "n_operators": len(self._ops),
            "operators": self.descriptors(),
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path
