"""Acquisition operators, the operator registry, and operator-distance measures."""

from .base import ForwardOperator, OperatorDescriptor
from .ct import CTProjectionOperator, make_angles
from .decomposition import (
    DecompositionReport,
    aggregate_reports,
    decompose_samples,
    orthogonality_defect,
)
from .distance import DISTANCES, compute_distance, distance_matrix
from .mri import MASK_FAMILIES, MRIUndersamplingOperator, build_mask
from .registry import OperatorRegistry, build_operator, expand_grid

__all__ = [
    "DISTANCES",
    "MASK_FAMILIES",
    "CTProjectionOperator",
    "DecompositionReport",
    "ForwardOperator",
    "MRIUndersamplingOperator",
    "OperatorDescriptor",
    "OperatorRegistry",
    "aggregate_reports",
    "build_mask",
    "build_operator",
    "compute_distance",
    "decompose_samples",
    "distance_matrix",
    "expand_grid",
    "make_angles",
    "orthogonality_defect",
]
