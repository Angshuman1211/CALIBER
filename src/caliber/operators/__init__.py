"""Acquisition operators, the operator registry, and operator-distance measures."""

from .base import ForwardOperator, OperatorDescriptor
from .ct import CTProjectionOperator, make_angles
from .distance import DISTANCES, compute_distance, distance_matrix
from .mri import MASK_FAMILIES, MRIUndersamplingOperator, build_mask
from .registry import OperatorRegistry, build_operator, expand_grid

__all__ = [
    "DISTANCES",
    "MASK_FAMILIES",
    "CTProjectionOperator",
    "ForwardOperator",
    "MRIUndersamplingOperator",
    "OperatorDescriptor",
    "OperatorRegistry",
    "build_mask",
    "build_operator",
    "compute_distance",
    "distance_matrix",
    "expand_grid",
    "make_angles",
]
