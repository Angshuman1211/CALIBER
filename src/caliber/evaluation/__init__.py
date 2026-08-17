"""Coverage, efficiency, and region-level detection metrics with logging (T2.4)."""

from .metrics import (
    bound_agreement,
    coverage_curve,
    coverage_gap,
    effective_sample_size,
    empirical_coverage,
    evaluate_groups,
    flag_area,
    miss_rate,
    monotonicity,
    region_fdr,
    region_power,
)

__all__ = [
    "bound_agreement",
    "coverage_curve",
    "coverage_gap",
    "effective_sample_size",
    "empirical_coverage",
    "evaluate_groups",
    "flag_area",
    "miss_rate",
    "monotonicity",
    "region_fdr",
    "region_power",
]
