"""Coverage, efficiency, and region-level detection metrics with logging (T2.4)."""

from .bank import (
    CANDIDACY_QUANTILE,
    candidacy_gate,
    cluster_bootstrap_coverage,
    clusters_for_precision,
    derive_score_and_label,
    design_quantile,
)
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
from .paired import (
    PairedGap,
    clusters_for_paired_precision,
    is_crossed,
    paired_coverage_gap,
    paired_gap_curve,
    paired_method_gap,
)

__all__ = [
    "CANDIDACY_QUANTILE",
    "bound_agreement",
    "candidacy_gate",
    "cluster_bootstrap_coverage",
    "clusters_for_precision",
    "coverage_curve",
    "coverage_gap",
    "derive_score_and_label",
    "design_quantile",
    "effective_sample_size",
    "empirical_coverage",
    "evaluate_groups",
    "flag_area",
    "miss_rate",
    "monotonicity",
    "region_fdr",
    "region_power",
    # Track 1 (T1.4) — the paired estimand for the headline figure
    "PairedGap",
    "clusters_for_paired_precision",
    "is_crossed",
    "paired_coverage_gap",
    "paired_gap_curve",
    "paired_method_gap",
]
