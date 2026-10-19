"""Region definition, operator-aware scoring (Stage 3), and the ground-truth criterion."""

from .attribution import (
    Q6Verdict,
    attribute_components,
    component_ablation,
    content_alignment,
    fill_ratio,
    identity_residual,
    q6_verdict,
    stratified_auc,
)
from .labels import (
    CRITERION_VERSION,
    DESIGN_QUANTILE,
    absolute_threshold_from_design,
    MIN_FREE_CONTENT,
    RegionTruth,
    compute_region_truth,
    label_hallucinated,
    positive_rate_sweep,
)
from .regions import RegionGrid
from .score import (
    COMBINATION_VERSION,
    SCORE_VERSION,
    ScoreComponents,
    ScoreScales,
    ScoreWeights,
    combine_components,
    compute_score_components,
)
from .weights import (
    WeightSelection,
    lipschitz_ceiling,
    select_weights,
    weight_curve,
)

__all__ = [
    "CRITERION_VERSION",
    "DESIGN_QUANTILE",
    "absolute_threshold_from_design",
    "MIN_FREE_CONTENT",
    "COMBINATION_VERSION",
    "SCORE_VERSION",
    "Q6Verdict",
    "attribute_components",
    "component_ablation",
    "content_alignment",
    "fill_ratio",
    "identity_residual",
    "q6_verdict",
    "stratified_auc",
    "WeightSelection",
    "lipschitz_ceiling",
    "select_weights",
    "weight_curve",
    "ScoreScales",
    "ScoreWeights",
    "combine_components",
    "RegionGrid",
    "RegionTruth",
    "ScoreComponents",
    "compute_region_truth",
    "compute_score_components",
    "label_hallucinated",
    "positive_rate_sweep",
]
