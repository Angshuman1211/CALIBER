"""Region definition, operator-aware scoring (Stage 3), and the ground-truth criterion."""

from .labels import CRITERION_VERSION, RegionTruth, compute_region_truth, label_hallucinated
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

__all__ = [
    "CRITERION_VERSION",
    "COMBINATION_VERSION",
    "SCORE_VERSION",
    "ScoreScales",
    "ScoreWeights",
    "combine_components",
    "RegionGrid",
    "RegionTruth",
    "ScoreComponents",
    "compute_region_truth",
    "compute_score_components",
    "label_hallucinated",
]
