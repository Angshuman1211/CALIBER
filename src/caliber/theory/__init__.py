"""Track 3 numerical verification (T3.5): small synthetic checks of the L1/L2 derivations.

Everything here runs on CPU in seconds to minutes and touches no dataset: operators come
from the shared registry, images from the synthetic phantom, scores from the real
``caliber.scoring`` code. ``stats`` holds the distribution-distance estimators the
derivations are stated in; ``harness`` holds the controlled score-generating setup.
"""

from .stats import (
    DistanceReport,
    crossing_probability,
    distribution_distances,
    joint_cdf_gap,
    kde_density_difference_sup,
    ks_distance,
    tv_distance,
    w1_distance,
)

__all__ = [
    "crossing_probability",
    "joint_cdf_gap",
    "DistanceReport",
    "distribution_distances",
    "kde_density_difference_sup",
    "ks_distance",
    "tv_distance",
    "w1_distance",
]
