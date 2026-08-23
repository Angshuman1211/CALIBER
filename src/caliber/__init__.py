"""CALIBER — operator-shift-robust conformal hallucination flagging.

A solver-agnostic audit layer that sits on top of pretrained generative-prior
reconstruction and issues per-region reliability flags that stay valid when the
acquisition operator changes between calibration and deployment.

Layout mirrors the five-stage method pipeline of Blueprint Sec. 10:

    operators/    the known acquisition operator A, its range/null-space split,
                  the shared operator registry, and operator-distance measures
    solvers/      pretrained diffusion and flow-matching back-ends behind one
                  black-box posterior-sampler interface           (Stage 1, 2)
    scoring/      region definition, operator-aware scoring, and the
                  ground-truth hallucination criterion                (Stage 3)
    scorebank/    the cached, shared data format all three tracks read
    calibration/  group-conditional and weighted recalibration       (Stage 4)
    evaluation/   coverage, efficiency, detection, and gap-vs-distance (Stage 5)
    data/         dataset access and preprocessing
"""

__version__ = "0.1.0"

from . import (  # noqa: F401
    calibration,
    config,
    data,
    evaluation,
    operators,
    scorebank,
    scoring,
    solvers,
)

__all__ = [
    "__version__",
    "calibration",
    "config",
    "data",
    "evaluation",
    "operators",
    "scorebank",
    "scoring",
    "solvers",
]
