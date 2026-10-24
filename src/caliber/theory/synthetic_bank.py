"""A real-schema score bank built from the T3.5 harness (T3.6).

T3.6 compares bound to measurement on *banks*. Track 2's fastMRI and CT banks are local to
their machine (NYU data agreement; ``banks/README.md``), so this builds a bank-shaped table
from the controlled harness instead: same column names, a three-way ``design`` / ``calib`` /
``test`` split that is identical at every operator (D-022, D-051), the item as the cluster, and
the same paired images and noise seeds at every operator (the crossed design, D-045). The T3.6
pipeline then runs on it through the **real** ``label_spec`` / ``derive_score_and_label`` path,
which is the point: the code that will meet Track 2's banks is the code exercised here.

Not a result about the project: the solvers are the harness's (white-noise fill or the toy
structured prior), the images are phantoms, and every "patient" is one phantom.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ..operators.base import ForwardOperator
from .harness import HarnessConfig, run_trials


def assign_splits(n_items: int, design_fraction: float = 0.15, calib_fraction: float = 0.45):
    """``design`` first, then ``calib``, then ``test``; the same for every operator."""
    n_design = int(round(design_fraction * n_items))
    n_calib = int(round(calib_fraction * n_items))
    return ["design"] * n_design + ["calib"] * n_calib + ["test"] * (n_items - n_design - n_calib)


def build_bank(
    operators: Sequence[ForwardOperator],
    cfg: HarnessConfig,
    n_items: int = 60,
    design_fraction: float = 0.15,
    calib_fraction: float = 0.45,
) -> pd.DataFrame:
    """One row per (item, operator, region), in the score-bank column layout."""
    splits = assign_splits(n_items, design_fraction, calib_fraction)
    frames = []
    for op in operators:
        desc = op.descriptor()
        ts = run_trials(op, cfg, n_items, with_truth=True)
        n_regions = ts.components.shape[2]
        for t in range(n_items):
            truth = ts.truth[t]
            frames.append(
                pd.DataFrame(
                    {
                        "dataset": "synthetic_harness",
                        "split": splits[t],
                        "sample_id": f"item{t:04d}",
                        "patient_id": f"item{t:04d}",
                        "slice_idx": 0,
                        "operator_id": desc.operator_id,
                        "modality": desc.modality,
                        "operator_family": desc.family,
                        "calib_group": desc.calib_group,
                        "region_id": range(n_regions),
                        "s_free": ts.components[t, 0].numpy(),
                        "s_disagree": ts.components[t, 1].numpy(),
                        "s_residual": ts.components[t, 2].numpy(),
                        "err_total": truth.err_total.numpy(),
                        "err_free": truth.err_free.numpy(),
                        "free_content": truth.free_content.numpy(),
                        "ann_coverage": truth.ann_coverage.numpy(),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)
