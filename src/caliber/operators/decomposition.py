"""Stage 2 — the range / null-space split, measured on solver output.

Blueprint Sec. 10, Stage 2; weekly cards Week 2 [B]: *begin operator
decomposition — range vs. null-space split*.

The projectors themselves are Week 1 infrastructure and live in
:mod:`caliber.operators.base`: ``P_R = A^+ A`` and ``P_N = I - P_R``. What this
module adds is the thing Week 2 could not do until a real solver existed —
apply the split to *reconstructions a pretrained prior actually produced* and
measure where their content, their error, and their posterior spread land.

Three quantities, answering three different questions:

* **energy** — how the reconstruction itself divides. A property of the image
  and the operator, not of the solver's quality.
* **error** — how the deviation from ground truth divides. The project's premise
  is that it is null-dominated: what the measurements pin down cannot be got
  badly wrong, so the error that matters is error the operator cannot see.
* **spread** — how disagreement across posterior draws divides. This is the
  assumption ``s_disagree`` rests on (``scoring/score.py`` measures spread in
  the free component only), and it is the one of the three a *solver* can
  violate: a sampler that wanders in the range component is not exploring a
  posterior, it is failing data consistency.

Nothing here is written into the score bank. The bank's schema is a cross-track
contract (D-021, ``score_bank_spec.md`` §6) and these are diagnostics about the
pipeline rather than evidence about an item, so they belong in a run's audit
artefact — which is where ``scripts/stage2_decomposition.py`` puts them.

**The orthogonality caveat is load-bearing.** ``P_R`` is an orthogonal
projection only when the inverse used to build it is the true pseudo-inverse.
That holds for mask-based MRI and *not* for CT, whose ``decomposition_pinv`` is
conjugate gradient at a finite iteration budget (D-006, D-012). Every share
reported here is therefore normalised by ``||P_R v||^2 + ||P_N v||^2`` rather
than by ``||v||^2``, and the gap between those two is reported explicitly as
:attr:`DecompositionReport.orthogonality_defect`. A share normalised by
``||v||^2`` would quietly absorb that gap and then read as a finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .base import ForwardOperator

# Below this norm a vector carries no direction worth splitting, and every share
# computed from it is 0/0. Reported as NaN rather than 0.0: a share of zero is a
# measurement, and "there was nothing to measure" is not one.
_NEGLIGIBLE = 1e-12


def _norm(x: Tensor) -> float:
    return float(torch.linalg.vector_norm(x))


def _share(part: float, other: float) -> float:
    """``part^2 / (part^2 + other^2)`` — the energy fraction, NaN when undefined."""
    total = part**2 + other**2
    if total < _NEGLIGIBLE**2:
        return float("nan")
    return part**2 / total


@dataclass(frozen=True)
class DecompositionReport:
    """One item's Stage-2 split. Norms are relative; shares are in ``[0, 1]``.

    ``*_share`` is always the **null** share, because that is the quantity every
    claim in this project is about. Its complement is the range share.
    """

    operator_id: str
    n_samples: int

    # -- energy: how the reconstruction divides ------------------------------
    range_energy: float
    null_energy: float
    null_energy_share: float

    # -- error: how the deviation from truth divides (NaN without ground truth)
    range_error: float = float("nan")
    null_error: float = float("nan")
    null_error_share: float = float("nan")

    # -- spread: how posterior disagreement divides (NaN for a single draw) ---
    range_spread: float = float("nan")
    null_spread: float = float("nan")
    null_spread_share: float = float("nan")

    # -- how far from orthogonal the split is, and how well the data is fit ---
    orthogonality_defect: float = float("nan")
    data_consistency_error: float = float("nan")

    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_row(self) -> dict[str, Any]:
        row = {
            "operator_id": self.operator_id,
            "n_samples": self.n_samples,
            "range_energy": self.range_energy,
            "null_energy": self.null_energy,
            "null_energy_share": self.null_energy_share,
            "range_error": self.range_error,
            "null_error": self.null_error,
            "null_error_share": self.null_error_share,
            "range_spread": self.range_spread,
            "null_spread": self.null_spread,
            "null_spread_share": self.null_spread_share,
            "orthogonality_defect": self.orthogonality_defect,
            "data_consistency_error": self.data_consistency_error,
        }
        row.update(self.extra)
        return row


def orthogonality_defect(operator: ForwardOperator, x: Tensor) -> float:
    """``| ||x||^2 - ||P_R x||^2 - ||P_N x||^2 | / ||x||^2`` — 0 iff the split is orthogonal.

    Exactly zero for mask-based MRI. Small but non-zero for CT, whose
    ``decomposition_pinv`` is CG at a finite iteration budget (D-012). Reported
    rather than gated: the registry already gates contraction, and this is the
    number that says how far a share may be trusted.
    """
    rng, null = operator.decompose(x)
    total = _norm(x) ** 2
    if total < _NEGLIGIBLE**2:
        return float("nan")
    return abs(total - _norm(rng) ** 2 - _norm(null) ** 2) / total


def decompose_samples(
    samples: Tensor,
    operator: ForwardOperator,
    x_true: Tensor | None = None,
    y: Tensor | None = None,
) -> DecompositionReport:
    """Split one item's posterior draws into range and null components.

    ``samples`` is ``(n_posterior, H, W)`` — the same tensor the runner hands to
    :func:`caliber.scoring.score.compute_score_components`, so this measures the
    decomposition scoring actually used rather than a re-derivation of it.

    ``x_true`` and ``y`` are optional. Without ground truth the error split is
    NaN; without measurements the data-consistency error is. Ground truth is
    absent at deployment, which is why the split is optional and not a required
    argument: the energy and spread halves are computable from data the method
    is allowed to see.
    """
    if samples.dim() != 3:
        raise ValueError(
            f"expected samples of shape (n_posterior, H, W), got {tuple(samples.shape)}"
        )

    n = int(samples.shape[0])
    mean_recon = samples.mean(dim=0)
    rng, null = operator.decompose(mean_recon)
    scale = _norm(mean_recon) or 1.0

    range_energy, null_energy = _norm(rng), _norm(null)

    report: dict[str, Any] = {
        "operator_id": operator.descriptor().operator_id,
        "n_samples": n,
        "range_energy": range_energy / scale,
        "null_energy": null_energy / scale,
        "null_energy_share": _share(null_energy, range_energy),
        "orthogonality_defect": orthogonality_defect(operator, mean_recon),
    }

    if x_true is not None:
        # The deviation is decomposed, not the two images separately. P_R and
        # P_N are linear so it is the same split either way, but this keeps one
        # projector application instead of two and never subtracts two large
        # numbers to obtain a small one.
        delta = mean_recon - x_true
        d_rng, d_null = operator.decompose(delta)
        truth_scale = _norm(x_true) or 1.0
        d_rng_norm, d_null_norm = _norm(d_rng), _norm(d_null)
        report["range_error"] = d_rng_norm / truth_scale
        report["null_error"] = d_null_norm / truth_scale
        report["null_error_share"] = _share(d_null_norm, d_rng_norm)

    if n > 1:
        # RMS deviation from the posterior mean, split per draw and then pooled.
        # Pooling first and splitting once would measure the spread of the mean
        # deviation, which is zero by construction.
        r_sq = n_sq = 0.0
        for s in samples:
            s_rng, s_null = operator.decompose(s - mean_recon)
            r_sq += _norm(s_rng) ** 2
            n_sq += _norm(s_null) ** 2
        range_spread = (r_sq / n) ** 0.5
        null_spread = (n_sq / n) ** 0.5
        report["range_spread"] = range_spread / scale
        report["null_spread"] = null_spread / scale
        report["null_spread_share"] = _share(null_spread, range_spread)

    if y is not None:
        y_scale = _norm(y) or 1.0
        report["data_consistency_error"] = _norm(operator.residual(mean_recon, y)) / y_scale

    return DecompositionReport(**report)


def aggregate_reports(reports: list[DecompositionReport]) -> dict[str, Any]:
    """Mean of every numeric field over items, NaN-safe, keyed as :meth:`to_row`.

    ``operator_id`` is carried through only when every report shares one, so an
    aggregate over a mixed grid cannot be mistaken for a per-operator number.
    """
    if not reports:
        raise ValueError("nothing to aggregate")
    rows = [r.to_row() for r in reports]
    ids = {r["operator_id"] for r in rows}
    out: dict[str, Any] = {
        "operator_id": ids.pop() if len(ids) == 1 else "<mixed>",
        "n_items": len(rows),
        "n_samples": rows[0]["n_samples"],
    }
    numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k != "n_samples"]
    for key in numeric:
        values = torch.tensor([float(r[key]) for r in rows], dtype=torch.float64)
        finite = values[torch.isfinite(values)]
        out[key] = float(finite.mean()) if finite.numel() else float("nan")
    return out
