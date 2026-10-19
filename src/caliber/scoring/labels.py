"""Ground-truth hallucination criterion (objective O2, Week 2, Blueprint Sec. 14).

The weekly card for Week 2 makes this a named deliverable: define the
ground-truth hallucination criterion, the region definition, and the
aggregation rule used to compute the region-level false-negative risk, and
document how MRI pathology annotations and CT anatomical annotations support
evaluating it.

Track 1 owns the criterion. Track 2 owns computing and caching the *continuous*
quantities it thresholds, so that changing the criterion later costs no GPU
time. Nothing here is written as a boolean: the bank stores per-region error
magnitudes and annotation coverage, and the label is derived at analysis time.

Note the asymmetry this criterion has to live with. A region can be wrong for
two reasons — the prior invented structure, or the prior smoothed structure
away. Only the first is hallucination in the Blueprint's sense (Sec. 4).
Restricting attention to the free component is what distinguishes them, and is
why ``err_free`` is recorded separately from ``err_total``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..operators.base import ForwardOperator
from .regions import RegionGrid

CRITERION_VERSION = "v0-error-threshold"


@dataclass(frozen=True)
class RegionTruth:
    """Continuous per-region ground-truth quantities; thresholded at analysis time."""

    err_total: Tensor  # region RMSE between reconstruction and truth
    err_free: Tensor  # region RMSE restricted to the free (null-space) component
    free_content: Tensor  # ||P_N x_true|| per region / ref -- the A1 denominator
    ann_coverage: Tensor  # fraction of the region covered by an annotation, or zeros


def compute_region_truth(
    samples: Tensor,
    x_true: Tensor,
    operator: ForwardOperator,
    grid: RegionGrid,
    annotation: Tensor | None = None,
    ref_intensity: float | None = None,
) -> RegionTruth:
    """Per-region error magnitudes for one item, scaled by reference intensity."""
    mean_recon = samples.mean(dim=0)
    x_true = x_true.abs() if x_true.is_complex() else x_true
    ref = ref_intensity or float(x_true.abs().mean()) or 1.0

    err = (mean_recon - x_true).abs()
    err_total = grid.reduce(err, how="rms") / ref

    free_err = operator.project_null(mean_recon - x_true)
    free_err = free_err.abs() if free_err.is_complex() else free_err.abs()
    err_free = grid.reduce(free_err, how="rms") / ref

    # The ground truth's *own* null-space content, per region (A1 / D21).
    # `err_free` alone cannot separate "the reconstruction invented structure"
    # from "this region simply has a lot the operator cannot see": a region of
    # dense anatomy carries large ||P_N x_true|| before any solver runs. Track 1
    # needs the denominator to make the criterion relative (err_free /
    # free_content > tau) instead of an absolute constant, which D-020 showed
    # transfers neither across modalities nor across substrates within one.
    # Recorded, never thresholded here -- the bank stays re-thresholdable.
    free_true = operator.project_null(x_true)
    free_true = free_true.abs() if free_true.is_complex() else free_true.abs()
    free_content = grid.reduce(free_true, how="rms") / ref

    if annotation is None:
        ann = torch.zeros(grid.n_regions, dtype=torch.float32)
    else:
        ann = grid.mask_coverage(annotation).to(torch.float32)

    return RegionTruth(
        err_total=err_total.to(torch.float32),
        err_free=err_free.to(torch.float32),
        free_content=free_content.to(torch.float32),
        ann_coverage=ann,
    )


# Below this much free content a region has essentially nothing the operator
# cannot see, and `err_free / free_content` is 0/0 — an arbitrarily large ratio
# from an empty patch of background.
#
# **This default is a numerical guard, and it is NOT the value a reported label
# should use (D60).** It sits two orders of magnitude below live data — the
# smallest region `free_content` on the M4Raw bank is 0.162 — so it never fires,
# and the denominator artefact it was meant to prevent happens anyway across the
# live range: `-free_content` alone ranks the ratio-only label at AUROC 0.826,
# better than the numerator's 0.741.
#
# For a reported label, pass `min_free_content` as a **quantile of
# `free_content` on the design split** — the candidacy gate. A region where the
# operator hides almost nothing cannot host a null-space hallucination, so it is
# not a candidate, and asking the relative question there is what produces the
# artefact. At the 33rd percentile on the M4Raw bank the denominator's AUROC
# falls to 0.499 — exactly chance — while `s_free`'s stays at a believable 0.931.
MIN_FREE_CONTENT = 1e-3


def label_hallucinated(
    truth: RegionTruth,
    threshold: float = 0.10,
    use_free_component: bool = True,
    require_annotation: bool = False,
    annotation_min_coverage: float = 0.10,
    relative: bool = False,
    min_free_content: float = MIN_FREE_CONTENT,
    absolute_floor: float | None = None,
) -> Tensor:
    """Derive the boolean hallucination label from cached continuous quantities.

    Two forms, and which one carries the headline claim is Track 1's call (O2):

    * **absolute** (default) — ``err_free > threshold``. This is the v0
      criterion, and D-020 recorded that its threshold transfers neither across
      modalities nor across substrates within one.
    * **relative** (``relative=True``) — ``err_free / free_content > threshold``,
      the form agreed at the Week 1 sign-off (A1 / D21) and the reason
      ``free_content`` was added to the bank. The denominator is the ground
      truth's *own* null-space content, so the criterion asks whether the
      reconstruction put more into the free component than was there to begin
      with, rather than whether it put in more than some fixed constant. A
      region of dense anatomy carries large ``err_free`` before any solver runs;
      dividing by ``free_content`` is what stops that from reading as invention.

    ``min_free_content`` is the **candidacy gate and the one to set** (D60): pass
    a quantile of ``free_content`` from the design split, not the numerical
    default. See the note on the constant below for why.

    ``absolute_floor`` adds a second, conjunctive gate, ``err_free >
    absolute_floor``. **Superseded by the candidacy gate and not recommended**
    (D60). It does remove the denominator artefact, but it *worsens* a second
    one: by D38's identity ``err_free -> s_free`` as ``free_content`` falls, so
    in low-content regions the label's numerator *is* the score —
    ``spearman(err_free, s_free) = +0.964`` in the bottom tercile against
    ``+0.491`` in the top. Gating on large ``err_free`` selects exactly those
    regions, which is why it drives ``s_free``'s AUROC to 0.973–0.985: the score
    is being graded against itself. The candidacy gate removes both artefacts and
    makes this gate redundant — adding it on top changes nothing measurable.

    The ratio is driven more by its denominator than by its numerator. Measured
    on the real-prior bank: ``-free_content`` alone ranks the relative label at
    AUROC 0.826 while ``err_free`` alone manages only 0.741, and
    ``spearman(ratio, free_content) = -0.520`` against
    ``spearman(ratio, err_free) = +0.450``. Track 2 measured 0.89-0.95 for the
    denominator on 44 test patients, better than any score in the project. A
    region with little hidden content to begin with yields a large ratio from a
    small error, so the criterion systematically labels smooth tissue as
    hallucinated, and anything correlated with low anatomical density scores well
    against it.

    The two gates fail in opposite directions and the conjunction kills both:
    the ratio alone is denominator-dominated; the absolute test alone is
    anatomy-confounded, which is why D21 introduced the ratio in the first place
    (and P-02, where ``err_free > 0.10`` flags every region).

    ``absolute_floor`` is **substrate-dependent** and must not be hard-coded: it
    is an error magnitude, and D-020 established that those transfer across
    neither modalities nor substrates. Set it as a quantile of ``err_free`` on
    the **design split** and freeze it, the same discipline D37 applies to
    ``tau`` and D28 to the weights. On the M4Raw bank the 62nd percentile
    (0.403) takes the denominator's AUROC from 0.826 to 0.696 while raising
    ``s_free``'s from 0.902 to 0.974, at a positive rate of 0.167 -- still well
    clear of D32's nine-positive floor.

    ``require_annotation`` restricts positives to clinically annotated regions,
    which is the stricter reading of "does the flag fire where it matters"
    (Sec. 14). Both readings are reported.

    Note that ``threshold`` means different things in the two modes — an error
    magnitude in one, a dimensionless ratio in the other — so a value tuned for
    one is meaningless in the other. There is deliberately no shared default.
    """
    err = truth.err_free if use_free_component else truth.err_total
    if relative:
        content = truth.free_content
        # Guarded division: regions with no free content to speak of are not
        # candidates at all, rather than being divided into a large ratio.
        usable = content > min_free_content
        ratio = torch.where(usable, err / content.clamp_min(min_free_content),
                            torch.zeros_like(err))
        label = (ratio > threshold) & usable
        if absolute_floor is not None:
            # Conjunctive, not a replacement: see the note above on why neither
            # gate is sufficient alone (D57).
            label = label & (err > absolute_floor)
    else:
        label = err > threshold
        if absolute_floor is not None:
            raise ValueError(
                "absolute_floor applies to the relative criterion; the absolute form "
                "already thresholds err_free directly via `threshold`"
            )
    if require_annotation:
        label = label & (truth.ann_coverage >= annotation_min_coverage)
    return label


def positive_rate_sweep(
    err_free: Tensor,
    free_content: Tensor,
    thresholds: Sequence[float],
    min_free_content: float = MIN_FREE_CONTENT,
) -> list[tuple[float, float]]:
    """Positive rate of the relative criterion at each candidate ``tau``.

    The Week 1 sign-off fixed the criterion's *form* and deferred ``tau`` until a
    real prior existed (T2.2), with its positive rate to be **reported, not
    targeted**. This is that report: Track 2 supplies the curve, Track 1 picks
    the point on it.

    Returns ``[(tau, positive_rate), ...]`` over regions with usable free
    content.
    """
    usable = free_content > min_free_content
    if not bool(usable.any()):
        raise ValueError("no region has enough free content to apply the relative criterion")
    ratio = err_free[usable] / free_content[usable]
    return [(float(t), float((ratio > t).to(torch.float32).mean())) for t in thresholds]


#: Default quantile of `err_free` on the design split for the absolute criterion (D62).
#:
#: The criterion is `err_free > eps`, and `eps` is set here rather than carried as a
#: constant because D-020 established that error magnitudes transfer across neither
#: modalities nor substrates. P-02's failure -- `err_free > 0.10` flagging every region --
#: was a property of the *constant*, not of the absolute form: a quantile cannot flag
#: everything by construction.
#:
#: 0.75 puts a quarter of candidate regions on the positive side, which keeps D32's
#: nine-positives-per-group floor comfortable on every bank the project has produced,
#: including the thin CT groups introduced by P-34's `ct/{family}/v{views}/a{arc}`.
DESIGN_QUANTILE = 0.75


def absolute_threshold_from_design(
    df,
    quantile: float = DESIGN_QUANTILE,
    err_col: str = "err_free",
    split_col: str = "split",
    design_split: str = "design",
) -> float:
    """``eps`` for the absolute criterion, as a quantile of ``err_free`` on the design split.

    **Why the design split and nothing else** (D62, following D48): choosing the threshold
    on calibration rows biases the conformal threshold, and choosing it on test rows biases
    the coverage number. Both fail silently. A missing design side is therefore a reason to
    stop rather than to fall back, and this raises.

    **Why a quantile and not a constant.** D-020 measured that an absolute error threshold
    transfers across neither modalities nor substrates. A quantile is re-derived per
    substrate from that substrate's own design rows, which is the only form that can be
    stated once and applied to both MRI and CT.

    **What this replaces.** D37's relative ``tau``, D57's conjunctive floor and D60's
    candidacy gate were three attempts to make a function of ``err_free`` and
    ``free_content`` anatomy-neutral within an operator. Measured per operator, none is:
    the ratio carries a low-anatomy bias (AUROC 0.90-0.94), the absolute form carries a
    high-anatomy one (0.05-0.42), and the gated ratio sits between (0.66). The correction
    moved out of the label and into the analysis -- see
    :func:`caliber.evaluation.detection_by_content`.
    """
    if split_col not in df.columns:
        raise ValueError(
            f"frame has no {split_col!r} column; the absolute threshold is set on the "
            "design split and nowhere else (D62)"
        )
    design = df[df[split_col].astype(str) == design_split]
    if design.empty:
        raise ValueError(
            f"no rows with {split_col}={design_split!r}. The threshold must be chosen on "
            "the design side: calibration rows bias the conformal threshold and test rows "
            "bias the coverage number, both silently (D-051). Regenerate the bank with the "
            "dataset's design_fraction set."
        )
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    return float(design[err_col].quantile(quantile))
