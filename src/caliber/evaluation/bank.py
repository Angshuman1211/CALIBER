"""Turning a cached bank into scores, labels, and honest precision on them (T2.4).

Two jobs, both of which were previously done by hand inside `scripts/analyse_bank.py`:

**Deriving the score and the label.** Both come from cached columns, so changing either
costs no GPU time (Sec. 17). Having one definition matters more than it looks: P-19 was
caused by a second copy of this logic drifting from the first, and the fix is not to be
more careful but to have one copy.

**Saying how precise the result is.** Regions are not independent — they are clustered
within volumes, and volumes within patients — so the usual interval over regions is far
too narrow. P-23 measured what that costs: with 12 test volumes, a per-operator coverage
estimate has a 95% interval about half the unit interval wide, and a bank can then show
neither that a method works nor that it fails. The bootstrap here resamples *clusters*,
which is the only version of the interval that means anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from ..scoring.labels import RegionTruth, label_hallucinated
from ..scoring.score import ScoreComponents, ScoreScales, ScoreWeights, combine_components


def _components(frame: pd.DataFrame) -> ScoreComponents:
    return ScoreComponents(
        s_free=torch.tensor(frame["s_free"].to_numpy()),
        s_disagree=torch.tensor(frame["s_disagree"].to_numpy()),
        s_residual=torch.tensor(frame["s_residual"].to_numpy()),
        ref_intensity=1.0,
    )


def derive_score_and_label(
    df: pd.DataFrame,
    calibration_operator: str,
    weights: ScoreWeights | None = None,
    threshold: float = 1.75,
    relative: bool = True,
    absolute_floor: float | None = None,
    min_free_content: float | None = None,
    require_annotation: bool = False,
    annotation_min_coverage: float = 0.10,
) -> pd.DataFrame:
    """Add ``score`` and ``label`` columns to a bank frame, from cache alone.

    The combination is T1.1's standardised affine rule with the scales estimated on the
    **calibration operator** and then frozen (D26, D27): re-estimating them per deployment
    operator would pull the score distributions towards each other and shrink the very
    coverage gap the analysis exists to measure.

    ``threshold`` defaults to Track 1's ratified ``tau = 1.75`` under the relative
    criterion (their D37, cross-track D-037/D-038).

    **The reported label uses the candidacy gate** (D-060, Track 1's ruling, superseding
    D-059's gate): ``min_free_content`` set to the 33rd percentile of ``free_content`` on the design
    split, via :func:`design_quantile`. The ratio alone is driven more by its denominator
    than its numerator and labels smooth tissue as hallucinated (D-059). D-059's first fix,
    ``absolute_floor`` on ``err_free``, removed that but selected exactly the regions where
    ``err_free`` approaches ``s_free`` (D-037's identity), so the score began to measure the
    label's own numerator (AUROC 0.99-1.00). A region where the operator hides almost
    nothing cannot host a null-space hallucination, so it is not a candidate: one gate, on
    ``free_content``. ``absolute_floor`` stays available only to reproduce that comparison.
    Leaving both ``None`` reproduces numbers measured before D-059.
    """
    if calibration_operator not in set(df["operator_id"]):
        raise ValueError(
            f"calibration operator {calibration_operator!r} is not in this bank; "
            f"have {sorted(set(df['operator_id']))}"
        )
    out = df.copy()
    calib_rows = out[out["operator_id"] == calibration_operator]
    scales = ScoreScales.from_components(_components(calib_rows))
    out["score"] = combine_components(
        _components(out), weights or ScoreWeights(), scales
    ).numpy()

    truth = RegionTruth(
        err_total=torch.tensor(out["err_total"].to_numpy()),
        err_free=torch.tensor(out["err_free"].to_numpy()),
        free_content=torch.tensor(out["free_content"].to_numpy()),
        ann_coverage=torch.tensor(out["ann_coverage"].to_numpy()),
    )
    gate = {} if min_free_content is None else {"min_free_content": min_free_content}
    out["label"] = label_hallucinated(
        truth, threshold=threshold, relative=relative, absolute_floor=absolute_floor,
        require_annotation=require_annotation, annotation_min_coverage=annotation_min_coverage,
        **gate,
    ).numpy()
    return out


#: D-061's ruling: the reported label is **absolute**, ``err_free > eps``, with ``eps`` this
#: quantile of ``err_free`` on the bank's *own* design split, frozen. D-061 names no quantile;
#: 0.75 is the one its table tested. Every reporting script defaults to it (P-19).
LABEL_QUANTILE = 0.75
#: The labels a script can be asked for. Only ``absolute`` is reported; the other two
#: reproduce numbers from before D-061, for comparison.
LABELS = ("absolute", "candidacy", "ratio")


@dataclass(frozen=True)
class LabelSpec:
    """A named label, valued on a bank: the arguments for :func:`derive_score_and_label`."""

    name: str
    text: str
    kwargs: dict = field(default_factory=dict)


def label_spec(df: pd.DataFrame, label: str = "absolute", quantile: float | None = None,
               tau: float = 1.75, require_annotation: bool = False) -> LabelSpec:
    """The label a script reports, valued on ``df``'s design split and frozen.

    - ``absolute`` (D-061, the reported label): ``err_free > eps``, ``eps`` the ``quantile``
      (default :data:`LABEL_QUANTILE`) of ``err_free`` over the design rows. No denominator:
      D-061 found no function of ``err_free`` and ``free_content`` that is anatomy-neutral
      within an operator, so the confound is handled in the analysis, by stratifying on
      ``free_content``, not in the outcome. A quantile cannot flag everything by
      construction, which was P-02's failure with a fixed constant.
    - ``candidacy`` (D-060, superseded): the ratio among regions above a design quantile
      of ``free_content``.
    - ``ratio`` (D37 alone): the ratio, with no gate. It needs no design side, which is why
      a bank built before D-051 can still be read with it.

    ``require_annotation`` gives D-061's secondary reading for MRI: a positive must also lie
    in an annotated region (fastMRI+). That is the one anchor not defined by the project's
    own error arithmetic. It does not exist on CT (P-03).
    """
    if label == "absolute":
        q = LABEL_QUANTILE if quantile is None else quantile
        try:
            eps = design_quantile(df, "err_free", q)
        except ValueError as exc:
            raise ValueError(
                f"{exc} For a bank built before D-051, pass --label ratio for the ratio-only "
                "label, knowing it is not the reported one (D-061)."
            ) from None
        spec = LabelSpec("absolute", f"err_free > {eps:.4f} (q={q:g} of err_free on the "
                         "design split, D-061)", {"threshold": eps, "relative": False})
    elif label == "candidacy":
        q = CANDIDACY_QUANTILE if quantile is None else quantile
        gate = candidacy_gate(df, q)
        spec = LabelSpec("candidacy", f"ratio > {tau:g} among candidates with free_content > "
                         f"{gate:.4f} (D-060, superseded by D-061)",
                         {"threshold": tau, "relative": True, "min_free_content": gate})
    elif label == "ratio":
        spec = LabelSpec("ratio", f"ratio > {tau:g}, no gate (D37; superseded by D-061)",
                         {"threshold": tau, "relative": True})
    else:
        raise ValueError(f"unknown label {label!r}; the labels are {list(LABELS)}")
    if require_annotation:
        spec = LabelSpec(spec.name + "+annotation",
                         spec.text + ", and in an annotated region (ann_coverage >= 0.10)",
                         {**spec.kwargs, "require_annotation": True})
    return spec


def add_label_args(ap) -> None:
    """The label options every reporting script takes, so they mean the same everywhere."""
    ap.add_argument("--label", choices=LABELS, default="absolute",
                    help="D-061's absolute label (default); candidacy and ratio reproduce "
                         "earlier numbers, for comparison only")
    ap.add_argument("--label-quantile", type=float, default=None,
                    help=f"absolute: quantile of err_free on the design split (default "
                         f"{LABEL_QUANTILE}); candidacy: of free_content ({CANDIDACY_QUANTILE})")
    ap.add_argument("--tau", type=float, default=1.75,
                    help="the ratio threshold, for the candidacy and ratio labels (D37)")
    ap.add_argument("--require-annotation", action="store_true",
                    help="D-061's secondary MRI reading: positives only in annotated regions")


def label_from_args(df: pd.DataFrame, args) -> LabelSpec:
    return label_spec(df, args.label, args.label_quantile, args.tau, args.require_annotation)


# D-060's ruled candidacy quantile. Every script that reports a label-dependent number takes
# its default from here, so the ruled value lives in one place (P-19).
CANDIDACY_QUANTILE = 0.33


def candidacy_gate(df: pd.DataFrame, quantile: float = CANDIDACY_QUANTILE) -> float | None:
    """The value to pass as ``min_free_content``: D-060's gate, valued on the design split.

    ``quantile = 0`` opts out and returns ``None``, which reproduces the ratio-only label of
    before D-059 -- defective, and only for comparison. A bank with no design side is
    refused by :func:`design_quantile` rather than gated on something else.
    """
    if quantile == 0:
        return None
    try:
        return design_quantile(df, "free_content", quantile)
    except ValueError as exc:
        raise ValueError(
            f"{exc} For a bank built before D-051, pass a candidacy quantile of 0 to get the "
            "ratio-only label, knowing it is the defective one (D-059)."
        ) from None


def design_quantile(df: pd.DataFrame, column: str, quantile: float,
                    split_col: str = "split") -> float:
    """A label gate's value: a quantile of ``column`` over the **design** patients, to freeze.

    Gate values are magnitudes, and magnitudes do not transfer across substrates (D-020), so
    each is set on the bank it is used with -- but only from the design side (D-051).
    Fitting it on calib or test rows would let the label see the patients whose coverage it
    is about to score, which is the in-sample shape behind P-19, P-23 and D-058. A bank with
    no design side is refused rather than quietly fitted on something else.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    if split_col not in df.columns or not (df[split_col] == "design").any():
        raise ValueError(
            "no design rows: a label gate must come from the design split (D-051). Build the "
            "bank with data.dataset.design_fraction > 0."
        )
    return float(df.loc[df[split_col] == "design", column].quantile(quantile))


def cluster_bootstrap_coverage(
    flags,
    labels,
    clusters,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Coverage with a confidence interval that resamples **clusters**, not regions.

    ``clusters`` is the unit that is actually exchangeable — the volume, or better the
    patient. Resampling regions instead would treat 9,216 correlated regions as 9,216
    independent observations and report an interval perhaps thirty times too narrow.

    Returns the point estimate, the interval, and ``se``, the bootstrap standard error —
    which is what :func:`clusters_for_precision` extrapolates from.
    """
    flags = np.asarray(flags).astype(bool)
    labels = np.asarray(labels).astype(bool)
    clusters = np.asarray(clusters)
    if not (len(flags) == len(labels) == len(clusters)):
        raise ValueError("flags, labels and clusters must be the same length")

    keys = np.unique(clusters)
    caught = np.array([int((flags & labels)[clusters == k].sum()) for k in keys])
    positives = np.array([int(labels[clusters == k].sum()) for k in keys])
    total = positives.sum()
    if total == 0:
        return {"coverage": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "se": float("nan"), "n_clusters": len(keys), "n_positive": 0}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boot = caught[idx].sum(axis=1) / np.maximum(positives[idx].sum(axis=1), 1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {
        "coverage": float(caught.sum() / total),
        "lo": float(lo),
        "hi": float(hi),
        "se": float(boot.std(ddof=1)),
        "n_clusters": int(len(keys)),
        "n_positive": int(total),
    }


def clusters_for_precision(
    se_observed: float, n_observed: int, target_half_width: float, z: float = 1.96
) -> int:
    """Clusters needed for a ``±target_half_width`` interval, from an observed SE.

    Scales the observed standard error as ``1/sqrt(n)``. That is the right scaling for
    independent clusters and an **approximation** here: it assumes the clusters added
    resemble the ones measured, and it is extrapolated from however few were observed.
    Treat the answer as an order of magnitude — which is enough to tell "about seventy"
    from "about seven hundred", and that is the decision it exists to inform.
    """
    if not np.isfinite(se_observed) or se_observed <= 0 or target_half_width <= 0:
        raise ValueError("need a positive observed SE and target half-width")
    target_se = target_half_width / z
    return int(np.ceil(n_observed * (se_observed / target_se) ** 2))
