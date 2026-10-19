"""Detection reported within anatomy strata, per operator (T1.4, D62).

**Track 1 owns this module.** It is the other half of D62: the label stopped trying to be
anatomy-neutral, so the anatomy correction moved here, where it can be inspected.

## Why the correction belongs in the analysis

The ground-truth criterion names a region hallucinated from ``err_free``, and ``err_free``
scales with how much anatomy the region holds. D21 corrected for that by dividing by
``free_content``, which over-corrected: measured per operator, the ratio carries a
*low*-anatomy bias (AUROC of ``-free_content`` against the label, 0.90-0.94), the plain
absolute form carries a *high*-anatomy one (0.05-0.42), and the gated ratio sits between
them (0.66). Three attempts -- D37's ``tau``, D57's floor, D60's candidacy gate -- failed
the same way, because **no function of ``err_free`` and ``free_content`` alone is
anatomy-neutral within an operator**: both quantities scale with anatomical density.

Dividing in the label makes the *label* depend on the confound, which is the thing a
reviewer can neither see nor adjust. Stratifying in the analysis leaves the label a plain
statement about error magnitude and puts the control where it is visible. That is the
standard move and the project arrived at it late.

## What a stratified table is for

Two different questions, and the table answers both at once:

* **Does the flag work at all?** The pooled AUC, which is what everything before this
  reported.
* **Does it work for the same reason everywhere?** The per-stratum AUCs. A flag that scores
  well pooled and poorly in every stratum is ranking anatomy, not invention -- the Q6
  failure mode, caught once already by :func:`caliber.scoring.attribution.stratified_auc`
  and now reported routinely rather than on request.

**Pooling across operators is not allowed here**, and that is the lesson of the retraction
in D62: the candidacy gate looked anatomy-neutral at 0.499 pooled and was 0.662 at R8,
because two opposite per-operator biases cancelled. Every number below is computed within
one operator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..scoring.attribution import rank_auc

#: Terciles. Three strata keep enough positives per cell on the thinnest groups the
#: project has (P-34's `ct/{family}/v{views}/a{arc}`) while still separating "little
#: anatomy" from "a lot", which is the contrast that matters.
DEFAULT_STRATA = 3

#: Fewest positives and negatives in a cell before its AUC is reported rather than left
#: blank. Matches `attribution.MIN_CLASS_FOR_AUC`, for the same reason: below a dozen the
#: standard error exceeds 0.1 and the cell contributes noise.
MIN_CLASS = 12


def _artefact_strength(content: np.ndarray, labels: np.ndarray) -> float:
    """How well anatomy density alone separates the label, in whichever direction it runs.

    ``max(auc, 1 - auc)``, because the direction of the confound depends on the criterion
    and must not be assumed: a relative label is predicted by *low* content, an absolute one
    by *high*. Reported as a strength so one number bounds both.
    """
    auc = rank_auc(content, labels)
    return float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan")


def detection_by_content(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "label",
    content_col: str = "free_content",
    operator_col: str = "operator_id",
    n_strata: int = DEFAULT_STRATA,
    min_class: int = MIN_CLASS,
) -> pd.DataFrame:
    """AUC of ``score_col`` against the label, per operator and per anatomy stratum.

    Strata are equal-count quantile bins of ``content_col`` **computed within each
    operator**, because the content distribution itself moves with the operator -- a larger
    null space hides more of the same image -- so global bins would mix operators inside
    one stratum and reintroduce exactly the pooling that D62 retracted.

    Returns one row per (operator, stratum) plus an ``all`` stratum row per operator, with
    ``anatomy_auc`` alongside: how well ``content_col`` alone separates the label in that
    cell, in whichever direction it runs, which is the artefact's own score. A detection AUC
    that does not clear its cell's ``anatomy_auc`` is not evidence of detection.
    """
    for col in (score_col, label_col, content_col):
        if col not in df.columns:
            raise KeyError(f"frame has no {col!r} column")

    rows: list[dict[str, object]] = []
    groups = (
        df.groupby(operator_col, observed=True) if operator_col in df.columns
        else [("all", df)]
    )
    for operator, frame in groups:
        content = frame[content_col].to_numpy(dtype=float)
        labels = frame[label_col].to_numpy().astype(bool)
        scores = frame[score_col].to_numpy(dtype=float)

        edges = np.quantile(content, np.linspace(0.0, 1.0, n_strata + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        cells: list[tuple[str, np.ndarray]] = [("all", np.ones(len(frame), dtype=bool))]
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            sel = (content > lo) & (content <= hi) if np.isfinite(lo) else (content <= hi)
            cells.append((f"q{i + 1}", sel))

        for name, sel in cells:
            n_pos, n_neg = int(labels[sel].sum()), int((~labels[sel]).sum())
            thin = n_pos < min_class or n_neg < min_class
            rows.append(
                {
                    "operator_id": str(operator),
                    "stratum": name,
                    "n": int(sel.sum()),
                    "n_positive": n_pos,
                    "content_median": float(np.median(content[sel])) if sel.any() else np.nan,
                    "auc": np.nan if thin else rank_auc(scores[sel], labels[sel]),
                    # Direction-agnostic: the ratio label's artefact runs low-anatomy and
                    # the absolute label's runs high-anatomy (D62 measured 0.90-0.94 one way
                    # and 0.05-0.42 the other), so a fixed sign would score one of them as
                    # 1 - the truth and let it through.
                    "anatomy_auc": np.nan if thin else _artefact_strength(
                        content[sel], labels[sel]
                    ),
                    "thin": thin,
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["beats_anatomy"] = out["auc"] > out["anatomy_auc"]
    return out


def stratification_verdict(table: pd.DataFrame, tolerance: float = 0.02) -> str:
    """One line on whether the flag survives holding anatomy fixed, per operator.

    The question is not whether the pooled AUC is high -- D62's retraction is that a pooled
    number can hide two cancelling biases -- but whether the score beats the anatomy
    artefact *inside* strata, in every operator. Reported as text because this belongs
    beside any table that quotes a detection number.
    """
    usable = table[~table["thin"] & (table["stratum"] != "all")]
    if usable.empty:
        return (
            "stratified verdict: UNDECIDABLE -- no (operator, stratum) cell has enough of "
            "both classes. Coarsen the strata or report pooled numbers with that stated."
        )
    per_operator = usable.groupby("operator_id")["beats_anatomy"].all()
    failed = sorted(per_operator[~per_operator].index)
    margin = float((usable["auc"] - usable["anatomy_auc"]).min())
    if not failed:
        return (
            f"stratified verdict: the score beats the anatomy artefact in every usable "
            f"stratum of every operator, by at least {margin:+.3f}. Detection is not an "
            f"anatomy ranking."
        )
    return (
        f"stratified verdict: the score FAILS to beat the anatomy artefact in at least one "
        f"stratum of {len(failed)} operator(s) ({', '.join(failed)}); worst margin "
        f"{margin:+.3f}. A pooled AUC from this table is not evidence of detection "
        f"(D62's retraction)."
    )
