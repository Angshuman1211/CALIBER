"""Does a score component fire on invention, or on anatomy that was always there? (Q6)

**Track 1 owns this module** (T1.1, open question Q6). It reads a cached score
bank and touches no GPU, no dataset and no solver — the Sec. 17 compute strategy
working as intended.

## The question, and why it is load-bearing

``s_free`` measures ``‖P_N x̂‖`` — the magnitude of the reconstruction's hidden
content. Real anatomy *has* large null-space content whether or not the prior
invented any of it, so a component that simply ranks regions by anatomical
detail would look useful while flagging texture rather than invention. D35
sharpened this: ``s_residual`` is structurally zero for any data-consistent
solver, so the score is effectively ``s_free`` + ``s_disagree``. If ``s_free``
also turns out to track anatomy, what is left is cross-sample disagreement — a
posterior-spread detector, which is a known baseline rather than this project's
contribution.

## The identity this module is built on

Every bank the runner writes normalises the three free-component quantities by
**one** reference intensity: ``runner.py`` passes ``components.ref_intensity``
into ``compute_region_truth``. And all three are region-wise root-mean-squares
of the *same* posterior mean pushed through the *same* projector. Writing
``a = P_N x̂`` and ``b = P_N x_true`` restricted to one region,

    s_free       = ‖a‖ / ref
    free_content = ‖b‖ / ref
    err_free     = ‖a − b‖ / ref

so, exactly and not approximately,

    err_free² = s_free² + free_content² − 2 · s_free · free_content · cos θ

with ``cos θ = Re⟨a, b⟩ / (‖a‖‖b‖)`` — how much of the reconstruction's hidden
content points the same way as the truth's. :func:`content_alignment` inverts
the identity to recover it, and :func:`identity_residual` checks the identity
holds on a bank before anything is read off it, because if it does not, the
three columns were not written by one consistent normalisation and every number
below is meaningless.

Two consequences fall out of the algebra alone, before any data:

**1. The relative label is a function of one ratio.** With ``u = s_free /
free_content``, the criterion ``err_free / free_content > τ`` (D21, D37) becomes
``u² + 1 − 2u·cos θ > τ²``. Holding ``cos θ`` roughly fixed, the label is
monotone in ``u`` — so the statistic that matches the label is ``s_free /
free_content``, *not* ``s_free``. ``free_content`` needs ground truth and is
unavailable at deployment, so a flag built on ``s_free`` alone is structurally
mismatched to its own label by exactly as much as ``free_content`` varies
between regions. That is Q6's mechanism, stated without reference to any
measurement. :func:`attribute_components` measures the size of the effect by
scoring the oracle ratio alongside the raw component.

**2. D37's falsifiable check has a precondition.** ``err_free > free_content``
in every region of the M4Raw bank was read as a mismatched prior (D-034, D37),
with "some regions should show ratio < 1" as the pass condition for a medical
prior. The algebra says ``ratio > 1 ⟺ cos θ < u/2``. A prior that *over-fills*
the null space (``u > 2``) therefore cannot produce ``ratio < 1`` however well
aligned it is. The check is only diagnostic where ``u < 2``, which is itself
measurable — :func:`fill_ratio` reports it.

## What "power" means here, and why it is the wrong axis

Conformal calibration pins the miss rate at ``α`` for every score it is given.
Power is therefore approximately ``1 − α`` by construction, and comparing
components on power compares them on how well the calibrator worked. What a
better score buys is a **smaller flagged area at the same reliability** —
Sec. 15.2's second metric. :func:`component_ablation` reports flag area and FDR
at a fixed ``α``, and Q6's per-component requirement should be read that way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..calibration.calibrators import apply_flags, calibrate_mondrian
from ..evaluation.metrics import flag_area, miss_rate, region_fdr, region_power

# -- pre-registered thresholds -------------------------------------------------
#
# These are fixed here, before the real bank is read, so Q6's verdict cannot be
# tuned to whatever the data turns out to say. Each has a reason.

#: Fewest positives (and negatives) in a stratum before its AUC is trusted. On
#: the 384-region development bank at tau=1.75 there are ~98 positives, so four
#: strata give ~24 each; below a dozen the Hanley-McNeil standard error exceeds
#: 0.1 and a stratum contributes noise rather than evidence.
MIN_CLASS_FOR_AUC = 12

#: An AUC this far above chance is doing real ranking work. At n_pos ~ 100,
#: n_neg ~ 290 the standard error is ~0.03, so 0.60 is ~3 SE from chance while
#: 0.55 is under 2 and cannot be distinguished from it on a bank this size.
AUC_INFORMATIVE = 0.60
AUC_NULL = 0.55

#: Partial Spearman of the component against the invention magnitude, with
#: anatomical content held fixed. Modest on purpose: the claim being tested is
#: "not zero", not "large".
PARTIAL_RHO_MIN = 0.20

#: Above this, a component is substantially an anatomy meter, whatever else it
#: also does.
ANATOMY_RHO_HIGH = 0.70

#: How far a component's detail-matched AUC must exceed the *anatomy control's
#: own* detail-matched AUC to count as carrying information the control does not.
#:
#: This paired comparison is the load-bearing instrument, and it is here because
#: the obvious one is not enough. Stratifying on a continuous confound only
#: attenuates it: with four quantile bins a component that is an exact monotone
#: function of ``free_content`` still scored 0.87 detail-matched on the
#: development control, which an absolute threshold reads as a strong detector.
#: Scoring ``free_content`` itself through the identical path gives 0.87 too, and
#: the difference — zero — is the honest answer. Both statistics are attenuated
#: by the same stratification, so the difference survives it.
#:
#: 0.05 rather than the 0.03 single-AUC standard error because the comparison is
#: paired on the same regions and the same labels, so the difference has far less
#: variance than either AUC alone.
AUC_EXCESS_MIN = 0.05

#: Below this, a component and the anatomy control are reported as tied.
AUC_EXCESS_TIE = 0.02

#: Fewest surviving strata for a confident detail-matched reading. One stratum
#: out of four is not a weaker version of the same evidence: it means the control
#: variable nearly determines the label, so the other strata had only one class
#: in them — which is itself a sign of confounding.
MIN_STRATA_USED = 2

#: Two candidate scores whose flagged areas differ by less than this factor are
#: reported as tied. 2% of the flagged area is well inside what re-splitting the
#: calibration set moves, so a smaller difference is not a reason to prefer one.
FLAG_AREA_TIE = 1.02

#: Strata for the detail-matched comparison. Four keeps ~24 positives per
#: stratum on the development bank; more strata is a finer control but a thinner
#: one, and this is the trade recorded rather than tuned.
DEFAULT_STRATA = 4

#: Denominators below this are not divided into. `free_content` is already
#: guarded at 1e-3 in `labels.MIN_FREE_CONTENT`; this is the same guard for the
#: products that appear in the identity.
EPS = 1e-12

CANONICAL_COMPONENTS = ("s_free", "s_disagree", "s_residual")


# -- rank statistics, in numpy so this module adds no dependency ---------------


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties averaged — the tie correction both stats below need."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _finite_pair(a, b) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep]


def rank_auc(scores, labels) -> float:
    """Mann-Whitney AUC: P(score of a hallucinated region > score of a clean one).

    Tie-corrected, so a component with many equal values (a structurally dead
    one, D35) returns 0.5 rather than an artefact of the sort order. NaN when
    either class is empty — the caller must decide what that means rather than
    receiving a number that looks like chance.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    keep = np.isfinite(scores)
    scores, labels = scores[keep], labels[keep]
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auc_standard_error(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil standard error, so no AUC in a table is read without its width."""
    if not math.isfinite(auc) or n_pos < 1 or n_neg < 1:
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc**2 / (1.0 + auc)
    var = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc**2)
        + (n_neg - 1) * (q2 - auc**2)
    ) / (n_pos * n_neg)
    return float(math.sqrt(max(var, 0.0)))


def spearman(a, b) -> float:
    """Rank correlation. Monotone-invariant, which matters because the score's
    units are arbitrary after standardisation (D26)."""
    a, b = _finite_pair(a, b)
    if a.size < 3:
        return float("nan")
    ra, rb = _average_ranks(a), _average_ranks(b)
    sa, sb = ra.std(), rb.std()
    if sa < EPS or sb < EPS:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def partial_spearman(a, b, control) -> float:
    """Rank correlation of ``a`` and ``b`` with ``control`` linearly removed.

    Computed on ranks, so it asks whether the component still tracks invention
    among regions of *comparable* anatomical content. This is the continuous
    counterpart of :func:`stratified_auc`; the two are reported together because
    they fail differently — the partial correlation assumes the confound acts
    linearly on ranks, the stratified AUC assumes nothing but needs enough
    regions per stratum.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    c = np.asarray(control, dtype=float).ravel()
    keep = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    a, b, c = a[keep], b[keep], c[keep]
    if a.size < 4:
        return float("nan")
    ra, rb, rc = _average_ranks(a), _average_ranks(b), _average_ranks(c)

    def residual(v: np.ndarray) -> np.ndarray:
        vc = v - v.mean()
        cc = rc - rc.mean()
        denom = float((cc * cc).sum())
        if denom < EPS:
            return vc
        return vc - cc * float((vc * cc).sum()) / denom

    res_a, res_b = residual(ra), residual(rb)
    sa, sb = res_a.std(), res_b.std()
    if sa < EPS or sb < EPS:
        return float("nan")
    return float((res_a * res_b).mean() / (sa * sb))


# -- the identity --------------------------------------------------------------


def fill_ratio(s_free, free_content, min_content: float = 1e-3) -> np.ndarray:
    """``u = s_free / free_content`` — hidden content produced over hidden content present.

    ``u < 1`` means the prior put *less* into the free component than the truth
    had there (a smoothing prior); ``u > 2`` means it over-filled so badly that
    ``err_free / free_content > 1`` is forced regardless of alignment, which is
    the precondition D37's pass/fail check needs and did not state.
    """
    s = np.asarray(s_free, dtype=float)
    c = np.asarray(free_content, dtype=float)
    return np.where(c > min_content, s / np.maximum(c, min_content), np.nan)


def content_alignment(s_free, free_content, err_free, min_content: float = 1e-3) -> np.ndarray:
    """``cos θ`` between the reconstruction's hidden content and the truth's.

    Inverted from the identity in the module docstring. The interpretation is
    the whole reason to compute it:

    * ``cos θ → 1`` — the prior recovered the hidden anatomy. Content in the
      free component is *reconstruction*, not invention, and a flag that fires
      on it is a false positive by the project's own definition (Sec. 4).
    * ``cos θ → 0`` — the prior's hidden content is orthogonal to the truth's:
      invention, in the sense the project means.
    * ``cos θ < 0`` — it invented structure where the truth had the opposite.

    Values outside ``[-1, 1]`` are returned as NaN rather than clipped. They
    cannot occur if the three columns share one normalisation, so they are a
    signal that the bank does not satisfy the identity and must be investigated,
    not a rounding nuisance to be hidden.
    """
    s = np.asarray(s_free, dtype=float)
    c = np.asarray(free_content, dtype=float)
    e = np.asarray(err_free, dtype=float)
    denom = 2.0 * s * c
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = (s**2 + c**2 - e**2) / denom
    usable = (c > min_content) & (s > min_content) & np.isfinite(cos)
    cos = np.where(usable, cos, np.nan)
    return np.where(np.abs(cos) <= 1.0 + 1e-4, np.clip(cos, -1.0, 1.0), np.nan)


def identity_residual(s_free, free_content, err_free) -> dict[str, float]:
    """How nearly the bank satisfies the identity, as a relative error.

    Run this before reading anything else off a bank. A large residual means the
    columns were not written by one consistent normalisation — for instance a
    bank whose truth quantities were scaled by ``mean|x_true|`` while its score
    components were scaled by ``mean|A⁺y|`` — and in that case every ratio in
    this module compares quantities in different units.

    The returned residual is defined only where ``cos θ`` is recoverable, and is
    reported as a fraction of ``err_free²``, because the identity is exact.
    """
    s = np.asarray(s_free, dtype=float)
    c = np.asarray(free_content, dtype=float)
    e = np.asarray(err_free, dtype=float)
    cos = content_alignment(s, c, e)
    ok = np.isfinite(cos)
    if not ok.any():
        return {"n": 0.0, "max_relative": float("nan"), "median_relative": float("nan")}
    lhs = e[ok] ** 2
    rhs = s[ok] ** 2 + c[ok] ** 2 - 2.0 * s[ok] * c[ok] * cos[ok]
    rel = np.abs(lhs - rhs) / np.maximum(lhs, EPS)
    return {
        "n": float(ok.sum()),
        "max_relative": float(rel.max()),
        "median_relative": float(np.median(rel)),
    }


# -- a label on direction rather than magnitude (D74) -------------------------

#: Design-split quantile of ``cos θ`` below which a region counts as invention.
#:
#: 0.25 is pre-registered, not fitted: the least-aligned quarter of candidate regions.
#: Any quantile makes the positive rate known in advance, which is what P-02's fixed
#: constant failed to do (it flagged every region).
ALIGNMENT_QUANTILE = 0.25

#: Below this spread in ``cos θ`` on the design split, direction carries no information
#: and a quantile cut on it is slicing noise. Measured for context: the M4Raw bank's
#: ``cos θ`` has sd 0.089 over a range of [-0.34, +0.27], which clears this by a long way.
MIN_ALIGNMENT_SPREAD = 0.01


@dataclass(frozen=True)
class MisalignmentLabel:
    """A hallucination label built on ``cos θ``, with what is needed to distrust it."""

    label: np.ndarray
    cos_theta: np.ndarray
    threshold: float
    n_total: int
    n_defined: int
    n_positive: int
    spread: float

    @property
    def positive_rate(self) -> float:
        return float(self.n_positive / self.n_total) if self.n_total else float("nan")

    @property
    def defined_fraction(self) -> float:
        return float(self.n_defined / self.n_total) if self.n_total else float("nan")

    @property
    def degenerate(self) -> bool:
        """Is ``cos θ`` too flat for a cut on it to mean anything?"""
        return not (np.isfinite(self.spread) and self.spread >= MIN_ALIGNMENT_SPREAD)

    def summary(self) -> str:
        return (
            f"cos theta < {self.threshold:.4f} (q={ALIGNMENT_QUANTILE:g} on the design "
            f"split, D74); {self.n_positive} positives at rate {self.positive_rate:.3f}; "
            f"cos theta defined on {self.defined_fraction:.1%} of regions, sd {self.spread:.4f}"
        )


def label_misaligned(
    s_free,
    free_content,
    err_free,
    threshold: float,
    min_content: float = 1e-3,
) -> MisalignmentLabel:
    """Label a region hallucinated when its hidden content points *away* from the truth's.

    Every earlier form of this label thresholded a **magnitude** -- ``err_free``, or
    ``err_free / free_content`` -- and D67 showed why that cannot work on this bank. By the
    identity, at ``cos θ ≈ 0``, ``err_free ≈ sqrt(s_free² + free_content²)``, so a
    magnitude label is a function of the score's own dominant component: measured here,
    ``spearman(err_free, hypot(s_free, free_content)) = +0.971``. The 0.95-1.00 AUROCs
    were the identity being measured, not a detector.

    ``cos θ`` is the other coordinate of the same decomposition, and it is **scale-free**:
    it says which way the invented content points, not how much of it there is. On the
    M4Raw bank ``spearman(cos θ, s_free) = -0.435``, against the magnitude label's 0.971.

    **The score is not independent of this label either**, and the honest claim is only
    that the dependence is moderate rather than algebraic. Nothing built from
    ``(s_free, free_content, err_free)`` can be independent of a score built from
    ``s_free``, because the identity ties all three. So an AUROC against this label still
    has to clear the paired anatomy control (:func:`stratified_auc`,
    :data:`AUC_EXCESS_MIN`) before it is a result. What changes is that it *can* now fail
    that check, where against a magnitude label it could not.

    **Do not restore the magnitude.** A label like ``s_free · sin θ`` -- invented mass
    rather than invented direction -- looks more physical and is a trap: ``sin θ ≈ 1``
    wherever ``cos θ ≈ 0``, which is almost everywhere on this bank, so the product
    collapses back to ``s_free`` and the tautology returns intact.

    Regions where ``cos θ`` is undefined (either norm below ``min_content``, or the bank
    failing the identity) are labelled **negative** and counted. They are not dropped:
    dropping them would make the label's support depend on the score's magnitude, which
    is the same confound by another route.
    """
    cos = content_alignment(s_free, free_content, err_free, min_content=min_content)
    defined = np.isfinite(cos)
    label = defined & (cos < threshold)
    spread = float(np.std(cos[defined])) if defined.any() else float("nan")
    return MisalignmentLabel(
        label=label,
        cos_theta=cos,
        threshold=float(threshold),
        n_total=int(cos.size),
        n_defined=int(defined.sum()),
        n_positive=int(label.sum()),
        spread=spread,
    )


def alignment_threshold(
    s_free, free_content, err_free, quantile: float = ALIGNMENT_QUANTILE
) -> float:
    """The ``cos θ`` cut, valued on whatever rows are passed -- **design rows only**.

    Fitting it on calibration or test rows would let the label see the patients whose
    coverage it is about to score, which is the in-sample shape behind P-19 and P-23.
    Callers pass the design split; :func:`caliber.evaluation.label_spec` enforces that.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    cos = content_alignment(s_free, free_content, err_free)
    defined = cos[np.isfinite(cos)]
    if defined.size == 0:
        raise ValueError(
            "cos theta is undefined on every design region, so no alignment threshold "
            "can be set; check identity_residual() on this bank first"
        )
    return float(np.quantile(defined, quantile))


# -- detail-matched comparison -------------------------------------------------


def within_cell_rank(df: pd.DataFrame, col: str, cell_cols: tuple[str, ...]) -> pd.Series:
    """Percentile rank of ``col`` within each acquisition/solver cell.

    Scores are not comparable across operators: the scales are frozen on the
    calibration operator precisely so the distributions are *allowed* to differ
    (D27), and that difference is the headline result. Pooling raw scores across
    operators would let operator-level shifts dominate every statistic here.
    Ranking within cell first removes that, and leaves the within-cell ordering
    — which is all an AUC ever uses — untouched.
    """
    present = [c for c in cell_cols if c in df.columns]
    if not present:
        return df[col].rank(pct=True)
    return df.groupby(present, dropna=False, observed=True)[col].rank(pct=True)


@dataclass(frozen=True)
class StratifiedAUC:
    """AUC within strata of a control variable, pooled over the usable strata."""

    value: float
    strata_used: int
    strata_requested: int
    per_stratum: tuple[float, ...]
    n_positive: int

    @property
    def usable(self) -> bool:
        return self.strata_used > 0 and math.isfinite(self.value)


def stratified_auc(
    scores,
    labels,
    control,
    n_strata: int = DEFAULT_STRATA,
    min_class: int = MIN_CLASS_FOR_AUC,
) -> StratifiedAUC:
    """AUC of ``scores`` against ``labels`` among regions of similar ``control``.

    This is Q6's decisive test. If a component separates hallucinated from clean
    regions only because it ranks regions by anatomical content, then holding
    that content approximately fixed destroys the separation and the
    within-stratum AUC collapses to chance. If the separation survives, the
    component is responding to something other than how much anatomy is present.

    Strata are equal-count quantile bins of ``control``. Strata without
    ``min_class`` of both classes are dropped rather than contributing an
    unstable estimate, and how many were dropped is returned, because a
    stratified AUC computed on one surviving stratum is not the same evidence as
    one computed on four.

    Pooling is weighted by positives, matching how the project aggregates
    region-level risk elsewhere (a region counts once, wherever it sits).
    """
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    control = np.asarray(control, dtype=float).ravel()
    keep = np.isfinite(scores) & np.isfinite(control)
    scores, labels, control = scores[keep], labels[keep], control[keep]

    if scores.size == 0:
        return StratifiedAUC(float("nan"), 0, n_strata, (), 0)

    edges = np.quantile(control, np.linspace(0.0, 1.0, n_strata + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    aucs: list[float] = []
    weights: list[int] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (control > lo) & (control <= hi) if np.isfinite(lo) else (control <= hi)
        if not sel.any():
            continue
        n_pos, n_neg = int(labels[sel].sum()), int((~labels[sel]).sum())
        if n_pos < min_class or n_neg < min_class:
            continue
        auc = rank_auc(scores[sel], labels[sel])
        if math.isfinite(auc):
            aucs.append(auc)
            weights.append(n_pos)

    if not aucs:
        return StratifiedAUC(float("nan"), 0, n_strata, (), 0)
    w = np.asarray(weights, dtype=float)
    pooled = float((np.asarray(aucs) * w).sum() / w.sum())
    return StratifiedAUC(pooled, len(aucs), n_strata, tuple(aucs), int(w.sum()))


# -- per-component attribution -------------------------------------------------


@dataclass(frozen=True)
class ComponentAttribution:
    """What one candidate score column responds to, and the verdict that follows."""

    component: str
    n: int
    n_positive: int
    auc: float
    auc_se: float
    auc_stratified: float
    strata_used: int
    auc_excess: float
    rho_anatomy: float
    rho_invention: float
    rho_invention_given_anatomy: float
    verdict: str
    reason: str

    def as_row(self) -> dict[str, object]:
        return {
            "component": self.component,
            "n": self.n,
            "n_positive": self.n_positive,
            "auc": self.auc,
            "auc_se": self.auc_se,
            "auc_detail_matched": self.auc_stratified,
            "auc_over_anatomy": self.auc_excess,
            "strata_used": self.strata_used,
            "rho_anatomy": self.rho_anatomy,
            "rho_invention": self.rho_invention,
            "rho_invention_given_anatomy": self.rho_invention_given_anatomy,
            "verdict": self.verdict,
            "reason": self.reason,
        }


def classify_component(
    auc: float,
    auc_stratified: float,
    auc_excess: float,
    rho_anatomy: float,
    partial_rho: float,
    strata_used: int,
) -> tuple[str, str]:
    """Map the measured statistics onto one of five verdicts, by a fixed rule.

    The rule is pre-registered in this module's constants rather than chosen
    after seeing the numbers, which is the same discipline D37 applied to ``τ``
    and D28 to the weights: the threshold is set on reasoning, then applied.

    * ``undecidable`` — the detail-matched comparison had no usable stratum, or
      a required statistic is undefined. Not a result.
    * ``uninformative`` — it does not separate either way.
    * ``invention-driven`` — separation survives holding anatomical content
      fixed *and* exceeds what the anatomy control itself achieves. The
      component is doing what the score design claims.
    * ``anatomy-confounded`` — it separates no better than a meter of how much
      anatomy is present. This is Q6's feared outcome.
    * ``ambiguous`` — reported as-is rather than forced into a box.

    The order matters. Chance-level separation is checked first, because a pure
    noise component also fails to beat the anatomy control and would otherwise be
    mislabelled as confounded when in truth it measures nothing at all.
    """
    if strata_used == 0 or not math.isfinite(auc) or not math.isfinite(auc_stratified):
        return "undecidable", "no usable detail-matched stratum, or a statistic is undefined"
    if auc < AUC_NULL and auc_stratified < AUC_NULL:
        return (
            "uninformative",
            f"AUC {auc:.3f} and detail-matched {auc_stratified:.3f} are both within noise "
            f"of chance",
        )
    if (
        auc_stratified >= AUC_INFORMATIVE
        and strata_used >= MIN_STRATA_USED
        and math.isfinite(auc_excess)
        and auc_excess >= AUC_EXCESS_MIN
        and math.isfinite(partial_rho)
        and partial_rho >= PARTIAL_RHO_MIN
    ):
        return (
            "invention-driven",
            f"detail-matched AUC {auc_stratified:.3f} over {strata_used} strata, "
            f"{auc_excess:+.3f} above the anatomy control, partial rho {partial_rho:.3f}",
        )
    if auc >= AUC_INFORMATIVE and auc_stratified < AUC_NULL:
        return (
            "anatomy-confounded",
            f"AUC {auc:.3f} collapses to {auc_stratified:.3f} once anatomical content is "
            f"held fixed",
        )
    if (
        math.isfinite(auc_excess)
        and auc_excess <= AUC_EXCESS_TIE
        and (not math.isfinite(partial_rho) or partial_rho < PARTIAL_RHO_MIN)
    ):
        return (
            "anatomy-confounded",
            f"detail-matched AUC {auc_stratified:.3f} is only {auc_excess:+.3f} against the "
            f"anatomy control's own {auc_stratified - auc_excess:.3f}, and the partial rho "
            f"against invention is {partial_rho:.3f}"
            + (
                f" (only {strata_used} of the strata survived)"
                if strata_used < MIN_STRATA_USED
                else ""
            ),
        )
    if (
        math.isfinite(rho_anatomy)
        and rho_anatomy >= ANATOMY_RHO_HIGH
        and math.isfinite(partial_rho)
        and partial_rho < PARTIAL_RHO_MIN
    ):
        return (
            "anatomy-confounded",
            f"rho(component, anatomy) {rho_anatomy:.3f} with partial rho against invention "
            f"only {partial_rho:.3f}",
        )
    return (
        "ambiguous",
        f"AUC {auc:.3f}, detail-matched {auc_stratified:.3f} ({auc_excess:+.3f} over the "
        f"anatomy control, {strata_used} strata), partial rho {partial_rho:.3f} -- no verdict "
        f"at the pre-registered thresholds",
    )


def attribute_components(
    df: pd.DataFrame,
    component_cols: tuple[str, ...] = CANONICAL_COMPONENTS,
    label_col: str = "label",
    anatomy_col: str = "free_content",
    invention_col: str = "err_free",
    cell_cols: tuple[str, ...] = ("operator_id", "solver_id"),
    n_strata: int = DEFAULT_STRATA,
) -> pd.DataFrame:
    """One row per candidate score column: what it tracks, and Q6's verdict.

    ``anatomy_col`` is the ground truth's *own* null-space content — the A1
    column, added for exactly this purpose (D21). ``invention_col`` is the
    free-component error. Both need ground truth, which is why they are the
    control and the target here and can never appear in a deployed score.

    Any column in ``component_cols`` that is missing is skipped silently, so a
    caller can pass a superset. A missing ``anatomy_col`` raises: a bank without
    it cannot answer Q6 at all, and silently answering a weaker question is how
    a confound survives into a manuscript.
    """
    for required in (label_col, invention_col):
        if required not in df.columns:
            raise KeyError(f"bank has no {required!r} column")
    if anatomy_col not in df.columns:
        raise KeyError(
            f"bank has no {anatomy_col!r} column -- Q6 needs the A1 column (D21); "
            "regenerate the bank with a runner that writes it"
        )
    if not np.isfinite(df[anatomy_col].to_numpy(dtype=float)).any():
        raise ValueError(
            f"{anatomy_col!r} is entirely null -- the A1 column is nullable and this bank "
            "predates it (or was coerced); Q6 cannot be answered from it"
        )

    labels = df[label_col].to_numpy().astype(bool)
    anatomy_rank = within_cell_rank(df, anatomy_col, cell_cols).to_numpy()
    rows: list[dict[str, object]] = []

    # The anatomy control scored through the identical path, as the reference
    # every component is measured against. A component that cannot beat "how
    # much anatomy is in this region" is not detecting invention, however good
    # its AUC looks in isolation.
    anatomy_reference = stratified_auc(
        anatomy_rank, labels, anatomy_rank, n_strata=n_strata
    ).value

    for col in component_cols:
        if col not in df.columns:
            continue
        score_rank = within_cell_rank(df, col, cell_cols).to_numpy()
        auc = rank_auc(score_rank, labels)
        n_pos, n_neg = int(labels.sum()), int((~labels).sum())
        strat = stratified_auc(score_rank, labels, anatomy_rank, n_strata=n_strata)
        rho_anatomy = spearman(df[col].to_numpy(), df[anatomy_col].to_numpy())
        rho_invention = spearman(df[col].to_numpy(), df[invention_col].to_numpy())
        partial = partial_spearman(
            df[col].to_numpy(), df[invention_col].to_numpy(), df[anatomy_col].to_numpy()
        )
        excess = strat.value - anatomy_reference
        verdict, reason = classify_component(
            auc, strat.value, excess, rho_anatomy, partial, strat.strata_used
        )
        rows.append(
            ComponentAttribution(
                component=col,
                n=int(len(df)),
                n_positive=n_pos,
                auc=auc,
                auc_se=auc_standard_error(auc, n_pos, n_neg),
                auc_stratified=strat.value,
                strata_used=strat.strata_used,
                auc_excess=excess,
                rho_anatomy=rho_anatomy,
                rho_invention=rho_invention,
                rho_invention_given_anatomy=partial,
                verdict=verdict,
                reason=reason,
            ).as_row()
        )
    return pd.DataFrame(rows)


# -- the ablation, on the axis that can actually move --------------------------


def component_ablation(
    df: pd.DataFrame,
    candidates: dict[str, str] | None = None,
    label_col: str = "label",
    group_col: str = "calib_group",
    alpha: float = 0.10,
    split_col: str = "split",
) -> pd.DataFrame:
    """Flag area and FDR at fixed ``α`` for each candidate score, via Mondrian.

    Q6 asks for per-component FDR and power. Power is the wrong axis and this
    function does not pretend otherwise: a conformal threshold pins the miss
    rate at ``α``, so every candidate's power is ``≈ 1 − α`` whatever it
    measures. Reporting power per component would compare calibrators, not
    scores. What a better score buys is **less flagged area for the same
    reliability**, which is Sec. 15.2's own second metric, so that is the
    comparison here. Power and miss rate are still returned, as the check that
    calibration did in fact hold — not as the comparison.

    Group-conditional calibration is used rather than a single pooled threshold
    so that coverage is equalised *within* each acquisition regime and the areas
    being compared are like for like (D34's lesson, applied to an ablation
    rather than to a baseline).

    ``reportable`` is ``False`` when any group was too thin to certify at
    ``α`` (D32). A row with ``reportable=False`` names an area that is not a
    guarantee, and the D2 tie test below refuses to read it.
    """
    candidates = candidates or {c: c for c in CANONICAL_COMPONENTS if c in df.columns}
    has_split = split_col in df.columns and df[split_col].nunique() > 1
    rows: list[dict[str, object]] = []

    for name, col in candidates.items():
        if col not in df.columns:
            continue
        work = df.rename(columns={col: "__score"})
        report = calibrate_mondrian(
            work, score_col="__score", label_col=label_col, group_col=group_col,
            alpha=alpha, split_col=split_col,
        )
        evaluate = work[work[split_col] == "test"] if has_split else work
        flagged = apply_flags(evaluate, report, score_col="__score", group_col=group_col)
        labels = flagged[label_col].to_numpy().astype(bool)
        flags = flagged["flag"].to_numpy().astype(bool)
        rows.append(
            {
                "candidate": name,
                "column": col,
                "n_evaluated": int(len(flagged)),
                "flag_area": flag_area(flags),
                "fdr": region_fdr(flags, labels),
                "power": region_power(flags, labels),
                "miss_rate": miss_rate(flags, labels),
                "in_sample": not has_split,
                "reportable": report.reportable,
                "uncertifiable_groups": ",".join(report.uncertifiable_groups),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("flag_area").reset_index(drop=True) if len(out) else out


# -- the overall reading -------------------------------------------------------


@dataclass(frozen=True)
class Q6Verdict:
    """Q6's answer for one bank, with the two findings that can invalidate it."""

    verdict: str
    reason: str
    free_component: str
    decomposition_earns_its_place: bool | None
    notes: tuple[str, ...] = ()

    def summary(self) -> str:
        lines = [f"Q6: {self.verdict} -- {self.reason}"]
        if self.decomposition_earns_its_place is False:
            lines.append(
                "  D2: cross-sample disagreement alone matches the combination's flagged "
                "area. The operator decomposition is not earning its place on this bank."
            )
        elif self.decomposition_earns_its_place is True:
            lines.append(
                "  D2: the combination flags less area than disagreement alone, so the "
                "free-component decomposition is contributing."
            )
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def q6_verdict(
    attribution: pd.DataFrame,
    ablation: pd.DataFrame | None = None,
    free_component: str = "s_free",
    disagree_component: str = "s_disagree",
    combination: str = "score",
) -> Q6Verdict:
    """Read Q6's answer off the two tables, by the pre-registered rule.

    Two independent things are decided. First, what ``s_free`` responds to —
    that is Q6 as written. Second, whether the combination buys anything over
    ``s_disagree`` alone at fixed reliability — D2's honest-negative check,
    which can fail even when ``s_free`` is invention-driven, and which is the
    outcome Q6 flagged as publishable rather than fatal.
    """
    notes: list[str] = []
    rows = attribution.set_index("component") if "component" in attribution else attribution
    if free_component not in rows.index:
        return Q6Verdict(
            "undecidable", f"{free_component!r} not in the attribution table",
            free_component, None,
        )
    row = rows.loc[free_component]
    verdict, reason = str(row["verdict"]), str(row["reason"])

    earns: bool | None = None
    if ablation is not None and len(ablation):
        area = ablation.set_index("candidate")["flag_area"].to_dict()
        report_ok = ablation.set_index("candidate")["reportable"].to_dict()
        if combination in area and disagree_component in area:
            if not (report_ok.get(combination) and report_ok.get(disagree_component)):
                notes.append(
                    "the D2 tie test was skipped: at least one candidate had a group too "
                    "thin to certify at this alpha (D32), so its flagged area is not a "
                    "guarantee and the comparison would be between a bound and a number"
                )
            else:
                earns = area[combination] < area[disagree_component] / FLAG_AREA_TIE
                if not earns:
                    notes.append(
                        f"flagged area: combination {area[combination]:.3f} vs disagreement "
                        f"alone {area[disagree_component]:.3f} -- within the "
                        f"{FLAG_AREA_TIE:g}x tie band or worse"
                    )
    return Q6Verdict(verdict, reason, free_component, earns, tuple(notes))
