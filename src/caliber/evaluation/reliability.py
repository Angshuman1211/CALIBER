"""T1-D: what a deployment certificate needs, and whether the splits behave as theory says.

Average coverage over a grid of operators is not what anyone deploys against. A radiologist
runs one scanner with one protocol, and the number that matters is the coverage at the
*worst* cell they might land in. This module computes two things that are cheap, fully
determined by the data, and hard to argue with:

**Worst-operator coverage.** The minimum over cells, reported with a *simultaneous* lower
bound rather than a pointwise interval. The distinction is the whole point: the observed
minimum of many noisy estimates is biased downwards, so quoting it as "the worst case" is
pessimistic in a way that is not a guarantee, while quoting a pointwise 95% interval on the
cell that happened to come last is anti-conservative -- that cell was selected *because* it
was low. :func:`worst_operator_coverage` therefore reports a Bonferroni-corrected lower
bound that holds across all cells at once, which is the only one of the three numbers a
certificate can be written from.

**And that bound is in units of the design effect, which must be passed in (D98).** For six
days this function fed Clopper-Pearson the raw region counts. A Clopper-Pearson bound is a
Beta band -- the same object under its other name -- so it was the thing
``prereg.BETA_REGION_COUNT_USE`` already forbade as a certificate, and the two names are why
that went unnoticed. On the CT bank the design effects in the same frames are 47 to 186, and
the shipped bound was too high by 0.20 to 0.38. ``design_effect_col`` now has no default and
the function refuses to guess, exactly as ``TransferBetaReference.interval`` refuses without
a unit, and for the same reason.

**Calibration-conditional reliability.** Split conformal does not promise 0.90 on every
calibration draw; it promises 0.90 on average over draws, and the spread around it is known
exactly. With ``n`` positive calibration scores and ``k = floor(alpha(n+1))``, the threshold
is the ``k``-th smallest, and coverage conditional on the calibration set is
``Beta(n + 1 - k, k)``. So the split-to-split spread is a prediction, not a free parameter,
and a method sitting off its own Beta law is mis-calibrated in a way a single average hides.

**The closed-form Beta law is the wrong yardstick for a measured spread, and it errs in
the dangerous direction.** This was checked rather than assumed, and the first guess was
wrong. Beta(n+1-k, k) is the law of coverage against the *population* -- the probability
that a fresh positive is caught -- which is the right object for a guarantee. What a split
actually reports is an empirical average over a *finite* test set, so its variance carries a
second term:

    Var(measured) ~= Var_Beta(threshold) + p(1 - p) / n_test

Simulation through this study's own split structure confirms the formula to within about 4%
across a fiftyfold range of sizes. At the project's balanced splits -- calibration and test
sets of similar size -- the measured spread runs about **1.36x the closed-form Beta SD**. So
comparing the two directly makes a *correctly calibrated* method look over-dispersed, and
would invite a repair to a method that has nothing wrong with it.

The overlap between splits matters too, and much less than expected. Thirteen re-cuts of
thirty patients share about a third of their calibration sets pairwise, which does bias the
sample variance downwards -- but by 2%, against the 36% inflation above. It is reported
because it is real, not because it changes a conclusion.

:func:`simulate_split_spread` is therefore the reference the verdict is taken from: it draws
exchangeable scores through the actual split membership and so carries both effects without
needing either to be modelled. :attr:`SpreadCheck.beta_sd_corrected` gives the closed-form
approximation beside it, and :attr:`SpreadCheck.ratio_to_beta` is kept only to show how far
off the uncorrected comparison is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from caliber.evaluation import prereg

#: Family-wise level for the simultaneous lower bound. One-sided: a certificate cares only
#: about coverage being too low.
FAMILY_WISE_ALPHA = 0.05

#: A spread check counts as informative only if its reference interval is tight enough to
#: rule out a mis-calibration of this factor.
#:
#: The sample SD of ``n`` split coverages has relative noise about ``1/sqrt(2(n-1))``, so at
#: the project's 13 splits the reference interval spans roughly 0.62x to 1.49x the correct
#: SD -- the check cannot distinguish correct calibration from a 36% error, which is the
#: size of the very correction this module exists to make. More splits barely help: 120 of
#: them still leave a 1.27x band, because the splits share patients and the estimator's
#: noise falls as the square root. A verdict of "consistent" from an interval that wide is
#: low power, not evidence, and :attr:`SpreadCheck.informative` is what keeps the two apart.
INFORMATIVE_BAND = 1.25

#: Below this many positives a cell's Beta reference is not reported. ``split_threshold``
#: already refuses to certify below ``k = 1``; this is the stricter reporting floor, because
#: a Beta(9, 1) reference has a 1st percentile near 0.6 and comparing anything against it is
#: theatre.
MIN_POSITIVES_FOR_REFERENCE = 30


def _beta_ppf(a: float, b: float, q: float) -> float:
    """Beta quantile. scipy if present, else a bisection on the regularised incomplete beta.

    The fallback exists because this module is imported by the matched-operator control,
    which has to run on the Track 2 partner's machine, and a missing scipy there should not
    stop a bank from being checked.
    """
    try:
        from scipy.stats import beta as _beta  # noqa: PLC0415

        return float(_beta.ppf(q, a, b))
    except Exception:  # pragma: no cover - exercised only where scipy is absent
        from math import lgamma  # noqa: PLC0415

        def cdf(x: float) -> float:
            # Series for I_x(a, b); adequate at the moderate a, b this module uses.
            if x <= 0.0:
                return 0.0
            if x >= 1.0:
                return 1.0
            total, term = 0.0, 1.0
            log_front = lgamma(a + b) - lgamma(a) - lgamma(b) + a * np.log(x) + b * np.log1p(-x)
            front = float(np.exp(log_front))
            for j in range(0, 4000):
                total += term
                term *= (a + b + j) * x / (a + 1 + j)
                if abs(term) < 1e-14 * max(total, 1.0):
                    break
            return min(1.0, max(0.0, front * total / a))

        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if cdf(mid) < q:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


# -- the Beta law --------------------------------------------------------------


@dataclass(frozen=True)
class BetaReference:
    """The distribution split conformal's coverage follows, conditional on calibration.

    Derivation, for the convention this project uses -- scores on hallucinated regions,
    threshold the ``k``-th *smallest* of the positives, flag iff ``score >= tau``:

        tau = S_(k),  k = floor(alpha (n + 1))
        coverage = P(S_new >= tau) = 1 - F(S_(k))
        F(S_(k)) ~ Beta(k, n + 1 - k)        [probability integral transform]
        => coverage ~ Beta(n + 1 - k, k)

    The mean is ``(n + 1 - k)/(n + 1)``, which is ``>= 1 - alpha`` by the flooring -- the
    familiar statement that split conformal is conservative by at most ``1/(n+1)``.
    """

    n_positives: int
    alpha: float
    k: int
    a: float
    b: float

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    @property
    def sd(self) -> float:
        a, b = self.a, self.b
        return float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0))))

    @property
    def certifiable(self) -> bool:
        return self.k >= 1

    @property
    def reportable(self) -> bool:
        return self.certifiable and self.n_positives >= MIN_POSITIVES_FOR_REFERENCE

    def quantile(self, q: float) -> float:
        return _beta_ppf(self.a, self.b, q)

    def interval(self, level: float = 0.98) -> tuple[float, float]:
        tail = (1.0 - level) / 2.0
        return (self.quantile(tail), self.quantile(1.0 - tail))

    def contains(self, coverage: float, level: float = 0.98) -> bool:
        """Is an observed coverage consistent with this cell being correctly calibrated?"""
        if not np.isfinite(coverage) or not self.reportable:
            return False
        lo, hi = self.interval(level)
        return lo <= coverage <= hi

    def as_row(self) -> dict[str, object]:
        lo, hi = self.interval() if self.reportable else (float("nan"), float("nan"))
        return {
            "n_positives": self.n_positives,
            "k": self.k,
            "beta_a": self.a,
            "beta_b": self.b,
            "beta_mean": self.mean,
            "beta_sd": self.sd,
            "beta_lo": lo,
            "beta_hi": hi,
            "reportable": self.reportable,
        }


def beta_reference(n_positives: int, alpha: float | None = None) -> BetaReference:
    """``Beta(n + 1 - k, k)`` for ``n`` positives at level ``alpha``.

    ``alpha`` defaults to the frozen :data:`caliber.evaluation.prereg.ALPHA`.
    """
    a_level = prereg.ALPHA if alpha is None else float(alpha)
    n = int(n_positives)
    if n < 0:
        raise ValueError("n_positives must be non-negative")
    k = int(np.floor(a_level * (n + 1)))
    return BetaReference(n, a_level, k, a=float(n + 1 - k), b=float(max(k, 0)))


# -- the Beta law at a transfer cell -------------------------------------------
#
# The Beta law above is the MATCHED cell. D89's identity says what happens when the
# threshold comes from somewhere else. Writing F_a, F_b for the positive-score CDFs at
# the calibration and deployment operators:
#
#     tau_a = S_(k) of n draws from F_a      U := F_a(tau_a) ~ Beta(k, n + 1 - k)
#     coverage(a -> b) = 1 - F_b(tau_a) = 1 - G_ab(U),   G_ab := F_b o F_a^{-1}
#
# So a transfer cell's coverage is **the same Beta variable pushed through the operator
# pair's probability-probability map**, and the matched cell is the case G = identity.
# Three consequences, all verified by simulation before this was written:
#
#   1. The law is exact, not asymptotic -- the identity above holds draw by draw.
#   2. The SPREAD rescales by G'(u) = f_b/f_a at the threshold, in EITHER direction.
#      Across a 6.6x range of density ratios the first-order prediction held to ~1%.
#      A certificate that quotes the matched Beta band at a transfer cell is therefore
#      wrong by that factor, and it can be wrong by being too NARROW.
#   3. At a fixed calibration draw, every deployment operator in one row is a monotone
#      function of the same scalar U, so the row's cells are exactly comonotone. That is
#      D86's "one draw displaces every cell together", as a theorem rather than an
#      observation -- but only WITHIN a row. Nine independent rows would land on the same
#      side with probability 2 * (1/2)^9 = 0.4%, so a one-directional displacement at all
#      72 pairs is evidence that the nine operators share a calibration cohort, which is a
#      property of how the bank was built and should be stated rather than assumed.


@dataclass(frozen=True)
class TransferBetaReference:
    """The coverage law at a transfer cell: the matched Beta pushed through ``G_ab``.

    Built by sampling ``U`` from the exact Beta and evaluating the *empirical* ``G_ab``,
    so no density is estimated anywhere and no asymptotics are used. :attr:`slope` is
    ``G'`` at the nominal level by a central difference on ``G`` -- which is the density
    ratio ``f_b/f_a`` at the threshold, obtained without estimating either density.
    """

    matched: BetaReference
    samples: np.ndarray
    slope: float
    n_independent: int | None = None

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples))

    @property
    def sd(self) -> float:
        return float(np.std(self.samples, ddof=1))

    @property
    def sd_ratio(self) -> float:
        """Observed spread against the matched cell's. 1.0 iff ``G`` is the identity."""
        return self.sd / self.matched.sd if self.matched.sd > 0 else float("nan")

    @property
    def matched_band_is_too_narrow(self) -> bool:
        """**The case a deployment certificate must not get wrong.**"""
        return self.sd_ratio > 1.0

    @property
    def reportable(self) -> bool:
        """The SHAPE results -- :attr:`slope` and :attr:`sd_ratio` -- may be reported.

        Both are dimensionless, so neither depends on which unit built the Beta. This is
        exactly why they are not enough: see :attr:`certifiable`.
        """
        return self.matched.reportable and self.samples.size > 0

    @property
    def rescalable(self) -> bool:
        """**Whether a band may be emitted at a stated unit.** Not whether it certifies.

        ``n_independent`` has no default. Every caller of :func:`beta_reference` passes a
        count of REGIONS, and regions inside one subject are not independent --
        :data:`prereg.CLUSTER_UNIT` has said so since D83, and the comment block above
        :class:`ClusteredCoverage` already records that 16,400 CT regions behave like
        roughly 35 units. None of that is new.

        What this property fixes is narrow and local. ``make_figures`` draws the
        region-count band explicitly as "if every *region* were independent" and
        ``t1a_safety_rule`` emits it as the idealisation; this class shipped with no such
        label and read as a certificate. So an interval now requires the unit to be named.

        **A band at the subject count is still not a gate.** The project's position is that
        neither count is a measurement -- the region count rejects a correctly calibrated
        bank, the subject count accepts almost anything -- and the certificate path is the
        cluster bootstrap with :data:`prereg.REFIT_THRESHOLD_PER_REPLICATE`. See
        :data:`prereg.BETA_REGION_COUNT_USE`.
        """
        if not self.reportable or self.n_independent is None:
            return False
        return int(self.n_independent) >= MIN_POSITIVES_FOR_REFERENCE

    def interval(self, level: float = 0.98) -> tuple[float, float]:
        """The independence-assumed idealisation, rescaled to a stated unit. **Not a gate.**

        Raises unless :attr:`rescalable`. The samples are drawn at the matched cell's n,
        so the band they span is narrow by ``sqrt(n_positives / n_independent)``; the
        rescaling below stretches it about its own mean by exactly that factor. The shape
        of the pushed-through distribution is kept -- only its scale is corrected.
        """
        if not self.rescalable:
            raise ValueError(
                "no interval without n_independent: a Beta band on the region count is "
                "the independent-regions idealisation, not a certificate; and at any "
                "count the gate is the cluster bootstrap "
                "(prereg.BETA_REGION_COUNT_USE)"
            )
        tail = (1.0 - level) / 2.0
        lo = float(np.quantile(self.samples, tail))
        hi = float(np.quantile(self.samples, 1.0 - tail))
        centre = float(np.mean(self.samples))
        stretch = float(np.sqrt(self.matched.n_positives / float(self.n_independent)))
        # Coverage is a proportion, so the stretched band is clipped rather than left to
        # run past 1 -- a certificate reading "up to 1.004" is not a stronger statement.
        return (float(np.clip(centre + (lo - centre) * stretch, 0.0, 1.0)),
                float(np.clip(centre + (hi - centre) * stretch, 0.0, 1.0)))

    def contains(self, coverage: float, level: float = 0.98) -> bool:
        """Consistency with the rescaled idealisation. **Not a calibration verdict.**"""
        if not np.isfinite(coverage) or not self.rescalable:
            return False
        lo, hi = self.interval(level)
        return lo <= coverage <= hi

    def as_row(self) -> dict[str, object]:
        lo, hi = self.interval() if self.rescalable else (float("nan"), float("nan"))
        return {
            "n_positives": self.matched.n_positives,
            "k": self.matched.k,
            "matched_mean": self.matched.mean,
            "matched_sd": self.matched.sd,
            "transfer_mean": self.mean,
            "transfer_sd": self.sd,
            "transfer_lo": lo,
            "transfer_hi": hi,
            "density_ratio_at_threshold": self.slope,
            "sd_ratio": self.sd_ratio,
            "matched_band_is_too_narrow": self.matched_band_is_too_narrow,
            "reportable": self.reportable,
            "n_independent": self.n_independent,
            "rescalable": self.rescalable,
        }


def probability_map(
    calibration_scores: Sequence[float],
    deployment_scores: Sequence[float],
    u: Sequence[float] | float,
) -> np.ndarray:
    """``G_ab(u) = F_b(F_a^{-1}(u))``, the operator pair's probability-probability map.

    Identity when the two samples come from the same distribution, which is the whole
    content of "the matched cell is the classical case".
    """
    cal = np.sort(np.asarray(calibration_scores, dtype=float))
    dep = np.sort(np.asarray(deployment_scores, dtype=float))
    cal = cal[np.isfinite(cal)]
    dep = dep[np.isfinite(dep)]
    if cal.size == 0 or dep.size == 0:
        raise ValueError("both score samples must be non-empty")
    levels = np.atleast_1d(np.asarray(u, dtype=float))
    x = np.quantile(cal, np.clip(levels, 0.0, 1.0))
    return np.searchsorted(dep, x, side="right") / float(dep.size)


def transfer_beta_reference(
    calibration_scores: Sequence[float],
    deployment_scores: Sequence[float],
    n_positives: int,
    alpha: float | None = None,
    n_draws: int = 40000,
    seed: int = 0,
    n_independent: int | None = None,
) -> TransferBetaReference:
    """The coverage law at a transfer cell, from the two operators' positive scores.

    ``n_positives`` is the size of the *calibration* set, which is what the Beta law's
    shape depends on; the deployment sample only supplies ``G_ab``.

    ``n_independent`` is the number of independent calibration units -- subjects, not
    regions (:data:`prereg.BETA_REFERENCE_UNIT`). It is optional because the
    dimensionless results do not need it: :attr:`TransferBetaReference.slope` and
    :attr:`~TransferBetaReference.sd_ratio` are ratios and the reference's ``n`` divides
    out of both. It is required for an interval, and there is no default, because the
    value is a property of the bank and assuming one is how the region count got in.
    """
    matched = beta_reference(n_positives, alpha)
    rng = np.random.default_rng(seed)
    if matched.k < 1:
        return TransferBetaReference(matched, np.empty(0), float("nan"),
                                     n_independent=n_independent)
    # U = F_a(tau_a) ~ Beta(k, n + 1 - k), the complement of the matched coverage law
    u = rng.beta(matched.b, matched.a, size=int(n_draws))
    samples = 1.0 - probability_map(calibration_scores, deployment_scores, u)

    u0 = float(matched.b / (matched.a + matched.b))
    h = min(0.02, 0.5 * min(u0, 1.0 - u0))
    if h <= 0:
        slope = float("nan")
    else:
        g = probability_map(calibration_scores, deployment_scores, [u0 - h, u0 + h])
        slope = float((g[1] - g[0]) / (2.0 * h))
    return TransferBetaReference(matched, samples, slope,
                                 n_independent=n_independent)


# -- worst-cell coverage -------------------------------------------------------


def clopper_pearson_lower(caught: int, total: int, alpha: float) -> float:
    """Exact one-sided lower confidence bound on a binomial proportion.

    Used rather than a normal interval because the cells that matter are the ones where
    coverage is near 1 and the count is small, which is exactly where a Wald interval
    leaves the unit interval and reports a bound below zero or above one.
    """
    if total <= 0:
        return float("nan")
    if caught >= total:
        return float(alpha ** (1.0 / total))
    if caught <= 0:
        return 0.0
    return float(_beta_ppf(caught, total - caught + 1, alpha))


@dataclass(frozen=True)
class WorstCell:
    """The three numbers, kept apart on purpose.

    **All three are in units of the design effect supplied (D98).** Before 24 November
    2026 this class fed Clopper-Pearson the raw region counts, which treats every
    hallucinated region as an independent Bernoulli trial. ``BETA_REGION_COUNT_USE``
    already forbade that -- a Clopper-Pearson bound *is* a Beta band, under the other of
    its two names, which is why nobody caught it for six days. The design effects sitting
    in the same frames are 47 to 186, and the shipped bound was too high by 0.20 to 0.38.
    """

    cell: str
    n_cells: int
    #: Lowest observed coverage. Biased **downwards** as an estimate of the true worst cell.
    observed_min: float
    #: Pointwise exact lower bound at that cell. Anti-conservative: the cell was selected.
    pointwise_lower: float
    #: Bonferroni lower bound holding across all cells at once. **The certificate number.**
    simultaneous_lower: float
    n_positives: int
    nominal: float
    #: Design effect at the worst cell, and the median over cells. Carried so the bound
    #: cannot be quoted without the quantity that sets its width.
    design_effect_at_worst: float = 1.0
    design_effect_median: float = 1.0
    #: The independent-regions value, kept only to be SHOWN as the idealisation it is
    #: (``BETA_REGION_COUNT_USE``). Never the certificate.
    simultaneous_lower_regions: float = float("nan")

    @property
    def certifiable_at_nominal(self) -> bool:
        """Can a guarantee at the nominal level be written for every cell at once?"""
        ok = np.isfinite(self.simultaneous_lower)
        return bool(ok and self.simultaneous_lower >= self.nominal)

    def as_row(self) -> dict[str, object]:
        return {
            "worst_cell": self.cell,
            "n_cells": self.n_cells,
            "observed_min_coverage": self.observed_min,
            "pointwise_lower": self.pointwise_lower,
            "simultaneous_lower": self.simultaneous_lower,
            "n_positives_at_worst": self.n_positives,
            "nominal": self.nominal,
            "certifiable_at_nominal": self.certifiable_at_nominal,
            "design_effect_at_worst": self.design_effect_at_worst,
            "design_effect_median": self.design_effect_median,
            "simultaneous_lower_regions": self.simultaneous_lower_regions,
            "unit": prereg.BETA_REFERENCE_UNIT,
            "prereg": prereg.fingerprint(),
        }


def worst_operator_coverage(
    per_cell: pd.DataFrame,
    cell_col: str = "operator_id",
    caught_col: str = "n_caught",
    total_col: str = "n_hallucinated",
    family_wise_alpha: float = FAMILY_WISE_ALPHA,
    design_effect_col: str | None = None,
) -> WorstCell:
    """Minimum coverage over cells, with a bound a certificate can actually cite.

    ``per_cell`` carries one row per cell with the count of hallucinated regions and how
    many of them were flagged. Counts, not rates: an exact bound needs the denominator.

    ``design_effect_col`` has **no default and must be named** (D98). Regions are not
    independent -- this project measured 47 to 186 regions per effective unit on the CT
    bank -- and a Clopper-Pearson bound on raw region counts is the Beta band that
    ``prereg.BETA_REGION_COUNT_USE`` forbids as a certificate, under its other name. Pass
    the column holding each cell's design effect; counts are divided by it before the
    bound is taken. To get the independent-regions idealisation deliberately, pass
    ``design_effect_col="__regions__"``, which is read from
    :attr:`WorstCell.simultaneous_lower_regions` and is not a certificate.
    """
    for col in (cell_col, caught_col, total_col):
        if col not in per_cell.columns:
            raise KeyError(f"per-cell frame has no {col!r} column")
    if design_effect_col is None:
        raise ValueError(
            "worst_operator_coverage needs design_effect_col named explicitly: regions are"
            " not independent (47 to 186 per unit on the CT bank), so a bound on raw"
            " region counts is the Beta band prereg.BETA_REGION_COUNT_USE bars from"
            " certifying. Pass the design-effect column, or '__regions__' to ask for the"
            " independent-regions idealisation on purpose."
        )
    regions_only = design_effect_col == "__regions__"
    if not regions_only and design_effect_col not in per_cell.columns:
        raise KeyError(f"per-cell frame has no {design_effect_col!r} column")

    usable = per_cell[per_cell[total_col] > 0].reset_index(drop=True)
    if usable.empty:
        raise ValueError("no cell has any hallucinated region, so there is no coverage to report")

    caught = usable[caught_col].to_numpy(dtype=float)
    total = usable[total_col].to_numpy(dtype=float)
    cov = caught / total
    m = int(len(usable))
    worst = int(np.argmin(cov))

    if regions_only:
        deff = np.ones(m, dtype=float)
    else:
        deff = np.clip(usable[design_effect_col].to_numpy(dtype=float), 1.0, None)
        if not np.all(np.isfinite(deff)):
            raise ValueError(
                f"{design_effect_col!r} holds a non-finite design effect; a certificate"
                " cannot be written from a width that is not known"
            )

    def bound(i: int, level: float, scale: np.ndarray) -> float:
        return clopper_pearson_lower(
            int(round(caught[i] / scale[i])), int(round(total[i] / scale[i])), level
        )

    ones = np.ones(m, dtype=float)
    return WorstCell(
        cell=str(usable[cell_col].iloc[worst]),
        n_cells=m,
        observed_min=float(cov[worst]),
        pointwise_lower=bound(worst, family_wise_alpha, deff),
        simultaneous_lower=min(
            bound(i, family_wise_alpha / m, deff) for i in range(m)
        ),
        n_positives=int(total[worst]),
        nominal=prereg.NOMINAL,
        design_effect_at_worst=float(deff[worst]),
        design_effect_median=float(np.median(deff)),
        simultaneous_lower_regions=min(
            bound(i, family_wise_alpha / m, ones) for i in range(m)
        ),
    )


# -- split-to-split spread -----------------------------------------------------


@dataclass(frozen=True)
class SpreadCheck:
    """Observed split-to-split spread against two references, and whether to believe it."""

    n_splits: int
    observed_mean: float
    observed_sd: float
    #: Closed-form Beta SD: threshold variability only, against an infinite test set.
    #: **Not** the reference for a measured spread -- see the module docstring.
    beta_sd: float
    #: Mean number of positive regions on the test side of a split.
    n_test: int
    #: Simulated SD under exchangeability, through this study's actual split membership.
    #: The reference :attr:`verdict` is taken from.
    simulated_sd: float
    simulated_sd_ci: tuple[float, float]
    overlapping: bool
    mean_shared_fraction: float

    @property
    def beta_sd_corrected(self) -> float:
        """``sqrt(Var_Beta + p(1-p)/n_test)`` -- the closed form plus test-set noise.

        Agrees with the simulation to a few per cent at balanced split sizes. Offered as
        the number a reader can check by hand; the simulation remains authoritative because
        it also carries the overlap and needs no independence assumption.
        """
        if self.n_test <= 0:
            return float("nan")
        p = prereg.NOMINAL
        return float(np.sqrt(self.beta_sd ** 2 + p * (1.0 - p) / self.n_test))

    @property
    def ratio_to_beta(self) -> float:
        """Observed over the **uncorrected** closed form. Expected near 1.36, not 1.0."""
        return self.observed_sd / self.beta_sd if self.beta_sd > 0 else float("nan")

    @property
    def ratio_to_corrected(self) -> float:
        c = self.beta_sd_corrected
        return self.observed_sd / c if np.isfinite(c) and c > 0 else float("nan")

    @property
    def ratio_to_simulated(self) -> float:
        if not np.isfinite(self.simulated_sd) or self.simulated_sd <= 0:
            return float("nan")
        return self.observed_sd / self.simulated_sd

    @property
    def consistent(self) -> bool:
        """Is the observed spread inside the simulated reference's interval?"""
        lo, hi = self.simulated_sd_ci
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return False
        return lo <= self.observed_sd <= hi

    @property
    def power_band(self) -> tuple[float, float]:
        """The reference interval as a multiple of the correct SD: what this check can see."""
        lo, hi = self.simulated_sd_ci
        if not np.isfinite(self.simulated_sd) or self.simulated_sd <= 0:
            return (float("nan"), float("nan"))
        return (lo / self.simulated_sd, hi / self.simulated_sd)

    @property
    def informative(self) -> bool:
        """Could this check have detected a mis-calibration of :data:`INFORMATIVE_BAND`?"""
        _, hi = self.power_band
        return bool(np.isfinite(hi) and hi <= INFORMATIVE_BAND)

    @property
    def verdict(self) -> str:
        """A passing check is reported as underpowered when that is what it is."""
        if not np.isfinite(self.simulated_sd):
            return "no_reference"
        if self.consistent:
            return (
                "consistent_with_exchangeability"
                if self.informative
                else "consistent_but_underpowered"
            )
        return "over_dispersed" if self.observed_sd > self.simulated_sd_ci[1] else "under_dispersed"

    @property
    def note(self) -> str:
        """The caveat that must travel with the number."""
        shared = (
            f"splits share on average {self.mean_shared_fraction:.0%} of their calibration "
            "patients, which deflates the observed SD by a further ~2%. "
            if self.overlapping
            else "splits are disjoint. "
        )
        lo, hi = self.power_band
        power = (
            f" With {self.n_splits} splits the check can only detect a spread below "
            f"{lo:.2f}x or above {hi:.2f}x the correct SD, so a consistent verdict is weak "
            "evidence of calibration."
            if np.isfinite(hi)
            else ""
        )
        return (
            shared + "The closed-form Beta SD omits test-set sampling noise and so "
            f"understates the expected spread here by about "
            f"{self.beta_sd_corrected / self.beta_sd - 1.0:.0%}; read ratio_to_simulated, "
            "not ratio_to_beta." + power
        )

    def as_row(self) -> dict[str, object]:
        return {
            "n_splits": self.n_splits,
            "observed_mean_coverage": self.observed_mean,
            "observed_sd": self.observed_sd,
            "beta_sd_threshold_only": self.beta_sd,
            "beta_sd_corrected": self.beta_sd_corrected,
            "n_test_positives": self.n_test,
            "simulated_sd": self.simulated_sd,
            "simulated_sd_lo": self.simulated_sd_ci[0],
            "simulated_sd_hi": self.simulated_sd_ci[1],
            "ratio_to_beta_uncorrected": self.ratio_to_beta,
            "ratio_to_beta_corrected": self.ratio_to_corrected,
            "ratio_to_simulated": self.ratio_to_simulated,
            "overlapping_splits": self.overlapping,
            "mean_shared_fraction": self.mean_shared_fraction,
            "power_band_lo": self.power_band[0],
            "power_band_hi": self.power_band[1],
            "informative": self.informative,
            "verdict": self.verdict,
            "note": self.note,
            "prereg": prereg.fingerprint(),
        }


def _shared_fraction(splits: Sequence[Sequence[str]]) -> float:
    """Mean pairwise Jaccard overlap of the calibration sets."""
    sets = [set(map(str, s)) for s in splits]
    if len(sets) < 2:
        return 0.0
    vals = []
    for i, a in enumerate(sets):
        for b in sets[i + 1 :]:
            union = a | b
            vals.append(len(a & b) / len(union) if union else 0.0)
    return float(np.mean(vals))


def simulate_split_spread(
    calib_splits: Sequence[Sequence[str]],
    test_splits: Sequence[Sequence[str]],
    positives_per_patient: Mapping[str, int],
    alpha: float | None = None,
    n_sim: int = 2000,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    """Reference distribution for the split-to-split SD, under exchangeability.

    Scores are drawn i.i.d. uniform. That is not an approximation: split conformal is
    distribution-free for continuous scores, so the coverage law depends on the sample sizes
    and the split structure and not on the score distribution at all. What the simulation
    supplies, and the closed form cannot, is the effect of the **same patients appearing in
    many splits** -- which is this study's situation and the reason the closed form is the
    wrong yardstick here.

    Returns ``(median_sd, (2.5th, 97.5th percentile))``.
    """
    a_level = prereg.ALPHA if alpha is None else float(alpha)
    if len(calib_splits) != len(test_splits):
        raise ValueError("calib_splits and test_splits must describe the same splits")
    rng = np.random.default_rng(seed)
    counts = {str(k): int(v) for k, v in positives_per_patient.items() if int(v) > 0}
    if not counts:
        return (float("nan"), (float("nan"), float("nan")))

    sds = np.empty(n_sim, dtype=float)
    for s in range(n_sim):
        draws = {p: rng.random(n) for p, n in counts.items()}
        per_split = []
        for calib, test in zip(calib_splits, test_splits, strict=True):
            pool = np.concatenate(
                [draws[str(p)] for p in calib if str(p) in draws] or [np.empty(0)]
            )
            held = np.concatenate(
                [draws[str(p)] for p in test if str(p) in draws] or [np.empty(0)]
            )
            if pool.size == 0 or held.size == 0:
                continue
            k = int(np.floor(a_level * (pool.size + 1)))
            if k < 1:
                continue
            tau = np.sort(pool)[k - 1]
            per_split.append(float((held >= tau).mean()))
        sds[s] = np.std(per_split, ddof=1) if len(per_split) > 1 else np.nan
    good = sds[np.isfinite(sds)]
    if good.size < n_sim // 10:
        return (float("nan"), (float("nan"), float("nan")))
    return (
        float(np.median(good)),
        (float(np.percentile(good, 2.5)), float(np.percentile(good, 97.5))),
    )


def split_spread_check(
    per_split_coverage: Sequence[float],
    calib_splits: Sequence[Sequence[str]],
    test_splits: Sequence[Sequence[str]],
    positives_per_patient: Mapping[str, int],
    alpha: float | None = None,
    n_sim: int = 2000,
    seed: int = 0,
) -> SpreadCheck:
    """Compare the measured split-to-split spread against both references.

    ``per_split_coverage[i]`` is the coverage measured on split ``i``, whose calibration and
    test patients are ``calib_splits[i]`` and ``test_splits[i]``.
    """
    cov = np.asarray(per_split_coverage, dtype=float)
    cov = cov[np.isfinite(cov)]
    if cov.size < 2:
        raise ValueError("at least two splits with finite coverage are needed for a spread")

    def _mean_positives(splits: Sequence[Sequence[str]]) -> float:
        return float(np.mean([
            sum(int(positives_per_patient.get(str(p), 0)) for p in part) for part in splits
        ]))

    mean_n = _mean_positives(calib_splits)
    mean_test = _mean_positives(test_splits)
    ref = beta_reference(int(round(mean_n)), alpha)
    sim, sim_ci = simulate_split_spread(
        calib_splits, test_splits, positives_per_patient, alpha, n_sim, seed
    )
    shared = _shared_fraction(calib_splits)

    return SpreadCheck(
        n_splits=int(cov.size),
        observed_mean=float(cov.mean()),
        observed_sd=float(cov.std(ddof=1)),
        beta_sd=ref.sd,
        n_test=int(round(mean_test)),
        simulated_sd=sim,
        simulated_sd_ci=sim_ci,
        overlapping=shared > 0.0,
        mean_shared_fraction=shared,
    )


# -- the calibration unit, read off the threshold's own motion ------------------
#
# Added 22 November 2026 (D94), as the dimensioned check amendment 12 demanded.
#
# `ClusteredCoverage.effective_n` already recovers the independent count, from a cluster
# bootstrap. This recovers it from a quantity every split records anyway -- where the
# threshold sat inside the calibration scores -- and needs no resampling at all.
#
# U := F_a(tau_a) is Beta(k, n + 1 - k) with k = floor(alpha*(n+1)), exactly, for n
# independent calibration scores. So U's across-split SPREAD reads n back, where the
# predicted spread RATIO divides n out (prereg.DIMENSIONLESS_PASS_WARNING).
#
# It returns a bracket and never a count. On T2-A's 13 splits the statistic separates tens
# from thousands by a factor of 300 and cannot separate 35 from 50; D93 Sec. 3 read a point
# estimate of "near 45" off the same number, which was more precision than it carries.


@dataclass(frozen=True)
class LevelRangeCount:
    """A bracket on the independent calibration count, from where tau sat across splits.

    :attr:`consistent` can have holes, because ``k = floor(alpha*(n+1))`` steps in
    integers and the likelihood is not monotone in ``n``. :attr:`bracket` is its span and
    :attr:`has_holes` says whether to read the span as solid.
    """

    observed_width: float
    n_splits: int
    #: Candidate counts whose central band contains the observed width.
    consistent: tuple[int, ...]
    #: Candidates ruled out, as (count, central band) -- kept so the exclusion is legible.
    excluded: tuple[tuple[int, tuple[float, float]], ...]
    alpha: float
    level: float

    @property
    def bracket(self) -> tuple[int, int] | None:
        """Span of the consistent counts. ``None`` if nothing fits -- read no number then."""
        if not self.consistent:
            return None
        return (int(min(self.consistent)), int(max(self.consistent)))

    @property
    def has_holes(self) -> bool:
        if not self.consistent:
            return False
        lo, hi = self.bracket or (0, 0)
        return any(lo < int(c) < hi for c, _ in self.excluded)

    def excludes(self, n: int) -> bool:
        """Whether this count is ruled out.

        Raises for a count that was never tried, rather than returning False.
        The obvious call is ``excludes(bank_region_count)``, and a bank's region
        count is almost never a round number on the candidate grid -- 5021, 3089
        and 8983 on the CT bank -- so a silent False would answer "not excluded"
        when it means "never tested", for every real call this method has.
        """
        tried = {int(c) for c in self.consistent}
        tried |= {int(c) for c, _ in self.excluded}
        if int(n) not in tried:
            raise ValueError(
                f"count {int(n)} was not among the candidates, so this result says"
                " nothing about it; pass it in `candidates` to ask about it"
            )
        return any(int(c) == int(n) for c, _ in self.excluded)

    def as_row(self) -> dict[str, object]:
        b = self.bracket
        return {
            "observed_width": self.observed_width,
            "n_splits": self.n_splits,
            "bracket_lo": b[0] if b else None,
            "bracket_hi": b[1] if b else None,
            "has_holes": self.has_holes,
            "n_consistent": len(self.consistent),
            "level": self.level,
            "unit": prereg.BETA_REFERENCE_UNIT,
        }


#: Candidate counts spanning subject-scale to region-scale, so one call shows both.
DEFAULT_LEVEL_RANGE_CANDIDATES: tuple[int, ...] = (
    10, 15, 20, 25, 30, 35, 41, 50, 60, 70, 85, 100, 125, 150, 200, 300, 500,
    1000, 2000, 5000, 10000, 20000, 50000,
)


def count_from_level_range(
    levels: Sequence[float],
    candidates: Sequence[int] | None = None,
    alpha: float | None = None,
    level: float = 0.90,
    n_sim: int = 20000,
    seed: int = 0,
) -> LevelRangeCount:
    """Bracket the independent calibration count from ``F_a(tau_a)`` across splits.

    ``levels`` is one value per split: the fraction of that split's calibration positives
    at or below its threshold. Pass the levels themselves, not the thresholds -- a
    threshold is in score units and carries no ``n``.

    The statistic is the range of ``levels``, compared against the range of the same
    number of ``Beta(k, n + 1 - k)`` draws. A count is kept when the observed range falls
    inside the central ``level`` of its simulated range.

    See :data:`prereg.COUNT_FROM_LEVEL_RANGE`. **This is not a gate.**
    """
    a = float(prereg.ALPHA if alpha is None else alpha)
    x = np.asarray(levels, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("count_from_level_range needs at least two split levels")
    width = float(x.max() - x.min())
    cands = tuple(int(c) for c in (
        DEFAULT_LEVEL_RANGE_CANDIDATES if candidates is None else candidates))
    rng = np.random.default_rng(seed)
    tail = (1.0 - float(level)) / 2.0
    keep: list[int] = []
    drop: list[tuple[int, tuple[float, float]]] = []
    for n in cands:
        k = int(np.floor(a * (n + 1)))
        if k < 1 or k > n:
            drop.append((n, (float("nan"), float("nan"))))
            continue
        d = rng.beta(k, n + 1 - k, size=(int(n_sim), int(x.size)))
        w = d.max(axis=1) - d.min(axis=1)
        lo = float(np.quantile(w, tail))
        hi = float(np.quantile(w, 1.0 - tail))
        if lo <= width <= hi:
            keep.append(n)
        else:
            drop.append((n, (lo, hi)))
    return LevelRangeCount(
        observed_width=width,
        n_splits=int(x.size),
        consistent=tuple(keep),
        excluded=tuple(drop),
        alpha=a,
        level=float(level),
    )


# -- the clustered control -----------------------------------------------------
#
# Added 20 November 2026, after this module's own Beta band was applied to a real bank and
# would have failed all nine cells for the wrong reason (D85).
#
# `beta_reference(n)` is exact for `n` *independent* calibration scores. Regions are not
# independent: on T2-A's CT bank, 16,400 calibration regions behave like roughly 35 units,
# because the unit is the subject. Handed the region count, the Beta band comes out at
# +/-0.011 and rejects a correctly calibrated bank; handed the subject count it is +/-0.10
# and would accept almost anything. Neither is a measurement, so neither should be the gate.
#
# The gate is a cluster bootstrap instead: resample subjects on both sides, refit the
# threshold inside every replicate, and read the interval off the replicates. That is the
# rule the pre-registration already froze (REFIT_THRESHOLD_PER_REPLICATE), applied to the
# control rather than only to the headline. The Beta band stays, reported beside it as the
# independence-assumed idealisation it is, which is also how the effective sample size is
# recovered: n_eff = p(1-p)/sd^2.


@dataclass(frozen=True)
class ClusteredCoverage:
    """Matched-cell coverage with an interval that moves both sides of the split."""

    coverage: float
    lo: float
    hi: float
    n_clusters_calib: int
    n_clusters_test: int
    n_positives_calib: int
    n_positives_test: int
    nominal: float
    level: float
    alpha: float = prereg.ALPHA

    @property
    def sd(self) -> float:
        width = self.hi - self.lo
        return width / (2.0 * 1.96) if np.isfinite(width) else float("nan")

    @property
    def independent_sd(self) -> float:
        """What the interval's SD would be if every region were an independent draw.

        **Both terms, not just the Beta one.** The bootstrap moves the calibration side and
        the test side, so the reference it should be compared against is
        ``sqrt(Var_Beta(n_cal) + p(1-p)/n_test)``. Comparing against the Beta SD alone makes
        genuinely independent regions look clustered by a factor of about sqrt(2) at
        balanced split sizes, which is how the first version of :attr:`design_effect` came
        out at 2.6 on a fixture built to have no clustering at all.
        """
        if self.n_positives_calib <= 0 or self.n_positives_test <= 0:
            return float("nan")
        p = self.nominal
        beta = beta_reference(self.n_positives_calib, self.alpha).sd
        return float(np.sqrt(beta ** 2 + p * (1.0 - p) / self.n_positives_test))

    @property
    def design_effect(self) -> float:
        """Observed variance over the independent-regions variance.

        1.0 means regions behave independently. On T2-A's CT bank it is of order 100: the
        unit is the subject, and there are roughly 400 regions in each.
        """
        ref, obs = self.independent_sd, self.sd
        if not (np.isfinite(ref) and np.isfinite(obs)) or ref <= 0:
            return float("nan")
        return float((obs / ref) ** 2)

    @property
    def effective_n(self) -> float:
        """Independent calibration units the spread implies."""
        deff = self.design_effect
        if not np.isfinite(deff) or deff <= 0:
            return float("nan")
        return float(self.n_positives_calib / deff)

    @property
    def contains_nominal(self) -> bool:
        return bool(np.isfinite(self.lo) and self.lo <= self.nominal <= self.hi)

    def as_row(self) -> dict[str, object]:
        return {
            "coverage": self.coverage,
            "ci_lo": self.lo,
            "ci_hi": self.hi,
            "ci_sd": self.sd,
            "n_clusters_calib": self.n_clusters_calib,
            "n_clusters_test": self.n_clusters_test,
            "n_positives_calib": self.n_positives_calib,
            "n_positives_test": self.n_positives_test,
            "independent_sd": self.independent_sd,
            "effective_n": self.effective_n,
            "design_effect": self.design_effect,
            "contains_nominal": self.contains_nominal,
        }


def clustered_coverage(
    calib_scores_by_cluster: Sequence[np.ndarray],
    test_scores_by_cluster: Sequence[np.ndarray],
    alpha: float | None = None,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> ClusteredCoverage:
    """Matched-operator coverage, with a cluster bootstrap over **both** sides.

    Each element of either sequence is one cluster's positive-region scores. The
    calibration side is resampled and the threshold **refitted** on every replicate, which
    is the term a test-side-only interval leaves out -- and on a bank where calibration and
    test sets are of similar size, that term is of the same order as the one it keeps, so
    omitting it understates the interval by roughly a factor of sqrt(2) and can turn a
    correctly calibrated cell into a failure.

    The test side is resampled too, because the measured coverage is an average over a
    finite set of test subjects and that noise is real.
    """
    a_level = prereg.ALPHA if alpha is None else float(alpha)
    calib = [np.asarray(c, dtype=float) for c in calib_scores_by_cluster if len(c)]
    test = [np.asarray(t, dtype=float) for t in test_scores_by_cluster if len(t)]
    n_cal_pos = int(sum(c.size for c in calib))
    n_tst_pos = int(sum(t.size for t in test))
    nan = float("nan")
    if not calib or not test:
        return ClusteredCoverage(nan, nan, nan, len(calib), len(test), n_cal_pos, n_tst_pos,
                                 prereg.NOMINAL, level, a_level)

    pooled_cal = np.concatenate(calib)
    k = int(np.floor(a_level * (pooled_cal.size + 1)))
    if k < 1:
        return ClusteredCoverage(nan, nan, nan, len(calib), len(test), n_cal_pos, n_tst_pos,
                                 prereg.NOMINAL, level, a_level)
    tau = np.sort(pooled_cal)[k - 1]
    pooled_tst = np.concatenate(test)
    point = float((pooled_tst >= tau).mean())

    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        ci = rng.integers(0, len(calib), size=len(calib))
        ti = rng.integers(0, len(test), size=len(test))
        drawn_cal = np.concatenate([calib[i] for i in ci])
        kb = int(np.floor(a_level * (drawn_cal.size + 1)))
        if kb < 1:
            reps[b] = np.nan
            continue
        tau_b = np.sort(drawn_cal)[kb - 1]
        drawn_tst = np.concatenate([test[i] for i in ti])
        reps[b] = float((drawn_tst >= tau_b).mean())

    good = reps[np.isfinite(reps)]
    if good.size < n_boot // 10:
        return ClusteredCoverage(point, nan, nan, len(calib), len(test), n_cal_pos, n_tst_pos,
                                 prereg.NOMINAL, level, a_level)
    tail = (1.0 - level) / 2.0
    return ClusteredCoverage(
        coverage=point,
        lo=float(np.percentile(good, 100 * tail)),
        hi=float(np.percentile(good, 100 * (1.0 - tail))),
        n_clusters_calib=len(calib),
        n_clusters_test=len(test),
        n_positives_calib=n_cal_pos,
        n_positives_test=n_tst_pos,
        nominal=prereg.NOMINAL,
        level=level,
        alpha=a_level,
    )


# -- the draw-marginal coverage the unsafe label is read from (D86) -----------

#: ``t_{0.975}`` by integer degrees of freedom 1..30, for the reported t inflation only.
#:
#: Hard-coded rather than computed: scipy's distributions are unavailable on this machine
#: (see :func:`_beta_ppf`), and a table a reader can check against any statistics text is
#: better here than a special function implemented from scratch for a diagnostic.
_T975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)

#: Normal two-sided 95% quantile, the project's reference width.
_Z975 = 1.959964


def _norm_ppf(q: float) -> float:
    """Normal quantile by bisection on ``erf``. scipy is unavailable here (see _beta_ppf)."""
    import math

    if not 0.0 < q < 1.0:
        return float("nan")
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _t975(df: float) -> float:
    """``t_{0.975}`` at possibly fractional ``df``. **Approximate**, and only a diagnostic.

    Interpolated in ``1/df`` rather than in ``df``: below about three degrees of freedom
    ``t_{0.975}`` is steeply convex, and linear interpolation there overstates it badly
    (at df 1.6 it gives 7.7 against a true value near 5.8, while ``1/df`` gives 6.4).
    Accurate enough for a reported inflation factor; it is never applied to an interval.
    """
    if not np.isfinite(df) or df <= 0:
        return float("nan")
    if df >= len(_T975):
        return _Z975
    if df <= 1.0:
        return _T975[0]
    lo = int(np.floor(df))
    if lo >= len(_T975):
        return _Z975
    t_lo, t_hi = _T975[lo - 1], _T975[lo]
    inv, inv_lo, inv_hi = 1.0 / df, 1.0 / lo, 1.0 / (lo + 1)
    frac = (inv - inv_hi) / (inv_lo - inv_hi) if inv_lo != inv_hi else 0.0
    return float(t_hi + frac * (t_lo - t_hi))


def mean_pairwise_overlap(splits: Sequence[Sequence[str]]) -> float:
    """Mean pairwise ``|a & b| / sqrt(|a| |b|)`` over the calibration sets.

    **Not** the Jaccard overlap that :func:`split_spread_check` reports. For two sample
    means over overlapping sets this quantity *is* their correlation, which is what the
    effective-n correction needs; Jaccard understates it (a third shared gives Jaccard 0.20
    against a correlation of 0.33). Both are kept because they answer different questions
    and the existing spread check is documented against the Jaccard one.
    """
    sets = [set(map(str, s)) for s in splits]
    if len(sets) < 2:
        return 0.0
    vals = []
    for i, a in enumerate(sets):
        for b in sets[i + 1 :]:
            denom = np.sqrt(len(a) * len(b))
            vals.append(len(a & b) / denom if denom else 0.0)
    return float(np.mean(vals))


@dataclass(frozen=True)
class DrawMarginalCoverage:
    """Coverage marginal over the calibration draw, which is what the label asks about.

    Split conformal promises nominal coverage *on average over the draw of the calibration
    set*, not on any one draw, and T2-A1 measured what one draw costs on T2-A's bank: the
    registry split sits a near-constant 1.38 SD low at all nine matched operators at once.
    Because that displacement is shared by the cells it does not average out over the 72
    ordered pairs, so the label has to be read from the mean over splits (D86).
    """

    coverage: float
    sd: float
    n_splits: int
    #: Mean pairwise calibration-set overlap; the correlation the mean must be corrected for.
    overlap: float
    nominal: float = prereg.NOMINAL
    level: float = 0.95
    min_splits: int = prereg.UNSAFE_LABEL_MIN_SPLITS

    @property
    def n_eff(self) -> float:
        """``n / (1 + (n - 1) rho)``: independent-equivalent number of draws."""
        n = float(self.n_splits)
        if n <= 0:
            return float("nan")
        return float(n / (1.0 + (n - 1.0) * self.overlap))

    @property
    def se_naive(self) -> float:
        """``sd / sqrt(n_splits)`` -- what the splits would give if they were independent."""
        return self.sd / np.sqrt(self.n_splits) if self.n_splits > 0 else float("nan")

    @property
    def se(self) -> float:
        """The reported standard error: ``sd / sqrt(n_eff)``."""
        ne = self.n_eff
        return float(self.sd / np.sqrt(ne)) if np.isfinite(ne) and ne > 0 else float("nan")

    @property
    def overlap_inflation(self) -> float:
        """How much the overlap correction widened the interval. 1.0 means disjoint splits."""
        return self.se / self.se_naive if self.se_naive > 0 else float("nan")

    @property
    def df(self) -> float:
        """Degrees of freedom behind the mean, after the overlap correction."""
        return self.n_eff - 1.0

    @property
    def t_inflation(self) -> float:
        """``t_{0.975}(df) / z_{0.975}``: what a Student interval would have added.

        Reported, not applied. The frozen interval is normal
        (:data:`caliber.evaluation.prereg.UNSAFE_LABEL_INTERVAL`) and the reason is recorded
        there; this is the number that lets a reader see the size of that choice.
        """
        t = _t975(self.df)
        return float(t / _Z975) if np.isfinite(t) else float("nan")

    def interval(self, level: float | None = None) -> tuple[float, float]:
        """Two-sided interval on the draw-marginal coverage, normal quantile."""
        lvl = self.level if level is None else float(level)
        se = self.se
        if not np.isfinite(se):
            return (float("nan"), float("nan"))
        z = _Z975 if abs(lvl - 0.95) < 1e-9 else _norm_ppf(0.5 + lvl / 2.0)
        return (float(self.coverage - z * se), float(self.coverage + z * se))

    @property
    def hi(self) -> float:
        """Upper end -- the quantity the unsafe label thresholds."""
        return self.interval()[1]

    @property
    def lo(self) -> float:
        return self.interval()[0]

    @property
    def reportable(self) -> bool:
        """Enough splits for the mean to beat the bias it removes."""
        return bool(self.n_splits >= self.min_splits and np.isfinite(self.se))

    @property
    def contains_nominal(self) -> bool:
        lo, hi = self.interval()
        return bool(np.isfinite(lo) and lo <= self.nominal <= hi)

    @property
    def note(self) -> str:
        if self.n_splits < self.min_splits:
            return (
                f"{self.n_splits} split(s), below the frozen minimum of {self.min_splits}; "
                "the across-split mean is noisier than the single-draw bias it removes, so "
                "the registry-split label is preferable here"
            )
        return (
            f"{self.n_splits} splits sharing {self.overlap:.0%} of their calibration "
            f"subjects pairwise, so n_eff = {self.n_eff:.1f} and the standard error is "
            f"{self.overlap_inflation:.2f}x the naive one; a Student interval on "
            f"{self.df:.1f} df would widen it a further {self.t_inflation:.2f}x"
        )

    def as_row(self) -> dict[str, object]:
        lo, hi = self.interval()
        return {
            "coverage_across": self.coverage,
            "coverage_across_lo": lo,
            "coverage_across_hi": hi,
            "across_sd": self.sd,
            "n_splits": self.n_splits,
            "overlap": self.overlap,
            "n_eff_splits": self.n_eff,
            "se": self.se,
            "se_naive": self.se_naive,
            "overlap_inflation": self.overlap_inflation,
            "t_inflation": self.t_inflation,
            "reportable": self.reportable,
            "contains_nominal": self.contains_nominal,
            "note": self.note,
        }


def draw_marginal_coverage(
    per_split_coverage: Sequence[float],
    calib_splits: Sequence[Sequence[str]] | None = None,
    overlap: float | None = None,
    level: float = 0.95,
) -> DrawMarginalCoverage:
    """The mean coverage over patient splits, with an interval the overlap is taken out of.

    ``per_split_coverage`` is one coverage per split for **one** cell. Either pass
    ``calib_splits`` -- the calibration subject ids per split, in the same order -- so the
    overlap can be measured, or pass ``overlap`` directly when it is already known for the
    whole run (it is a property of the split design, not of the cell, so computing it once
    and reusing it is both cheaper and less error-prone than recomputing per pair).

    Non-finite coverages are dropped and the remaining count is what ``n_splits`` reports,
    so a cell that was uncertifiable on some splits is described by the splits it has rather
    than silently credited with all of them.
    """
    vals = np.asarray(per_split_coverage, dtype=float)
    vals = vals[np.isfinite(vals)]
    if overlap is None:
        rho = mean_pairwise_overlap(calib_splits) if calib_splits is not None else 0.0
    else:
        rho = float(overlap)
    rho = float(np.clip(rho, 0.0, 0.999))
    if vals.size == 0:
        return DrawMarginalCoverage(float("nan"), float("nan"), 0, rho, level=level)
    sd = float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
    return DrawMarginalCoverage(
        coverage=float(vals.mean()), sd=sd, n_splits=int(vals.size), overlap=rho, level=level
    )


def across_split_columns(
    per_split: pd.DataFrame,
    calib_splits: Sequence[Sequence[str]] | None = None,
    overlap: float | None = None,
    level: float = 0.95,
    keys: Sequence[str] = ("calibration", "deployment"),
    coverage_col: str = "coverage",
    split_col: str = "split",
) -> pd.DataFrame:
    """Turn per-split per-pair coverage into the columns the unsafe label reads.

    ``per_split`` is long: one row per (ordered pair, split) with its coverage. The result
    is one row per ordered pair carrying ``coverage_across`` and ``coverage_across_hi``,
    which is exactly what :data:`caliber.evaluation.safety.LABEL_SOURCE_COLUMNS` asks for
    under the frozen label source, plus the diagnostics that make the overlap correction
    visible.

    The overlap is a property of the split design, so it is computed once here and shared
    across pairs rather than re-derived per row.
    """
    missing = [c for c in (*keys, coverage_col, split_col) if c not in per_split.columns]
    if missing:
        raise KeyError(f"per-split frame is missing {missing}")
    rho = overlap
    if rho is None:
        rho = mean_pairwise_overlap(calib_splits) if calib_splits is not None else 0.0
    rows = []
    for key, part in per_split.groupby(list(keys), dropna=False, sort=True):
        est = draw_marginal_coverage(
            part[coverage_col].to_numpy(dtype=float), overlap=rho, level=level
        )
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row.update(est.as_row())
        rows.append(row)
    return pd.DataFrame(rows)


#: Columns a per-split coverage table must carry, whoever produced it.
PER_SPLIT_REQUIRED = ("calibration", "deployment", "coverage")

#: Columns the registry-split table must carry. ``hi`` is the upper end of that split's own
#: both-sides interval, which is what the registry-split arm of the sensitivity thresholds.
REGISTRY_REQUIRED = ("calibration", "deployment", "coverage", "hi")


def pair_coverage_from_splits(
    per_split: pd.DataFrame,
    registry: pd.DataFrame,
    calib_splits: Sequence[Sequence[str]] | None = None,
    overlap: float | None = None,
    level: float = 0.95,
    keys: Sequence[str] = ("calibration", "deployment"),
    split_col: str = "salt",
) -> pd.DataFrame:
    """Join a per-split coverage table to its registry split, carrying *both* label sources.

    This is the contract between whoever measures coverage and whoever reads the unsafe
    label. The result holds ``coverage``/``coverage_hi`` from the registry split and
    ``coverage_across``/``coverage_across_hi`` from the splits, which is what
    :data:`caliber.evaluation.safety.LABEL_SOURCE_COLUMNS` asks for under either source --
    so the frozen arm and the arm it displaced can be scored on one frame and the
    sensitivity table can compare them without recomputing anything.

    **The across-split interval is rebuilt here from the per-split coverages, never read.**
    A table that has already aggregated reports ``sd / sqrt(n)``, and these splits are
    re-cuts of one cohort rather than independent replications: they share a large share of
    their calibration subjects, so that standard error is too small by
    :attr:`DrawMarginalCoverage.overlap_inflation`. Accepting a stored interval would
    quietly reinstate the independence this correction exists to remove, and it would do it
    in the direction that *narrows* the interval and so calls more pairs unsafe. Passing the
    long table is therefore the supported path; ``overlap`` or ``calib_splits`` supplies the
    sharing it cannot see.

    ``registry`` is an outer merge, so a pair present in one table and not the other keeps a
    row with the missing side blank rather than disappearing. A pair silently dropped here
    would shrink the denominator of the false-safe rate, which is the project's own recurring
    failure: a statistic that improves because its sample changed underneath it.
    """
    missing = [c for c in (*keys, "coverage") if c not in per_split.columns]
    if split_col not in per_split.columns:
        missing.append(split_col)
    if missing:
        raise KeyError(
            f"per-split frame is missing {missing}; needed {[*keys, 'coverage', split_col]}"
        )
    missing = [c for c in REGISTRY_REQUIRED if c not in registry.columns]
    if missing:
        raise KeyError(
            f"registry frame is missing {missing}; needed {list(REGISTRY_REQUIRED)}"
        )
    across = across_split_columns(
        per_split,
        calib_splits=calib_splits,
        overlap=overlap,
        level=level,
        keys=keys,
        coverage_col="coverage",
        split_col=split_col,
    )
    reg = registry[[*keys, "coverage", "hi"]].rename(columns={"hi": "coverage_hi"})
    return reg.merge(across, on=list(keys), how="outer")
