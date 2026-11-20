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
    """The three numbers, kept apart on purpose."""

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
            "prereg": prereg.fingerprint(),
        }


def worst_operator_coverage(
    per_cell: pd.DataFrame,
    cell_col: str = "operator_id",
    caught_col: str = "n_caught",
    total_col: str = "n_hallucinated",
    family_wise_alpha: float = FAMILY_WISE_ALPHA,
) -> WorstCell:
    """Minimum coverage over cells, with a bound a certificate can actually cite.

    ``per_cell`` carries one row per cell with the count of hallucinated regions and how
    many of them were flagged. Counts, not rates: an exact bound needs the denominator.
    """
    for col in (cell_col, caught_col, total_col):
        if col not in per_cell.columns:
            raise KeyError(f"per-cell frame has no {col!r} column")
    usable = per_cell[per_cell[total_col] > 0].reset_index(drop=True)
    if usable.empty:
        raise ValueError("no cell has any hallucinated region, so there is no coverage to report")

    caught = usable[caught_col].to_numpy(dtype=float)
    total = usable[total_col].to_numpy(dtype=float)
    cov = caught / total
    m = int(len(usable))
    worst = int(np.argmin(cov))

    return WorstCell(
        cell=str(usable[cell_col].iloc[worst]),
        n_cells=m,
        observed_min=float(cov[worst]),
        pointwise_lower=clopper_pearson_lower(
            int(caught[worst]), int(total[worst]), family_wise_alpha
        ),
        simultaneous_lower=min(
            clopper_pearson_lower(int(c), int(t), family_wise_alpha / m)
            for c, t in zip(caught, total, strict=True)
        ),
        n_positives=int(total[worst]),
        nominal=prereg.NOMINAL,
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
