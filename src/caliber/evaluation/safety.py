"""T1-A: the ordered-pair safety rule, and what it costs to be wrong about one.

The project's headline correlation -- Spearman +0.976 between operator distance and
coverage gap -- is weak on its own, and the objection is simple: within one operator
family, *any* monotone function of view count orders the gaps, so a high rank correlation
says little more than that the knob was turned in one direction. The repair is to stop
reporting a correlation and report a decision instead:

    Calibrate on operator A, deploy on operator B. Predict **safe** iff the signed
    operator distance from A to B is at most zero -- that is, iff B hides no more of the
    image than A did.

That is a rule a deployer can run before acquiring anything, because the signed distance
is a property of the two operators and needs no reconstruction, no prior, and no data.
What it needs is an honest error rate, and this module computes it.

Three things here are not conventional and are deliberate.

**The primary endpoint is the false-safe rate, not accuracy.** Of the operator pairs the
rule green-lights, what fraction actually under-cover? That is the number that decides
whether the rule is usable, because a false alarm costs flag area and a false *safe*
costs a missed hallucination. Accuracy averages the two and hides the asymmetry.

**It is reported against the trivial rule.** "Always say safe" has a false-safe rate equal
to the base rate of unsafe pairs. If 10% of pairs under-cover, a 10% false-safe rate is
not a result -- it is what saying nothing achieves. :attr:`SafetyRule.beats_trivial`
makes that comparison structural rather than something a reader has to do in their head,
and :attr:`SafetyRule.green_light_fraction` is reported beside it because a rule that
green-lights two pairs out of seventy-two can reach a perfect false-safe rate and still
be useless. Coverage and area, again, in a different costume.

**The permutation null shuffles operator labels, not the distance column.** Shuffling
distances independently would break the antisymmetry of ``null_space_growth`` and the
triangle structure of the matrix, producing a null world in which no sign rule could work
and against which anything looks significant. Permuting the operator *names* keeps the
matrix intact and destroys only its association with the measured outcomes, which is the
hypothesis actually under test. See :data:`caliber.evaluation.prereg.N_PERMUTATIONS`.

Every tunable choice is imported from :mod:`caliber.evaluation.prereg` and none is a
keyword argument, so no result produced here can have been obtained by choosing a
threshold or a tolerance after seeing the coverage numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from caliber.evaluation import prereg

#: Columns an ordered-pair frame must carry.
#:
#: ``coverage_hi`` is the upper end of the pair's cluster-bootstrap coverage interval, and
#: it is **required rather than optional** (D85). The label is a statement about whether a
#: cell under-covers, per-cell coverage on a real bank has a standard deviation of about
#: 0.05, and a point estimate cannot support that statement. Making the column optional
#: with a fall-back to the point estimate would let the superseded label be used silently,
#: which is the failure mode this module is built against. A caller with genuinely exact
#: coverage -- a simulation fixture -- passes ``coverage_hi == coverage``, which is a
#: zero-width interval and reduces to the old behaviour explicitly.
REQUIRED = ("calibration", "deployment", "signed_distance", "coverage", "coverage_hi")

#: Below this many operators a cluster bootstrap over operators is not reported: with
#: three clusters the interval is an artefact of the resampling, not a measurement.
MIN_OPERATORS_FOR_CI = 4

#: Fraction of the ordered pairs among the frame's own operators that must be present
#: before a permutation p-value is reported.
#:
#: The null permutes operator names and then looks the permuted pair up in the distance
#: matrix. On a complete grid every lookup lands. On a sparse one, some permutations ask
#: for directions the frame does not contain, those pairs drop out, and each replicate is
#: then computed on a different and smaller sample than the observed statistic was -- so
#: the p-value is not the exchangeability argument it claims to be. Rather than return a
#: number that looks like a p-value and is not, this refuses below the threshold. The CT
#: grid is complete (9 operators, all 72 ordered pairs), so this does not bind there.
MIN_PAIR_COMPLETENESS = 0.80


# -- the confusion matrix ------------------------------------------------------


@dataclass(frozen=True)
class SafetyRule:
    """One cell set's confusion matrix, endpoints, and null comparisons.

    "Safe" throughout means *the measured coverage at the deployment operator reached
    nominal*. "Predicted safe" means the rule said so in advance from the operators alone.
    """

    label: str
    n_pairs: int
    n_excluded: int
    true_safe: int
    false_safe: int
    false_alarm: int
    true_unsafe: int
    #: Signed Youden's J on the observed data; the permutation statistic.
    youden_j: float = float("nan")
    #: Two-sided permutation p-value on ``|J|``. NaN when not computed.
    p_value: float = float("nan")
    #: Cluster-bootstrap interval for the false-safe rate, over operators.
    false_safe_ci: tuple[float, float] = (float("nan"), float("nan"))
    #: Per-held-out-operator false-safe rates, leave-one-operator-out.
    loo_false_safe: tuple[float, ...] = field(default_factory=tuple)
    #: Share of the ordered pairs among this frame's operators that are present.
    pair_completeness: float = float("nan")

    # -- endpoints ------------------------------------------------------------

    @property
    def n_green_lit(self) -> int:
        """Pairs the rule called safe."""
        return self.true_safe + self.false_safe

    @property
    def n_unsafe(self) -> int:
        """Pairs that actually under-covered. Invariant under the permutation."""
        return self.false_safe + self.true_unsafe

    @property
    def false_safe_rate(self) -> float:
        """**The primary endpoint.** Of the pairs called safe, the fraction that were not.

        NaN when the rule green-lights nothing -- which is not a perfect score, it is a
        refusal to answer, and :attr:`green_light_fraction` is what shows that.
        """
        return self.false_safe / self.n_green_lit if self.n_green_lit else float("nan")

    @property
    def green_light_fraction(self) -> float:
        """Fraction of pairs the rule is willing to call safe. The utility side."""
        return self.n_green_lit / self.n_pairs if self.n_pairs else float("nan")

    @property
    def unsafe_miss_rate(self) -> float:
        """Of the pairs that under-covered, the fraction the rule green-lit.

        A different quantity from :attr:`false_safe_rate` -- same numerator, different
        denominator -- and the two are routinely conflated. This one is one minus recall;
        the endpoint above is one minus precision.
        """
        return self.false_safe / self.n_unsafe if self.n_unsafe else float("nan")

    @property
    def false_alarm_rate(self) -> float:
        """Of the pairs that were safe, the fraction the rule needlessly blocked."""
        n_safe = self.true_safe + self.false_alarm
        return self.false_alarm / n_safe if n_safe else float("nan")

    @property
    def base_rate_unsafe(self) -> float:
        """Fraction of all pairs that under-covered."""
        return self.n_unsafe / self.n_pairs if self.n_pairs else float("nan")

    @property
    def trivial_false_safe_rate(self) -> float:
        """What "always say safe" achieves -- the bar :attr:`false_safe_rate` must clear."""
        return self.base_rate_unsafe

    @property
    def beats_trivial(self) -> bool:
        """Lower false-safe rate than saying nothing, *and* green-lighting a real share.

        The green-light floor is deliberate. Without it a rule that admits one pair in
        fifty and gets it right would read as a strong result.
        """
        if not np.isfinite(self.false_safe_rate):
            return False
        return (
            self.false_safe_rate < self.trivial_false_safe_rate
            and self.green_light_fraction >= 0.10
        )

    @property
    def verdict(self) -> str:
        """One word, with the refusals named rather than scored."""
        if self.n_pairs < 4:
            return "too_few_pairs"
        if self.n_unsafe == 0:
            return "no_unsafe_pairs"
        if self.n_green_lit == 0:
            return "green_lights_nothing"
        if self.green_light_fraction < 0.10:
            return "green_lights_too_little"
        if self.false_safe_rate >= self.trivial_false_safe_rate:
            return "no_better_than_trivial"
        if np.isfinite(self.p_value) and self.p_value > 0.05:
            return "not_separable_from_permutation_null"
        return "positive"

    def as_row(self) -> dict[str, object]:
        """Flat record for a results table, prereg fingerprint included."""
        return {
            "cells": self.label,
            "n_pairs": self.n_pairs,
            "n_excluded": self.n_excluded,
            "true_safe": self.true_safe,
            "false_safe": self.false_safe,
            "false_alarm": self.false_alarm,
            "true_unsafe": self.true_unsafe,
            "false_safe_rate": self.false_safe_rate,
            "false_safe_lo": self.false_safe_ci[0],
            "false_safe_hi": self.false_safe_ci[1],
            "trivial_false_safe_rate": self.trivial_false_safe_rate,
            "green_light_fraction": self.green_light_fraction,
            "unsafe_miss_rate": self.unsafe_miss_rate,
            "false_alarm_rate": self.false_alarm_rate,
            "youden_j": self.youden_j,
            "p_value": self.p_value,
            "loo_false_safe_max": (
                max(self.loo_false_safe) if self.loo_false_safe else float("nan")
            ),
            "pair_completeness": self.pair_completeness,
            "beats_trivial": self.beats_trivial,
            "verdict": self.verdict,
            "prereg": prereg.fingerprint(),
        }


# -- internals -----------------------------------------------------------------


def _check(frame: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise KeyError(f"ordered-pair frame is missing {missing}; needed {list(REQUIRED)}")
    same = frame["calibration"].astype(str) == frame["deployment"].astype(str)
    if same.any():
        raise ValueError(
            f"{int(same.sum())} row(s) pair an operator with itself; a matched-operator "
            "cell is the control, not a transfer pair, and its distance is zero by "
            "construction so it would be green-lit for free"
        )
    dup = frame.duplicated(subset=["calibration", "deployment"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} ordered pair(s) appear more than once")
    hi, point = frame["coverage_hi"].to_numpy(dtype=float), frame["coverage"].to_numpy(float)
    both = np.isfinite(hi) & np.isfinite(point)
    if np.any(hi[both] < point[both] - 1e-9):
        bad = int((hi[both] < point[both] - 1e-9).sum())
        raise ValueError(
            f"{bad} pair(s) have coverage_hi below coverage; the upper end of an interval "
            "cannot sit under its own point estimate, so the columns are swapped or the "
            "interval was computed on different rows"
        )


def _predict_safe(signed: np.ndarray) -> np.ndarray:
    """The rule. ``<=`` so that an exactly-zero distance is called safe."""
    return signed <= prereg.SIGN_RULE_THRESHOLD


def _actually_safe(coverage_hi: np.ndarray, tolerance: float) -> np.ndarray:
    """A cell is safe unless its coverage interval's **upper end** misses nominal (D85).

    Thresholding the upper end means a cell is only called unsafe when even the optimistic
    reading of its coverage falls short. That is the conservative direction for this
    endpoint: it makes "unsafe" harder to earn, so the rule cannot be made to look accurate
    by manufacturing easy negatives out of cells that merely scattered low.
    """
    return coverage_hi >= prereg.NOMINAL - tolerance


def _counts(pred_safe: np.ndarray, truth_safe: np.ndarray) -> tuple[int, int, int, int]:
    """``(true_safe, false_safe, false_alarm, true_unsafe)``."""
    return (
        int((pred_safe & truth_safe).sum()),
        int((pred_safe & ~truth_safe).sum()),
        int((~pred_safe & truth_safe).sum()),
        int((~pred_safe & ~truth_safe).sum()),
    )


def _youden_j(ts: int, fs: int, fa: int, tu: int) -> float:
    """``sensitivity + specificity - 1`` for the *unsafe* call.

    Sensitivity is ``tu / (tu + fs)``, the share of under-covering pairs the rule blocked;
    specificity is ``ts / (ts + fa)``, the share of safe pairs it let through. Both
    denominators are counts of the measured outcomes, which the operator-label permutation
    leaves untouched -- that invariance is why this, and not the false-safe rate, is the
    statistic the null is computed on.
    """
    n_unsafe, n_safe = tu + fs, ts + fa
    if n_unsafe == 0 or n_safe == 0:
        return float("nan")
    return tu / n_unsafe + ts / n_safe - 1.0


def _distance_lookup(frame: pd.DataFrame) -> dict[tuple[str, str], float]:
    """``(calibration, deployment) -> signed distance``, with the antisymmetric fill.

    ``null_space_growth`` is exactly antisymmetric, so an unobserved reverse pair is
    ``-d`` rather than missing. Without the fill a permutation that asks for a direction
    the frame happens not to contain would drop the pair, and the null would be computed
    on a shrinking sample.
    """
    out: dict[tuple[str, str], float] = {}
    for a, b, d in zip(
        frame["calibration"].astype(str),
        frame["deployment"].astype(str),
        frame["signed_distance"].to_numpy(dtype=float),
        strict=True,
    ):
        out[(a, b)] = d
        out.setdefault((b, a), -d)
    return out


def _pair_completeness(frame: pd.DataFrame) -> float:
    """Observed ordered pairs, including antisymmetric fills, over the ``k(k-1)`` possible."""
    lookup = _distance_lookup(frame)
    k = len(_operators(frame))
    return len(lookup) / (k * (k - 1)) if k > 1 else float("nan")


def _operators(frame: pd.DataFrame) -> list[str]:
    return sorted(
        set(frame["calibration"].astype(str)) | set(frame["deployment"].astype(str))
    )


def _permutation_p(
    frame: pd.DataFrame,
    truth_safe: np.ndarray,
    observed_j: float,
    rng: np.random.Generator,
) -> float:
    """Two-sided p-value on ``|J|`` under permutation of the operator labels.

    Returns NaN -- not a number -- when the grid is too sparse for the permutation to be
    an exchangeability argument. See :data:`MIN_PAIR_COMPLETENESS`.
    """
    if not np.isfinite(observed_j):
        return float("nan")
    lookup = _distance_lookup(frame)
    operators = _operators(frame)
    if len(operators) < 3:
        return float("nan")
    if _pair_completeness(frame) < MIN_PAIR_COMPLETENESS:
        return float("nan")
    calib = frame["calibration"].astype(str).to_numpy()
    deploy = frame["deployment"].astype(str).to_numpy()

    hits = 0
    done = 0
    for _ in range(prereg.N_PERMUTATIONS):
        mapping = dict(zip(operators, rng.permutation(operators), strict=True))
        pairs = zip(calib, deploy, strict=True)
        drawn = np.array([lookup.get((mapping[a], mapping[b]), np.nan) for a, b in pairs])
        ok = np.isfinite(drawn)
        if ok.sum() < 4:
            continue
        j = _youden_j(*_counts(_predict_safe(drawn[ok]), truth_safe[ok]))
        if not np.isfinite(j):
            continue
        done += 1
        if abs(j) >= abs(observed_j):
            hits += 1
    if done == 0:
        return float("nan")
    # Add-one, so a p-value is never reported as exactly zero on a finite null.
    return (hits + 1) / (done + 1)


def _bootstrap_ci(
    frame: pd.DataFrame,
    truth_safe: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile interval for the false-safe rate, resampling **operators**.

    The 72 ordered pairs are not 72 independent observations; they are every arrangement
    of 9 operators, so a pair-level bootstrap would report an interval roughly sqrt(8)
    times too narrow. Resampling the operators and keeping every pair among the drawn
    multiset respects that dependence. It is the node bootstrap for dyadic data and is
    approximate -- with nine clusters the interval is wide, and that width is the
    measurement, not a defect.
    """
    operators = _operators(frame)
    if len(operators) < MIN_OPERATORS_FOR_CI:
        return (float("nan"), float("nan"))

    signed = frame["signed_distance"].to_numpy(dtype=float)
    pred_safe = _predict_safe(signed)
    by_pair: dict[tuple[str, str], tuple[bool, bool]] = {
        (a, b): (p, t)
        for a, b, p, t in zip(
            frame["calibration"].astype(str),
            frame["deployment"].astype(str),
            pred_safe,
            truth_safe,
            strict=True,
        )
    }

    rates = np.empty(prereg.N_BOOT, dtype=float)
    ops = np.asarray(operators, dtype=object)
    for b in range(prereg.N_BOOT):
        drawn = rng.choice(ops, size=ops.size, replace=True)
        fs = gl = 0
        for i, a in enumerate(drawn):
            for d in drawn[i + 1 :]:
                for key in ((a, d), (d, a)):
                    hit = by_pair.get(key)
                    if hit is None or not hit[0]:
                        continue
                    gl += 1
                    fs += not hit[1]
        rates[b] = fs / gl if gl else np.nan
    good = rates[np.isfinite(rates)]
    if good.size < prereg.N_BOOT // 10:
        return (float("nan"), float("nan"))
    return (float(np.percentile(good, 2.5)), float(np.percentile(good, 97.5)))


def _leave_one_operator_out(
    frame: pd.DataFrame, truth_safe: np.ndarray
) -> tuple[float, ...]:
    """False-safe rate on the pairs involving each operator, one operator at a time.

    The rule has no fitted parameter, so this is not cross-validation -- there is nothing
    to refit. It answers a different question: is the headline rate carried by one
    operator's pairs? A single operator whose held-out rate is far above the pooled one is
    the result's weakest point, and it is better found here than by a reviewer.
    """
    signed = frame["signed_distance"].to_numpy(dtype=float)
    pred_safe = _predict_safe(signed)
    calib = frame["calibration"].astype(str).to_numpy()
    deploy = frame["deployment"].astype(str).to_numpy()
    operators = sorted(set(calib) | set(deploy))

    out: list[float] = []
    for op in operators:
        involves = (calib == op) | (deploy == op)
        if involves.sum() < 2:
            continue
        gl = pred_safe[involves]
        if not gl.any():
            continue
        out.append(float((~truth_safe[involves][gl]).mean()))
    return tuple(out)


# -- entry points --------------------------------------------------------------


def safety_rule(
    frame: pd.DataFrame,
    label: str = "pooled",
    tolerance: float | None = None,
    seed: int = 0,
    with_null: bool = True,
) -> SafetyRule:
    """Evaluate the sign rule on one set of ordered pairs.

    ``frame`` carries one row per ordered pair: ``calibration``, ``deployment``,
    ``signed_distance`` (from calibration to deployment, positive meaning the deployment
    operator hides more), and ``coverage`` (measured at the deployment operator using the
    threshold fitted at the calibration operator).

    Rows whose coverage is not finite are **excluded and counted** -- an uncertifiable
    cell has no outcome to be right or wrong about, and scoring it either way would be an
    invention. ``tolerance`` defaults to the frozen
    :data:`caliber.evaluation.prereg.UNSAFE_TOLERANCE` and is exposed only so
    :func:`tolerance_sensitivity` can sweep it; the primary endpoint uses the frozen value.
    """
    _check(frame)
    tol = prereg.UNSAFE_TOLERANCE if tolerance is None else float(tolerance)

    coverage = frame["coverage"].to_numpy(dtype=float)
    coverage_hi = frame["coverage_hi"].to_numpy(dtype=float)
    usable = (
        np.isfinite(coverage)
        & np.isfinite(coverage_hi)
        & np.isfinite(frame["signed_distance"].to_numpy(dtype=float))
    )
    n_excluded = int((~usable).sum())
    sub = frame.loc[usable].reset_index(drop=True)
    if sub.empty:
        return SafetyRule(label, 0, n_excluded, 0, 0, 0, 0)

    signed = sub["signed_distance"].to_numpy(dtype=float)
    truth_safe = _actually_safe(sub["coverage_hi"].to_numpy(dtype=float), tol)
    ts, fs, fa, tu = _counts(_predict_safe(signed), truth_safe)
    j = _youden_j(ts, fs, fa, tu)

    rng = np.random.default_rng(seed)
    p = _permutation_p(sub, truth_safe, j, rng) if with_null else float("nan")
    ci = _bootstrap_ci(sub, truth_safe, rng) if with_null else (float("nan"), float("nan"))

    return SafetyRule(
        label=label,
        n_pairs=int(len(sub)),
        n_excluded=n_excluded,
        true_safe=ts,
        false_safe=fs,
        false_alarm=fa,
        true_unsafe=tu,
        youden_j=j,
        p_value=p,
        false_safe_ci=ci,
        loo_false_safe=_leave_one_operator_out(sub, truth_safe),
        pair_completeness=_pair_completeness(sub),
    )


def safety_by_family(
    frame: pd.DataFrame,
    family_col_calibration: str = "family_calibration",
    family_col_deployment: str = "family_deployment",
    seed: int = 0,
    with_null: bool = True,
) -> list[SafetyRule]:
    """The pooled rule **and** every ordered family cell, in that order.

    Reported together because they can disagree in sign, and the pooled number alone has
    hidden exactly that before: a pooled +0.097 on the alignment label concealed -0.152 at
    R4 against +0.100 at R8. Pooling across families is the same operation on a different
    axis, so both are always printed.
    """
    out = [safety_rule(frame, "pooled", seed=seed, with_null=with_null)]
    have = family_col_calibration in frame.columns and family_col_deployment in frame.columns
    if not have:
        return out
    for (fa, fb), part in frame.groupby(
        [family_col_calibration, family_col_deployment], dropna=False
    ):
        if len(part) < 4:
            continue
        out.append(
            safety_rule(
                part.reset_index(drop=True),
                f"{fa} -> {fb}",
                seed=seed,
                with_null=with_null,
            )
        )
    return out


def tolerance_sensitivity(
    frame: pd.DataFrame,
    tolerances: Sequence[float] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """The primary endpoint against the width of the band that counts as under-coverage.

    The frozen primary is ``tolerance = 0``: a cell is unsafe if its point estimate misses
    nominal at all. That is the conservative orientation, so this sweep can only make the
    rule look *better*, and it is reported to show how much of the headline rests on cells
    that miss nominal by a hair.
    """
    tols = prereg.UNSAFE_TOLERANCE_SENSITIVITY if tolerances is None else tolerances
    rows = []
    for tol in tols:
        r = safety_rule(frame, f"tolerance={tol}", tolerance=tol, seed=seed, with_null=False)
        rows.append(
            {
                "tolerance": tol,
                "frozen_primary": tol == prereg.UNSAFE_TOLERANCE,
                "n_unsafe": r.n_unsafe,
                "false_safe_rate": r.false_safe_rate,
                "trivial_false_safe_rate": r.trivial_false_safe_rate,
                "green_light_fraction": r.green_light_fraction,
                "beats_trivial": r.beats_trivial,
            }
        )
    return pd.DataFrame(rows)


def safety_table(results: Iterable[SafetyRule]) -> pd.DataFrame:
    """Stack :meth:`SafetyRule.as_row` into the table that goes in the paper."""
    return pd.DataFrame([r.as_row() for r in results])
