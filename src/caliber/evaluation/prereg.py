"""The Phase 4 pre-registration, as constants rather than prose (T1-A..E, D-070 Sec. 2).

Five of this project's eight statistics traps were Track 1's own, and every one of them
was a choice made *after* seeing a number: a bandwidth picked once the gaps were known, a
label orientation settled once the AUC had a sign, a pooled statistic reported because the
per-cell ones disagreed. The work plan's Sec. 2 asks for those choices to be frozen before
the CT bank finishes. A table in a Markdown file is not a freeze -- nothing stops it being
edited in the same commit as the result it governs.

So the choices live here, as module constants with no setters, and the analyses import them
instead of accepting them as arguments. Changing one is a diff against a file whose git
history is the audit trail, and :func:`fingerprint` lets a result record exactly which
version of the freeze produced it.

**Nothing in this module may be made configurable.** If an analysis needs a different
value, that is a new, separately named constant with its own dated rationale -- not a
keyword argument that lets the caller choose after seeing the outcome.
"""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

#: The date these choices were fixed. The CT bank (T2-A) had not finished running.
FROZEN_ON = "2026-11-18"

#: Amendments since the freeze, each with its date and why it was safe to make.
#:
#: An amendment is only legitimate while no number it governs exists. Both entries below
#: were made from operator-only data -- the signed distance matrix depends on the forward
#: operators and nothing else -- with T2-A's bank still unrun, and both are recorded here
#: rather than silently edited so that the audit trail shows what changed and when.
AMENDMENTS = (
    (
        "2026-11-18",
        "PERMUTATION_STATISTIC named as Youden's J; the false-safe rate is the endpoint "
        "but its denominator is not invariant under the permutation.",
    ),
    (
        "2026-11-20",
        "GAP_PREDICTION_BASELINES scoped by family and extended with "
        "null_space_dimension: the acquisition parameter is undefined on cross-family "
        "pairs. Found from the completed distance matrix, no coverage number in existence.",
    ),
    (
        "2026-11-20",
        "UNSAFE_LABEL moved from the coverage point estimate to the upper end of the "
        "cluster-bootstrap interval (D85). Per-cell coverage on T2-A's bank has an SD of "
        "about 0.05, so a point-estimate call at nominal is a coin flip near the boundary. "
        "Still the conservative direction; fixed before any ordered-pair coverage existed.",
    ),
    (
        "2026-11-20",
        "UNSAFE_LABEL_SOURCE fixed to the across-split mean rather than a single "
        "calibration draw (D86). T2-A1 showed one draw displaces every cell together by "
        "about 1.4 SD, so the error is common to all 72 pairs and does not average out "
        "across them. THIS AMENDMENT LOWERS THE PRIMARY ENDPOINT and so carries "
        "UNSAFE_LABEL_SOURCE_SENSITIVITY: the single-draw false-safe rate is reported "
        "beside the primary one, always, so the favourable direction stays visible.",
    ),
)

#: Miss-rate budget. Unchanged since Week 2; restated here so one import gives the
#: whole frozen configuration.
ALPHA = 0.1

#: Nominal coverage, the level a cell is judged against.
NOMINAL = 1.0 - ALPHA


# -- T1-A, the ordered-pair safety rule ---------------------------------------

#: The rule is ``predict safe iff signed_distance <= SIGN_RULE_THRESHOLD``.
#:
#: **Fixed at zero, not fitted.** Zero is the point where the deployment operator stops
#: hiding more of the image than the calibration operator did, so it is the only value
#: with a reading that does not depend on the grid. Fitting a cut-point on 72 pairs drawn
#: from 9 operators would be fitting one number to roughly 9 independent units, and the
#: false-safe rate it reported would be an in-sample one.
SIGN_RULE_THRESHOLD = 0.0

#: A cell counts as **actually unsafe** iff the **upper end of its cluster-bootstrap
#: coverage interval** is below :data:`NOMINAL` -- no tolerance band.
#:
#: **Amended 20 November 2026** (D85). The original form used the point estimate, frozen on
#: the stated grounds that it is the conservative orientation. On T2-A's bank that reasoning
#: does not survive contact with the numbers: per-cell coverage has a standard deviation of
#: about 0.05, so a point-estimate call at nominal is close to a coin flip for any cell
#: within a standard error of it, which is most of them. Thresholding the interval's upper
#: end is still the conservative direction -- it takes more evidence to call a cell unsafe,
#: so the rule cannot be flattered by easy negatives -- and it uses the interval the
#: project's own rules already require everywhere else. Fixed before any ordered-pair
#: coverage existed.
UNSAFE_TOLERANCE = 0.0

#: Named so that a table cannot be read as if it used the superseded point-estimate form.
UNSAFE_LABEL = "interval_upper_below_nominal"

#: Reported alongside the primary endpoint as a sensitivity curve, never in place of it.
UNSAFE_TOLERANCE_SENSITIVITY = (0.0, 0.01, 0.02, 0.05)

#: Which coverage the "actually unsafe" label is read from (D86).
#:
#: Split conformal promises nominal coverage *marginally over the calibration draw*, not on
#: any one draw. T2-A1 measured what one draw does on T2-A's bank: the registry split sits
#: about 1.4 SD low at every one of the nine operators at once, with the per-operator
#: amplitude set by that operator's own sensitivity (corr(shortfall, across-draw sd) =
#: +0.61). The displacement is therefore **common to the cells, not independent across
#: them** -- it moves all 72 ordered pairs the same way and cannot average out over them.
#: A label taken from one draw is biased in one direction by a known amount, which is not
#: what "conservative" was meant to buy.
UNSAFE_LABEL_SOURCE = "across_split_mean"

#: The minimum number of patient splits the across-draw mean may be taken over. Below this
#: the mean is noisier than the bias it removes and the single-draw label is preferable.
UNSAFE_LABEL_MIN_SPLITS = 8

#: The across-split label is the *more favourable* of the two for the sign rule, because the
#: single draw depresses coverage and so manufactures unsafe cells among the green-lit
#: pairs. The single-draw endpoint is therefore reported beside the primary one rather than
#: replaced by it, so a reader can see which way the choice cut.
UNSAFE_LABEL_SOURCE_SENSITIVITY = ("across_split_mean", "registry_split")

#: How the interval on the across-split mean is formed (D86).
#:
#: The splits are **re-cuts of one cohort**, not independent draws, so `sd / sqrt(n_splits)`
#: overstates the precision of their mean. The correction is the standard one for a mean of
#: correlated estimates, `n_eff = n / (1 + (n - 1) * rho)`, with `rho` the mean pairwise
#: calibration-set overlap -- at 13 splits sharing about a third, `n_eff` is near 2.6, so
#: the honest standard error is roughly 2.2x the naive one.
#:
#: The quantile is **normal, not Student t**, and that choice is frozen here rather than
#: left to the caller. `n_eff - 1` is about 1.6 degrees of freedom, where `t_{0.975}` is
#: near 6 and would widen the interval by a further ~3.3x -- which would shrink the unsafe
#: class until
#: the false-safe rate were near-vacuously zero, flattering the rule for an arithmetic
#: reason. `rho` is itself a modelled quantity, so compounding it with a t on 1.6 df stacks
#: one rough correction on another. The t width is **reported** beside the interval as a
#: diagnostic (`DrawMarginalCoverage.t_inflation`) so the choice is visible and a reader can
#: see what the stricter form would have done.
UNSAFE_LABEL_INTERVAL = "normal_overlap_corrected"

#: The permutation null shuffles **operator labels**, not the distance column.
#:
#: Shuffling distances independently would destroy the antisymmetry of
#: ``null_space_growth`` and the triangle structure of the matrix, giving a null in which
#: no sign rule could work and against which any result looks significant. Permuting the
#: operator names instead keeps the matrix intact and breaks only its association with
#: the outcomes, which is the hypothesis under test.
N_PERMUTATIONS = 10_000

#: Confidence intervals resample the **9 operators**, not the 72 pairs. The pairs are not
#: independent units; a pair-level bootstrap would report an interval roughly sqrt(8)
#: times too narrow. Nine clusters give a wide interval, and that width is the honest
#: answer rather than a defect to be engineered away.
N_BOOT = 10_000

#: Two-sided unless a one-sided test is pre-registered here by name. None is.
#:
#: Two-sided because a systematic *reversal* -- the rule reliably worse than chance -- is
#: itself a finding about the direction of the distance, not a null result to be discarded.
PERMUTATION_SIDED = "two-sided"

#: The statistic the permutation test is computed on: Youden's J for the unsafe call,
#: ``sensitivity + specificity - 1``.
#:
#: **Not the false-safe rate**, even though that is the primary endpoint. The false-safe
#: rate conditions on the set of pairs the rule green-lights, and that set changes size
#: under every permutation, so its null distribution mixes a change in the statistic with
#: a change in its denominator. J's two denominators are the counts of actually-safe and
#: actually-unsafe pairs, and the permutation does not touch the outcomes, so both are
#: invariant across replicates. The endpoint and the test statistic are therefore
#: different quantities on purpose, and both are reported.
PERMUTATION_STATISTIC = "youden_j"


# -- T1-B, gap prediction from operator distance ------------------------------

#: **One** distance measure, declared before fitting. The signed, directional one: the
#: hypothesis is about direction, and a symmetric measure cannot express it.
GAP_PREDICTION_DISTANCE = "null_space_growth"

#: Reported beside it for comparison, not selected between. Naming both as candidates and
#: picking the better one after the fact is the forking path Sec. 2 exists to close.
GAP_PREDICTION_DISTANCE_SYMMETRIC = "null_space_projection"

#: **One** functional form. Isotonic regression: the claim is monotonicity, and a
#: parametric curve would add a shape assumption the claim does not need. It also cannot
#: be tuned, which is the point.
GAP_PREDICTION_FORM = "isotonic"

#: The distance must beat **both** of these or T1-B is not reported as a result.
#: A constant predictor tests whether distance carries any information; a predictor built
#: from view count or arc fraction tests whether it carries information beyond the knob
#: that was turned, which is the objection that makes rho = +0.976 weak within a family.
GAP_PREDICTION_BASELINES = ("constant", "acquisition_parameter", "null_space_dimension")

#: **Amended 20 November 2026, before any coverage number existed** (see D84).
#:
#: The original two baselines were underspecified, and the completed distance matrix made
#: that visible on operator-only data. ``acquisition_parameter`` is well defined *within* a
#: family -- view count orders the six sparse-view operators, arc fraction orders the three
#: limited-angle ones -- and **undefined across families**: all three limited-angle
#: operators have 120 views, all six sparse-view operators span 180 degrees, so no single
#: scalar knob can order a limited-angle operator against a sparse-view one. Leaving that
#: unstated would have let the comparison be defined after the fact, which is the thing
#: this module exists to prevent.
#:
#: So the baseline is scoped, and ``null_space_dimension`` is added because it is defined
#: everywhere and therefore the only parameter-style baseline available on the cross-family
#: cells. It is also degenerate there -- the four operators at 120 views share a null-space
#: dimension of 64000 at 320^2 -- and that degeneracy is the point: it is what the
#: probe-based measure has to beat, and a baseline that cannot separate four of nine
#: operators is a weak one, which must be said rather than hidden by a good-looking margin.
BASELINE_SCOPE = MappingProxyType(
    {
        "constant": "all pairs",
        "acquisition_parameter": "within-family pairs only; undefined across families",
        "null_space_dimension": "all pairs; degenerate on the four operators at 120 views",
    }
)

#: Within-family and cross-family results are reported **separately and always both**.
#: Pooling them would let the 36 within-family pairs, where the acquisition parameter is a
#: strong baseline, average against the 36 cross-family pairs, where it does not exist.
GAP_PREDICTION_REPORT_BY_FAMILY_SCOPE = True


# -- T1-C, Tool 2's flag area --------------------------------------------------

#: Bandwidth and weighting are selected on **calibration operators only**, nested inside
#: each fold. Selecting on the held-out operator is the defect that made the first Tool 2
#: number unreportable.
TOOL2_SELECTION_SCOPE = "calibration_operators_only"

#: All weightings are reported, not the best one.
TOOL2_REPORT_ALL_WEIGHTINGS = True

#: Area is compared **only in cells where both arms are valid**. ``nearest`` is invalid in
#: 3 of 7 interior folds, and comparing against it there is what D-070 Sec. 3 corrected.
TOOL2_REQUIRE_BOTH_VALID = True

#: The baselines Tool 2 must be scored against. ``nearest`` is excluded deliberately.
TOOL2_BASELINES = ("max_quantile", "pooled", "group_conditional_oracle")


# -- T1-D / T3-A, thresholds inside the bootstrap ------------------------------

#: The calibrated threshold is recomputed inside **every** bootstrap resample, at the
#: clustering unit's level. Holding it fixed is D73's defect: the threshold is a single
#: order statistic of a few dozen positives, so its own variability is the larger term,
#: and ignoring it understated the interval by roughly 5x.
REFIT_THRESHOLD_PER_REPLICATE = True

#: Resampling unit for every coverage interval. Regions within a patient are not
#: independent.
CLUSTER_UNIT = "patient"


# -- T1-E, circularity ---------------------------------------------------------

#: The scale-only null control runs **alongside** the circularity analysis, not after it.
#: Run afterwards it becomes a check on a result already written up, which is how a
#: control stops being one.
CIRCULARITY_CONTROL_CONCURRENT = True

#: The control statistic: the null-space magnitude alone, carrying no posterior
#: information, under the same label orientation.
CIRCULARITY_CONTROL = "null_space_magnitude_only"


# -- phrasings that must not drift ---------------------------------------------

#: Claims that are measured and false, with what to write instead. ``tests/`` greps the
#: documentation for the left-hand sides.
FORBIDDEN_PHRASINGS = MappingProxyType(
    {
        "the curves collapse": (
            "The strong cross-domain form is measured and fails (D70/D72). Write what was "
            "measured: a shared monotone trend within a modality."
        ),
        "13 independent splits": (
            "30 patients resampled 13 ways is a robustness check against split choice, not "
            "13 replications. Write 'on every one of 13 patient splits'."
        ),
        "no conformal work in MRI risk estimation": (
            "False -- Shor and Bronstein use split conformal in Appendix H and CQR as a "
            "comparator. Write 'no prior work reports conformal coverage when the "
            "acquisition operator differs between calibration and deployment'."
        ),
    }
)


def as_dict() -> dict[str, object]:
    """Every frozen choice, for writing into a result's provenance block."""
    return {
        "frozen_on": FROZEN_ON,
        "amendments": [d for d, _ in AMENDMENTS],
        "alpha": ALPHA,
        "sign_rule_threshold": SIGN_RULE_THRESHOLD,
        "unsafe_tolerance": UNSAFE_TOLERANCE,
        "unsafe_label": UNSAFE_LABEL,
        "unsafe_label_source": UNSAFE_LABEL_SOURCE,
        "unsafe_label_min_splits": UNSAFE_LABEL_MIN_SPLITS,
        "unsafe_label_interval": UNSAFE_LABEL_INTERVAL,
        "n_permutations": N_PERMUTATIONS,
        "n_boot": N_BOOT,
        "permutation_sided": PERMUTATION_SIDED,
        "permutation_statistic": PERMUTATION_STATISTIC,
        "gap_prediction_distance": GAP_PREDICTION_DISTANCE,
        "gap_prediction_form": GAP_PREDICTION_FORM,
        "gap_prediction_baselines": list(GAP_PREDICTION_BASELINES),
        "baseline_scope": dict(BASELINE_SCOPE),
        "gap_prediction_report_by_family_scope": GAP_PREDICTION_REPORT_BY_FAMILY_SCOPE,
        "tool2_selection_scope": TOOL2_SELECTION_SCOPE,
        "tool2_baselines": list(TOOL2_BASELINES),
        "tool2_require_both_valid": TOOL2_REQUIRE_BOTH_VALID,
        "refit_threshold_per_replicate": REFIT_THRESHOLD_PER_REPLICATE,
        "cluster_unit": CLUSTER_UNIT,
        "circularity_control": CIRCULARITY_CONTROL,
    }


def fingerprint() -> str:
    """Short stable hash of the frozen choices, to stamp onto every result table.

    A table carrying a fingerprint that does not match the current module was produced
    under a different freeze, and the discrepancy is visible without reading git history.
    """
    blob = json.dumps(as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
