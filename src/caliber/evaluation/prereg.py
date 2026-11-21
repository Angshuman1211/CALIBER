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
    (
        "2026-11-21",
        "T1-B's GAP_DEFINITION, GAP_ERROR_METRIC, GAP_PREDICTION_PROTOCOLS and "
        "GAP_CLUSTER_UNIT named (D88). The freeze fixed the distance, the functional form "
        "and the baselines but never said what the *gap* is, which left the one quantity "
        "being predicted unfrozen. PARTIAL KNOWLEDGE AT DECLARATION, recorded in "
        "GAP_DECLARATION_KNEW: summary figures from T2.11 Sec. 1g were already reported to "
        "Track 1, including one monotonicity violation. Per-pair gaps were not in hand and "
        "no fit had been run.",
    ),
    (
        "2026-11-21",
        "GAP_PREDICTION_DIRECTION named explicitly as increasing. This CHANGES NOTHING: "
        "GAP_PREDICTION_FORM was already isotonic and the implementation already fitted a "
        "non-decreasing function, so the direction was frozen de facto from the start. The "
        "entry exists because the direction has now been REFUTED on the CT bank and a "
        "refutation must be legible without reading the implementation. No amendment can "
        "rescue it: a sign fitted after the sign is seen is free. GAP_BARRED_PREDICTORS "
        "and GAP_FACE_BANK_PREDICTION are added under the same date and are the only new "
        "constraints -- both narrow what may be claimed, neither widens it.",
    ),
    (
        "2026-11-21",
        "GAP_DIRECTION_SOURCE added: the direction of the fit stops being a frozen "
        "constant and becomes a function of the CALIBRATION THRESHOLDS, which every "
        "bank's ordered_pairs.csv already records and which take no coverage value as "
        "input. THIS AMENDMENT WIDENS WHAT MAY BE CLAIMED -- the first of the seven "
        "that does -- so it carries its own fence. GAP_DIRECTION_BANKS_BARRED names "
        "every bank whose per-pair gaps have already been seen; on those the rule may "
        "be shown and may not be reported, and the CT bank is named there. The motive "
        "is not the CT refutation but what D89's identity implies: the direction is a "
        "property of how tau orders the operators, so NO single frozen direction can "
        "be right for a project using two banks whose orderings differ. A constant was "
        "the wrong KIND of choice, which is a different defect from a wrong value.",
    ),
    (
        "2026-11-21",
        "GAP_ARM_SIGN_AGREEMENT_BARRED added. Track 2's T1-B run cited gap-sign "
        "agreement at 72 of 72 pairs between the registry and across-split arms as "
        "support for D86. It is not support: by D89's identity the gap's sign is "
        "sign(tau_a - tau_b) under either arm, so the statistic says only that the two "
        "arms rank the nine thresholds the same way. It is compatible with any amount "
        "of non-cancellation and it breaks under perfect cancellation if one adjacent "
        "threshold pair transposes. The magnitude comparison D88 actually predicted "
        "stands and is strengthened; the sign count comes out of the support set.",
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

#: **One direction, and it is part of the claim.** Isotonic regression as implemented fits
#: a *non-decreasing* function, so "isotonic" alone already froze this; naming it makes the
#: hypothesis falsifiable on its face rather than only in the source.
#:
#: **Refuted on the CT bank, 21 November 2026.** Within every family cell the gap *falls*
#: with the signed distance (rho -0.72, -0.70, -0.70, -1.00). The cause is not subtle and
#: is derived in D89: coverage of positives at ``b`` is ``1 - F_b(tau)``, so
#: ``gap = F_b(tau_a) - F_b(tau_b)`` and ``sign(gap) == sign(tau_a - tau_b)`` exactly. The
#: threshold rises with ill-posedness because the score tracks reconstruction error, and
#: the signed distance is positive when ``b`` is the more ill-posed operator -- so a
#: positive distance forces a negative gap. The frozen direction is backwards by
#: arithmetic, not by bad luck.
#:
#: A decreasing fit is reported beside it as a DIAGNOSTIC and is barred from the verdict.
#: Letting it set ``reportable`` would be fitting one bit to the data that refuted the
#: hypothesis, on the same grid, and calling the result a confirmation.
GAP_PREDICTION_DIRECTION = "increasing"

#: Directions the output may *show*. Only the frozen one may be *reported as a result*.
GAP_PREDICTION_DIRECTIONS = ("increasing", "decreasing")

#: **Where the direction comes from, as of the seventh amendment.** D89's identity makes
#: ``sign(gap) == sign(tau_a - tau_b)`` a theorem, so the sign of the gap-versus-distance
#: relation is fixed by how the calibration threshold orders the operators along the
#: distance's own axis -- a property of the bank, not of the hypothesis. The CT bank and
#: the face-prior bank order tau oppositely (rho -0.70..-1.00 against +0.976), which is why
#: one frozen constant could not have been right for both.
#:
#: The rule takes the recorded thresholds and the signed distance and **nothing else**: no
#: coverage, no gap, no fitted quantity. It therefore spends one bit on the outcome's
#: *global* sign, against the 72 per-pair values that :data:`GAP_BARRED_PREDICTORS` forbids
#: -- and that one bit is the same for all 72 pairs of a bank, so it is checkable as one
#: number rather than hidden in a fit.
GAP_DIRECTION_SOURCE = "threshold_ordering"

#: The rule, spelled out so it can be run by hand from one CSV.
GAP_DIRECTION_RULE = (
    "let d be the signed distance and tau the calibration threshold, both per pair",
    "if rho(d, tau_deployment - tau_calibration) > 0 the fit is non-increasing",
    "if it is < 0 the fit is non-decreasing",
    "inputs are thresholds and distances only; no coverage or gap may enter",
)

#: **Banks on which the rule may be shown and may not be reported**, because their per-pair
#: gaps were already seen when the rule was written. Choosing a direction after seeing the
#: gaps it has to fit is free, whatever the stated justification, and a fence that names the
#: bank is the only kind that cannot be argued around later.
GAP_DIRECTION_BANKS_BARRED = ("lidc_ct",)

#: **When the rule declines to answer.** Near-flat tau leaves the ordering undetermined and
#: the correlation in :data:`GAP_DIRECTION_RULE` will still return a sign, from noise. The
#: across-operator tau range must exceed the median per-operator tau sampling spread --
#: T2-A4's quantity, measured, not assumed -- by at least this factor, or the direction is
#: recorded as undetermined and no fit is reported under this source at all.
GAP_DIRECTION_DETERMINACY_MIN_RATIO = 3.0

#: **A statistic barred from the support set, with the reason it cannot work.**
GAP_ARM_SIGN_AGREEMENT_BARRED = (
    "gap-sign agreement between label arms may not be cited as evidence for D86",
    "by D89 the gap's sign is sign(tau_a - tau_b) under either arm, so the count"
    " measures only whether the two arms rank the thresholds the same way",
    "it stays at 100% under arbitrarily large non-cancellation and falls below it"
    " under perfect cancellation if one adjacent threshold pair transposes",
    "report instead corr(level_difference(a,b), level_difference(b,b)), which is"
    " the common-mode quantity D86's argument actually rests on",
)

#: **Quantities barred from the baseline set because they are the endpoint in other units.**
#:
#: Added before anyone proposed them, which is the only time such a bar is worth anything.
#: Having derived ``gap = F_b(tau_a) - F_b(tau_b)``, the temptation is to enter
#: ``tau_a - tau_b`` as a predictor; it would beat every baseline and the distance, and the
#: win would be arithmetic. Its sign agreement with the gap is 100% by theorem, not by
#: measurement, and a tautology entered as a predictor is a further form of the failure
#: mode this module exists to prevent. (The project's running tally of instances lives
#: in the decision log, which is the only place it can be kept consistently.)
#:
#: They stay barred from the baselines and are used only in
#: :func:`caliber.evaluation.gap.gap_sign_from_thresholds`, as a CHECK ON THE BANK: an
#: agreement rate below 100% means the recorded thresholds and coverages contradict the
#: definition of coverage, which is a defect no coverage result can survive. A check that
#: can only find faults is not a result that can only flatter.
GAP_BARRED_PREDICTORS = ("threshold_difference", "tau_difference", "calibration_quantile")

#: **Stated before the number exists.** The signed distance ordered the gaps at rho = +0.976
#: on the face-prior bank and at rho -0.70 to -1.00 within the CT family cells. If D89's
#: account is right, the sign of that relation is set by how the calibration threshold
#: orders the operators, so the face bank must have the OPPOSITE threshold ordering: tau
#: must *fall* as the face operators hide more of the image. The face bank's own
#: ``ordered_pairs.csv`` already holds the thresholds, so this costs one groupby and no new
#: bank pass. If tau rises there too, the account in D89 is wrong and the sign difference
#: between the two banks is unexplained -- which must then be reported as unexplained.
GAP_FACE_BANK_PREDICTION = (
    "tau is decreasing in the signed distance measure on the face-prior bank",
    "equivalently: the two banks disagree on the threshold ordering, not on the mechanism",
    "falsified if tau rises with the distance on the face bank as it does on CT",
)

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

#: What the gap *is*. **Declared 21 November 2026** (D88): the freeze named the distance,
#: the functional form and the baselines, but never the quantity being predicted -- so the
#: one remaining degree of freedom was the response variable itself.
#:
#: ``matched_minus_transfer``: ``coverage(b -> b) - coverage(a -> b)``. The claim is that
#: distance predicts *how much coverage you lose by having calibrated somewhere else*, and
#: what you would otherwise have had is the matched cell at the deployment operator. This
#: form subtracts it, so the per-operator level of the bank -- which is a property of the
#: bank, not of the shift -- is removed from the response rather than left in it. It also
#: vanishes identically at zero distance, which anchors a monotone fit at the origin.
GAP_DEFINITION = "matched_minus_transfer"

#: Reported beside it, always: ``NOMINAL - coverage(a -> b)``, the operational reading, which
#: is what a deployment certificate actually cares about. On T2-A's bank the two differ by
#: the matched-cell spread (0.892 to 0.905, so at most about 0.013) against gaps reaching
#: 0.5, so the choice should not change the verdict -- which is precisely why it is safe to
#: name now and would not have been safe to name after fitting.
GAP_DEFINITION_SENSITIVITY = ("matched_minus_transfer", "nominal_minus_transfer")

#: Mean absolute error, and **not** the criterion isotonic regression optimises.
#:
#: PAVA minimises squared error, so scoring the fit by RMSE would grade the method on its
#: own objective while the baselines are graded on someone else's. MAE also keeps the one
#: extreme pair from setting the comparison on its own: with 72 points and a worst gap near
#: 0.5, a squared-error summary is largely that pair. RMSE is reported beside it.
GAP_ERROR_METRIC = "mae"

#: Both held-out protocols, reported separately; neither is selected over the other.
#:
#: ``cross_family`` fits on one family and predicts the other, in both directions. It is the
#: test with teeth, because no acquisition parameter orders a limited-angle operator against
#: a sparse-view one (D84), so it is the only place the probe-based measure can earn its
#: keep. ``leave_one_operator_out`` holds out all 16 ordered pairs involving one operator,
#: nine folds, and asks whether an operator's gaps are predictable from the other eight.
GAP_PREDICTION_PROTOCOLS = ("cross_family", "leave_one_operator_out")

#: The resampling unit for every interval on a gap-prediction error: the **operator**.
#:
#: The 72 ordered pairs are not 72 independent observations. They are generated by 9
#: operators, each appearing in 16 of them, so resampling pairs would report an interval
#: for a sample size the design does not have -- the same error as treating 13 overlapping
#: re-cuts as 13 draws (D86). Operators are the exchangeable unit and there are nine.
GAP_CLUSTER_UNIT = "operator"

#: **What Track 1 had already been told about the gaps when the above was declared.**
#:
#: An honest freeze states its own contamination. These summary figures from T2.11 Sec. 1g
#: had been reported before ``GAP_DEFINITION`` was named, so the declaration is "before the
#: per-pair gaps" and not "before any knowledge of them". The last entry matters most: a
#: monotonicity violation was already known, so the sv 120 anomaly **cannot be reported as
#: a discovery of the fit**, and no sv-120 term may be added to the model.
GAP_DECLARATION_KNEW = (
    "naive transfer below nominal at 36 of 72 ordered pairs, 22 of them below 0.80",
    "worst single pair 0.377, at limited-angle 90 deg -> sparse-view 120",
    "matched arm 0.892 to 0.905 across the nine operators",
    "downhill transfers over-cover: from sparse-view 180, 0.964 at sv 30 and 0.994 at la 90",
    "uphill transfers under-cover: from sparse-view 30, 0.731 at sv 180 and 0.676 at sv 120",
    "sv 120 is out of order -- hardest to cover from almost every row (a known violation)",
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
        "gap_prediction_direction": GAP_PREDICTION_DIRECTION,
        "gap_prediction_directions": list(GAP_PREDICTION_DIRECTIONS),
        "gap_direction_source": GAP_DIRECTION_SOURCE,
        "gap_direction_rule": list(GAP_DIRECTION_RULE),
        "gap_direction_banks_barred": list(GAP_DIRECTION_BANKS_BARRED),
        "gap_direction_determinacy_min_ratio": GAP_DIRECTION_DETERMINACY_MIN_RATIO,
        "gap_arm_sign_agreement_barred": list(GAP_ARM_SIGN_AGREEMENT_BARRED),
        "gap_barred_predictors": list(GAP_BARRED_PREDICTORS),
        "gap_face_bank_prediction": list(GAP_FACE_BANK_PREDICTION),
        "gap_prediction_baselines": list(GAP_PREDICTION_BASELINES),
        "baseline_scope": dict(BASELINE_SCOPE),
        "gap_prediction_report_by_family_scope": GAP_PREDICTION_REPORT_BY_FAMILY_SCOPE,
        "gap_definition": GAP_DEFINITION,
        "gap_definition_sensitivity": list(GAP_DEFINITION_SENSITIVITY),
        "gap_error_metric": GAP_ERROR_METRIC,
        "gap_prediction_protocols": list(GAP_PREDICTION_PROTOCOLS),
        "gap_cluster_unit": GAP_CLUSTER_UNIT,
        "gap_declaration_knew": list(GAP_DECLARATION_KNEW),
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
