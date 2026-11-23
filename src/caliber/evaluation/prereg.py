"""The Phase 4 pre-registration, as constants rather than prose (T1-A..E, D-070 Sec. 2).

Eight of this project's eleven statistics traps were Track 1's own, and all but the last
were a choice made *after* seeing a number: a bandwidth picked once the gaps were known, a
label orientation settled once the AUC had a sign, a pooled statistic reported because the
per-cell ones disagreed. The eleventh (D93) is the exception that this module cannot
prevent on its own -- a statistic frozen here, correct, and still blind to the error beside
it -- which is why DIMENSIONLESS_PASS_WARNING asks for a dimensioned check as well.

The work plan's Sec. 2 asks for those choices to be frozen before the CT bank finishes.
A table in a Markdown file is not a freeze -- nothing stops it being edited in the same
commit as the result it governs.

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
        "input. THIS AMENDMENT WIDENS WHAT MAY BE CLAIMED -- the only entry in this "
        "tuple that does, which is why it carries its own fence. GAP_DIRECTION_BANKS_BARRED names "
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
    (
        "2026-11-22",
        "TRANSFER_SPREAD_PREDICTION added, before the number exists, on the same footing "
        "as GAP_FACE_BANK_PREDICTION. D92 derives the coverage law at a transfer cell -- "
        "the matched Beta pushed through the pair's probability-probability map -- and it "
        "predicts the across-split SPREAD of every transfer cell from the two operators' "
        "positive-score distributions, with no free parameter. Track 2's 13-split table "
        "can falsify it. This narrows what may be claimed: without the entry the ratio "
        "could be fitted after the fact and reported as agreement.",
    ),
    (
        "2026-11-22",
        "GAP_DIRECTION_DETERMINACY_SCOPE and GAP_DIRECTION_POOLED_RHO_BARRED added. "
        "Track 2 found the defect: on the CT bank the rule's pooled rho is +0.05 while "
        "every family cell is +0.55 to +1.00, and the determinacy test passed anyway "
        "because it checked tau's RANGE against tau's noise and never asked whether the "
        "correlation it reads was resolved. Pooling two families whose tau levels are "
        "offset washes out a unanimous within-cell signal; on a fixture reproducing the "
        "bank's pattern the pooled sign flipped in 52% of bootstrap resamples while "
        "within-cell unanimity broke in 0.1%. The rule now reads within-family cells and "
        "requires their signs to agree. THIS AMENDMENT NARROWS WHAT MAY BE CLAIMED: it "
        "adds a condition the rule must pass, and both banks pass it.",
    ),
    (
        "2026-11-22",
        "BETA_REFERENCE_UNIT added and TRANSFER_SPREAD_PREDICTION's equal-count clause "
        "retracted. Two corrections, both mine. (a) The clause asserted the bank has "
        "~10,760 positives per operator and that the reference cell is therefore not "
        "load-bearing; the counts are 3,157 to 21,762 and 10,760 is their MEAN, so the "
        "choice is load-bearing and Track 2 measured it: the calibration operator's "
        "matched cell gives rho +0.977 and slope 1.08, the deployment operator's +0.88 "
        "and 0.66. The original instinct was right and the correction that softened it "
        "was the error. (b) The n fed to the Beta is the positive count, but CLUSTER_UNIT "
        "has said 'patient' since D83, D73 recorded that the threshold is an order "
        "statistic of 'a few dozen' positives, and the comment block above "
        "ClusteredCoverage already states that 16,400 CT regions behave like roughly 35 "
        "units and that NEITHER count should be the gate. None of that is new here and it "
        "is not written up as new. What is new is narrow: TransferBetaReference.interval "
        "and .contains, added yesterday, shipped without the label every other use of the "
        "Beta in this codebase carries, and read as a certificate. They now refuse unless "
        "a unit is passed in and are documented as the rescaled idealisation, not a gate. "
        "NARROWS: it removes a certificate that should never have been one.",
    ),
    (
        "2026-11-22",
        "DIMENSIONLESS_PASS_WARNING added, before the follow-up number exists. "
        "TRANSFER_SPREAD_PREDICTION passed at rho +0.977 while the reference it was "
        "computed from had an n wrong by a factor of order 100 -- because the predicted "
        "quantity is a RATIO of two spreads and n divides out of both sides. A "
        "dimensionless prediction passing is therefore not evidence that the reference it "
        "was built from is sound, and this is the form of the failure mode the project had "
        "not yet seen: not a statistic chosen after the fact, but a true statistic that is "
        "blind to the error beside it.",
    ),
    (
        "2026-11-22",
        "DIMENSIONLESS_PASS_REGIME and COUNT_FROM_LEVEL_RANGE added; both narrow. "
        "(a) Amendment 12 said the reference's n divides out of the predicted ratio. "
        "Measured over seven score-distribution families, that holds to about 3% only "
        "where |G'| is of order 1 or larger; where |G'| is far below 1 the ratio at 41 "
        "units and at 21,762 differ by up to 3.3x, because the pushforward is n-free only "
        "while G is linear across U\'s range and that range is set by n. The remedy "
        "amendment 12 froze stands; its premise now carries the regime it holds in, which "
        "is the regime the dangerous cells live in. (b) The dimensioned check that "
        "amendment 12 demanded has been run, on a number Track 2 had already reported "
        "without either of us noticing what it measured: tau sat between the 0.03 and 0.18 "
        "levels of F_a over 13 splits, and 13 draws of Beta(4, 38) have median min 0.034 "
        "and median max 0.177, while the region count gives [0.097, 0.103]. The check "
        "passes at the subject count and excludes the region count by a factor of order "
        "300. COUNT_FROM_LEVEL_RANGE names the method and fixes that it yields a BRACKET; "
        "it is frozen here before the face bank, where it has not been applied.",
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

#: **The scope the rule's correlation is read at, as of the tenth amendment.** Track 2
#: found that the ratio above is not sufficient: on the CT bank tau's range clears its
#: noise by 12.5x, so the determinacy test passed, while the pooled correlation the rule
#: actually reads was +0.05 -- indistinguishable from zero, and about to hand back a
#: direction drawn from noise. Every family cell was +0.55 to +1.00.
#:
#: The cause is Simpson's: the two families' tau levels are offset, so pooling the 72
#: pairs mixes a between-family level difference into a within-family trend and the trend
#: is washed out. On a fixture built to reproduce the bank's pattern the pooled sign
#: flipped in 52% of bootstrap resamples; within-cell unanimity broke in 0.1%.
#:
#: So the rule reads the four within-family cells, not the pool, and the cells must agree.
GAP_DIRECTION_DETERMINACY_SCOPE = "within_family_cell"

#: **The pooled correlation is barred from setting the direction**, for the reason above.
#: It may be shown; a rule that reads it is not this rule.
GAP_DIRECTION_POOLED_RHO_BARRED = (
    "the pooled rho(d, tau_b - tau_a) over all ordered pairs may not set the direction",
    "the two families' tau levels are offset, so the pool mixes a between-family level"
    " difference into a within-family trend and can wash it out entirely",
    "measured on the CT bank: pooled +0.05 against family cells +0.55 to +1.00",
    "the direction is determinate only when every family cell's rho has the same sign",
    "a near-zero pooled rho alongside unanimous cells is not a contradiction and not a"
    " reason to withhold the direction; it is a reason not to read the pool",
)

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

#: **Stated before the number exists (D92).** The coverage law at a transfer cell is the
#: matched Beta pushed through ``G_ab = F_b o F_a^{-1}``, so the across-split spread of
#: cell ``(a, b)`` must be the spread of cell ``(a, a)`` -- the matched cell of the
#: *calibration* operator, which shares the same calibration draw -- times ``G'`` at the
#: nominal level, which is the density ratio ``f_b/f_a`` at the threshold. Both sides are
#: computable from artefacts that already exist: the left from the 13 splits, the right
#: from the two operators' positive scores. 72 predictions, no fitted parameter.
#:
#: The ratio is predicted, not the absolute spread: the finite test set inflates each
#: spread by about 36% and the split overlap deflates it by about 2% (see
#: ``reliability.SpreadCheck``), and those corrections apply to numerator and denominator
#: alike only to the extent that the two cells have similar positive counts. Cells whose
#: counts differ materially are therefore excluded rather than explained.
TRANSFER_SPREAD_PREDICTION = (
    "sd[coverage(a->b)] / sd[coverage(a->a)] across splits equals f_b/f_a at the"
    " threshold, for every ordered pair",
    "the reference cell is the matched cell of the CALIBRATION operator, because that"
    " is the one sharing the calibration draw",
    "RETRACTED 2026-11-22, and the retraction left visible because the clause licensed"
    " the wrong reference: this entry used to say the choice is not load-bearing on a"
    " bank with ~10,760 positives per operator. The CT bank's counts are 3,157 to 21,762"
    " and 10,760 is their MEAN, so the premise was false and the choice IS load-bearing."
    " Measured by Track 2: the calibration operator's matched cell gives rho +0.977 and"
    " log-log slope 1.08, the deployment operator's +0.88 and 0.66. Use the calibration"
    " operator's, for the reason in the line above and not for the retracted one",
    "falsified if the measured ratios do not track the predicted ones across the 72"
    " pairs, once cells with materially different positive counts are dropped",
    "a ratio above 1 means the matched Beta band is too NARROW at that cell, which is"
    " the direction a deployment certificate must not get wrong",
    "this prediction is a RATIO, so it is blind to the n of the reference it is built"
    " from -- see DIMENSIONLESS_PASS_WARNING and BETA_REFERENCE_UNIT. It passed at rho"
    " +0.977 on a reference whose n was wrong by a factor of order 100",
)


# -- the Beta reference's sample size -----------------------------------------

#: **What counts as one independent observation in the Beta law.** Not the region.
#:
#: **This is not a new finding and must not be written up as one.** The comment block above
#: :class:`~caliber.evaluation.reliability.ClusteredCoverage` already records it: *"Regions
#: are not independent: on T2-A's CT bank, 16,400 calibration regions behave like roughly 35
#: units, because the unit is the subject."* That module also already measures it per cell --
#: ``ClusteredCoverage.design_effect``, ``.effective_n``, and an ``.independent_sd`` that
#: carries **both** variance terms -- and states the project's position outright: handed the
#: region count the Beta band is +/-0.011 and rejects a correctly calibrated bank, handed the
#: subject count it is +/-0.10 and accepts almost anything, so **neither is a measurement and
#: neither should be the gate**. The gate is the cluster bootstrap.
#:
#: So this constant does not introduce a unit. It records that one class shipped without the
#: label every other use of the Beta in this codebase carries, and closes that gap.
#:
#: **The effective count is not frozen here**, because a constant would be a number invented
#: for the convenience of having one when the code already computes it per cell from the
#: bootstrap. Read it from ``ClusteredCoverage.effective_n``.
#:
#: One cheap cross-check is worth recording, because it needs no bootstrap and no resampling
#: at all: **tau's own motion across splits implies the same order of magnitude.** Track 2
#: reports tau sitting between the 0.03 and 0.18 levels of ``F_a`` over 13 splits, and
#: :func:`~caliber.evaluation.reliability.count_from_level_range` brackets that span at
#: **10 to 70** -- which contains both the cohort size and the module's ~35 from the
#: bootstrap, and excludes the region count by a factor of order 300.
#:
#: **Read it as a bracket and not a count.** D93's first draft put a point estimate of
#: "near 45" on it and claimed agreement within 30%; the statistic does not resolve 35 from
#: 50 and :data:`COUNT_FROM_LEVEL_RANGE` now says so (D94). An earlier estimate of 174 came
#: from comparing the bootstrap width against a ONE-term reference, which is exactly the
#: error ``independent_sd``'s docstring documents; it is withdrawn.
BETA_REFERENCE_UNIT = "calibration_subject"

#: **No Beta band is a certificate, at either count.** The project's existing position,
#: restated here because one class was written as though it were.
BETA_REGION_COUNT_USE = (
    "a Beta reference on the region count may be SHOWN as the independent-regions"
    " idealisation, and must be labelled as one wherever it appears",
    "rescaling it to the subject count does not make it a certificate either: that band"
    " accepts almost anything, which is the other way to be wrong",
    "no Beta band may set a gate; the clustered bootstrap with"
    " REFIT_THRESHOLD_PER_REPLICATE is the only certificate path",
    "a rescaled band requires the unit to be passed in and stated, with no default,"
    " because the value is a property of the bank and assuming one is how regions got in",
)

#: **The form of the failure mode this project had not yet seen.** Frozen before the
#: follow-up number exists, because the warning is worth nothing after it does.
#:
#: Every previous instance was a statistic that came out well because it had been chosen,
#: framed or inverted after the fact. This one is different and worse: the statistic is
#: correct, the prediction was frozen before the number existed, the number came back at
#: rho +0.977 -- and it was blind to an error of order 100 standing right next to it,
#: because a ratio of two spreads has the reference's ``n`` in both numerator and
#: denominator and divides it out.
#:
#: A dimensionless quantity agreeing therefore certifies the SHAPE of a law and says
#: nothing about its SCALE. Any such agreement must be reported with the statement that
#: it does not validate the reference it was computed from, and with one dimensioned
#: check beside it.
DIMENSIONLESS_PASS_WARNING = (
    "a dimensionless prediction passing does not certify the reference it was built from",
    "TRANSFER_SPREAD_PREDICTION is a ratio of two spreads, so the reference's n divides"
    " out and an n wrong by a factor of 100 leaves the predicted ratio unchanged",
    "report every dimensionless agreement with one dimensioned check beside it",
    "for this prediction the dimensioned check is the matched cell's own across-split"
    " spread against its Beta sd at the independent count, which does not divide out",
    "the second clause above holds only in a regime, measured after the fact and stated"
    " in DIMENSIONLESS_PASS_REGIME: n divides out while G is linear over U's range, and"
    " U's range is itself set by n",
)

#: **When the spread ratio may be read without a unit.** Measured 22 November, after
#: amendment 12 asserted the stronger thing.
#:
#: ``sd_ratio`` is ``sd[1 - G(U)] / sd[U]``. If ``G`` were linear this is ``|G'|`` and no
#: ``n`` survives. ``G`` is not linear, and the width of the window it is sampled over is
#: ``sd[U]``, which *is* set by ``n`` -- so the ratio is not exactly dimensionless. The
#: departure was measured over seven score-distribution families (gamma, lognormal,
#: bimodal, crossing, heavy-tailed) at n = 41 against n = 21,762.
DIMENSIONLESS_PASS_REGIME = (
    "the ratio is n-free only while G is linear across U's range, and U's range is itself"
    " set by n, so sd_ratio is not exactly dimensionless",
    "where |G'| is of order 1 or larger the ratio at the two counts agrees to about 3%",
    "where |G'| is far below 1 the two counts disagree by up to 3.3x, because a narrow"
    " U window also makes the empirical G's own sampling noise the dominant term",
    "so the ratio may be reported without a unit only for cells where it is near 1 or"
    " above -- which are exactly the cells matched_band_is_too_narrow is about",
    "a cell whose ratio is far below 1 is harmless and its ratio is not quotable",
)

#: **The dimensioned check, named before it is applied to the second bank.**
#:
#: ``U := F_a(tau_a)`` is ``Beta(k, n + 1 - k)`` exactly, for ``n`` independent
#: calibration scores. Its across-split SPREAD therefore carries ``n``, where the
#: predicted spread *ratio* divides it out. Every split already records where its
#: threshold sat, so this costs no new run and no bootstrap.
COUNT_FROM_LEVEL_RANGE = (
    "the across-split range of F_a(tau_a) is a dimensioned read of the Beta's effective"
    " n, and is the check DIMENSIONLESS_PASS_WARNING requires beside the spread ratio",
    "it returns a bracket, never a count: it separates tens from thousands decisively"
    " and cannot separate 35 from 50",
    "k = floor(alpha*(n+1)) steps in integers, so the likelihood is not monotone in n and"
    " the consistent set can have holes; the bracket is its span and is reported as such",
    "it is calibration-side only, so it carries no test-set term -- which is what makes it"
    " a clean read, and also what stops it from being a coverage certificate",
    "no bracket from it may set a gate; the gate remains the clustered bootstrap",
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
        "gap_direction_determinacy_scope": GAP_DIRECTION_DETERMINACY_SCOPE,
        "gap_direction_pooled_rho_barred": list(GAP_DIRECTION_POOLED_RHO_BARRED),
        "gap_arm_sign_agreement_barred": list(GAP_ARM_SIGN_AGREEMENT_BARRED),
        "transfer_spread_prediction": list(TRANSFER_SPREAD_PREDICTION),
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
        "beta_reference_unit": BETA_REFERENCE_UNIT,
        "beta_region_count_use": list(BETA_REGION_COUNT_USE),
        "dimensionless_pass_warning": list(DIMENSIONLESS_PASS_WARNING),
        "dimensionless_pass_regime": list(DIMENSIONLESS_PASS_REGIME),
        "count_from_level_range": list(COUNT_FROM_LEVEL_RANGE),
        "circularity_control": CIRCULARITY_CONTROL,
    }


def fingerprint() -> str:
    """Short stable hash of the frozen choices, to stamp onto every result table.

    A table carrying a fingerprint that does not match the current module was produced
    under a different freeze, and the discrepancy is visible without reading git history.
    """
    blob = json.dumps(as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
