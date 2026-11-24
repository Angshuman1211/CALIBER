"""The Phase 4 pre-registration, as constants rather than prose (T1-A..E, D-070 Sec. 2).

Nine of this project's twelve statistics traps were Track 1's own, and all but the last two
were a choice made *after* seeing a number: a bandwidth picked once the gaps were known, a
label orientation settled once the AUC had a sign, a pooled statistic reported because the
per-cell ones disagreed. The last two are the forms this module cannot prevent on its own.
The eleventh (D93) is a statistic frozen here, correct, and still blind to the error beside
it -- which is why DIMENSIONLESS_PASS_WARNING asks for a dimensioned check as well. The
twelfth (D96) is worse, because freezing it early is exactly what hid it: the dimensioned
check that warning demanded was itself constructed so that it could not fail, and it was
read as a pass the day after it was frozen. Being frozen in advance makes a statistic
honest about *when* it was chosen. It says nothing about whether it can come out the other
way, and only a calculation under the hypothesis it is supposed to reject can say that.

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
    (
        "2026-11-23",
        "TOOL2_BANDWIDTH_GRID added, before T1-C has been run on either bank. "
        "TOOL2_SELECTION_SCOPE already required the kernel bandwidth to be chosen inside "
        "each fold on calibration operators only, but named no candidate set, and a grid "
        "assembled after seeing which bandwidth won would make that nested selection "
        "cosmetic -- the same defect as selecting on the held-out operator, one level up. "
        "Five geometric multiples of D33's median non-zero distance, every one of them "
        "reported under TOOL2_REPORT_ALL_WEIGHTINGS so the grid cannot be mined for a "
        "best case. NARROWS: it removes a free choice that was previously unconstrained.",
    ),
    (
        "2026-11-23",
        "COUNT_FROM_LEVEL_RANGE RETRACTED as the dimensioned check, one day after it was "
        "frozen. Amendment 13(b) read Track 2's level range as a pass that excluded the "
        "region count by a factor of order 300. It was not a pass. The wording frozen here "
        "-- the fraction of a split's CALIBRATION positives at or below its threshold -- "
        "is k/n by construction, since tau is the k-th smallest of them; Track 2 measured "
        "it at 0.0990 to 0.1001 over 234 cells, a constant carrying no n. The only "
        "computable version is test-side, and its across-split variance is the sum of the "
        "calibration Beta's and the test sample's, so the count it returns is bounded "
        "above by min(n_calib, n_test) and equals the harmonic combination of the two. "
        "With the calibration side AT the region count and the test side at subject scale, "
        "the bracket excluded the region count in 200 of 200 seeds -- with the region "
        "count true. The statistic rejects the region count whatever the truth, so 18 of "
        "18 cells excluding it across two banks is a property of the statistic. Statistics "
        "trap twelve, and Track 1's ninth: the first one that was not a choice made after "
        "seeing a number, but a check constructed so that it could not fail. The "
        "CONCLUSION survives on the clustered bootstrap, which measures subject clustering "
        "directly and can fail, and which was always the only certificate path -- and it "
        "corrects the conclusion too: effective n is per-operator, 13 to 154 on CT and 19 "
        "to 46 on face, tracking each operator's positive count, so it is not one number "
        "and BETA_REFERENCE_UNIT names a unit and not a count. FACE_BANK_DIRECTION_"
        "COINCIDENCE and CLAIM_STAMP_BINDS added under the same date; both NARROW. The "
        "first fences a face-bank fit that agrees with the rule and with the refuted "
        "frozen direction at once, and requires the CT bank's opposite sign to be "
        "disclosed beside it. The second records that all 72 face rows were reportable=True "
        "under a DEVELOPMENT ONLY stamp, and conjoins the two in the writer. Also "
        "corrected: GAP_DIRECTION_SOURCE's note quoted the GAP correlations in the place "
        "of the THRESHOLD-ORDERING ones, which D89's identity requires to have the "
        "opposite sign, on both banks.",
    ),
    (
        "2026-11-24",
        "BASELINE_COLLINEARITY and SPREAD_RESIDUAL_UNATTRIBUTABLE added; both NARROW. "
        "Track 3's T3.6 row 5 reports that the score-law shift follows the view RATIO and "
        "not the operator distance, which predicts D96's open residual is the view ratio "
        "in disguise. The test was pre-specified before running and it PASSED on both "
        "stated conditions: the view ratio orders the face excess at +0.429 against the "
        "distance's +0.418, and the partial correlation of distance given the view ratio "
        "collapses to +0.068. The pass is NOT claimed, because the pre-specification was "
        "defective: it asked the question in one direction only. Symmetrically, the view "
        "ratio given the distance is +0.128, also near zero -- at +0.916 collinearity "
        "neither variable survives controlling for the other, and 1 of 28 count-matched "
        "pairs can separate them. A one-sided partial correlation cannot settle "
        "attribution between collinear predictors, and that is now barred project-wide. "
        "The same measurement limits T1-B, which is the more consequential half: within a "
        "family -- the only scope where acquisition_parameter is defined -- the knob and "
        "the distance correlate at +0.92 to +0.97 on BOTH banks, so the baseline is a "
        "near-duplicate of the predictor it controls and no within-family margin may be "
        "read as the measure beating the knob. The freeze already called that objection "
        "'weak'; this gives it a number and moves T1-B's claim onto the 36 cross-family "
        "pairs, where no scalar knob is defined. The remedy is a GRID change -- interior "
        "operators moving both knobs at once -- and must be settled before the next bank "
        "is built, because no reanalysis of the existing 72 pairs can recover a "
        "contrast the grid never varied independently.",
    ),
    (
        "2026-11-24",
        "EXACT_BINOMIAL_IS_A_BETA_BAND added; NARROWS. T1-D's worst-operator coverage -- "
        "the number reliability.py's own docstring calls 'the only one of the three a "
        "certificate can be written from' -- was computing Clopper-Pearson on raw REGION "
        "counts, i.e. treating every hallucinated region as an independent Bernoulli "
        "trial. BETA_REGION_COUNT_USE has barred exactly that since 22 November; the "
        "function violated a bar that was already in force, for six days, because an "
        "exact binomial bound IS a Beta quantile under a different name. Measured on both "
        "banks, matched and transfer: the shipped bound was too high by 0.20 to 0.38, "
        "against design effects of 47 to 186 sitting in the same frames. No published "
        "verdict flips -- certifiable_at_0.90 was already False in all four cases -- but "
        "the numbers were wrong and the failure mode was live: a fixture that certifies "
        "at 0.90 on independent regions fails once the measured clustering is carried, "
        "which is a FALSE CERTIFICATE and is now a test. design_effect_col takes no "
        "default and the function refuses rather than assumes, as "
        "TransferBetaReference.interval does. T1-C is unaffected: its validity runs "
        "through cluster_bootstrap_coverage, which was checked rather than assumed. This "
        "is the third naming defect in three days (max_quantile's two readings, "
        "reportable's two senses, now two names over one object), so the entry also "
        "records the general form: a bar written in one vocabulary does not enforce "
        "itself over code written in another.",
    ),
    (
        "2026-11-24",
        "PRIOR_ADMISSIBILITY_UNCHANGED, BANK_DISAGREEMENT_BLOCKS_T1B and "
        "TOOL2_CEILING_FAILS added; NARROWS, and one of the three WITHDRAWS A FINDING OF "
        "MY OWN FROM YESTERDAY. Track 2 re-stamped the face-prior bank and the answer is "
        "no claims, by the prior's admissibility and not by a stale manifest, so the rule "
        "question came to Track 1 with a default. The default is taken: a natural-image "
        "prior still carries nothing for CT. It has to be taken, because the request "
        "reached me as the only route to a positive T1-B, and a rule relaxed after "
        "learning which bank holds the result is this project's recurring failure one "
        "level up from a statistic chosen after seeing a number. The second constant is "
        "the correction: D99 said the claim stamp was 'the binding constraint and nothing "
        "else', and Track 2's cut table shows the two CT banks disagree in SIGN on the "
        "gap-distance correlation (+0.546 to +1.000 against -0.886 to -0.980, opposite "
        "direction verdicts) and in kind on the spread trend (absent at every count-match "
        "cut on LIDC, +0.28 to +0.60 at every cut on the face-prior bank). So the stamp "
        "was never the only block and lifting it would not have unblocked T1-B. Third, "
        "T1-C's two runs: the weighted arm is valid at 3 of 7 interior folds and so is "
        "pooled, but the undeployable group-conditional oracle reaches only 4 of 7, which "
        "makes at least 3 folds unreachable by any operator-aware method. Checked before "
        "believing it: the conformal correction floors rather than ceils, validity is a "
        "lenient hi >= nominal off a subject-clustered bootstrap, and no cluster spans "
        "both sides -- so this is a within-operator deficit at design effects of 47 to "
        "186, a scope result rather than a verdict on distance weighting.",
    ),
    (
        "2026-11-24",
        "BANK_RESULTS_ARE_NOT_INTERCHANGEABLE added; NARROWS, and it takes a headline "
        "away. No measurement was made for this entry: every number in it was already in "
        "the documents, and what is new is putting them in one table. PROJECT_STATUS.md "
        "section 3c, 'Tool 2 -- the primary contribution -- works where it was argued "
        "to', is measured on runs/w07_ct_onboarding, which is the face-prior bank that "
        "carries no claims; T2.9's command block names it and the document predates the "
        "LIDC bank by two days. Both banks have nine operators and so seven interior "
        "folds, which is why the tables look comparable. Checked across every quantity "
        "where both banks have a value: they disagree in sign or in kind on all eight, "
        "including the direction verdict and the own-operator ceiling (6 of 7 against 4 "
        "of 7). Two of the eight were already filed as open puzzles. The bar is needed "
        "because that 6-of-7 is the most attractive Tool 2 number in the project and is "
        "what a paper draft reaches for. The split it leaves is the honest one: the core "
        "demonstration and the group-conditional fix hold on the stamped bank at 9 of 9 "
        "operators, and the four operator-aware refinements built on top of them are 0 of "
        "4 supported there.",
    ),
    (
        "2026-11-24",
        "GRID_CANNOT_DECORRELATE_DISTANCE added, TOOL2_CEILING_FAILS rewritten and "
        "BASELINE_COLLINEARITY corrected in two places; "
        "NARROWS, and it WITHDRAWS A READING OF MY OWN ON THE TEST I SET FOR IT. I gave "
        "Track 2 a falsification in advance -- if the weighted arm is valid anywhere the "
        "oracle is not, the oracle is not a per-fold bound and the ceiling reading goes. "
        "weighted_h0.25 is valid at la120 and la90 where the oracle is not, so it goes. "
        "What replaces it is not the opposite headline: 'some operator-aware arm is valid "
        "at 6 of 7' is read off the same criterion and fails the same way. Measured on "
        "the delivered CSVs: the own-operator arm sits at 0.8520 +/- 0.0144 over all "
        "seven folds, under nominal at 7 of 7 in one direction, and its two family ranges "
        "overlap, so the operator explains none of the deficit; its valid/invalid flips "
        "sort perfectly by interval width, the three narrowest intervals being exactly "
        "the three invalid folds. Nominal is unreachable on all four sparse-view folds by "
        "all eight arms, 8 of 29 valid stamps have point coverage below nominal, and 1 of "
        "56 interior cells lands within 0.02 of nominal. Within a fold there is one "
        "deployment operator, so every arm is one threshold and all eight sit on one "
        "area-coverage curve by construction -- an arm ranking is not available from this "
        "design. Second half: the 120-view subset looked like a route to T1-B at "
        "rho(distance, |dtau|) = +0.9429, and it is not one. The distance is a rank-"
        "duplicate of the arc ratio there at +1.0000, the arc ratio scores the identical "
        "+0.9429, and under the correct 4!-label null the observed value is the attainable "
        "maximum at p = 1/24. Another recurrence of the statistics trap, and the first "
        "caught by a rival-predictor check rather than by a sign. w13 is ruled not to be "
        "built for this purpose: it moves the blocking correlation from +0.950 to +0.952. "
        "Following that through BASELINE_COLLINEARITY took down two more of my own "
        "entries. It said the cross-family cells are where no scalar knob is defined, so "
        "T1-B's claim had to live there. A scalar is defined there -- every operator has "
        "both a view count and an arc, and only the IMPLEMENTED knob is undefined because "
        "it switches meaning by family -- and on those 36 pairs the arc ratio reaches "
        "+0.8524 against |dtau| while the distance reaches -0.5046. The trivial predictor "
        "beats the geometric one, in the direction the hypothesis wants, with the distance "
        "running backwards. So T1-B on this bank is refuted on its own response variable "
        "in its own designated cell, which is stronger than the blockage D97 recorded. "
        "The face-prior bank was deliberately not consulted for this ruling. "
        "CORRECTED WITHIN THE DAY, before any commit: I first called the "
        "own-operator offset an unexplained calibration defect and said T1-C was "
        "uninterpretable until a cheap same-operator run resolved it. That run "
        "exists -- T2.11 1e, which Track 2 named in the very message that carried "
        "the table. The offset is the registry split being a low draw (rank 1 to 4 "
        "of 13 at every operator); over 13 patient splits the arm averages 0.897, "
        "nominal at all nine to within 0.008. Against the matched control on the "
        "SAME split the own-operator arm meets or beats it at 7 of 7. The arm is "
        "sound; nominal 0.90 was simply the wrong reference. The T1-C verdict is "
        "unchanged -- pooled and weighted still reach the same-split reference at "
        "0 of 4 sparse-view folds -- so this corrects the reason, not the result. "
        "My own standing rule covers it: read the partner document that already "
        "has the data.",
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
#: the face-prior bank order tau oppositely, which is why one frozen constant could not
#: have been right for both.
#:
#: **The two correlations have opposite signs and an earlier draft of this note quoted the
#: wrong one (D96).** The identity above says so: ``sign(gap) == sign(tau_a - tau_b)``, and
#: the rule reads ``tau_b - tau_a``, so rho against the gap and rho against the threshold
#: difference must differ in sign. Measured on both banks, with the gap taken as
#: matched-minus-transfer:
#:
#: ====================  =========================  ==================
#: bank                  rho(d, tau_b - tau_a)      rho(d, gap)
#: ====================  =========================  ==================
#: CT (lidc_ct)          +0.55 to +1.00             -0.66 to -1.00
#: face-prior            -0.89 to -0.98             +0.89 to +0.98
#: ====================  =========================  ==================
#:
#: The rule's input is the first column. The note previously gave the second column's
#: numbers in the first column's place, so anyone running the rule by hand -- which
#: :data:`GAP_DIRECTION_RULE` explicitly invites -- would have found the opposite sign on
#: both banks and concluded they had made an error. The conclusion the sentence draws is
#: unaffected: the two banks do order tau oppositely.
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

#: **The coincidence that makes the face bank's fit unreadable as a confirmation (D96).**
#:
#: The prediction above passed: face tau falls with the distance, rho -0.89 to -0.98 across
#: all four family cells, unanimous. Run :data:`GAP_DIRECTION_RULE` on that and it calls a
#: NON-DECREASING fit -- which is the same shape as :data:`GAP_PREDICTION_DIRECTION`, the
#: direction frozen in Week 2 and REFUTED on the CT bank. On the face bank the rule's
#: answer and the refuted constant's answer coincide.
#:
#: That is not a defect in the rule; it is a property of this pair of banks. But it means a
#: face-bank isotonic fit agrees with the original hypothesis and with the rule at once, so
#: it cannot tell them apart, and a reader would be entitled to read it as the hypothesis
#: vindicated. The CT bank, where the two disagree and the comparison would be
#: informative, is the bank barred from reporting a direction.
FACE_BANK_DIRECTION_COINCIDENCE = (
    "on the face bank the rule's direction and the refuted GAP_PREDICTION_DIRECTION are"
    " the same shape, so the fit there discriminates between them not at all",
    "a face-bank fit may not be reported as evidence for GAP_PREDICTION_DIRECTION, and"
    " the coincidence must be stated wherever the face fit appears",
    "the CT bank's rule output is the OPPOSITE sign (+0.55 to +1.00 against face's -0.89"
    " to -0.98) and must be disclosed beside it, barred from reporting or not",
    "the bar on the CT bank is a bar on letting it SET the direction; it was never a"
    " licence to omit that the two banks answer oppositely",
)

#: **Provenance beats the per-cell flag, and the two disagreed (D96).** ``ordered_pairs.csv``
#: carries a per-cell ``reportable`` column, which is a STATISTICAL question -- is this
#: estimate determined well enough to quote -- and a ``claim_stamp``, which is a PROVENANCE
#: question: may any number from this bank be quoted at all. On the face bank all 72 rows
#: came back ``reportable=True`` under a ``DEVELOPMENT ONLY`` stamp, so the obvious filter,
#: ``df[df.reportable]``, selected 72 development rows for the write-up.
#:
#: ``t1a_safety_rule.py`` now conjoins the column and keeps the statistical bit as
#: ``reportable_statistical``, so an old filter becomes conservative instead of wrong.
CLAIM_STAMP_BINDS = (
    "a per-cell reportable flag is a statement about the estimate, never about whether the"
    " bank's numbers may be published; claim_stamp is the one that binds",
    "the face-prior bank is DEVELOPMENT ONLY for coverage as of 23 November 2026, so no"
    " face coverage number -- including its direction-rule cells -- may be written up",
    "a frozen prediction that passes on a development-only bank has passed, and is"
    " reported as passing ON A DEVELOPMENT BANK, with the stamp beside it",
    "the conjunction is applied in the writer rather than left to each consumer, because"
    " every consumer getting it right is not a property anyone can check",
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
#: **The cheap cross-check that was recorded here is withdrawn (D96).** D94 offered tau's
#: across-split motion as a bootstrap-free confirmation, bracketed at 10 to 70. That
#: bracket cannot exclude the region count -- see :data:`COUNT_FROM_LEVEL_RANGE`, retracted
#: under amendment 15 -- so it confirms nothing and is no longer cited here. D93's earlier
#: point estimates, "near 45" and 174, were already withdrawn for their own reasons.
#:
#: **What survives is the bootstrap, and it says the count is not one number.** Measured on
#: both banks at amendment 13's freeze, per operator, matched cells:
#:
#: - CT: effective n 13 to 154, median 44; design effect 47 to 186, median 125.
#: - face: effective n 19 to 46, median 28; design effect 61 to 211, median 110.
#: - transfer cells: design effect median 54 (CT), 29 (face).
#:
#: It tracks each operator's positive count. So this constant names the UNIT -- the thing
#: one independent draw is -- and must not be read as naming a count. Treating the count as
#: one frozen number would repeat amendment 7's defect exactly: a constant where the right
#: object is a function of the bank, which is a different error from a wrong value.
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

#: **Every name the forbidden band travels under (D98, 24 November 2026).**
#:
#: ``BETA_REGION_COUNT_USE`` has barred a region-count Beta band from certifying since 22
#: November. ``worst_operator_coverage`` went on computing one for six days, because it
#: called it a Clopper-Pearson interval -- which is the same object: the exact binomial
#: bound *is* a Beta quantile, and the implementation literally calls ``_beta_ppf``. The
#: bar was in force, the code violated it, and the two names are the entire reason nobody
#: saw it. The shipped bound was too high by 0.20 to 0.38 on both banks.
#:
#: **Third instance in three days of one concept under two names, or one name over two
#: concepts.** ``max_quantile`` was one name with two opposite readings (D95). ``reportable``
#: was one word with two orthogonal senses, statistical and provenance, which disagreed on
#: all 72 face rows (D96). This is two names over one object. The lesson is not "choose
#: better names": it is that a rule stated in one vocabulary does not enforce itself over
#: code written in another, so a bar on a quantity has to name the quantity's aliases.
EXACT_BINOMIAL_IS_A_BETA_BAND = (
    "a Clopper-Pearson or exact binomial bound on region counts IS the Beta band"
    " BETA_REGION_COUNT_USE bars from certifying; the names differ and the object does not",
    "so is any Wilson, Jeffreys or Agresti-Coull interval taken over regions: what is"
    " barred is treating regions as independent trials, by whatever interval",
    "a bound that a certificate will cite must take its unit as an argument with no"
    " default, and refuse rather than assume -- as TransferBetaReference.interval does",
    "the independent-regions value may be carried and SHOWN beside it, labelled as the"
    " idealisation, which is what makes the gap between them legible",
    "a bar written in one vocabulary does not enforce itself over code written in"
    " another; a bar on a quantity must name that quantity's aliases",
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

#: **RETRACTED as a dimensioned check, 23 November 2026 (D96, amendment 15).** It is kept
#: here, and the code is kept, because what it does measure is worth measuring -- but it is
#: NOT the check amendment 12 demanded and it never could have been.
#:
#: The frozen wording was *the fraction of that split's calibration positives at or below
#: its threshold*. Track 2 implemented it and found it vacuous: ``tau`` IS the k-th
#: smallest of those positives, so that fraction is ``k/n`` by construction. Measured over
#: 234 cells it is 0.0990 to 0.1001 -- a constant, carrying no ``n``.
#:
#: The only computable version is **test-side**: the split's test positives at or below
#: ``tau``, which is ``1 - matched coverage``. Its across-split variance is then
#:
#:     ``var[level] = var_Beta(n_calibration) + var_test(n_test)``
#:
#: and variances add, so the count read off the total lies **below both** counts, never at
#: the calibration one. Once the test side is subject-clustered -- which it is on both
#: banks -- the bracket cannot rise above subject scale whatever the calibration side does.
#:
#: The estimand is the harmonic combination, confirmed over 60 seeds a row:
#:
#: ==========  ========  ==========  ================  =============  ==============
#: n_calib     n_test    harmonic    median bracket    holds n_calib  holds harmonic
#: ==========  ========  ==========  ================  =============  ==============
#: 40          50        22          (10, 50)          67%            78%
#: 5000        50        50          (10, 100)         **0%**         85%
#: 5000        3300      1988        (1000, 2000)      23%            62%
#: 40          3300      40          (10, 70)          92%            92%
#: 100         10000     99          (15, 200)         92%            92%
#: ==========  ========  ==========  ================  =============  ==============
#:
#: So it recovers the calibration count only where the test side is much the larger -- the
#: last two rows -- and on these banks the two sides are both subject-scale, which is the
#: one regime where the bracket looks right and proves nothing.
#:
#: Measured, worst case first: with the calibration side at the REGION count (5000) and the
#: test side at 50 subjects, the bracket excluded 5000 in **200 of 200 seeds**. The check
#: rejects the region count with probability 1 whether or not the region count is the
#: truth, so Track 2's "region count excluded at 9 of 9 on each bank" is a property of the
#: statistic and not a finding about the bank.
#:
#: What amendment 13(b) read as a pass was a statistic that could not fail. The conclusion
#: it was taken to support survives on :data:`BETA_REFERENCE_UNIT`'s own evidence -- the
#: clustered bootstrap's design effects, which measure subject clustering directly and can
#: fail -- and that was always the only certificate path.
COUNT_FROM_LEVEL_RANGE = (
    "RETRACTED as a read of the calibration Beta's n: the calibration-side version is k/n"
    " by construction and the test-side version confounds the two counts",
    "variances add, so the count it returns is bounded above by min(n_calibration,"
    " n_test) and equals neither; it is a read on the MATCHED COVERAGE estimate's own"
    " effective count, which is a different and smaller quantity",
    "it may not be cited as the dimensioned check DIMENSIONLESS_PASS_WARNING requires,"
    " and it may not be cited against the region count at all",
    "measured: at a region-count calibration side with a subject-scale test side it"
    " excluded the region count in 200 of 200 seeds, with the region count true",
    "it returns a bracket, never a count: it cannot separate 35 from 50",
    "k = floor(alpha*(n+1)) steps in integers, so the likelihood is not monotone in n and"
    " the consistent set can have holes; the bracket is its span and is reported as such",
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

#: **How much room the grid leaves the comparison to be won in (D97, 24 November 2026).**
#:
#: The docstring above calls the knob objection "the objection that makes rho = +0.976 weak
#: within a family". That was qualitative. Measured, on both banks, within the only scope
#: where ``acquisition_parameter`` is defined at all:
#:
#: ===================  ======  ==============================  ======
#: family cell          pairs   rho(abs distance, abs log knob)  banks
#: ===================  ======  ==============================  ======
#: sparse_view          30      +0.923 (CT), +0.957 (face)      both
#: limited_angle        6       +0.970 (CT), +0.970 (face)      both
#: ===================  ======  ==============================  ======
#:
#: The baseline is a near-duplicate of the predictor it is supposed to control. The cause is
#: the grid's shape, not the measure: it is two one-dimensional arms -- sparse view turns the
#: view count at a full arc, limited angle turns the arc at 120 views -- so within an arm the
#: distance is close to a monotone function of the single knob that moved. On the face bank's
#: 72 pairs only **3** have distance and view-ratio ranks differing by more than a quarter of
#: the sample, and among the 28 count-matched pairs only **1** does.
#:
#: This is a statement about the GRID, and it is why T1-B's within-family margin cannot be
#: read as the measure beating the knob. It is also why D96's open residual cannot be
#: attributed: see :data:`SPREAD_RESIDUAL_UNATTRIBUTABLE`.
BASELINE_COLLINEARITY = (
    "within a family the operator distance and the acquisition knob correlate at +0.92 to"
    " +0.97 on both banks, so the grid cannot separate them",
    "no within-family result may be reported as the distance carrying information BEYOND"
    " the knob; on this grid that comparison is not identified",
    "WITHDRAWN 24 Nov (D102): 'the cross-family cells are where the measure is not"
    " duplicated by a knob, because no scalar knob is defined there -- those 36 pairs are"
    " where T1-B's claim has to live'. A scalar IS defined there. Every operator in the"
    " grid has both a view count and an angular range; only the IMPLEMENTED knob is"
    " undefined cross-family, because it switches meaning by family and so compares 120"
    " views against 90 degrees. Read the arc uniformly and the baseline exists on all 36",
    "and it is not degenerate -- it wins. On those 36 pairs rho(|log arc ratio|, |dtau|)"
    " = +0.8524 against the distance's -0.5046: a three-level scalar read off the operator"
    " name predicts the threshold gap better than the geometric measure, and in the"
    " direction the hypothesis wants, while the distance runs backwards. Coarse ties are"
    " noted -- the arc ratio takes 3 values, 12 pairs each -- and its group means separate"
    " cleanly (0.2395, 0.3682, 0.7250)",
    "so T1-B on this bank is not merely unidentified, it is REFUTED on its own response"
    " variable in its own designated cell. That is a stronger statement than D97's and it"
    " replaces it",
    "WITHDRAWN 24 Nov (D102): 'the remedy is a grid change and not an analysis change:"
    " interior operators that move both knobs at once break the duplication'. Measured"
    " before building, they do not -- see GRID_CANNOT_DECORRELATE_DISTANCE. The remedy"
    " named here was wrong about which confound binds",
    "what stands from the original entry: no reanalysis of the existing 72 pairs can"
    " recover what the grid did not vary independently, and it must be settled before the"
    " next bank is built",
)

#: **The one residual D96 left open, and why it stays open (D97).**
#:
#: The face bank's spread excess grows with distance at matched positive counts, rho +0.418.
#: Track 3's T3.6 row 5 reports independently that the score-law shift follows the view
#: *ratio* and not the distance, which predicts the residual is the view ratio in disguise.
#: Pre-specified before running: a pass needed the view ratio to order the excess at >= 0.40
#: AND the partial rho of distance given the view ratio to fall below 0.20. Both held --
#: +0.429 and +0.068.
#:
#: **The pre-specification was defective and the pass is not claimed.** It asked the
#: question in one direction only. Run symmetrically, the partial correlation of the view
#: ratio GIVEN the distance is +0.128, also near zero: at +0.916 collinearity neither
#: variable survives controlling for the other, which is what collinearity does, and is not
#: evidence for either. A partial correlation near zero means the two cannot be separated,
#: not that the controlled-for variable is the cause.
SPREAD_RESIDUAL_UNATTRIBUTABLE = (
    "the face bank's distance-growing spread excess is consistent with Track 3's view-ratio"
    " finding and is NOT shown to be explained by it",
    "both partial correlations are near zero (+0.068 and +0.128) because the predictors are"
    " +0.916 collinear, and 1 of 28 count-matched pairs can tell them apart",
    "it is reported as an open residual with a stated size, not as a mechanism found or a"
    " mechanism excluded",
    "a one-sided partial correlation may not settle attribution between collinear"
    " predictors anywhere in this project; the symmetric pair must be shown",
)

#: **The face-prior bank's scope: the rule stands (D100, 24 November 2026).**
#:
#: Track 2 re-stamped the bank in show mode and it reports
#: ``totalsegmentator + natural_image  claims [] -> none (unchanged)``. That is not a stale
#: manifest. The stamp is recomputed from the substrate, the sampler and the checkpoint,
#: and :func:`~caliber.reportability.prior_claims` returns nothing because a natural-image
#: prior is not a medical prior for CT (D80, P-18). D80's coverage-only case is a medical
#: prior of ANOTHER ANATOMY, which is the LIDC bank; a face prior is not that case.
#:
#: Track 2 put the rule question with a default -- admit a non-medical prior for coverage,
#: or leave it. **Left.** Two reasons, and the second would still hold if the first were
#: answered the other way.
PRIOR_ADMISSIBILITY_UNCHANGED = (
    "a non-medical prior carries no claim for a medical modality; the rule is unchanged",
    "the request arrived looking like the only route to a positive T1-B -- it was not"
    " even that, as BANK_DISAGREEMENT_BLOCKS_T1B shows, but it looked like it at the time"
    " and that is what makes the ruling worth recording",
    "a rule relaxed after learning which bank the result lives on is this project's own"
    " recurring defect moved up one level: not a statistic chosen after seeing a number,"
    " but an admissibility criterion chosen after seeing which bank gives the number",
    "the ruling was taken while the stamp still looked like the only block, so it was"
    " made at APPARENT cost; the cost then turned out to be illusory, and that is luck"
    " rather than judgement -- it is not a reason to trust the next such ruling more",
    "nothing about the coverage mathematics is in doubt -- split conformal is"
    " distribution-free and does not care what the prior saw; what the face prior changes"
    " is what the score distribution IS, and a coverage claim is a claim about that"
    " distribution",
)

#: **Why the stamp was never the only thing blocking T1-B, which D99 got wrong.**
#:
#: D99 concluded that "the binding constraint is the face bank's claim stamp and nothing
#: else". That was too narrow, and Track 2's 24 November cut table is what shows it. The
#: two CT banks do not merely differ in power; they disagree in SIGN and in KIND.
#:
#: ===========================  ======================  =========================
#: quantity                     LIDC (coverage-stamped) face-prior (unstamped)
#: ===========================  ======================  =========================
#: rho(distance, tau_b - tau_a) +0.546 to +1.000        -0.886 to -0.980
#: direction the rule then       non-increasing          non-decreasing
#: calls
#: rho(distance, log spread     0.00, 0.00, -0.11,      +0.60, +0.52, +0.42,
#: ratio), by count-match cut    -0.09, +0.12            +0.28, +0.40
#: ===========================  ======================  =========================
#:
#: So the bank that may be quoted and the bank that carries the effect return opposite
#: direction verdicts, and the spread trend that is present at every cut on one is absent
#: at every cut on the other. Lifting the stamp would license quoting a magnitude measured
#: where the sign of the underlying relation is reversed.
BANK_DISAGREEMENT_BLOCKS_T1B = (
    "D99's finding that the claim stamp was the single binding constraint is WITHDRAWN",
    "the binding constraint is that the effect does not replicate on the bank where it"
    " would count, and that is not repaired by a stamp, a grid or an estimator",
    "the two banks' direction verdicts are opposite, so neither bank can stand in for the"
    " other on any quantity whose sign enters the claim",
    "the face-prior excess also SHRINKS as positive counts are matched, 1.28 to 1.05,"
    " while its distance trend persists -- the level is count mismatch and the trend is"
    " not, and only the trend is a candidate finding",
    "a magnitude computed on the face-prior bank may be reported as a development value"
    " that motivates a measurement, never as an estimate of the LIDC bank's effect",
)

#: **T1-C on T2-A's bank: the ceiling fails, so the question is not answerable here
#: (D100).** Track 2's two runs of ``tool2_holdout.py --t1c``, unmodified:
#:
#: =========================  =============  ========  ======  ======  ============
#: run                        interior folds weighted  pooled  oracle  max_quantile
#: =========================  =============  ========  ======  ======  ============
#: full grid                  7              3         3       4       7
#: --exclude-nearest 1        6              3         3       4       6
#: =========================  =============  ========  ======  ======  ============
#:
#: The decisive cell is the oracle's. ``group_conditional_oracle`` calibrates on the
#: held-out operator's OWN calibration rows -- it is not deployable and exists to show the
#: headroom -- and it is invalid at 3 of 7 folds. Three things were checked rather than
#: assumed before reading that as real: ``split_threshold`` applies the finite-sample
#: correction ``k = floor(alpha * (n + 1))`` and so errs conservative; ``valid`` is
#: ``hi >= nominal`` off ``cluster_bootstrap_coverage`` on ``sample_id``, so it is
#: subject-clustered and not the region-count unit error of amendment 17; and the harness
#: refuses a fold whose clusters appear on both sides. A LENIENT criterion on a
#: cluster-robust interval failing at the ceiling is therefore a genuine deficit, and it is
#: WITHIN an operator, where there is no operator shift to blame.
TOOL2_CEILING_FAILS = (
    "no validity claim and no efficiency claim for Tool 2 on this bank; the script's own"
    " refusal stands",
    "the 4-of-7 ceiling reading is WITHDRAWN (24 Nov), on the falsification set with it:"
    " weighted_h0.25 is valid at la120 and la90 where the oracle is not, so the oracle is"
    " not a per-fold bound and no fold is shut by it",
    "the replacement is not the opposite headline either. 'some operator-aware arm is"
    " valid at 6 of 7' and 'the ceiling reaches 4 of 7' are both read off a criterion that"
    " measures neither: of 29 valid stamps over the 7 interior folds, 8 have POINT"
    " coverage below nominal and pass on interval width alone",
    "the own-operator arm sits at 0.8520 +/- 0.0144 across all 7 folds, below nominal at 7"
    " of 7 in the same direction, and the operator explains none of it -- its limited-angle"
    " range 0.8478-0.8666 overlaps its sparse-view range 0.8252-0.8664",
    "CORRECTED the same day it was written: an earlier reading called that offset an"
    " unexplained calibration defect and declared T1-C uninterpretable until it was"
    " resolved. It is explained, it was already measured, and Track 2 named the section in"
    " the same message that carried the table -- T2.11 1e. Every T1-C number is on the"
    " REGISTRY split, which ranks 1 to 4 of 13 draws at every operator; the same arm over"
    " 13 patient splits averages 0.897, nominal at all nine to within 0.008. Split"
    " conformal promises coverage over the draw of the calibration set, and the spread"
    " comes from the 41 calibration subjects a test-side interval never resamples",
    "so nominal 0.90 is the WRONG reference for a T1-C cell. Against the matched control"
    " measured on the same split, the own-operator arm meets or beats it at 7 of 7 (mean"
    " +0.0195, from +0.0005 at la90 to +0.0578 at sv90): the arm is sound and was being"
    " held to a level this split does not sit at",
    "the verdict survives the correction, which is why this is a correction and not a"
    " retraction. Against the same-split matched control, pooled and weighted_h0.25 reach"
    " it at 3 of 7 -- all three limited-angle folds, 0 of 4 sparse-view -- while"
    " max_quantile reaches it at 7 of 7. No efficiency claim is rescued",
    "that arm's valid/invalid pattern is a pure function of interval width: the 3 narrowest"
    " intervals are EXACTLY the 3 invalid folds in the full grid, and the 2 narrowest"
    " exactly the 2 invalid in the sparse run. Validity there detects n, not the method --"
    " and the registry-draw correction explains why cleanly, since a common offset across"
    " folds leaves interval width as the only thing that can still vary",
    "nominal 0.90 is unreachable on all 4 sparse-view folds by all 8 arms including the"
    " oracle (best 0.8959), and overshot by 0.11 to 0.32 of image area on all 3"
    " limited-angle folds; exactly 1 of 56 interior cells lands within 0.02 of it, from"
    " below. Read against the same-split matched control instead, max_quantile reaches the"
    " reference at 4 of 4 sparse-view folds -- so this row is a fact about the split's"
    " level, not about what the arms can do",
    "the arms are not rival methods on this grid: each fold has ONE deployment operator, so"
    " every arm is a single threshold there and all 8 lie on that fold's own area-coverage"
    " curve by construction (0 of 42 interior cells clear their own interval width above"
    " the chord of their area-neighbours). That is the design, not a result, and it is why"
    " an arm ranking cannot be read off this grid at all",
    "T1-C presumes within-operator calibration transfers across subjects; at design"
    " effects of 47 to 186 on this bank it does not, which makes this a SCOPE result and"
    " not a verdict on distance weighting",
    "the only arm valid at every unseen operator is the trivially-safe max_quantile, which"
    " is valid by construction and buys nothing",
    "an area number may not be quoted beside an invalid arm: a broken arm flags less and"
    " looks cheaper, which is the retired nearest arm's defect (coverage 0.547 at the"
    " smallest area of any arm)",
)

#: **The two CT banks are not interchangeable evidence (D101).**
#:
#: Checked on every quantity where both banks have a recorded value. They disagree in sign
#: or in kind on all eight, with no exception:
#:
#: ==============================  =====================  =====================
#: quantity                        face-prior (no claims) LIDC (coverage)
#: ==============================  =====================  =====================
#: Tool 2 weighted arm valid at     6 of 7                 3 of 7
#: own-operator ceiling valid at    6 of 7                 4 of 7
#: coverage-loss direction          loses, 0.890 -> 0.693  over-covers, 0.964
#: rho(distance, tau_b - tau_a)     -0.886 to -0.980       +0.546 to +1.000
#: direction the rule then calls    non-decreasing         non-increasing
#: rho(distance, spread ratio)      +0.28 to +0.60         -0.11 to +0.12
#: marginal positives, edge share   0.96                   0.05
#: coverage spread over 13 splits   0.034                  0.051
#: ==============================  =====================  =====================
#:
#: **The reason this is a bar and not a remark.** The project's most attractive Tool 2
#: number -- valid in 6 of 7 interior folds, beating its baseline on a sparse grid -- is on
#: the face-prior bank, and it is the number a paper draft will reach for. It was written
#: into ``PROJECT_STATUS.md`` §3c as "the primary contribution works where it was argued
#: to" for a week before anyone noticed which bank it came from.
BANK_RESULTS_ARE_NOT_INTERCHANGEABLE = (
    "a result measured on one CT bank is not evidence about the other in EITHER direction",
    "the face-prior bank's Tool 2 result may not be cited as the Tool 2 result; on the"
    " coverage-stamped bank there is no validity claim and no efficiency claim",
    "nor may the LIDC bank's negatives be cited as refuting a method -- T1-C's failure is"
    " within-operator and therefore a scope result (TOOL2_CEILING_FAILS)",
    "the eight disagreements are not eight findings; two were already recorded as open"
    " puzzles and the pattern is what is new, so this constant adds no measurement",
    "what survives on the stamped bank is the demonstration and the group-conditional fix:"
    " nominal coverage at 9 of 9 operators, 0.892 to 0.905. The operator-aware refinements"
    " -- distance ordering, safety rule, gap prediction, distance-weighted recalibration --"
    " are 0 of 4 supported there",
)

#: **A product grid cannot decorrelate operator distance from its knobs (D102).**
#:
#: The bank of record varies two knobs, views and arc, as a product. Any subset that holds
#: one of them fixed therefore varies only the other, and the geometric distance comes out
#: a rank-duplicate of whichever knob is still moving. Measured on the only subset where
#: both of the trivial scalars are constant -- the six unordered pairs among the four
#: operators at 120 views, where view count and null-space dimension are both unchanged:
#:
#: ===================================  =========
#: statistic on those 6 pairs            value
#: ===================================  =========
#: rho(distance, \|log arc ratio\|)       +1.0000
#: rho(distance, \|dtau\|)                +0.9429
#: rho(\|log arc ratio\|, \|dtau\|)        +0.9429
#: ===================================  =========
#:
#: The distance and the arc ratio rank the six pairs identically, and they predict the
#: threshold gap equally well, so on the one decorrelated subset the bank offers, the
#: distance adds exactly nothing over a number read off the config file.
GRID_CANNOT_DECORRELATE_DISTANCE = (
    "a product grid in two knobs has no subset that holds more than one of them constant,"
    " so there is no subset of the bank of record on which distance is free of every"
    " trivially available scalar at once",
    "the 120-view subset swaps the views confound for the arc confound: rho(distance, |log"
    " arc ratio|) = +1.0000 there, and the arc ratio matches the distance exactly at"
    " +0.9429 against |dtau|. It is not a route to T1-B",
    "and that number could not have come out much otherwise: permuting the 4 operator"
    " labels -- the correct null for 6 pairs built from 4 objects, not the 720 orderings"
    " of the pairs, which is anti-conservative here -- reaches +0.9429 in 1 of 24"
    " relabellings and never exceeds it. The estimate IS the attainable maximum and"
    " p = 1/24 is the test's floor rather than a finding. Another recurrence of the"
    " statistics trap, and the first caught by a rival-predictor check rather than"
    " by a sign -- a sign check would have passed it",
    "w13 is NOT to be built for this purpose: adding v90/a120 and v60/a150 moves"
    " rho(distance, null-space dimension difference) from +0.950 over 72 ordered pairs to"
    " +0.952 over 110 -- 38 new pairs and about 25 GPU hours for 0.002 on the statistic"
    " that does the blocking. What they do fix, arc duplication inside limited_angle"
    " (+0.970 to -0.307), was never the binding confound",
    "what T1-B needs is a family at the SAME view count AND the same angular range,"
    " differing only in which views are taken (uniform, non-uniform, disjoint blocks,"
    " half-step offset). There both trivial scalars are constant by construction. Check it"
    " on distances before building anything, as the w13 request was checked",
    "the single acquisition knob is retired: _acquisition_knob returns arc for"
    " limited_angle and assumes 120 views, so v90/a120 against v120/a120 has knob"
    " difference 0 at 4 of 20 pairs. A knob that collapses distinct operators to zero makes"
    " the trivial baseline artificially weak and flatters distance, so freeze it as the"
    " tuple (views, arc, null-space dimension). The fix moves AGAINST T1-B, which is the"
    " direction a baseline should ever be fixed in",
    "null_space_projection is the one candidate that is not a restatement of a knob:"
    " +0.201 against views and +0.369 against null-space dimension over 110 pairs, against"
    " growth's +0.920 and +0.952, and it ranks pairs differently (spearman +0.317 between"
    " the two measures). But its dynamic range is a third of growth's (cv 0.127 against"
    " 0.378, max/min 1.94 against 9.36), so part of that decorrelation is compression. A"
    " candidate to TEST against the gap, not a fix to adopt",
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

#: **The bandwidth grid, as multiples of D33's median non-zero distance.** Frozen before
#: any T1-C number exists, because :data:`TOOL2_SELECTION_SCOPE` says the bandwidth is
#: selected inside each fold -- and a grid chosen after seeing which bandwidth won would
#: make that nested selection cosmetic. Five geometric points centred on the existing
#: default: wide enough to show whether the weighting is sensitive to the kernel at all,
#: coarse enough that picking one is not a fit. Every point is reported
#: (:data:`TOOL2_REPORT_ALL_WEIGHTINGS`), so the grid cannot be mined for a best case.
TOOL2_BANDWIDTH_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)


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
        "exact_binomial_is_a_beta_band": list(EXACT_BINOMIAL_IS_A_BETA_BAND),
        "baseline_collinearity": list(BASELINE_COLLINEARITY),
        "spread_residual_unattributable": list(SPREAD_RESIDUAL_UNATTRIBUTABLE),
        "prior_admissibility_unchanged": list(PRIOR_ADMISSIBILITY_UNCHANGED),
        "bank_disagreement_blocks_t1b": list(BANK_DISAGREEMENT_BLOCKS_T1B),
        "tool2_ceiling_fails": list(TOOL2_CEILING_FAILS),
        "grid_cannot_decorrelate_distance": list(GRID_CANNOT_DECORRELATE_DISTANCE),
        "bank_results_are_not_interchangeable": list(
            BANK_RESULTS_ARE_NOT_INTERCHANGEABLE),
        "face_bank_direction_coincidence": list(FACE_BANK_DIRECTION_COINCIDENCE),
        "claim_stamp_binds": list(CLAIM_STAMP_BINDS),
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
        "tool2_bandwidth_grid": list(TOOL2_BANDWIDTH_GRID),
        "circularity_control": CIRCULARITY_CONTROL,
    }


def fingerprint() -> str:
    """Short stable hash of the frozen choices, to stamp onto every result table.

    A table carrying a fingerprint that does not match the current module was produced
    under a different freeze, and the discrepancy is visible without reading git history.
    """
    blob = json.dumps(as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
