# Phase 4 Work Plan — the paper, divided across the three tracks

**Date:** 18 November 2026 · **Week 12** · **Decisions:** D-070, D82, D80, D81
**Target:** MICCAI 2027, with an arXiv preprint as soon as the CT result lands. MIDL is not the target (§6).
**Supersedes** the task lists in `PROJECT_STATUS.md` §8.
**Audience:** all three track owners. Each track's section is self-contained; read yours and §2.

> **The strategy in one sentence.** Run the LIDC CT bank now to unlock reportable numbers,
> rule M4Raw reportable for coverage so the MRI result survives, then spend the CPU time on
> three analyses that turn a correlation into a safety rule, re-score Tool 2 against a *valid*
> baseline, and close T3.6 at its minimum defensible version.

---

## 1. What changed, and why this plan exists

Three things moved in the last week.

**The prior-art risk is retired.** Shor & Bronstein (arXiv 2610.26558) was the one unchecked
paper that could have pre-empted the novelty claim. It does not: a ridge regressor, not a
conformal procedure; its one coverage figure has no shift in it; it never varies the acquisition
operator and states no acceleration factor anywhere. It also lists *"additional anatomies, field
strengths, accelerations, sampling patterns"* as future work — the adjacent paper names our
contribution as its own open problem. Full reasoning in D-070.

**The headline is re-framed from a correlation to a decision rule.** rho = +0.976 is weak on its
own, because within a family any monotone function of view count orders the gaps. It becomes a
pre-deployment safety rule with a false-safe rate.

**Two of our own results were scored against the wrong comparator.** Tool 2 was reported beating
`nearest`, which is invalid in 3 of 7 interior folds — D51's own rule says that comparison does
not mean what it looks like. And the abstract skeleton quoted an MRI number from a bank that fails
its substrate gate, which D82 now resolves.

---

## 2. Everyone reads this: the pre-registration

**Freeze these before the LIDC bank finishes.** Five of this project's eight statistics traps were
Track 1's own, and every one of them was a choice made after seeing a number. The point of writing
these down now is that the bank is still running, so none of them can be.

| Analysis | Frozen in advance |
|---|---|
| **Sign rule** (T1-A) | Threshold fixed at **0**, not fitted. Operator-level leave-one-out. Permutation null over operator labels. Reported per family *and* pooled, never pooled alone |
| **Gap prediction** (T1-B) | **One** distance and **one** functional form declared before fitting. Report signed *and* symmetric. Baselines: a constant predictor and a views/arc-fraction predictor — the distance must beat both |
| **Tool 2 area** (T1-C) | Bandwidth and weighting selected on **calibration operators only**, nested. All 11 weightings reported. Area compared **only in cells where both arms are valid** |
| **T3.6 table** (T3-A) | The calibrated threshold is recomputed **inside every bootstrap resample**, at patient level. Not held fixed — that is D73's defect, and it understated by ~5× |
| **LIDC bank** (T2-A) | Patient-level splits. HU mapping fixed to the prior's training normalisation **before** the run. A matched-operator control must return ~ 0.90, or the bank is wrong and nothing else is read. **Superseded in part by D85**: the control's interval must move the calibration side, as Sec. 2's own refit rule already required; on the test-side-only interval it failed, on the correct one it passes 9 of 9 |
| **Circularity** (T1-E) | Label orientation fixed in advance. The scale-only null control is run **alongside**, not after. **Track 2 has already run it** (T2-D): `s_free` beats the posterior spread at 9 of 9 operators on both face-prior banks |
| **Unsafe label** (T1-A) | A cell is unsafe iff the **upper end** of its cluster-bootstrap coverage interval misses nominal -- not the point estimate. Amended 20 Nov (D85) because per-cell coverage has an SD of ~0.05, so a point-estimate call is a coin flip near the boundary |

**Two phrasings that must not drift.**
- Never *"the curves collapse"* — the strong cross-domain form is measured and fails (D70/D72).
- Never *"13 independent splits"* — 30 patients resampled 13 ways is a robustness check against
  split choice, not 13 replications. Write "on every one of 13 patient splits".

---

## 3. Track 2 — Systems and Data

Track 2 is on the critical path for exactly one thing, and everything else waits on it.

| # | Task | Compute | Blocks | Due |
|---|---|---|---|---|
| **T2-A** | ~~LIDC-prior CT bank on TotalSegmentator~~ **BUILT 19 Nov 20:18** (900 pairs, 360,000 regions, 25 h). Its control failed as written and **Track 1 has now ruled: PASS at 9 of 9** once the interval includes the calibration draw (D85). One condition outstanding - see below | done | T1-A, T1-B, T1-C, T3-A | **unblocked** |
| ~~**T2-A1**~~ | **DONE 20 Nov. Excluded nominal at 3 of 9** (la 120, la 90, sv 30), two further seeds agreeing. D85's reverting clause fired; **D86 rules the clause itself was the mis-specified part and the bank IS read** - the gate was a 9-cell pointwise test whose firing order is anti-correlated with each cell's actual shortfall | CPU, 1 h | - | done |
| ~~**T2-A2**~~ | **DONE 21 Nov, and the hypothesis failed.** The calibration half's mean per-volume HU range does not order tau (rho -0.05 to +0.36, median +0.08), though *within* a subject a wider range does go with lower positive scores (-0.19 to -0.44): D-071 reaches the scores, but a half's mean is the wrong aggregate for a quantile. **Superseded by T2-A4**, which tests the null first instead of trying a third variable (D87) | CPU, minutes | explains T1-D's open row | done |
| ~~**T2-A3**~~ | **DONE 21 Nov.** 81 cells x 13 splits, `pair_coverage_split_mean.csv`. Registry below the across-split mean at **72 of 72** ordered pairs, median -0.061 - the single-draw bias D86 predicted, at full strength. T1-A reads it via `--pair-coverage-dir` (D87); the interval is rebuilt there, not read, because their `split_sd / sqrt(13)` omits the 41% calibration overlap and so runs 2.44x too narrow | CPU | **T1-A** | done |
| **T2-A4** | **Is tau's spread anything but an order statistic?** Draw 13 subject-level subsamples of the observed size from the pooled positives, recompute tau in each, and compare the simulated sd against the observed. Reproduces it -> nothing to explain and D-071 is exonerated; undershoots -> try the **lower tail** of the per-volume range distribution, not its mean. Run on both banks, so the face/CT contrast (0.034 vs 0.051) is tested as tail shape | CPU, minutes | closes T2-A2's question | **now** |
| **T2-A5** | **CG residual per operator** on the CT projector - the measurement that settles whether the limited-angle scores are numerically sound. The violation *count* cannot: `|cos theta| > 1` is a threshold on a continuous error, so the 1,744 are its visible tail. `label_comparison.py`'s guard stays unchanged until this exists (D87) | CPU | gates T2-C's interpretation, and T1-B's slope | **now** |
| T2-B | ~~Implement D82's `reportable_claims`~~ **DELIVERED** (`src/caliber/reportability.py`, `scripts/restamp_bank.py`). **Re-stamping the existing banks is still outstanding** - a bank written before the field reads as carrying *nothing*, so the MRI numbers T1-F quotes are refused until `restamp_bank.py` is run on them | CPU, ~1 h | the paper's tables | re-stamp still due |
| T2-C | Re-measure `cos θ` on the LIDC bank and report it beside the smoke figure | CPU | T1-E | after T2-A |
| T2-D | Protocol replication: re-run the "STD vs \|error\| correlation / ROC for high-error regions" protocol on our own banks | CPU | T1-E | within 2 weeks |
| T2-E | Figures for the paper, from the LIDC bank | CPU | T1-F | after T1-A..C |
| T2-F | ~~DPS medical-prior bank~~ | — | — | **deferred, D-070** |

**T2-A is built, ruled on twice, and read.** The control failed on the registry split
(0.831, nominal contained at 4 of 9) because its interval resampled test subjects only. The
rule requiring otherwise is committed from **25 Oct** (`c06c51f`, D73), so restoring the
calibration-draw term is not a post-hoc repair -- but D85's quadrature estimate of that term
used one pooled `sd_cal = 0.047` for all nine cells, and T2-A1 measured the real thing at
**0.024 to 0.066**, smallest at exactly the three cells nearest nominal. Measured, the control
excludes nominal at **3 of 9**, and D85's reverting clause fired.

**D86 rules the bank read, and the clause the mis-specified part.** The gate was a 9-cell
pointwise test, and which cells it fires on is anti-correlated with their actual shortfall:
sparse-view 120 sits 0.105 below nominal and passes, limited-angle 120 sits 0.044 below and
fails. The evidence that settles it is Track 2's own 13-split table, which needs no rule
reinterpreted: **every operator's across-draw mean is nominal** (0.892-0.905, widest miss
0.008) and the registry split is one draw sitting a near-constant **-1.38 SD** low at all nine
at once, with `P(lowest of 13 at least this low) = 0.68`. Limited-angle 90 deg is reported as
*unresolved* on the registry split rather than passed. D85's condition 2 is reversed for the
label: a single draw displaces every cell together, so it cannot average out over the 72
ordered pairs, and T1-A's unsafe label is now the across-split mean -- a change that **lowers
Track 1's own endpoint** and therefore carries a mandatory single-draw sensitivity. D86.

**The original gate wording stands as history:** `totalsegmentator.py` carries no `reportable = False`, and `ct_chest`
is admitted under D80, so **this bank produces the project's first stamped numbers with no further
ruling**. Sequencing note: it is ~21 h at 320² (83 s/pair on the smoke run × 900 pairs), not the
16 h estimated, and the DPS bank would contend for the same card — hence T2-F.

**T2-A's own guardrails**, beyond the pre-registration above: patient-level splits (not slice);
and the cross-anatomy setting (chest prior, whole-body anatomy) may lower `cos θ` and revive the
identity, so T2-C must be read **before** any score result from this bank is interpreted.

**T2-B** is the one-line-plus-a-test change, not a boolean flip:

```python
reportable = False                            # the unqualified claim
reportable_claims = frozenset({"coverage"})   # D82
```

with a test that a detection number from a coverage-only bank is refused. P-18 is why.

---

## 4. Track 1 — Methodology, Calibration, Authoring

All CPU, all on the cached bank, all blocked only by T2-A.

| # | Task | Expected outcome | Due |
|---|---|---|---|
| **T1-0** | ~~Signed operator-distance matrix, 9 CT operators at 320^2~~ | **DONE 20 Nov** - 72/72, perfect total order, zero sign violations (D84). Grid matches `w12_ct_lidc_gpu.yaml` exactly | done |
| **T1-A** | **Ordered-pair safety rule.** Predict "safe" iff signed distance ≤ 0, over all **72** ordered CT pairs. Confusion matrix; **false-safe rate is the primary endpoint** | Likely positive on CT; **will fail on MRI at matched R, and that is reported, not hidden** | T2-A + 3 d |
| **T1-B** | **Cross-family and leave-one-operator-out gap prediction.** Fit a monotone map from distance to gap on sparse-view only, predict limited-angle, then reverse. Hold out every pair involving one operator | Moderate. Must beat a constant, a views/arc-fraction predictor **where that is defined**, and null-space dimension. Within- and cross-family reported separately (D84) | T2-A + 5 d |
| **T1-C** | **Tool 2 re-scored against valid baselines**: max-quantile, pooled, group-conditional oracle. Area compared only where both are valid | The D-070 correction. Target headline: "validity at an unseen operator with X% less area than the trivially-safe rule" | T2-A + 5 d |
| **T1-D** | **Worst-operator coverage** (the minimum over cells — what a deployment certificate needs) and **calibration-conditional reliability**: split-to-split spread against the Beta law split conformal follows under exchangeability | Cheap, quantitative, hard to dispute. Mondrian should match its Beta law; naive transfer should be shifted off it | T2-A + 5 d |
| **T1-E** | **Circularity section** with the **scale-only null control** (`‖P_N x̂‖` alone, no posterior information, same magnitude label) | One section, not a paper. Defer the standalone critique | 2 weeks |
| **T1-F** | **Paper draft** on the D-070 contribution order | the deliverable | see §6 |
| **T1-G** | Phase 3 gate (T1.6) | housekeeping | after T1-A..D |

**Contribution order for T1-F**, fixed by D-070: (i) the failure mode, (ii) the repair,
(iii) the pre-deployment predictor and safety rule, (iv) the evaluation caution, (v) the negative
result on the proposed score — in an honest limitations or "what did not work" subsection, not
buried.

**Related-work precision.** Write *"no prior work reports conformal coverage when the acquisition
operator differs between calibration and deployment"*. Do **not** write "no conformal work in MRI
risk estimation" — Shor & Bronstein use split conformal in Appendix H and CQR as a comparator.
Cite their future-work sentence.

**T1-0 was a missing prerequisite, and it is why T1-A can start the day the bank lands.**
The signed distance is a property of two forward operators -- no image, no prior, no
reconstruction -- so it never needed T2-A. It was not free: ~11 h of CPU for the 36 unordered
pairs by 2 measures at 320^2, with the reverse direction filled by antisymmetry rather than
computed. It contended with nothing on the GPU, and it is now **done**.

**T1-A and T1-D are implemented and tested ahead of the bank** (D83): `prereg.py` freezes
every choice as a module constant that the analyses import rather than accept as an
argument, `safety.py` is the ordered-pair rule, `reliability.py` is the worst-cell bound and
the Beta-law check, and `scripts/t1a_safety_rule.py` runs the lot against a finished bank in
one command. 47 tests; full suite 603 passed. Nothing has a result -- the coverage side of
every one of them is T2-A's bank.

**T1-0 landed and it changed T1-B, not T1-A.** The signed distance turns out to be a
*perfect* total order on the nine CT operators -- 72 ordered pairs, zero sign violations --
so the sign rule reads as "deploy on an operator no more ill-posed than the one you
calibrated on". Two consequences worth knowing before the bank lands. The rule green-lights
exactly 50% of ordered pairs by construction, because an antisymmetric measure must, so
that number is not evidence of anything. And the three limited-angle operators interleave
*between* `v120` and `v90`, which no acquisition parameter can express -- all three have 120
views, all six sparse-view operators span 180 degrees, and the four operators at 120 views
share a null-space dimension. **The cross-family cells are the only place where the
parameter baselines are undefined or degenerate, so they are where T1-B is actually
tested.** Full reasoning and the amended baseline specification in D84.

**Two things were measured during that work that change what T1-D can claim.**
- The closed-form Beta SD **understates** a measured spread by about 36%, because it is the
  law against an infinite test set; the correction is `sqrt(Var_Beta + p(1-p)/n_test)`.
  Comparing against the uncorrected form would report a correctly calibrated method as
  over-dispersed. The split-overlap effect runs the other way and is ~2%.
- At 13 splits the spread check can only detect a spread below 0.62x or above 1.49x the
  correct value, so it cannot see a 36% error at all. **T1-D's worst-operator bound is the
  reportable half**; the Beta-law spread comparison catches only gross mis-calibration and
  is now labelled `consistent_but_underpowered` when it passes.

**Already done this week:** D82 (M4Raw scoped to coverage), D-070 (this plan), the Shor &
Bronstein clearance, D83 (the freeze, T1-A and T1-D implemented), **T1-0 complete** and
D84 (the total order, and T1-B's baseline scoped before any coverage number existed).

---

## 5. Track 3 — Analytical Foundations

Track 3 has had nothing in flight for a month and owns the longest pole. **T3.6 at its minimum
defensible version is achievable and is the whole ask.**

| # | Task | Notes | Due |
|---|---|---|---|
| **T3-A** | **T3.6 table, per ordered operator pair** | the six rows below | T2-A + 2 weeks |
| **T3-B** | **T3.7**: write up §2b and §3b | the analytical paper's core | after T3-A |
| T3-C | Re-check every step that uses the **orthogonal-error approximation** on the medical-prior bank | `cos θ` moved from ≈0 to ≈0.6 — this invalidates an assumption, not just a number | with T3-A |
| T3-D | T3.4 pass: departure of a real prior from §2b as a function of null-space dimension and operator family | lower priority | after T3-B |

**T3-A, the six rows** — specified in Track 3's own documents, so this is not a blank page:

1. **Measured gap**, with patient-level bootstrap CIs that **recompute the calibrated threshold
   inside every resample**.
2. **d_K** between calibration and test score laws, with CI. Check `gap ≤ d_K + 1/(n+1)` and report
   the **tightness ratio**.
3. **W₁** between score laws.
4. **A coupled upper bound on W₁.** The bank scores the same items under both operators, so the
   mean of `|s(x; A₂) − s(x; A₁)|` over items bounds W₁ for that coupling. Fully computable, and
   the honest bridge from operators to score-law shift.
5. **The operator distance**, plus Spearman of distance against `d_K` and against `W₁`.
6. *Optional:* score-level TV by histogram across several bandwidths. By data processing it is at
   most the data-level TV in the Barber bound, so a loose score-level TV implies the classical
   bound is looser still.

**Two warnings carried from T3.5.** Report **d_K and W₁, not D** — `D` moved by ≈2× across a 16×
bandwidth change, so it is a free parameter unless the smoothing scale is stated. And reporting
`d_K` beside the gap is **close to a tautology** for split conformal with continuous scores; its
value is as an implementation check and a tightness ratio. **The substantive claim must link
operator quantities to the shift in the score law** — row 5 is where that lives.

---

## 6. Timeline, around the exam block

End-semester examinations run **27 November to 3 December**, so the plan is built around one
working window before them and a resumption after.

| Date | |
|---|---|
| **18 Nov** | IoT exam. Pre-registration frozen (Sec. 2, D83); T1-0 started |
| 19 Nov | **T2-A's bank built** (900 pairs). Its control fails as written; nothing downstream read |
| **20 Nov** | T1-0 complete (D84). T2-B and T2-D delivered. D85 ruled the control a pass at 9 of 9 on a quadrature estimate; **T2-A1 measured it and got 3 of 9, so D85's reverting clause fired**. **D86 rules the bank read**: the 13-split per-operator means are nominal at all nine (0.892-0.905) and the registry split is one draw sitting a near-constant -1.38 SD low, which is what the lowest of 13 looks like (p = 0.68). D85's condition 2 reversed for the label: T1-A's unsafe label now comes from the across-split mean, with the single-draw rate reported beside it because the change *lowers* Track 1's own endpoint. `cos θ = +0.882` on the LIDC bank against +0.072 on the face prior - the CT confirmation of T2.10 §5 |
| **21 Nov** | **D86 acted on: the bank is read, and Sec. 1g's transfer tables are in.** The headline pair of numbers: the matched arm holds at **0.892-0.905**, the naive arm is below nominal at **36 of 72** ordered pairs with **22 below 0.80** and a worst of **0.377** (la 90 deg -> sv 120). Direction is the *reverse* of the face-prior bank, and sv 120 is out of order - hardest to cover from almost every row, while holding **zero** identity violations. **D87** discharges D86's live condition: the violations do concentrate in the three failed cells (p = 0.048) and the reading stands anyway, on split invariance. Both of Track 1's Sec. 1f predictions failed (fixed epsilon, HU range). T1-A now reads T2-A3's artefact; 699 tests pass. T1-E's LIDC row separates the spread from the scale-only control at **9 of 9** (0.920 vs 0.801) |
| **19–26 Nov** | **The working window — 8 clear days.** LIDC bank lands (~21 h), then T1-A..D and T3-A on the cached bank |
| 27 Nov | NLP exam |
| 28 Nov | **CV exam** |
| 30 Nov | CC exam |
| 3 Dec | MTM exam, ends 13:00 |
| **4 Dec onwards** | Writing resumes. T1-F, T3-B |
| ~25–26 Feb 2027 | **MICCAI 2027** (tracker estimate; the Society still lists "To be announced") |

**MIDL 2027 is not the target.** Its full-paper deadline is 4 December and abstract registration
is 30 November — the day of the CC exam — with four examinations in the eight days before. A
submission written in that window would be worse than no submission, and MIDL's notification
(12 Feb) sits before the likely MICCAI deadline anyway, so nothing is foreclosed by skipping it.

**MICCAI is a comfortable target, not a scramble.** Roughly fourteen weeks, of which eight days
before the exams carry the compute and the analyses. That is what makes T3.6 achievable at more
than its minimum, and it removes the forking-path pressure that a two-week deadline creates.

**Post the arXiv preprint when the CT result lands**, without waiting for MICCAI. It establishes
priority, it is what D11 has asked for since August, and it costs nothing.

**What has to happen before 27 November**, because after that nothing moves for a week:
1. T2-A finishes and its matched-operator control passes (§3).
2. T1-A, T1-B, T1-C produce tables — even rough ones.
3. T3-A's six rows are computed, even if unwritten.

Everything else can wait for 4 December.

**Note for Track 2 and Track 3:** this schedule is Track 1's. If your examination dates differ,
say so now — T2-A is the gate for both other tracks, and if it cannot start this week the whole
window moves.

## 7. What we are not doing, and why

| Stopped | Reason |
|---|---|
| Score-vs-dispersion re-tests | Answered. D68 negative on the direction label, D73 12 cells to 1 |
| Curve-collapse analyses | The test's own null control also "collapses" — no power here |
| New weighting schemes | D54/D76 settled; sweeping more is forking-path risk |
| Detection AUROCs as results | D67, and the medical prior did not rescue them |
| Energy-weighted MRI distance | It would be chosen after seeing which MRI gaps need explaining — forking-path risk, and five of eight traps were ours |
| DPS medical-prior bank | ~20 h GPU contending with T2-A, to re-answer an answered question |
| Standalone evaluation-critique paper | Needs a broad empirical study we cannot do now |

---

## 8. Risk register

| Risk | Owner | Mitigation |
|---|---|---|
| LIDC bank fails or the matched-operator control misses ≈0.90 | T2 | Control is checked **first**; nothing downstream is read until it passes |
| ~~Cross-anatomy LIDC lowers `cos θ` and revives the identity~~ | T2 + T1 | **Happened, and ruled on 21 Nov (D87).** `cos θ = +0.882`, and the 1,744 identity-violating regions **do** concentrate in the three cells that failed T2-A1 (4.4x the per-cell rate of the passing six; exact one-sided rank-sum p = 0.048) - the pattern D86 said would send the bank back. The reading stands anyway, because the defect is **split-invariant** (reconstructions are computed once; re-cutting only re-partitions ids) while the T2-A1 failure is split-specific, and la 90 deg holds 41% of the violations yet carries the *highest* across-draw mean of the nine. The 0.0014 drop-test offered as exoneration is a bound, not evidence: its arithmetic ceiling is 14% of the shortfall it was pointed at. Live residue is a **T1-B** confound, not a coverage one - solver quality co-varies with ill-posedness, so part of the distance-gap slope may be the solver |
| The 19–26 Nov window is lost (bank fails, GPU unavailable) | T2 | Everything slips to 4 Dec. MICCAI still comfortable; only the arXiv date moves |
| T3.6 slips again | T3 | Minimum version is six rows on a cached bank; no new theory required |
| ~~T1-0 does not finish before the bank~~ | T1 | **Retired** - finished 20 Nov, 72/72 |
| Existing banks read as carrying no claims until re-stamped | T2 | `restamp_bank.py` exists and has been run on 6 banks; M4Raw and the LIDC smoke bank now carry coverage |
| ~~T2-A1's proper interval still excludes nominal somewhere~~ | T1 + T2 | **Happened, 20 Nov: 3 of 9.** Resolved by D86 against Track 2's own 13-split table rather than by more subjects - the extra 100 subjects (21 GPU h) were explicitly *not* spent, because the pattern is not a power problem: the five cells furthest below nominal all passed the gate and the three closest to it failed |
| **The multiplicity rule has no datable provenance** | T1 | Committed in `1e94d37`, **20 Nov 16:20** - a day after the bank and hours after T2-A1's result. Git confirms the content, not the 18 Nov date, so a reader cannot be shown the rule predates the gate firing. Not fixable after the fact; the mitigation is that D86 rests on Track 2's 13-split table instead, which needs no rule of ours. The refit rule *is* datable, at `c06c51f`, **25 Oct**. **Standing lesson: freeze files get committed the day they are written, not the day they are needed** |
| This bank's split-to-split spread (0.051) is unexplained | T2 | Two candidate causes measured and ruled out. Travels as a limitation; it is T1-D's first row |
| The sign rule's MRI false-safe rate is embarrassing | T1 | **Report it.** It is what makes the CT result credible, and it motivates the stated MRI limitation |
| Checkpoint licence (O7 release) | T2 | Fine for a cited preprint; resolve before any code/weights release |

---

## Appendix — caveats this plan carries

- The expected outcomes in §3–§5 are **judgement, not measurement**. Nothing here has been run.
- The prior-art gap rests on one targeted search plus the Shor & Bronstein read (D-070). It is
  not exhaustive.
- MICCAI 2027 dates are tracker estimates, not official.
- The Shor & Bronstein read covered the v1 HTML in full, not the PDF or TeX; the covered quantity
  behind their 0.94 figure is inferred rather than named in that sentence.
