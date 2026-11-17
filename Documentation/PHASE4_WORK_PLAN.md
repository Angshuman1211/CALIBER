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
| **LIDC bank** (T2-A) | Patient-level splits. HU mapping fixed to the prior's training normalisation **before** the run. A matched-operator control must return ≈ 0.90, or the bank is wrong and nothing else is read |
| **Circularity** (T1-E) | Label orientation fixed in advance. The scale-only null control is run **alongside**, not after |

**Two phrasings that must not drift.**
- Never *"the curves collapse"* — the strong cross-domain form is measured and fails (D70/D72).
- Never *"13 independent splits"* — 30 patients resampled 13 ways is a robustness check against
  split choice, not 13 replications. Write "on every one of 13 patient splits".

---

## 3. Track 2 — Systems and Data

Track 2 is on the critical path for exactly one thing, and everything else waits on it.

| # | Task | Compute | Blocks | Due |
|---|---|---|---|---|
| **T2-A** | **LIDC-prior CT bank on TotalSegmentator**, full grid | **~21 h GPU** | T1-A, T1-B, T1-C, T3-A | **start today** |
| T2-B | Implement D82's `reportable_claims` and re-stamp the M4Raw and CT banks | CPU, ~1 h | the paper's tables | with T2-A |
| T2-C | Re-measure `cos θ` on the LIDC bank and report it beside the smoke figure | CPU | T1-E | after T2-A |
| T2-D | Protocol replication: re-run the "STD vs \|error\| correlation / ROC for high-error regions" protocol on our own banks | CPU | T1-E | within 2 weeks |
| T2-E | Figures for the paper, from the LIDC bank | CPU | T1-F | after T1-A..C |
| T2-F | ~~DPS medical-prior bank~~ | — | — | **deferred, D-070** |

**T2-A is the whole gate.** `totalsegmentator.py` carries no `reportable = False`, and `ct_chest`
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
| **T1-A** | **Ordered-pair safety rule.** Predict "safe" iff signed distance ≤ 0, over all **72** ordered CT pairs. Confusion matrix; **false-safe rate is the primary endpoint** | Likely positive on CT; **will fail on MRI at matched R, and that is reported, not hidden** | T2-A + 3 d |
| **T1-B** | **Cross-family and leave-one-operator-out gap prediction.** Fit a monotone map from distance to gap on sparse-view only, predict limited-angle, then reverse. Hold out every pair involving one operator | Moderate. Must beat a constant and a views/arc-fraction predictor or it is not reported as a result | T2-A + 5 d |
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

**Already done this week:** D82 (M4Raw scoped to coverage), D-070 (this plan), the Shor &
Bronstein clearance.

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
| **18 Nov** | IoT exam, 10:00–13:00. **T2-A starts this evening** · pre-registration frozen (§2) |
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
| Cross-anatomy LIDC lowers `cos θ` and revives the identity | T2 → T1 | T2-C measured before any score result is interpreted |
| The 19–26 Nov window is lost (bank fails, GPU unavailable) | T2 | Everything slips to 4 Dec. MICCAI still comfortable; only the arXiv date moves |
| T3.6 slips again | T3 | Minimum version is six rows on a cached bank; no new theory required |
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
