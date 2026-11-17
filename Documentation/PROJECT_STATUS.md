# CALIBER — Project Status, Results, Problems and Gaps

**Date:** 17 November 2026 · **Week 11** · **Phase 3 of 4**
**Scope:** the whole project, all three tracks, written to be read end to end.
**Audience:** the three track owners, and anyone who needs the state of the project without reading twenty documents.

> **One sentence.** The central claim is demonstrated, measured in two modalities and far
> stronger on the medical prior than it ever was on the face prior; the primary calibration
> contribution has a positive result on sparse CT grids; the operator-aware *score* has a
> definitive negative result against a simpler baseline; **and not one number in the project
> is currently stamped reportable**, which is the single thing standing between the work and
> a paper.

---

## 1. The verdict, in one page

### What is true

| | |
|---|---|
| **The phenomenon is real and graded** | A conformal threshold calibrated at one acquisition operator loses coverage at another, monotonically with the shift, in MRI and CT |
| **The fix works** | Group-conditional (Mondrian) calibration restores nominal coverage at every operator in both modalities, at a measured cost in flagged area and FDR |
| **It is predictable in advance** | On CT the coverage gap is ordered by a signed operator distance computable from the two operators alone, ρ = +0.976 |
| **Tool 2 earns its keep where it was argued to** | On a sparse CT grid the distance-weighted recalibration beats the nearest-operator baseline, 0.929 coverage against 0.800, valid in 6 of 7 interior folds against 4 |
| **The medical prior changed the regime, as predicted** | `cos θ` rose from +0.07 to +0.60 (MRI) and +0.98 (CT smoke); the identity that made every magnitude label circular weakened from Spearman 0.97 to 0.65 |

### What is false, or dead

| | |
|---|---|
| **The score does not beat a dispersion baseline** | Settled negative on both routes that were built to test it (D68 direction label 0 of 3; D73 12 cells to 1 of 32) |
| **Detection is not reportable** | Even on the medical prior the score beats the anatomy control at 1 of 8 operators, and `s_disagree` alone out-ranks the combined score |
| **The strong cross-domain claim fails** | The two modalities' coverage curves do not collapse onto one; the MRI distance barely orders its gaps |
| **The curve-collapse test is uninformative here** | It "collapses" in all three readings — and so does the `group_conditional` control, which should show no trend at all. A pass that the control also passes is a lack of power, not a demonstration |

### What does not exist

- **T3.6** (bound against measurement) and **T3.7** (analytical authoring) — two of twenty-one tasks, and the whole of the project's third front.
- **A single reportable number.** Every bank fails a gate (§5).

---

## 2. What the project is

**Thesis.** Distribution-free (conformal) reliability certificates for image reconstruction
assume the acquisition operator `A` is fixed between calibration and deployment. The project
shows the guarantee silently fails when `A` shifts, restores validity by conditioning,
demonstrates this across two modalities with structurally different operators, and derives a
coverage-gap bound in quantities computable from the operators alone.

**Hallucination, defined.** With `P_R = A⁺A` and `P_N = I − P_R`, a reconstruction splits as
`x̂ = P_R x̂ + P_N x̂`. The range part is pinned by the measurements; the null part is the
prior's to invent, and **is invisible to every data-consistency check**. That is what makes
the problem hard and why a different kind of guarantee is needed.

**Three fronts.** I — empirical core (accelerated MRI). II — cross-domain (sparse-view and
limited-angle CT). III — theory (an operator-distance coverage bound, staged L1–L4).

**Three tools, in increasing ambition.** 1 — group-conditional calibration (guaranteed
backbone, exact for a known finite operator set). 2 — operator-distance weighted
recalibration (the primary novel contribution). 3 — the coverage-gap bound.

**Architecture.** A score bank: reconstructions and non-conformity scores computed once on
GPU across the operator grid and cached. Every calibration experiment, coverage curve and
numerical check then runs on the cache with no GPU. This is what makes the widened scope
affordable and decouples calibration and theory work from reconstruction runs.

---

## 3. Results that have landed

All figures below are **development output** unless §5 says otherwise.

### 3a. The coverage failure and its restoration

**CT, face prior**, calibrated at 180 views, 13-split means, nominal 0.90:

| Deployed at | Naive transfer | Group-conditional | Below nominal in |
|---|---|---|---|
| 120 views | 0.890 | 0.902–0.923 | — |
| 90 views | 0.847 | ″ | 13/13 |
| 60 views | 0.770 | ″ | 13/13 |
| 45 views | 0.728 | ″ | 13/13 |
| **30 views** | **0.693** | ″ | 13/13 |
| 150° arc | 0.888 | ″ | — |
| 120° arc | 0.837 | ″ | — |
| 90° arc | 0.796 | ″ | 13/13 |

**MRI, face prior.** Fewer robust failures: `equispaced1d R2` 13/13, `random1d R8` 11/13,
`equispaced1d R8` 10/13. Group-conditional holds 0.884–0.901 wherever positives exist.

**MRI, BraTS medical prior** (2,160 M4Raw pairs) — **much stronger**. Calibrated on a 1-D
mask and deployed on 2-D `variable_density2d` masks, naive transfer covers **0.000–0.159**,
below nominal in **13 of 13 splits in both directions**; group-conditional restores
**0.80–0.91**. The worst failure sits at **signed distance 0** — a mask-family change at
matched acceleration.

**What the fix costs** (CT, single split): at 30 views flag area 0.220 → 0.313 and region FDR
0.029 → 0.106; at 90° 0.308 → 0.423 and 0.084 → 0.202. Naive transfer's near-zero FDR is not
precision — it flags too little and misses a quarter to a third of the positives.

### 3b. The gap is ordered by a quantity available before deployment

| Measure | ρ with the coverage gap |
|---|---|
| CT, signed `null_space_growth` | **+0.976** (13-split means, 8 pairs) |
| CT, symmetric measures | +0.762 |
| MRI, from `random1d R8` | +0.556 |
| MRI, from `random1d R2` | +0.316 |

**Signed, not symmetric, and that is load-bearing.** The failure is directional — deploying
towards sparser sampling loses coverage, deploying towards denser over-flags — and a
symmetric metric maps both to the same positive distance, collapsing two opposite failures
onto one point.

**MRI is reported as a limitation with a mechanism, not as noise.** Its worst failures sit at
*matched* acceleration across mask families, where any scalar operator distance is ≈ 0. This
holds on the medical prior too. **The curve claim stays with CT.**

### 3c. Tool 2 — the primary contribution — works where it was argued to

CT, 7 interior folds (the two endpoints are outside what weighting can reach):

| Arm | Full grid: coverage / area / valid | Nearest excluded: coverage / area / valid |
|---|---|---|
| `matched` (oracle) | 0.890 / 0.260 / 6 | 0.890 / 0.260 / 6 |
| `nearest` | 0.840 / 0.276 / 5 | **0.800 / 0.240 / 4** |
| **`weighted` (Tool 2)** | **0.950 / 0.354 / 6** | **0.929 / 0.343 / 6** |
| `group_weighted` (fallback [7]) | 0.840 / 0.276 / 5 | 0.800 / 0.240 / 4 |
| `mondrian` | 1.000 / 1.000 / 7 | 1.000 / 1.000 / 7 |

- **Tool 2's advantage grows on the sparse grid**, which is exactly D75's hypothesis. With
  the nearest neighbour dropped, `nearest` collapses at v60 (0.390), v90 (0.768) and 90°
  (0.797); Tool 2 holds v90 at 0.976 and 90° at 0.988.
- **It buys that coverage with area**: 0.343 against 0.240, and 0.83 of all regions at 90°.
  Stated, not hidden.
- **It is not valid everywhere** — it misses v45 on the full grid (0.820) and v60 on the
  sparse one (0.777).
- **`mondrian`'s "valid in every fold" is trivial.** A held-out group has no calibration
  rows, so that arm flags everything. It is not a result.
- **Fallback [7] collapses to `nearest`** — its retained-group mass is 1.000, so the group
  weighting puts everything on one group. A negative result for [7].

### 3d. An exact identity, and a caution for the field

For any region, with `s = s_free`, `c = ‖P_N x‖` the truth's own hidden content and
`err = ‖P_N(x̂ − x)‖`:

> **`err² = s² + c² − 2·s·c·cos θ`**

Exact whenever the three columns share one normalisation; verified on the banks to a maximum
relative residual of 0.

**The consequence.** Under a mismatched prior `cos θ ≈ 0`, so `err ≈ √(s² + c²)` — and any
criterion thresholding an error *magnitude* becomes approximately a function of the score
being graded. Measured: Spearman 0.99 (0.9957 on the DPS bank). **Detection AUROCs of
0.95–1.00 obtained this way measure the identity, not a detector.** Four criteria were tried
and all four threshold a magnitude, so all four inherit it.

**And the medical prior proved it is the prior's fault, not the decomposition's**: `cos θ`
+0.07 → **+0.60** (MRI) and **+0.98** (CT smoke); the identity weakened from Spearman 0.97 to
0.65 (MRI) and 0.99 to 0.49 (CT). This was **written as a prediction before the run** (D79)
and confirmed.

This is probably the most transferable thing in the project: it is a caution about how the
field evaluates hallucination detectors, with the identity that predicts when it bites.

---

## 4. Results that have NOT landed

### 4a. The score does not beat a simpler baseline — settled

Posterior dispersion ranks hallucinations within **0.001–0.012 AUROC** of `s_disagree` at
every operator. Two routes were built to test whether the decomposition contributes; both are
now answered.

**Route 1 — the solver ablation (D68).** DDNM replaces the range component structurally, so
the posterior spread is already confined to the null space. DPS leaves more room. Premise
confirmed independently by Stage 2:

| Solver | Null-error share, median (range) | Orthogonality defect |
|---|---|---|
| DDNM | 0.991 (0.966–0.998) | 3.5×10⁻⁶ |
| DPS | **0.694** (0.388–0.845) | 3.7×10⁻⁶ |

| Label | Verdict | DPS: score − `s_disagree` (R2, R4, R8) |
|---|---|---|
| magnitude | **NOT ANSWERED** — positive withheld, the excess is the identity | +0.059, +0.077, +0.059 |
| direction (D74) | **ANSWERED, NEGATIVE** — beats the baseline at 0 of 3 | **−0.034, −0.014, −0.002** |

**Route 2 — the coverage axis (D73).** Over 32 cells across both modalities, the baseline
transfers better at **12** and CALIBER at **1** — and that one interval clears zero by 0.002.

**On the medical prior it is no better**: the score beats the anatomy control at 1 of 8
operators, and `s_disagree` alone out-ranks the combined score.

### 4b. Detection is not reportable

D67's ruling stands and the medical prior does not overturn it. Under the direction label on
CT the score beats the anatomy control at **0 of 9** operators (against 9 of 9 under the
magnitude label — which is the identity asserting itself).

### 4c. The cross-domain claim is the weak form only

The strong form — both modalities' gap-versus-distance curves collapsing onto one — is
measured and fails (D70, D72). The claim is: *the same failure and the same fix appear in
both modalities, with the distance ordering demonstrated where the distance is a genuine
scalar.* **Do not write "collapse".**

**And the collapse test itself is currently uninformative.** It reports "collapses" in all
three readings — but so does the `group_conditional` control, which by construction should
show no trend in either modality. A test its own null control also passes has no power here.

### 4d. The weights are not selected, and the result is not fully weight-insensitive

D54: 8 design patients cannot separate any weighting, so the score ships **equal weights by
default**. D76's sweep then asked whether the result needs them: the restoration is
established at **9 of 11** weightings (+0.075 at equal weights). So it is *mostly* insensitive
but **not at every weighting** — which is a weaker statement than the one hoped for, and has
to be reported as such.

---

## 5. The crux: nothing is reportable

Reportability is a property of the bank (D-018, D-035), gated on three things: the substrate
is real, the sampler is a generative prior rather than a reference, and **the prior is
admissible for the modality**. Current state:

| Bank | Substrate | Prior | Reportable? | Failing gate |
|---|---|---|---|---|
| fastMRI MRI (1,791 pairs, 458,496 regions) | fastMRI | face (celebahq) | **No** | prior |
| CT (900 pairs, 230,400 regions) | TotalSegmentator | face | **No** | prior |
| DPS MRI (597 pairs, 152,832 regions) | fastMRI | face | **No** | prior |
| M4Raw + BraTS (2,160 pairs) | M4Raw | **medical, admitted (D80)** | **No** | **substrate (D-015)** |
| CT + LIDC smoke (18 pairs) | TotalSegmentator | **medical, admitted** | smoke only | size |
| phantom demo | synthetic | reference | **No** | both, by design |

**So the results in §3 are all development output.** That is why the project can feel like
nothing has landed when a great deal has.

### 5a. The M4Raw deadlock

D80 ruled the prior admissible — anatomy gates *claims*, not banks. That left D-015's
substrate question as the only gate on the BraTS bank. And it is **stuck**: T2.10 §10 says the
call is Track 1's; D80 said making the case is Track 2's. Neither has moved in three weeks.

The argument for: coverage under operator shift compares two calibrations scored against the
same label on the same reconstructions, so it uses **no annotation** — which is precisely what
D-015 found M4Raw lacking. The argument against is also real: 30 patients at 0.3 T, three
scans per contrast, and **three design patients setting ε**.

### 5b. The unlock that bypasses it

**`totalsegmentator.py` sets no `reportable = False`** — unlike `fastmri.py` and
`synthetic.py`, which both do. TotalSegmentator is therefore already a reportable substrate,
and **the only gate failing on the CT banks is the prior**.

The LIDC CT prior is **already ported, verified to 1.3×10⁻⁵ against the authors' own Flax
model, registered as `ct_chest`, and admitted for CT under D80**. A smoke bank exists.

> **A full CT bank on the LIDC prior yields the project's first reportable numbers — with no
> ruling, no deadlock and no M4Raw argument. It is a GPU run, not a decision.**

And CT is where the results are strongest anyway: ρ = +0.976, the Tool 2 win, the graded
failure. Two caveats to state rather than hide: LIDC is **chest** CT against
TotalSegmentator's whole body (cross-anatomy — under D80 that is coverage-reportable,
detection not, which is exactly the claim wanted), and the project's fixed-HU-window
normalisation differs from LIDC's per-volume min-max.

---

## 6. Track-by-track status

### Track 1 — Methodology, Calibration and Scientific Direction

**All seven tasks delivered.** Log runs to **D81**.

| Task | State |
|---|---|
| T1.1 score design | Complete. Weights equal by default (D54); D76 measures what that costs |
| T1.2 shift-robust calibration | Three tools + fallback [7]. Tool 2 positive on sparse CT (§3c); [7] degenerates to `nearest` |
| T1.3 operator distance | Delivered, amended for direction (D47). CT ρ = +0.976 |
| T1.4 experimental design | Delivered — paired estimand, grids, ablation matrix, stratified reporting, curve-collapse test, coverage-axis comparison |
| T1.5 cross-track integration | Standing; the adjudication channel is most of the work now |
| T1.6 analysis | Phase 1 and Phase 2 gates delivered. **Phase 3 gate not written** |
| T1.7 authoring | Method section and preprint cut delivered (D77). Blocked only on a reportable number |

**Open:** rule on M4Raw (or redirect to CT+LIDC, §5b) · Phase 3 gate · preprint final cut ·
review `select_guidance.py`'s thresholds before any DPS bank is reported (T2.10 asks).

### Track 2 — Systems, Data Engineering and Cross-Domain Validation

**Through T2.10.** Problem log to **P-40**. Strongest position of the three.

Delivered: both operator pipelines, three solvers behind one interface, the full fastMRI and
CT banks, the evaluation harness, the corrected label, the cross-modality result, the CT
harness and figures, Track 1's Week 10 runs and the D-069 re-runs, and the **medical prior
ported and verified** (BraTS and LIDC, each matching the authors' Flax model to ~1×10⁻⁵).

**Open:** finish the DPS bank on the medical prior (~20 h GPU, resumable; completes D68 there
at unequal budgets, 100 vs 24 steps) · argue M4Raw's substrate case *or* run the full CT bank
on LIDC · download and convert the LDCT 512² checkpoint (the converter already holds its
settings) · figures and packaging · the O7 release, where the checkpoints' **absent licence**
becomes a real question.

### Track 3 — Analytical Foundations and Numerical Verification

**T3.1–T3.5 delivered. T3.6 and T3.7 do not exist.** This is the project's largest structural
gap: two of twenty-one tasks, and the entire third front.

What is delivered is substantial: the formal review, the distance candidates, the L1 stability
result (now stated on the **Kolmogorov** distance of the label-conditional law, tighter than
the original TV statement), the ill-posedness scaling, and six numerical checks that **found
and repaired an error in T3.3** and measured the density constant `D` for the first time.

**T3.6 is specified in Track 3's own documents** — it is not a blank page:

- compare bound to measurement **per ordered operator pair**, on **label-conditional** laws,
  with item-level resampling;
- estimate the §3b constants `L_ρ`, `M_ρ`, `M_S`;
- bound label flips caused by the candidacy gate moving with `A`;
- **carry `d_K` and `W₁`, not `D`** — T3.5 measured `D` moving by ≈2× across a 16× bandwidth
  change, so it is a free parameter unless the smoothing scale is stated;
- report `d_K` per ordered pair beside the gap;
- needs a bank with ground truth and enough items — Track 2's fastMRI banks exist.

**T3.7** writes up §2b and §3b.

**Two more:** a further T3.4 pass on how a real prior departs from §2b as a function of
null-space dimension and operator family; and a **new item from T2.10** — the medical prior
moves `cos θ` from ≈0 to ≈0.6, so **any step using the orthogonal-error approximation needs
re-checking on that bank**. That one invalidates an assumption, not just a number.

---

## 7. Problems and recurring failure modes

### 7a. The one that keeps recurring

**Eight times now: a statistic returns the number someone hoped for, because of an artefact in
the framing.**

| Instance | Returned | Why |
|---|---|---|
| P-19 | a script "working" for two weeks | nothing executed it |
| P-23 | coverage exactly 0.900 | thresholds fitted in-sample |
| `tool2_holdout` | matched arm exactly 0.900 | same patients both sides |
| D60 gate | anatomy AUROC 0.499, "chance" | **pooled** across operators; 0.662 within R8 |
| D65 verdict | score beats anatomy by +0.329 | direction-sensitive measure scored one way |
| D72 F-test | p = 0.031, "no collapse" | slopes 0.3025 vs 0.3029 |
| D73 interval | null \|delta\| to 0.128 vs a 0.02 floor | bootstrap held the threshold fixed |
| D74 first read | excess +0.097, "beats anatomy" | **pooled**; −0.152 at R4, +0.100 at R8 |

**Five of the eight were Track 1's own**, and two were written in the same session as the
module built to prevent them. The rules that catch them:

1. **Never report a pooled statistic across operators.** Show the per-cell table.
2. **Assert the frames are disjoint on the clustering unit** — the patient, not the region.
3. **Run the null fixture before trusting any new interval.** An interval that holds anything
   fixed while resampling is understating; D73's was understating by ~5×.
4. **A rejecting test is not a finding.** Compare the magnitude against the measured noise
   floor — the patient split (~0.05 coverage), never the mask seed (0.001–0.007).
5. **Equivalence needs the interval inside the floor**, not the estimate inside it.
6. **Check direction-sensitive measures both ways.**
7. **Read the partner document before predicting.** The D-033 question was asked when the
   answer was already in `operators/decomposition.py`.

### 7b. Standing scope limits

- **Known, linear/affine operators** at calibration and deployment. Not unknown or estimated.
- **Exact guarantee only** in the discrete known-operator regime. Continuous change targets a
  gap *bound*, not exact coverage.
- Operator shift changes the measurement **process**, so it is **not covariate shift** and the
  textbook weighted-CP correction does not transfer.
- **Solver held fixed** within any one coverage curve.
- **No external validation of the label.** The one published alternative (Tivnan 2024)
  disagrees at κ ≈ 0 and mostly flags background.
- **Volume, not patient, is the clustering unit on MRI**, so MRI intervals are optimistic by
  roughly the volumes-per-patient factor.
- **The CT forward operator is ours**, not a scanner's, and its pseudo-inverse is approximate.
- **Single-coil MRI** via root-sum-of-squares reduction.

### 7c. Live risks

| Risk | Severity |
|---|---|
| **Nothing reportable** — every bank fails a gate | **Blocking the paper.** Mitigated by §5b |
| **T3.6/T3.7 absent** — the third front has no output | High; longest pole, nothing in flight |
| **The M4Raw deadlock** — both tracks believe it is the other's move | High, and purely organisational |
| Checkpoints carry **no stated licence** | Fine for a cited preprint; real for the O7 release |
| The DPS medical-prior bank is **partial** (~20 h GPU, resumed in stretches) | Medium |
| BraTS is skull-stripped; M4Raw is not — skull and scalp are outside the prior's training distribution | Medium, stated |
| `equispaced1d R2` has **no positives** under ε, so one grid cell is undefined | Low, dropped from rank correlations |
| The region grid is **fixed by the network** at 240² (15×15) vs the fastMRI bank's 16×16 — no per-region number carries between them | Low, stated |

---

## 8. What to do next

**Critical path, in order.**

1. **Run the full CT bank on the LIDC prior.** It is a GPU run, it needs no ruling, and it
   produces the project's first reportable numbers on the modality where the results are
   strongest. *This replaces the M4Raw deadlock as the critical path.*
2. **Start T3.6** against the spec Track 3's own documents already give (§6). It is the
   longest pole and the only track with nothing in flight.
3. **Finish the DPS medical-prior bank** — completes D68 on the medical prior.
4. **Write the Phase 3 gate** (T1.6) once (1) lands.
5. **Final preprint cut** — the text is written; it needs a reportable table.
6. Decide M4Raw's substrate status at leisure, since it is no longer blocking.

**Publication outlook.** One paper, not the three the blueprint planned. The failure, the fix,
Tool 2's sparse-grid win, the distance ordering, and the circularity caution — at a workshop
or mid-tier methods venue. The second (analytical) paper needs T3.6, which does not exist.

---

## Appendix A — key numbers

| Quantity | Value |
|---|---|
| MRI bank | 1,791 pairs · 458,496 regions · 6.4 h GPU |
| CT bank | 900 pairs · 230,400 regions · ~16 h GPU |
| DPS bank (face prior) | 597 pairs · 152,832 regions |
| M4Raw + BraTS bank | 2,160 pairs |
| Patient splits | 13 |
| Operator grid | 9 MRI · 9 CT |
| CT coverage, naive, 180 → 30 views | 0.890 → **0.693** |
| CT coverage, group-conditional | 0.902–0.923 |
| MRI + medical prior, naive | **0.000–0.159**, 13/13 both directions |
| MRI + medical prior, group-conditional | 0.80–0.91 |
| CT distance ordering, signed | **ρ = +0.976** (symmetric +0.762) |
| Tool 2 vs `nearest`, interior mean coverage | 0.950 vs 0.840 (full) · **0.929 vs 0.800** (sparse) |
| Tool 2 valid interior folds | 6/7 both settings |
| D68 null-error share | DDNM 0.991 · DPS 0.694 |
| D68, direction label | **negative**, 0 of 3 |
| D73, 32 cells | baseline 12 · CALIBER 1 |
| `cos θ`, face → medical prior | +0.07 → **+0.60** (MRI) · +0.98 (CT smoke) |
| Identity Spearman, face → medical | 0.97 → 0.65 (MRI) · 0.99 → 0.49 (CT) |
| Weight sensitivity | restoration established at 9 of 11 weightings |
| Tests | 836 (Track 2's machine) · 9 (site) |

## Appendix B — where things are written

| Document | Holds |
|---|---|
| `problems_and_decisions.md` | Track 1's log, **D1–D81**, open questions, cross-track contracts |
| `decision_log.md` | Cross-track decisions, **D-001–D-069** |
| `T2_problems_and_decisions.md` | Track 2's problem log, **P-01–P-40**, plus the index of every cross-track decision |
| `T1.1`–`T1.7` | Track 1's per-task records; `T1.7_preprint_draft.md` is the narrowed paper |
| `T2.1`–`T2.10` | Track 2's weekly records; `T2.9` is Track 1's Week 10 runs, `T2.10` the medical prior |
| `T3.1`–`T3.5` | Track 3's records. **T3.6 and T3.7 do not exist** |
| `site/` | The presentation site, with two demos that compute rather than recite |

> **Numbering warning.** The two logs number independently: `D12` (Track 1) and `D-012`
> (cross-track) are unrelated decisions. Always say which log.
