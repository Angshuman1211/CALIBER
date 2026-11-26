# CALIBER — Project Status, Results, Problems and Gaps

**Date:** 17 November 2026 · **Week 11** · **Phase 3 of 4**  · *reviewed through D106, 26 November — Phase 3's gate is closed, so the project is in Phase 4, and **T1-F is cut**: the only Track 1 item left is the Phase 4 gate at submission*
**UPDATED 24-26 November 2026 (Week 12) through D102. Every section has now been read
against D85-D102.** The body below is still the 17 November prose; the eighteen decisions
that landed after it (D85-D102, the LIDC bank's build and stamp, and negative results on all
four of Track 1's measurement tasks) are marked inline rather than rewritten, so that what
was claimed stays visible beside what replaced it.

**Sections carrying an inversion mark:** §1 (three rows and the "does not exist" list), §2,
§3a, §3b, §3c, §4a, §4c, §4d, §5, §5a, §5b, §6, §7a, §7c. §8 and both appendices were
rewritten outright. **Only §3d, §4b and §7b survived their review unchanged** — the three
named here, and no count is given of the rest because the two lists above are the count.
§6 is in the list because it still said "log runs to D101" and called T1-B
*blocked*, which D102 upgraded to *refuted*; it had been marked on 24 Nov and went stale
again within two days, which is what a file of this age does between readings.

**The sweep was worth more than the banner it replaces.** On 24 Nov it had written §2, §4
and §7 off as "merely unreviewed" — and §4c turned out to state the D102-refuted distance
ordering as the project's surviving cross-domain claim, §4d to quote a face-prior number
with no bank label, §5a to describe a deadlock D82 had already resolved, §5b to justify the
LIDC run by two claims that run withdrew, §7a to miscount its own table, and §7c to list
four risks of which three are closed and one is obsolete. **Stale and inverted are not the
same thing, and only reading finds out which a section is.** Read the decision log from D85
for anything load-bearing. **D102 withdrew two readings of Track 1's own**, so where this
file and the decision log disagree, the log is right.
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
| ~~**It is predictable in advance**~~ **REFUTED on the stamped bank, D101/D102** | ρ = +0.976 is 8 pairs of 13-split means on the **face-prior bank**, which carries no claims. On the coverage-stamped LIDC bank the safety rule built on this ordering is **refuted** (D89), gap prediction is reportable on **0 of 4** cells (D91), and D102 goes further: on the 36 cross-family pairs where D97 placed the claim, the **arc ratio read off the operator name beats the geometric distance** against the threshold gap, +0.8524 against **−0.5046** — the distance runs backwards |
| ~~**Tool 2 earns its keep where it was argued to**~~ **WITHDRAWN, D100/D101** | That run is on the **face-prior bank** against `nearest`, a baseline D-070 §3 retired as contaminated. On the LIDC bank there is no validity claim and no efficiency claim. ~~The undeployable ceiling reaches 4 of 7~~ **that reading is withdrawn too (D102)**: `weighted_h0.25` is valid where the oracle is not, and the valid/invalid flips sort perfectly by *interval width*, so "valid at k of 7" is a sample-size statistic in either direction |
| **The medical prior changed the regime, as predicted** | `cos θ` rose from +0.07 to +0.60 (MRI) and +0.98 (CT smoke); the identity that made every magnitude label circular weakened from Spearman 0.97 to 0.65 |

### What is false, or dead

| | |
|---|---|
| **The score does not beat a dispersion baseline** | Settled negative on both routes that were built to test it (D68 direction label 0 of 3; D73 12 cells to 1 of 32) |
| **Detection is not reportable** | Even on the medical prior the score beats the anatomy control at 1 of 8 operators, and `s_disagree` alone out-ranks the combined score |
| **The strong cross-domain claim fails** | The two modalities' coverage curves do not collapse onto one; the MRI distance barely orders its gaps |
| **The curve-collapse test is uninformative here** | It "collapses" in all three readings — and so does the `group_conditional` control, which should show no trend at all. A pass that the control also passes is a lack of power, not a demonstration |

### What does not exist

- ~~**T3.6** (bound against measurement) and **T3.7** (analytical authoring) — two of
  twenty-one tasks, and the whole of the project's third front.~~ **OBSOLETE (26 Nov).**
  Both documents exist and both tasks are run; Appendix B of this same file already said so
  while this line said the opposite. T3.6 reports the bound holding at **72 of 72** ordered
  pairs on the **stamped LIDC bank**, close to attained where under-coverage is real
  (tightness median 0.72). What does not exist is now only the *positive* form of T3.6's
  row 5: within family the operator distance does not order the score-law shift (ρ −0.16
  to +0.18 on 66 of 72 pairs), which is the same verdict D102 reached from the other side.
- **Nothing else.** All twenty-one tasks have a document; what is missing is claims, not
  deliverables.
- ~~**A single reportable number.** Every bank fails a gate (§5).~~ **OBSOLETE, D82 and D-072.** The LIDC CT bank carries **coverage** claims. What is true instead, per D101: the core demonstration and the group-conditional fix are supported on that bank at 9 of 9 operators, and **0 of 4** operator-aware refinements are.

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
recalibration (the primary novel contribution *as designed*). 3 — the coverage-gap bound.

> **Read "primary novel contribution" as a statement about ambition, not about results
> (D101, D102).** It is what the project set out to contribute and it is why the design is
> shaped this way. On the one bank that carries claims, Tool 1 is supported at 9 of 9
> operators and **Tool 2 has no positive result at all** — no validity claim and no
> efficiency claim (§3c). Tool 3's bound holds at 72 of 72 pairs but its ordering row is
> negative. A reader who takes this sentence as a results summary gets the project exactly
> backwards, which is the error §3c sat in for a week.

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

> **The bank above carries no claims; this is the one that does (added D102).** Everything
> in §3a to this point is the **face-prior** CT bank, correctly labelled but
> development-only (a natural-image prior is not a medical prior for CT — D80, P-18). The
> **LIDC medical-prior bank** is the one stamped for coverage, and it shows the same two
> results:
>
> - **The fix holds.** Group-conditional coverage is nominal at **9 of 9** operators,
>   **0.892–0.905**, mean **0.897**, to within 0.008 — *as the mean over 13 patient
>   splits*, which is the quantity split conformal actually promises (coverage over the
>   draw of the calibration set). **Carry that qualifier.** On the single registry split
>   the bank was built with, the same arm is **0.795–0.856** (mean 0.831) and reaches 0.90
>   at none of the nine; that split ranks 1–4 of 13 at every operator, and the spread comes
>   from the 41 calibration subjects a test-side interval never resamples (T2.11 §1e).
>   **Report the per-draw spread beside the mean:** sd **0.051** by operator, individual
>   draws spanning 0.795–0.991. The guarantee is marginal over calibration draws, so the
>   mean is the right estimate of it — but a single deployment draw can sit near 0.80, and
>   why the spread is wider on this bank than on the face-prior one (0.051 against 0.034)
>   is **not established**. State it as a limit; do not fill it quietly.
> - **Naive transfer fails.** Below nominal at **36 of 72** ordered pairs, **22 below
>   0.80**, worst **0.377** (limited-angle 90° → sparse-view 120), on the across-split mean
>   (T2.11 §1g).
>
> **The direction is the reverse of the face-prior bank's**, and the two banks may not be
> cited as evidence about each other (`BANK_RESULTS_ARE_NOT_INTERCHANGEABLE`, D101).

### 3b. ~~The gap is ordered by a quantity available before deployment~~ — **WITHDRAWN for the stamped bank (D102)**

> **This section has §3c's defect, one section earlier, and it went unnoticed in the same
> sweep.** The headline **+0.976** is 8 pairs on the **face-prior bank**, which carries no
> claims, and the row below did not say so. On the **coverage-stamped LIDC bank the gap is
> not ordered by the distance at all**: over all 72 ordered pairs ρ(distance, coverage) is
> **−0.03**, and on the 36 cross-family pairs where D97 placed the claim the **arc ratio
> read off the operator name beats the geometric distance** against the threshold gap —
> **+0.8524 against −0.5046**, with the distance running *backwards*. T1-B is refuted on
> that bank rather than blocked. Treat everything below as a face-prior development result.

| Measure | ρ with the coverage gap |
|---|---|
| CT **face-prior bank**, signed `null_space_growth` | **+0.976** (13-split means, 8 pairs) — *development only* |
| CT, symmetric measures | +0.762 |
| MRI, from `random1d R8` | +0.556 |
| MRI, from `random1d R2` | +0.316 |

**Signed, not symmetric, and that is load-bearing.** The failure is directional — deploying
towards sparser sampling loses coverage, deploying towards denser over-flags — and a
symmetric metric maps both to the same positive distance, collapsing two opposite failures
onto one point.

**MRI is reported as a limitation with a mechanism, not as noise.** Its worst failures sit at
*matched* acceleration across mask families, where any scalar operator distance is ≈ 0. This
holds on the medical prior too. ~~**The curve claim stays with CT.**~~ **It does not survive
on the CT bank that carries claims either (D102)** — see the banner above.

**Six places in this document were inverted rather than stale, not five.** §3b was the sixth
and was found by sweeping the sections the banner had declared unreviewed.

### 3c. ~~Tool 2 — the primary contribution — works where it was argued to~~ — **WITHDRAWN (D100/D101)**

> **WITHDRAWN as a claim, D100 and D101 (24 Nov). Kept as development output.** The bank
> here is `runs/w07_ct_onboarding` — the **face-prior** bank, `claims [] -> none`. Two
> further defects in the comparison, both found after this was written: the baseline
> `nearest` was retired by D-070 §3 (it posts the smallest area of any arm at coverage
> 0.547, so beating it while flagging *more* area is not efficiency), and one `weighted`
> bandwidth is reported where `TOOL2_REPORT_ALL_WEIGHTINGS` now requires the whole frozen
> grid. On the coverage-stamped LIDC bank, 24 Nov: weighted valid at **3 of 7**, pooled 3,
> `max_quantile` 7. ~~Ceiling **4 of 7**.~~ **D102: the arm counts below are not a
> method ranking.** Each fold has one deployment operator, so every arm is a single
> threshold there and all eight sit on that fold's own area–coverage curve by
> construction. The own-operator arm is 4.8 points under nominal at **7 of 7** folds
> (0.8520 ± 0.0144, families overlapping) — because every cell here is on the **registry
> split**, a low draw ranked 1–4 of 13 (T2.11 §1e, whose 13-split mean is **0.897**, nominal
> at all nine). **So 0.90 is the wrong reference for this table.** Against the matched
> control on the same split the own-operator arm meets or beats it at **7 of 7**, and the
> arms' ranking is unchanged: `pooled` and `weighted` reach that reference on all three
> limited-angle folds and **0 of 4** sparse-view. Read the areas and coverages; do not read
> the `valid` column as a verdict.

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
operators, and `s_disagree` alone out-ranks the combined score. That is the **M4Raw + BraTS
MRI** bank, 8 operators (T2.10). §4b's *0 of 9* is the **CT** bank's 9 operators (T2.9 §4) —
two banks, not a disagreement.

### 4b. Detection is not reportable

D67's ruling stands and the medical prior does not overturn it. Under the direction label on
CT the score beats the anatomy control at **0 of 9** operators (against 9 of 9 under the
magnitude label — which is the identity asserting itself).

### 4c. The cross-domain claim is the weak form only

The strong form — both modalities' gap-versus-distance curves collapsing onto one — is
measured and fails (D70, D72). ~~The claim is: *the same failure and the same fix appear in
both modalities, with the distance ordering demonstrated where the distance is a genuine
scalar.*~~ **The second half of that is REFUTED, D101/D102 (26 Nov).** The surviving claim
is: *the same failure and the same fix appear in both modalities* — and nothing about the
distance. Where the distance is a genuine scalar is exactly where D102 measured it and the
**arc ratio read off the operator name beat it**, +0.8524 against −0.5046 on the 36
cross-family pairs, the distance running backwards; the one subset that looked decorrelated
is a rank duplicate of the arc at ρ = +1.0000. **Do not write "collapse", and do not write
"ordered by the distance" either.**

**And the collapse test itself is currently uninformative.** It reports "collapses" in all
three readings — but so does the `group_conditional` control, which by construction should
show no trend in either modality. A test its own null control also passes has no power here.

### 4d. The weights are not selected, and the result is not fully weight-insensitive

D54: 8 design patients cannot separate any weighting, so the score ships **equal weights by
default**. D76's sweep then asked whether the result needs them: the restoration is
established at **9 of 11** weightings (+0.075 at equal weights). So it is *mostly* insensitive
but **not at every weighting** — which is a weaker statement than the one hoped for, and has
to be reported as such.

> **Bank label, added 26 Nov.** That sweep is `random1d R2 → R8` on the **fastMRI
> face-prior** bank (T2.9 §5), which carries **no claims**. It has never been run on a
> stamped bank, and `BANK_RESULTS_ARE_NOT_INTERCHANGEABLE` (D100) forbids carrying it over
> in either direction — including this negative direction. So the honest statement is
> narrower still: **weight sensitivity is unmeasured on any bank that may be quoted.** The
> limitation goes in the paper as unmeasured, not as measured-and-mostly-fine.

---

## 5. The crux: nothing is reportable

> **NO LONGER TRUE, D82 and D-072 (and see D100). Superseded twice over.** A bank now
> carries a **set of claims**, not a boolean: a defect gates the claims it reaches, not the
> whole bank. And the full LIDC CT bank (900 pairs, 360,000 regions, built 19 Nov) **carries
> coverage claims** — so the project does have a quotable bank, for coverage. The boolean
> in the table below is the pre-D82 model; it reads `False` for a bank that is
> coverage-reportable, which is the same defect D100 fixed in `tool2_holdout.py`'s header.
> What D101 establishes is narrower and sharper than this section's claim: the core result
> *is* on the stamped bank, and the operator-aware refinements are not.

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
| **CT + LIDC, full (900 pairs, 360,000 regions, 19 Nov)** | TotalSegmentator | **medical, admitted (D80)** | **coverage, yes** (D82) | detection only |
| phantom demo | synthetic | reference | **No** | both, by design |

**So the results in §3 are all development output.** That is why the project can feel like
nothing has landed when a great deal has.

### 5a. ~~The M4Raw deadlock~~ — **RESOLVED by D82**

> **Resolved, and this section is the pre-D82 text (noted 26 Nov).** **D82 — "M4Raw is
> reportable for coverage claims and not for detection"** made the call that this section
> says nobody was making. It applies D80's structure to the *substrate* instead of the
> prior: the defect gates which claims the bank can carry, not whether the bank exists.
> M4Raw lacks pathology annotations, so it cannot carry detection; coverage under operator
> shift uses no annotation, so it can carry that. The two rulings are now one principle with
> two instances. The `reportable` boolean became a scoped claim set rather than being
> flipped, because flipping it would have let an M4Raw *detection* number be stamped — the
> exact error of P-18. The table above still shows the old boolean for this bank.

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

~~And CT is where the results are strongest anyway: ρ = +0.976, the Tool 2 win, the graded
failure.~~ **Two of those three are withdrawn (26 Nov)** — ρ = +0.976 is 8 pairs on the
face-prior bank (§3b) and the Tool 2 win is §3c, and the run this section argued for is
what withdrew them. The conclusion held anyway and for a better reason: **the graded failure
and its group-conditional fix are stamped on CT at 9 of 9 operators.** This is the §8 defect
in miniature — a correct decision justified by numbers that did not survive the thing it
decided to build. Two caveats to state rather than hide: LIDC is **chest** CT against
TotalSegmentator's whole body (cross-anatomy — under D80 that is coverage-reportable,
detection not, which is exactly the claim wanted), and the project's fixed-HU-window
normalisation differs from LIDC's per-volume min-max.

---

## 6. Track-by-track status

### Track 1 — Methodology, Calibration and Scientific Direction

**All seven tasks delivered.** ~~Log runs to **D81**.~~ ~~**Log runs to D101.**~~ **Log runs
to D104** (26 Nov), and the **Phase 3 gate is delivered** — three of four gates done, Q5
repeated, §23's publication sequence re-scoped so the narrowed cut is *the* method paper
(D104). Three of the
four Phase 4 measurement tasks have since returned **negatives on the coverage-stamped
bank** — T1-A refuted (D89), T1-B reportable on 0 of 4 cells and then ~~blocked~~
**refuted on its own response variable in its own designated cell (D91, D100, and D102:
arc ratio +0.8524 against the distance's −0.5046)**,
T1-C no validity or efficiency claim (D100) — while the core demonstration and the
group-conditional fix hold there. See D101 for the split.

| Task | State |
|---|---|
| T1.1 score design | Complete. Weights equal by default (D54); D76 measures what that costs |
| T1.2 shift-robust calibration | Three tools + fallback [7]. ~~Tool 2 positive on sparse CT (§3c)~~ **that positive is on the face-prior bank; on LIDC there is no validity or efficiency claim (D100)**; [7] degenerates to `nearest` |
| T1.3 operator distance | Delivered, amended for direction (D47). ~~CT ρ = +0.976~~ — **that is 8 pairs on the face-prior bank and the ordering is REFUTED on the stamped one** (D101, D102): arc ratio +0.8524 against the distance's −0.5046 cross-family, the distance backwards. The measure is delivered; its *ordering role* is not supported |
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

~~**T3.1–T3.5 delivered. T3.6 and T3.7 do not exist.** This is the project's largest structural
gap: two of twenty-one tasks, and the entire third front.~~

> **OBSOLETE — all seven Track 3 tasks are delivered (26 Nov, D103).** This was the **fourth**
> place in this one file claiming two documents that exist do not; the other three were §1,
> §7c and Appendix B. **T3.6** is run on the stamped LIDC bank as well as four harness banks:
> the matched bound holds at **72 of 72** ordered pairs with no separated interval and is close
> to attained where under-coverage is real (tightness median 0.72). **T3.7** states and proves
> Theorem 1, Proposition 2, Theorem 3, Corollary 4 and Proposition 5. **The front that was the
> largest structural gap is the one that delivered on time** — and its row 5 is negative:
> within a family the operator distance does not order the score-law shift (ρ −0.16 to +0.18
> on 66 of 72 pairs), which is the same verdict Track 1 reached from the threshold-gap side
> (D102). One negative, two tracks, two response variables. The specification below is kept
> because it is what T3.6 was built against.

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

**Nine times now: a statistic returns the number someone hoped for, because of an artefact in
the framing.** The owner column is there so the count below can be checked against the table
instead of being taken on trust — see rule 10, which this section broke.

| Instance | Owner | Returned | Why |
|---|---|---|---|
| P-19 | Track 2 | a script "working" for two weeks | nothing executed it |
| P-23 | Track 2 | coverage exactly 0.900 | thresholds fitted in-sample |
| `tool2_holdout` | Track 1 | matched arm exactly 0.900 | same patients both sides |
| D60 gate | Track 1 | anatomy AUROC 0.499, "chance" | **pooled** across operators; 0.662 within R8 |
| D65 verdict | Track 1 | score beats anatomy by +0.329 | direction-sensitive measure scored one way |
| D72 F-test | Track 1 | p = 0.031, "no collapse" | slopes 0.3025 vs 0.3029 |
| D73 interval | Track 1 | null \|delta\| to 0.128 vs a 0.02 floor | bootstrap held the threshold fixed |
| D74 first read | Track 1 | excess +0.097, "beats anatomy" | **pooled**; −0.152 at R4, +0.100 at R8 |
| **D102 subset** | Track 1 | ρ = +0.9429, p = 0.0167, on a direction stated in writing first | the **rival predictor ties it exactly** (arc ratio, also +0.9429, a rank duplicate at +1.0000), and the null permuted 6 pair values drawn from **4 operators** as if independent; permuting the 4 labels instead gives p = 1/24 and shows +0.9429 is the statistic's **attainable maximum** |

**Seven of the nine are Track 1's own** — count the owner column; the previous text said
"five of the eight" and was wrong about its own table, which is why the column now exists.
Two were written in the same session as the module built to prevent them. The rules that
catch them:

1. **Never report a pooled statistic across operators.** Show the per-cell table.
2. **Assert the frames are disjoint on the clustering unit** — the patient, not the region.
3. **Run the null fixture before trusting any new interval.** An interval that holds anything
   fixed while resampling is understating; D73's was understating by ~5×.
4. **A rejecting test is not a finding.** Compare the magnitude against the measured noise
   floor — the patient split (~0.05 coverage), never the mask seed (0.001–0.007).
5. **Equivalence needs the interval inside the floor**, not the estimate inside it.
6. **Check direction-sensitive measures both ways.**
7. **Read the partner document before predicting.** The D-033 question was asked when the
   answer was already in `operators/decomposition.py`. **Broken a second time in D102 §2**
   (24 Nov): T1-C's flat own-operator coverage was called an unexplained calibration defect
   and a run was requested to explain it, when Track 2 had attributed it in the same message
   that carried the table — "the registry-split matched shortfall of §1e" — and the run
   already existed as T2.11 §1e. The line was quoted and then reasoned past. Corrected the
   same day, before any commit.
8. **Run the rival predictor, not just the null.** D102's +0.9429 had the right sign, a
   direction pre-stated in writing and an exact p-value, and it was still nothing — a
   scalar readable straight off the operator name scored *identically*. A sign check and a
   permutation test both passed it. Ask what the dumbest thing that would do this well is,
   and compute that too.
9. **Permute the right objects.** Pairwise data from `n` items has far fewer degrees of
   freedom than it has pairs. Permuting D102's 6 pair values gave p = 0.0167; permuting the
   4 operator labels gave p = 1/24, the test's floor. **If a statistic's observed value is
   the largest its null can produce, it has no resolution** — and that is a stronger
   refutation than a non-significant p.
10. **Check an ordinal before writing it.** "Seventh recurrence" went into a frozen constant
   with nothing in the repo supporting the count. This table had already held eight, and its
   own prose miscounted its own rows. Counts of this project's own history are hand-written
   every time; derive them from a table or do not assert them.

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
| ~~**Nothing reportable** — every bank fails a gate~~ | **OBSOLETE** (D82, D-072). The LIDC CT bank carries coverage |
| **One stamped layer, and it is the unsurprising one** — on the bank that may be quoted, Tool 1 holds at 9 of 9 operators and **0 of 4** operator-aware refinements are supported (D101, D102) | **This is now the paper risk**, and it replaces the row above. It is a scope result, not a dead end (§8) |
| **Every negative result Track 1 has is on one bank** — a single registry split of one 900-pair LIDC bank, with a per-draw coverage sd of 0.051 | Medium, and unmitigated. `BANK_RESULTS_ARE_NOT_INTERCHANGEABLE` cuts both ways: it forbids generalising the refutations too |
| ~~**T3.6/T3.7 absent** — the third front has no output~~ | **CLOSED.** Both documents exist and both tasks are run, T3.6 on the stamped bank (bound holds 72 of 72). Its row 5 is negative, which is a result, not an absence |
| ~~**The M4Raw deadlock** — both tracks believe it is the other's move~~ | **RESOLVED by D82** — coverage yes, detection no. §5a is the pre-ruling text |
| ~~Checkpoints carry **no stated licence**~~ | **RESOLVED** (T2.11). The repository carries `LICENSE` (Apache-2.0) and a `NOTICE`; **the release contains no weights**, so there is nothing to relicense. The converter and the download command ship instead |
| The DPS medical-prior bank is **partial** (~20 h GPU, resumed in stretches) | Medium |
| BraTS is skull-stripped; M4Raw is not — skull and scalp are outside the prior's training distribution | Medium, stated |
| `equispaced1d R2` has **no positives** under ε, so one grid cell is undefined | Low, dropped from rank correlations |
| The region grid is **fixed by the network** at 240² (15×15) vs the fastMRI bank's 16×16 — no per-region number carries between them | Low, stated |

---

## 8. What to do next

> **Rewritten 24 Nov (D102).** Items 1 and 2 of the old list are **done**, and the
> publication paragraph listed two things that have since been withdrawn. The old text is
> preserved in git history.

**Critical path, in order.**

1. ~~Run the full CT bank on the LIDC prior~~ — **done 19 Nov**, and it carries the coverage
   stamp. This is the bank every reportable CT number now comes from.
2. ~~Start T3.6~~ — **done**; T3.6 and T3.7 both exist.
3. **Write the Phase 3 gate** (T1.6). Still not written, and it is now the oldest open
   Track 1 deliverable.
4. ~~**T1-E, the circularity section**, with the scale-only null control. Local, no bank
   pass.~~ **DONE 26 Nov (D105)** — `T1.7_manuscript_method_and_analysis.md` §3.6. The
   scale-only null beats the posterior spread at **9 of 9** operators on both face-prior
   banks and loses on both medical-prior ones, so **the sign is set by the prior and not the
   modality**; the posterior's contribution is the **margin** (0.042, 0.119), not the level
   (0.937, 0.920). This is the one Track 1 contribution that stands without CALIBER's score
   being good.
5. **Final preprint cut (T1-F).** It now *has* a reportable table — the one in §3a's LIDC
   block — but see the warning below before drafting a word of it.
6. **Finish the DPS medical-prior bank** — completes D68 on the medical prior. *Status not
   re-checked in this sweep; confirm with Track 2 before relying on it.*
7. Decide M4Raw's substrate status at leisure, since it is no longer blocking.

**Nothing on this list is blocked on a Track 2 run, and nothing is blocked on a new bank.**

**Publication outlook.** One paper, not the three the blueprint planned.

> **The paper's content is not what this section used to say.** It listed *"Tool 2's
> sparse-grid win, the distance ordering"* — **both withdrawn** (D101, D102), and both
> measured on the bank that carries no claims. Writing them up is the specific failure D101
> was logged to prevent, and `BANK_RESULTS_ARE_NOT_INTERCHANGEABLE` exists in code to stop
> it.

What the paper can actually claim, all of it on stamped banks:

- **The failure**: a conformal threshold calibrated at one acquisition operator loses
  coverage at another — CT LIDC naive transfer below nominal at 36 of 72 ordered pairs,
  worst 0.377; MRI with a medical prior at 0.000–0.159, 13/13 in both directions.
- **The fix**: group-conditional (Mondrian) calibration restores nominal — CT LIDC
  0.892–0.905 at 9 of 9 operators over 13 patient splits, MRI 0.80–0.91 — at a measured
  cost in flagged area and FDR.
- **Two modalities with structurally different operators**, and the cost of the fix
  quantified rather than asserted.
- **The circularity caution** (§3d), which is a genuine service to the field.
- **The operator-aware layer as a negative result**: four refinements, 0 of 4 supported on
  the stamped bank, with the reasons measured — including that a scalar read off the
  operator name beats the geometric distance. Scope-limiting is part of the contribution,
  and this is the honest form of it.

The second (analytical) paper now has T3.6 and T3.7 to build on.

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
| CT **face-prior** coverage, naive, 180 → 30 views | 0.890 → **0.693** — *development only* |
| CT **face-prior** coverage, group-conditional | 0.902–0.923 — *development only* |
| **CT LIDC** (stamped) coverage, group-conditional | **0.892–0.905**, mean 0.897, 9 of 9 — *mean over 13 patient splits*; on the registry split alone 0.795–0.856 |
| **CT LIDC** (stamped) naive transfer | below nominal at **36 of 72** ordered pairs, 22 below 0.80, worst **0.377** |
| MRI + medical prior, naive | **0.000–0.159**, 13/13 both directions |
| MRI + medical prior, group-conditional | 0.80–0.91 |
| CT distance ordering, signed | **ρ = +0.976** on the **face-prior** bank — **refuted on the stamped bank** (D102): arc ratio +0.8524 beats distance −0.5046 |
| ~~Tool 2 vs `nearest`~~ | **Withdrawn, D101** — retired baseline, face-prior bank, single bandwidth |
| ~~Tool 2 valid interior folds, 6/7~~ | **Withdrawn, D101/D102** — face-prior bank, and the `valid` flips sort by interval width |
| D68 null-error share | DDNM 0.991 · DPS 0.694 |
| D68, direction label | **negative**, 0 of 3 |
| D73, 32 cells | baseline 12 · CALIBER 1 |
| `cos θ`, face → medical prior | +0.07 → **+0.60** (MRI) · +0.98 (CT smoke) |
| Identity Spearman, face → medical | 0.97 → 0.65 (MRI) · 0.99 → 0.49 (CT) |
| Weight sensitivity | restoration established at 9 of 11 weightings — **face-prior fastMRI bank, unmeasured on any stamped bank** (§4d) |
| Tests | **1,244 passing, 7 skipped** (26 Nov) · 9 (site) |

## Appendix B — where things are written

| Document | Holds |
|---|---|
| `problems_and_decisions.md` | Track 1's log, **D1–D106**, open questions, cross-track contracts |
| `decision_log.md` | Cross-track decisions, **D-001–D-072** |
| `T2_problems_and_decisions.md` | Track 2's problem log, **P-01–P-45**, plus the index of every cross-track decision |
| `T1.1`–`T1.7` | Track 1's per-task records; `T1.7_preprint_draft.md` is the narrowed paper |
| `T2.1`–`T2.11` | Track 2's weekly records; `T2.9` is Track 1's Week 10 runs, `T2.10` the medical prior, **`T2.11` the LIDC CT bank and its claim stamps** |
| `T3.1`–`T3.7` | Track 3's records. **T3.6 and T3.7 now exist** (the bound-versus-measurement comparison and the analytical writeup) |
| `site/` | The presentation site, with two demos that compute rather than recite |

> **Numbering warning.** The two logs number independently: `D12` (Track 1) and `D-012`
> (cross-track) are unrelated decisions. Always say which log.
