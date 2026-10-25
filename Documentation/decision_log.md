# Cross-Track Decision Log

Required by task T1.5 and the Week 1 card: *establish a cross-track decision log
and adjudication process for discrepancies between empirical, methodological,
and analytical results* (Sec. 19).

**Adjudication process.** When two tracks disagree — a measured coverage gap
that contradicts the bound, a distance measure that behaves differently in
theory and in data — the disagreement is logged here as OPEN, discussed at the
weekly sync, and resolved by Track 1, which owns the central claim (T1.6). An
unexplained empirical finding is referred back to Track 3 for analytical
explanation rather than dropped (Sec. 19, cross-verification).

Every entry: date, decision, alternatives considered, reason, affected
Blueprint items.

---

## D-001 · Score bank stores components, not a combined score
**Date:** Week 1 · **Status:** ACCEPTED · **Proposed by:** Track 2

The three score components (`s_free`, `s_disagree`, `s_residual`) are stored as
separate columns rather than a single pre-combined non-conformity score.

*Alternative:* store the combined score only — a smaller, simpler table.

*Reason:* Track 1 finalises the score definition in Week 3 (T1.1) and will
re-weight it repeatedly during ablation. Storing components makes every
re-weighting free; storing the total would require re-running GPU
reconstruction each time, which contradicts Sec. 17.

*Affects:* T1.1, T2.3, Sec. 17, Week 10 ablation.

---

## D-002 · Ground-truth labels are derived at analysis time
**Date:** Week 1 · **Status:** ACCEPTED · **Proposed by:** Track 2

The bank stores continuous per-region error magnitudes (`err_total`,
`err_free`) and annotation coverage, not boolean hallucination labels.

*Reason:* the O2 criterion is a Week 2 decision and will be revisited. A bank
of booleans would need regenerating on every revision.

*Affects:* O2, T2.3, Week 2 criterion definition.

---

## D-003 · `err_free` recorded separately from `err_total`
**Date:** Week 1 · **Status:** ACCEPTED · **Proposed by:** Track 2

A region can be wrong because the prior invented structure or because it
smoothed structure away. Only the first is hallucination in the Blueprint's
sense (Sec. 4). Restricting the error to the free component distinguishes them,
so both are stored.

*Open:* does the Week 2 criterion use `err_free`, `err_total`, or both? Current
default is `err_free`.

*Affects:* O2, Sec. 4, Week 2.

---

## D-004 · One solver per run, enforced in config and schema
**Date:** Week 1 · **Status:** ACCEPTED · **Proposed by:** Track 2

`experiment.solver` rejects a list, and score-bank validation reports any
`run_id` containing more than one `solver_id`.

*Reason:* Sec. 13 requires the solver held fixed within a coverage curve so the
operator is the only varying factor. Making the violation structurally
impossible is cheaper than catching it in review. The Week 10 solver ablation
runs one experiment per solver and merges afterwards.

*Affects:* Sec. 13, T1.4, Week 10 ablation.

---

## D-005 · CT uses our own projector, not LoDoPaB's supplied sinograms
**Date:** Week 1 · **Status:** ACCEPTED · **Proposed by:** Track 2

CT operators are built by re-projecting LoDoPaB ground-truth images with the
project's registered `CTProjectionOperator`.

*Alternative:* consume LoDoPaB's low-dose sinograms directly, which would be
more physically faithful.

*Reason:* the shipped sinograms fix the acquisition at one geometry, making the
operator grid — the entire object of study — impossible to construct.

*Cost:* the CT forward model is ours, not a scanner's. Must be stated in
Assumptions and Honest Scope.

*Affects:* Sec. 13, Sec. 14, T2.1, Week 7.

---

## D-006 · CT range/null-space split is approximate, and says so
**Date:** Week 1 · **Status:** ACCEPTED · **Proposed by:** Track 2

CT `pinv` is filtered backprojection, not the Moore-Penrose inverse, so `P_R`
is not an exact projector. Operators carry `exact_pinv: false` and the health
check exempts them from the idempotence gate.

*Open for Track 3:* does L1/L2 need an exact projector, or does an approximate
one only loosen constants? If exactness is needed, the CT inverse must be
upgraded (CG on the normal equations) in Week 7.

*Affects:* L1, L2, T3.3, T3.4, Sec. 13, Week 7.

---

## D-007 · CT `null_space_dim` is a dimension-count bound
**Date:** Week 1 · **Status:** OPEN · **Proposed by:** Track 2

Recorded as `ambient - measurements`, flagged in params as
`null_space_dim_method: dimension_count`. The true rank needs spectral
estimation.

*Why it matters:* L2 scales the bound with null-space dimension. If it uses
this bound rather than the estimated rank, the scaling claim is about a proxy,
not the real quantity — and the write-up must say which.

*Needs:* Track 3 to state what L2 requires. `spectrum.estimate_spectrum`
supplies the estimate if needed.

*Affects:* L2, T3.4, Week 7, Week 9.

---

## D-008 · Single-coil MRI via root-sum-of-squares
**Date:** Week 1 · **Status:** ACCEPTED · **Proposed by:** Track 2

Multi-coil volumes are reduced by RSS; the studied operator is single-coil
Cartesian undersampling.

*Reason:* consistent with the Sec. 17 compute profile on a 6 GB GPU, and avoids
coil sensitivity estimation becoming a confound in the operator definition.

*Cost:* narrows the empirical claim. Belongs in the Experimental Setup.

*Affects:* Sec. 17, T2.1, T2.7.

---

## D-009 · Region definition v0 is a 16px patch grid
**Date:** Week 1 · **Status:** **RESOLVED — Week 1 sign-off** · **Proposed by:** Track 2 · **Ruled by:** Track 1

**Confirmed as proposed.** The patch grid stays the risk unit; annotations are an
evaluation *overlay*, never the region definition. `region_area` therefore stays
constant and the weighted-rate problem does not arise.

Track 1's reasoning, in full, is **D20** in `Documentation/problems_and_decisions.md`.
Summary: §11 requires one region definition across both modalities, and MRI lesion
boxes and CT organ masks are not commensurable objects; hallucination lives in the
null space, which does not respect organ boundaries, so an annotation marks where a
hallucination *matters*, not where it *is*; and equal-area regions keep every
region-level rate a plain proportion instead of an area-weighted one.

*C4 is satisfied by* intersecting flagged patches with annotation masks —
`ann_coverage` and `label_hallucinated(require_annotation=True)` already do this.
Report both readings; annotation-restricted is secondary.

*Open sub-item:* region size (16 px) is a declared **ablation axis** for T1.4, not a
settled constant — it trades detection resolution against per-region calibration
sample size.

*Affects:* O2, T1.2, T2.4, Week 2.

---

## D-010 · Working default distance is `null_space_projection` / `ambient`
**Date:** Week 1 · **Status:** **RESOLVED — locked as primary** · **Proposed by:** Track 2 · **Ruled by:** Track 1 (T1.3)

**Confirmed as proposed, and now locked** rather than a working default. Track 2's
default and Track 3's T3.2 §5 recommendation had independently converged; this
ratifies the convergence. Full reasoning is **D19** in
`Documentation/problems_and_decisions.md`.

*Why it survives:* computable from the two operators alone; defined identically for
a k-space mask and a projection-angle set; and on Cartesian MRI under `ambient`
normalisation it reduces to √(fraction of k-space lines whose sampled status
changed) — interpretable to a non-specialist.

*T3.2's saturation warning does not apply* — the implemented measure is the
**Frobenius** form. Verified graded and monotone at 128², `random1d`, calibrating at
R=2: R = 2, 3, 4, 6, 8, 12, 16 → 0.000, 0.405, 0.500, 0.580, 0.612, 0.644, 0.650.
The operator-norm form of family (A), which pins at 1 under dimension mismatch, is
**not implemented and must not be added** under a confusable name.

*Two consequences for Track 2:*
1. **The axis compresses** (it is a square root): R=2→3 moves 0.405, R=8→16 moves
   0.037. Operator grids must be designed to spread **distance** evenly rather than
   acceleration evenly. Report `d²` as the interpretable secondary axis.
2. **Naming collision:** `operator_norm` in `operators/distance.py` is ‖A−B‖_F —
   family (B) — not the family-(A) operator norm T3.2 warns about. Worth a comment
   at the call site before someone conflates them in a figure caption.

*Ill-posedness normalisation stays out of the distance*, deliberately: L1 bounds the
gap by `2ML·d_op` and L2's job is to express `L`, `M` via null-space dimension and
spectral decay. Folding ill-posedness into `d` would make L2's scaling
unidentifiable. It is applied at plot time as a separate **named** factor, which is
what leaves the Week 9 collapse test something to test.

*`sampling_support` is retained as a free MRI-only diagnostic* — verified numerically
identical to the primary in MRI (both 0.612372 at R=2→8), heuristic in CT.
*`chordal_subspace` is an alias* (same number ÷ √2), not an alternative.

*Affects:* T1.3, T3.2, T3.3, Week 3, Week 9.

---

## D-011 · Randomised spectral estimation barely separates CT operators
**Date:** Week 1 · **Status:** OPEN · **Raised by:** Track 2 · **For:** Track 3

Measured on 32x32 phantoms with `spectrum.estimate_spectrum`. The fitted
singular-value decay exponent for parallel-beam CT:

| components `k` | 180 views | 60 views | 20 views |
|---|---|---|---|
| 16 | 1.688 | 1.687 | 1.676 |
| 64 | 1.623 | 1.620 | 1.521 |
| 200 | 1.788 | 1.777 | 1.524 |

A randomised range finder recovers the top of the spectrum, and the top looks
nearly identical across view counts. What distinguishes these operators is the
decay tail and the null space — precisely what the estimator resolves worst.

*Why it matters:* L2 relates the coverage-gap bound to null-space dimension
**and spectral decay**. If the decay term cannot be estimated reliably at this
scale, L2 either rests on `null_space_dim` alone — exact for MRI, a
dimension-count bound for CT, see D-007 — or must state that its spectral term
uses an estimate of unestablished accuracy.

*Also:* `spectral_divergence` is correspondingly weak as a primary distance
candidate. It should not be selected in Week 3 without evidence it separates
operators better than this.

*Needs:* Track 3 to decide by Week 6, when L2 starts. Options are exact SVD on
small synthetic operators only (T3.5 already works at that scale), a better
tail estimator, or dropping the spectral term from the headline statement.

*Affects:* L2, T3.2, T3.4, Sec. 12.1, Weeks 3, 6, 9.

---

## D-012 · The FBP-based CT null projector amplifies; it must be replaced
**Date:** Week 1 · **Status:** RESOLVED — fixed · **Raised by:** Track 2 · **Fixed:** Week 7 (ahead of schedule)

Measured on real LoDoPaB validation data (`w07_ct_smoke`, 128x128, 24 items):

| bank | median `err_free` | median `err_total` | ratio |
|---|---|---|---|
| MRI (phantom) | 0.104 | 0.134 | **0.78** |
| CT (LoDoPaB) | 0.664 | 0.501 | **1.33** |

For a true orthogonal projection `‖P_N e‖ ≤ ‖e‖`, so the ratio cannot exceed 1.
CT's is 1.33. The CT `P_N = I − A⁺A` built from filtered backprojection is not
a contraction: **FBP amplifies rather than projects.**

*Consequence, and it is not cosmetic.* Both the CT free-component score
`s_free` and the CT ground-truth label `err_free` are then dominated by FBP
reconstruction error rather than by invented structure. They measure the wrong
thing. Concretely, at `err_free > 0.05` **100% of CT regions label as
hallucinated**, which makes region FDR identically zero and the CT flag
vacuous. The v0 criterion threshold cannot be shared across modalities as
things stand — and sharing it is what the cross-modality claim requires.

*This is the empirical answer to the question D-006 left open for Track 3:* the
approximation does not merely loosen constants, it inverts the inequality the
free component is supposed to satisfy.

### The part that threatens the headline result

The defect is not constant across the grid — it grows with sparsity
(128x128, 180°):

| views | `‖P_N x‖/‖x‖` | idempotence error |
|---|---|---|
| 180 | 0.667 | 0.635 |
| 120 | 0.753 | 0.807 |
| 90 | 0.867 | 1.202 |
| 60 | **1.107** | 2.167 |
| 45 | **1.287** | 2.709 |
| 30 | **1.738** | 5.893 |

The projector degrades **monotonically in the same direction as operator
distance**, and the idempotence error is large at *every* view count — FBP is
nowhere near a projector even at 180 views.

So a CT coverage-gap-vs-distance curve could come out monotone for the wrong
reason: not because reliability degrades with operator shift, but because our
own decomposition degrades with operator shift. The two effects are currently
confounded, and the Week 9 curve-collapse test cannot distinguish them.

**This must be fixed before any CT coverage curve is reported**, not merely
before publication. Until it is, CT numbers are for plumbing only.

### Fix, landed — scoped to the contraction property, not idempotence

FBP is replaced with a genuine pseudo-inverse for the **decomposition**:
conjugate gradient on the normal equations `AᵀA x = Aᵀy`
(`CTProjectionOperator.pinv_cg`). Starting from `x0 = 0`, every CG iterate
stays in the Krylov subspace of `AᵀA` and `b = Aᵀy`, which lives entirely in
`row-space(A)` — so CG never touches `null-space(A)` and heads, within row
space, towards the minimum-norm least-squares solution, `A⁺y`.

**FBP stays the display reconstruction**, exactly as planned — `pinv()` is
unchanged and still what solvers and score-bank baselines read. Only the
decomposition (`project_range` / `project_null`, everything `err_free` and
`s_free` are built from) is repointed, through a new
`ForwardOperator.decomposition_pinv` hook that defaults to `pinv` and that CT
overrides.

**What this fixes: contraction.** Measured across the Sec. 16 operator grid
(64x64 to 256x256, 8 to 180 views, sparse-view and limited-angle),
`null_projection_gain()` clears `≤ 1` robustly — often from as few as 5-10 CG
iterations. `exact_pinv` is unchanged (still False for CT) and
`OperatorRegistry.check_health()`'s idempotence gate (`projector_ok`) stays
keyed on it, so it stays bypassed for CT exactly as before — that gate was
never what D-012 was about. What *is* now gated, promoted from a runner
warning to a hard failure, is `contraction_ok`.

**What this does not fix, and was never going to: idempotence.**
`projection_idempotence_error()` settles in the low tens-of-percent range at
best, and is *not* monotone in CG iteration count — applying an
incompletely-converged solve twice, as the idempotence test does, compounds
approximation error in a way more iterations of the first solve doesn't
simply undo. CT's decomposition is a cheap iterative approximation, not an
exact projector, before or after this fix; only its worst-case behaviour
(amplification) has changed.

**Why iterations are capped, not driven toward convergence.** More CG
iterations do not reliably help (see above) and are not free: measured
several seconds per single decomposition call on CPU at 256x256 with 100+
views, past which returns visibly diminish. Worse, unregularized CG on a
rank-deficient system is not merely slow past the point where it has
extracted the recoverable signal — it is numerically unsafe: an early,
undamped version of this fix diverged by 6+ orders of magnitude beyond ~500
iterations on a 16-view operator, because once the Krylov recurrence exhausts
`row-space(A)`'s well-determined part, further iterations amplify rounding
error in near-null directions instead of resolving anything. `pinv_cg` now
tracks the best iterate by residual norm and stops outright once the residual
turns upward, rather than trusting whatever iteration count is configured to
be safe. Iteration count and tolerance are exposed as `cg_iterations` /
`cg_tol` constructor params (default 20 / 1e-6 — chosen because the
contraction property saturates by ~20 iterations while idempotence keeps
wobbling non-monotonically well past it, so more is cost without reliable
benefit), recorded in the operator's
`params` for reproducibility, tunable per experiment, and deliberately
excluded from `operator_id` (decomposition numerics, not physical identity —
same treatment as `null_space_dim_method`).

*Verified — including on the real data the original defect was found on.*
`null_projection_gain() ≤ 1` at the `test_operators.py` fixture scale (64x64,
32 views, 180°) and spot-checked across the wider grid above. Re-running
`w07_ct_smoke.yaml` (real LoDoPaB validation slices, 128x128, 180/90/45 views,
1536 region rows) end to end after the fix:

| operator | median `err_free`/`err_total` | max row-level ratio | rows with ratio > 1 |
|---|---|---|---|
| v180 | 0.024 | 0.79 | 0 / 512 |
| v90 | 0.038 | 0.76 | 0 / 512 |
| v45 | 0.065 | 0.70 | 0 / 512 |

Against D-012's original measurement of a **1.33** median ratio (`err_free`
exceeding `err_total`, which a genuine decomposition cannot produce), every
one of 1536 real-data region rows across all three operators now has
`err_free` a small, bounded fraction of `err_total`, with zero violations of
`err_free ≤ err_total`. This is the direct empirical resolution of D-012 on
the substrate the defect was originally found on. Refreshing P-02's specific
numbers under the new decomposition — the O2 threshold decision Track 1 owns
— is the follow-up.

*Performance, measured on the same run.* `w07_ct_smoke.yaml` end to end
(24 item×operator pairs, 128x128): 345s at the initial `cg_iterations=50`
attempt, 162s after tuning the default down to 20 — confirms the "more
iterations, no reliable benefit" finding above wasn't just a synthetic-grid
artefact. Measured on CPU; expect substantially faster on the project's
target GPU.

*Detection, promoted from a warning to a gate.*
`ForwardOperator.null_projection_gain()` and `contraction_ok` in
`OperatorRegistry.check_health()` now fail the run (`runner.py`) rather than
merely warn — expected to read ≤ 1 for every registered operator, MRI and CT
alike, and a run should stop if that regresses.

*Affects:* O1, O2, L1, L2, D-006, T2.1, T2.5, T3.3, Sec. 13, Weeks 7-9.

---

## D-013 · TotalSegmentator masks do not align with LoDoPaB images
**Date:** Week 1 · **Status:** RESOLVED (Week 1 sign-off) — Option A adopted · **Raised by:** Track 2

> **Resolution.** Option A: TotalSegmentator becomes the CT substrate. Track 1's Week 1
> sign-off records that it does not object, and D20 removed the coupling that made this
> a cross-track question at all — annotations are an evaluation *overlay*, not the region
> definition, so the choice of CT substrate no longer changes what a region is. D-005
> already re-projects ground-truth images with our own operator, so no projection data is
> needed and LoDoPaB's observation archives stay unused (D-005, D-013).

Sec. 14 lists LoDoPaB-CT as the cross-domain substrate and TotalSegmentator as
the source of "region-level ground truth for evaluating flags in the second
modality". Inspecting both datasets shows those two roles do not compose:
**TotalSegmentator's masks segment TotalSegmentator's own CT volumes.** It is a
different cohort from LoDoPaB, which derives from LIDC-IDRI. There is no
correspondence between subjects, so LoDoPaB slices cannot be annotated with
TotalSegmentator masks.

Without region-level annotations the CT flag cannot be evaluated on whether it
fires *where it matters*, and Front II stops mirroring Front I — which is the
whole point of the cross-domain front.

### Option A — use TotalSegmentator as the CT substrate (recommended)

Each of its 1228 subjects ships `ct.nii.gz` plus 117 aligned organ masks
(verified: 11 of the first 12 subjects carry liver, spleen, kidney, lung, or
vertebra labels). Because D-005 already means we re-project ground-truth images
with our own operator, **no projection data is needed at all** — only clean CT
images and aligned region labels, which is exactly what this provides.

This mirrors the MRI setup one-for-one: image plus region annotation.

*Cost:* TotalSegmentator is resampled to 1.5 mm isotropic and spans varied body
regions and scanners, so it is heterogeneous where LoDoPaB is uniform. It is
also not a low-dose reconstruction benchmark, so we lose the ability to compare
against published LoDoPaB numbers — which we were not going to do anyway, since
we define our own operators.

*Consequence:* LoDoPaB becomes optional. It could still serve as an
unannotated second CT source for the coverage curve, since coverage against
operator distance does not require annotations — only the FDR/power metrics do.

### Option B — keep LoDoPaB and generate masks

Run the TotalSegmentator *model* (a separate pip package, not this dataset) over
LoDoPaB ground-truth images. Keeps LoDoPaB as substrate but adds an inference
dependency, GPU time we budgeted for reconstruction, and label noise from a
model rather than curated ground truth.

*Recommendation:* Option A. Decide at the Week 1 kickoff, since it determines
which ~49 GB of archives can be deleted and what the Week 7 loader targets.

*Affects:* O5, Sec. 11, Sec. 14, T2.1, T2.5, Weeks 7-9.

---

## D-014 · LoDoPaB must be split by patient, not by slice
**Date:** Week 1 · **Status:** ACCEPTED · **Proposed by:** Track 2

`patient_ids_rand_*.csv` supplies one patient ID per slice in slice order:
35,820 train / 3,522 validation / 3,553 test slices across far fewer patients.

A slice-level random calibration/test split therefore places the *same patient*
on both sides. Adjacent slices of one patient are near-duplicates, so this
inflates apparent coverage and violates the exchangeability the conformal
guarantee rests on — in a way that would make our headline numbers look better
than they are, which is the worst direction for an error to run.

Splits are assigned by patient. The same applies to fastMRI, where slices come
from volumes: split by volume, never by slice.

*Note:* `LoDoPaBDataset` currently indexes by file and slice offset and does not
yet read these CSVs. Wiring that is a Week 7 item, or earlier if CT moves up.

*Affects:* Sec. 5.2, Sec. 16, T2.1, T1.2, Weeks 3, 7.

---

## D-015 · M4Raw as an open stopgap while fastMRI access is pending
**Date:** Week 1 · **Status:** ACCEPTED · **Proposed by:** Track 2

fastMRI needs a signed Data Sharing Agreement and an approved application. That
cannot be automated and it gates Week 3. Rather than idle, MRI development
proceeds against **M4Raw** ([Zenodo 7523691](https://zenodo.org/records/7523691),
CC-BY-4.0): multi-channel brain k-space, 240 validation volumes, no application.

*Why it substitutes cleanly:* M4Raw ships ISMRMRD `.h5` with the same `kspace`
and `reconstruction_rss` keys fastMRI uses, so `FastMRIDataset` reads it with
little or no change. Retrospective undersampling — the thing the whole project
does — needs only fully-sampled k-space, which M4Raw has.

*Where it is not a substitute:*
- 0.3 T low-field brain at ~1.2 mm, against fastMRI's 3 T knee. Different
  noise regime and anatomy.
- **No pathology annotations.** There is no fastMRI+ equivalent, so the
  annotation-restricted evaluation path (region FDR/power against clinically
  meaningful regions, Sec. 14) cannot run on M4Raw. Coverage against operator
  distance *can*, since that needs no annotations.

*So:* M4Raw carries pipeline development, solver wiring, and the coverage-curve
machinery. It does not carry any reported MRI result, and it cannot close O2's
annotation-linked evaluation. fastMRI remains required.

*Affects:* T2.1, T2.2, T2.3, Weeks 2-3.

---

## D-016 · Natural-image diffusion prior wired before medical priors
**Date:** Week 1 · **Status:** ACCEPTED · **Proposed by:** Track 2

`google/ddpm-celebahq-256` (Apache-2.0, 434 MB, 113.7 M params, verified
loadable) is fetched to `checkpoints/`.

*Reason:* Week 2 must wire DPS and DDNM (T2.2). Debugging a guidance
implementation against an unfamiliar medical prior conflates two unknowns. A
standard 256x256 unconditional prior is the setting DPS and DDNM were published
on, so the implementation can be validated against known-good behaviour first,
then pointed at medical weights.

*Second use:* this is also the prior for the Week 10 natural-image robustness
appendix (Sec. 11.1), so it is not throwaway work.

*Still outstanding:* the medical priors. Song et al. [27] hosts knee-MRI and CT
score models on Google Drive, not scriptable. **Note:** their MRI model is
fastMRI-derived, so the NYU agreement plausibly reaches the checkpoint as well
as the data — confirm before use, and certainly before the open-source release
(O7).

*Searched, Week 2 — no open substitute exists.* The Hugging Face Hub was searched for an
openly-licensed medical MRI diffusion/score prior that could replace the Google Drive
dependency. There is none. What the Hub carries for fastMRI is **end-to-end reconstruction
networks** (VarNet, CIRIM, MoDL, KIKI-net, UNet — the `wdika/REC_*` and
`zaccharieramzi/*` families), which are the wrong object twice over: they are
deterministic reconstructors rather than generative priors, so they cannot be driven as
black-box posterior samplers at all, and they are trained *at fixed 4x/8x undersampling*,
which bakes in the acquisition operator that this project requires to be the free
variable. The generic "medical diffusion" results are text-to-image X-ray LoRAs and
hobby repos with single-digit downloads and no provenance. So D-016's deferral is not a
matter of not having looked: Song et al. remains the reference option, with its licensing
question unresolved.

*Affects:* T2.2, T2.5, O7, Weeks 2, 10, 12.

---

## D-017 · fastMRI acquired by direct download, staged off-machine
**Date:** Week 1 · **Status:** RESOLVED (Week 1 close-out) · **Proposed by:** Track 2

> **Resolution.** The outstanding check is answered: the download links were **NYU's own,
> emailed after they approved the application** — not a Kaggle mirror or any third-party
> redistribution. The agreement's redistribution clause is therefore not engaged, and
> results computed on `singlecoil_val` are reportable. `singlecoil_val` is extracted,
> verified (199 volumes / 7,135 slices), and is the reported MRI substrate; M4Raw reverts
> to what D-015 always scoped it as — a development stopgap, not the lawful fallback.
>
> The archive was deleted after verification to reclaim 16 GB (P-14). The extracted data
> is intact; re-obtaining the tar means requesting fresh links from NYU, since the emailed
> ones are time-limited.
>
> One clause still binds at release: §6 — fastMRI forbids redistribution, so the O7
> open-source release ships code, configs and score banks, never images. Confirm before
> Week 12 that derived score banks are within terms.

Download links for fastMRI were obtained and the transfer is running on Kaggle
rather than on the 6 GB laptop. The local disk has ~30 GB free against a 7 GB
knee singlecoil split plus a 30.1 GB TotalSegmentator extraction, so staging
the fetch elsewhere and copying in only `singlecoil_val` is the difference
between fitting and not fitting.

*Reason:* fastMRI was the critical path and the only item with external lead
time. It is now moving, which unblocks the Week 3 deliverable.

*Consequences for the code, already handled:* volume discovery is recursive
(`data.fastmri.find_volumes`), because archives staged through another
environment rarely arrive with the same nesting. Configs stay pointed at the
split directory rather than `data/raw/fastmri`, so that train and val cannot be
swept into one index — the D-014 hazard in its fastMRI form.

**Outstanding check, and it is not a formality.** The NYU Data Sharing
Agreement is a legal agreement between the recipient and NYU, and its terms
forbid third-party redistribution — which is why `data_access.md` §0 rules out
mirrors. Confirm the download source is NYU's own approved links, and that
staging the data through a Kaggle account does not itself constitute
redistribution under the agreement signed. If either fails, the data is
unusable for a reported result no matter how well the pipeline works, and
M4Raw remains the only lawful MRI substrate.

*Affects:* T2.1, O7, Sec. 16, Weeks 3-4, 12.

---

## D-018 · Reportability is a property of the bank, not of whoever ran it
**Date:** Week 1 · **Status:** ACCEPTED · **Proposed by:** Track 2

`SliceDataset.reportable` is False for phantoms and M4Raw; the runner combines
it with `solver.family == "reference"` and writes a `provenance` block into
every manifest. The runner prints `NOT REPORTABLE` at run time and
`analyse_bank.py` prints it again at analysis time.

*Reason:* three of the project's substrates and samplers are development-only
by design — phantoms (D-001 smoke path), M4Raw (D-015), and the reference
samplers. Each produces a completely valid score bank. A bank is a table of
floats and carries no visible mark of where it came from, so the distinction
survived only in whoever remembered which config was run. Over twelve weeks and
three tracks reading each other's banks, that is not a safe place to keep it.

*Alternative:* a naming convention on `run_id`. Rejected — conventions are not
enforced, and the failure is silent in exactly the direction that flatters the
results.

*Affects:* D-015, D-016, T2.3, T2.7, O7, Sec. 17.

---

## D-019 · fastMRI+ boxes live on the 320x320 reference grid, not on k-space
**Date:** Week 1 · **Status:** ACCEPTED — fixed · **Found by:** Track 2

Rasterising the fastMRI+ bounding boxes on the inverse-FFT grid put **every
annotation in the wrong place**. Knee k-space inverts to 640x368; the boxes are
drawn on fastMRI's 320x320 `reconstruction_esc` reference image.

Measured across the shipped file — 16,154 boxes, 969 volumes:

| assumed grid | boxes surviving a 256 centre crop |
|---|---|
| 320x320 reference | **100.0%** |
| 640x368 k-space | 25.5% |

`max(x + width) = 319` and `max(y + height) = 320` exactly, which fixes the
grid at 320x320 and leaves no room for a different reading.

*Why it had no symptom.* A misplaced mask is still a valid mask: right dtype,
right shape, plausible area. The first real box tested rasterised to **zero
pixels** after the crop, and nothing downstream would have objected — regions
would simply have been scored against annotations describing different
anatomy. This is exactly the failure `data_access.md` §2 warns about, and the
loader was committing it.

*Fix.* The image is cropped to the annotation grid first, then image and mask
take an identical second crop to the working resolution, so the two agree by
construction rather than by coincidence. The grid is read from
`reconstruction_esc` / `reconstruction_rss` in the volume when present, since
it differs between fastMRI splits, and falls back to `ANNOTATION_GRID` when
absent. Guarded by `test_boxes_are_placed_on_the_annotation_grid_not_the_kspace_grid`
and by a check against the real file.

*Consequence for the working resolution.* At 256x256 all boxes survive the
crop; at 128x128 only 76.7% do. The Sec. 17 compute profile prefers 128, and
the annotation-restricted evaluation (Sec. 14) wants boxes. **Whoever sets the
MRI working resolution is trading region FDR/power sample size against GPU
time**, and should do so knowingly — the CT smoke run uses 128, the MRI configs
use 256.

*Affects:* O2, T2.1, T2.4, Sec. 14, Sec. 17, Weeks 3-5.

---

## D-020 · The v0 label threshold is substrate-dependent, not just modality-dependent
**Date:** Week 1 · **Status:** OPEN — needs the Week 2 O2 decision · **Raised by:** Track 2

First run of the MRI pipeline on real raw k-space (`w02_m4raw_smoke`, M4Raw,
128x128, 6 operators, 4,608 region rows). At the v0 criterion `err_free > 0.10`,
**100% of MRI regions label as hallucinated** — region FDR is identically zero
and the flag is vacuous.

That is the same symptom D-012 reports for CT, but it has a different cause,
and the difference matters.

| substrate | median `err_free` | p1 | ratio `err_free`/`err_total` | frac > 0.10 |
|---|---|---|---|---|
| phantom (MRI) | 0.104 | 0.030 | 0.78 | 52.5% |
| M4Raw (MRI) | 0.206 | 0.116 | **0.97** | 100% |
| LoDoPaB (CT) | 0.664 | — | **1.33** | 100% |

**The MRI decomposition is sound.** Its ratio is 0.97, i.e. `≤ 1` as an
orthogonal projection requires, and all six operators passed the contraction
check. CT's 1.33 is the D-012 defect; MRI has no equivalent. This is useful
evidence *for* D-012 being a CT-specific artefact of filtered backprojection
rather than something wrong with the free component in general.

**What is wrong here is the threshold, not the projector.** Real brain anatomy
carries far more null-space content than a smooth phantom, so the whole
`err_free` distribution shifts up — the 1st percentile on M4Raw (0.116) already
clears a threshold set at 0.10. The v0 constant was implicitly calibrated to
phantoms.

*So the criterion cannot be a shared absolute constant.* D-012 established that
the threshold does not transfer across modalities; this establishes it does not
transfer across substrates *within* one modality. Anything the Week 2 O2
decision picks has to be relative — a quantile of the error distribution, a
per-item normalisation, or a criterion defined so the positive rate is a
controlled quantity rather than an outcome.

*The pipeline itself is fine.* At `err_free > 0.25` (17.8% positives) the
central phenomenon reproduces on real k-space for the first time: naive
transfer holds 0.944 coverage at the calibration operator and collapses to
0.217 at R=4 and 0.014 at R=8, monotonicity rho = +0.829 against operator
distance. Not reportable — M4Raw plus a reference sampler, doubly so under
D-018 — but it is the first evidence the claim survives contact with real
measurements rather than phantoms.

*Needs:* Track 1 to fix the criterion in Week 2 (O2), knowing that a constant
will not do. Interacts with **D-003** (which error the criterion uses) and
**D-009** (region definition).

*Affects:* O2, T1.1, T1.2, T2.3, T2.4, D-003, D-009, D-012, Weeks 2-3.

---

### Track 1 ruling — Week 1 sign-off · **Status: RESOLVED, with the diagnosis revised**

Full entry: **D21** in `Documentation/problems_and_decisions.md`.

**The finding is real, but the cause is not the threshold — and that changes the fix.**
`w02_m4raw_smoke` runs `reference_nullspace`, whose draws are `A⁺y + P_N(texture)`.
`A⁺y` carries **zero** null-space content by construction, so
`P_N(mean_recon − x_true)` is dominated by `−P_N(x_true)`. On this sampler `err_free`
is therefore measuring **the true image's own hidden content**, not what the
reconstruction invented. Real anatomy has much of it; a smooth phantom has little.
That *is* the phantom-vs-M4Raw gap.

Reproduced on synthetic images alone (128², R=4, reference sampler, 16 px grid),
without touching M4Raw:

| true image | mean `err_free` | ‖P_N x_true‖/ref | labelled at 0.10 |
|---|---|---|---|
| smooth (phantom-like) | 0.0715 | 0.0771 | 14.1% |
| textured (anatomy-like) | 0.1878 | 0.2046 | **100%** |

Per-region `err_free` vs ‖P_N x_true‖: **r = 0.95**, mean ratio 0.92. This also
accounts for the `err_free`/`err_total` ratio moving 0.78 → 0.97 in the table above:
as the truth's null-space content grows, essentially *all* of a prior-free
reconstruction's error becomes free-component error.

**Consequence:** no constant, absolute *or* relative-to-global-intensity, fixes this
while the sampler has no prior — the thresholded quantity is a property of the
anatomy. (Note `err_free` is *already* divided by `ref_intensity` in `labels.py`, so
"make it relative" has in effect been tried.) **So the threshold must not be set from
this run.** Lock the criterion's form now; set τ once a real prior lands (T2.2).

**Criterion form v1:** `err_free(r) / (free_content(r) + ε) > τ`, where
`free_content(r) = ‖P_N x_true‖_r / ref_intensity` — *what fraction of this region's
hidden content did the reconstruction get wrong*, with `>1` meaning it invented more
than was ever there. Dimensionless, substrate-independent, modality-independent.
Using `x_true` is legitimate: a **label** may use ground truth, only the **score** may not.

**Ask to Track 2 (A1):** add one nullable column `free_content` to the bank. Per
score-bank spec §6 an added nullable column is backwards-compatible — no
`SCHEMA_VERSION` bump, no re-run, and the criterion stays re-derivable from cache
forever (Rule 2).

**Sanity property:** on `reference_nullspace` the v1 ratio sits near 0.92 everywhere,
correctly reporting *"this sampler does not invent, it fails to fill the null space."*

**τ is reported, not targeted.** Choose it once on held-out data after T2.2, then
report the induced positive rate per substrate as a measured property. Tuning τ to
hit a target rate would make the positive rate definitional and hollow out region-level
FDR and power.

---

## D-021 · `calib_group` keeps its granularity and moves into the registry
**Date:** Week 1 · **Status:** RESOLVED · **Ruled by:** Track 1 (T1.2) · **Track 1 entry:** D22

Granularity unchanged — `mri/{family}/R{acceleration}`, `ct/{family}/v{n_views}` —
and the definition moves out of `runner.calibration_group` into the **operator
registry** as a field, so Track 1 can re-group without a Track 2 code change. Both
halves as Track 2 recommended (B2/C2).

**Noise is not a group axis in v1.** It multiplies the group count, and D10 (Track 1
log) already warns that high-acceleration regimes are where per-regime calibration
data is scarcest — exactly where Tool 1's declared trade-off bites hardest. If T1.4's
protocol varies noise as a studied axis it *becomes* a group axis and the registry
field must reflect it. Recorded so that stays a decision, not a silent default.

*Affects:* T1.2, T2.1, T2.3, score-bank `calib_group`, Week 4.

---

## D-022 · Split is fixed per patient in the registry, and must be operator-independent
**Date:** Week 1 · **Status:** RESOLVED · **Ruled by:** Track 1 (T1.2) · **Track 1 entry:** D23

Split assignment is deterministic per `sample_id`, made at **patient/volume** level
(D-014), stored in the registry, identical across every run, every operator, and both
modalities. Never per-run. Track 2's B3 recommendation, confirmed.

**Track 1 adds the binding reason.** The split must be **operator-independent** — the
same patient on the same side in *every* regime. Otherwise group-conditional
calibration compares different patient populations across regimes, and a coverage
difference between R=4 and R=8 is confounded with anatomy rather than attributable to
acquisition. That is the Sec. 13 controlled-comparison assumption applied to the
split; violating it would undermine the headline claim, not merely inconvenience the
Week 10 merge.

*Affects:* T1.2, T2.1, T2.3, D-014, bank `split` column, Week 10 merge.

---

## D-023 · One calibration group per (family, acceleration); seeds are members
**Date:** Week 1 · **Status:** RESOLVED · **Ruled by:** Track 1 (T1.2) · **Track 1 entry:** D24

A mask family at fixed acceleration is **one** group; `mask_seed` values are members
of it. Track 2's C3 recommendation, confirmed.

Seeds from the same family at the same acceleration are exchangeable draws from a
single acquisition regime, and that exchangeability is exactly what licenses using
them as calibration data for it. Splitting by seed would discard the only
within-regime exchangeability the design has, while starving each group (D10, Tool 1).

**This also defines the Week 4 grid's axes** — the part that matters beyond grouping.
A *mask-family shift* = change family at fixed acceleration. An *acceleration shift* =
change R at fixed family. Both are between-group moves. **Seed variation is
within-group and is therefore the noise floor a real shift must clear**: a coverage
change smaller than seed-to-seed variation is not a finding, and T1.4 should report
that floor alongside the curves.

*Affects:* T1.2, T1.4, T2.1, Week 4 grid.

---

## D-024 · `NullSpaceNoiseSampler` leaks ~4% of its "hallucination" into the measured component
**Date:** Week 2 · **Status:** RESOLVED — fixed, and it moved a headline number · **Raised by:** Track 1 (T1.1) · **Owner:** Track 2 · **Track 1 entry:** D29

> **Resolution.** The `.real` cast is gone from both reference samplers; draws stay in the
> operator's own complex domain and magnitudes are taken only where something is measured.
> `s_residual` is now **exactly 0.000000** at every operator, where it had been 0.082 at
> R=2 and 0.032 at R=10. `tests/test_splits.py` asserts data consistency for both samplers.
> **This changed a result, not just a number** — see D-028.

`solvers/reference.py` builds each draw as `base_real + free`, where
`free = operator.project_null(texture)` is cast with `.real` before it is added.
A random 1-D Cartesian mask is **not conjugate-symmetric**, so `P_N` of a real
image genuinely has an imaginary part — measured at **33% of its norm** at
128², R=4. Discarding it means the perturbation no longer lies in the null space.

Measured, same operator and texture:

| injection | relative change in `A x` |
|---|---|
| full complex `P_N delta` | 9.6e-08 (float noise) |
| `.real` part only | **4.1e-02** |

**Why it matters.** The docstring calls the sampler "a *faithful caricature* of
the failure the project studies... the measured component is pinned to the data
and every posterior draw differs only inside the operator's null space." With the
`.real` cast that is not what it does: draws perturb the measured component too,
so `s_residual` is non-zero as a pure artefact, and the smoke numbers taken from
this sampler — the README coverage table, the D-020 coverage figures — carry that
confound.

**Not urgent** (D-018: nothing from this sampler is reportable) but it should be
fixed before the sampler is used to characterise anything, and before anyone
reads its `s_residual` as evidence.

**Fix:** keep the projection complex and take magnitudes only at the point of
measurement, as `tests/test_score_combination.py` now does. The same `.real`
pattern is worth grepping for elsewhere in the operator paths.

*Found by* the WI-4 injection test failing its own data-consistency precondition —
the test working as intended.

*Affects:* T2.2, `solvers/reference.py`, README smoke table, D-018, D-020.

---

## D-025 · `data/README.md` removed; all data documentation consolidated into `data_access.md`
**Date:** Week 2 · **Status:** DONE · **Raised and actioned by:** Track 1 (T1.5)

`data/README.md` had grown into a status board — per-archive `OK`/`downloading`
markers, "~30 GB free", "downloading off-machine", "nothing has been deleted,
the call is yours". Two problems with that in a tracked file.

**It described one machine.** `data/` is gitignored apart from this README, so
on any other clone the file documents contents that are not there. It reads as
a report on whoever last ran the downloads.

**It had already gone stale against itself.** The layout section listed m4raw as
extracted with 240 volumes while the archive inventory five sections later still
showed the same archive as downloading — in the same file, in the commit named
"M4Raw landed". Status tables in version control drift the moment they are
correct.

**Actioned.** The durable half moved into `data_access.md`, which is the
provenance record and already carried the status board: archive inventory with
measured sizes and the ~55 GB recoverable (§6), why `observation_*` and
`ground_truth_train` are not needed (§6), extraction commands (§6), M4Raw's
layout and its `patient_id` attribute (§1a), LoDoPaB's `patient_ids_rand_*.csv`
and why slice-level splits leak (§3), and the TotalSegmentator/LoDoPaB cohort
mismatch (§4).

**Final outcome went one step further than first proposed.** The intermediate
state was a trimmed `data/README.md` holding the layout tree; on review that was
cut too, and `data/` now carries no README at all. One data document, not two.
The layout tree moved into `data_access.md` §6 with it.

No content was dropped; the volatile status markers were, deliberately — the
per-archive `OK`/`downloading` flags, free-disk figures, and "downloading
off-machine" notes. `fetch_data.py --list` / `--verify` answer those, and cannot
go stale.

*Track 2 owns both files* — flagged here rather than done silently. Say if the
inventory should live back in `data/` and it goes back.

*Affects:* T2.1, `data/README.md` (deleted), `Documentation/data_access.md`, `.gitignore`.

---

## D-026 · Reference `split_conformal_threshold` over-certifies thin groups
**Date:** Week 2 · **Status:** RESOLVED (Week 4) · **Raised by:** Track 1 (T1.2) · **Owner:** Track 2 · **Track 1 entry:** D32

Two issues in `calibration/__init__.py`, found while T1.2 superseded it.

**Docstring contradicts code.** The docstring says the threshold is the
`ceil((n+1)(1-alpha))`-th smallest score among hallucinated regions; the code
takes the `ceil((n+1)*alpha)`-th. The code is the nearer of the two.

**The ceiling certifies groups it should refuse.** For exchangeable scores the
correct constant is `k = floor(alpha*(n+1))`, giving `P(miss) = k/(n+1) <= alpha`
exactly. Rounding up breaks this where it matters most: at `n=1, alpha=0.1` the
ceiling returns `k=1`, so the threshold is that single positive score and the
true miss rate is **1/2** — five times the promise, with nothing in the return
value indicating a problem. Measured at ~0.5 over 2,000 trials.

**Impact: none published.** The function is marked not-for-reported-results and
the Week 4 harness only exercises it. `caliber.calibration.conformal.split_threshold`
now floors and returns `certifiable=False` for thin groups.

**Suggested fix:** delegate the reference to `conformal.split_threshold`, or
floor and fix the docstring. Worth doing before someone reuses it in good faith.

*Affects:* T1.2, `calibration/__init__.py`, Week 4 harness.

**Resolved, Week 4, by taking the suggested fix.** Both reference functions are now thin
wrappers over `conformal.split_threshold` rather than a second implementation of the same
rule, which is what let the two drift in the first place.

Two behaviours changed, both toward correctness:

* **`ceil` to `floor`.** Thin groups now come back `certifiable=False` instead of a
  plausible-looking threshold. On the Week 3 bank the thresholds move by about **0.001**
  and no reported coverage number changes, because every group there carries 2,000–3,800
  positives. That is precisely why this survived four weeks: the defect is invisible where
  the data is thick and wrong where a reviewer would look hardest.
* **The empty case flipped sign.** With no positives at all the old code returned `+inf` —
  flag nothing, a 100% miss rate presented as a successful calibration. The delegate
  returns `-inf`: flag everything, valid and visibly useless. This was not in the original
  report and is the more dangerous of the two, since `+inf` fails in the flattering
  direction.

The float return cannot carry `certifiable`, so `mondrian_calibration_results` was added
alongside it, and `analyse_bank.py` and `sizing.py` now print a **NOT CERTIFIABLE** line
naming any group with fewer than `min_positives(alpha)` positives. Flooring alone would
have made the numbers correct while leaving them just as silent.

Pinned by four tests in `test_calibration.py`, including the `n=1, alpha=0.1` case from
this entry and an equivalence check against the delegate across three sizes and three
alphas — so the two cannot drift apart again.

---

## D-027 · fastMRI `singlecoil_val` has 199 volumes but only 96 patients; the split must be by `patient_id`
**Date:** Week 1 · **Status:** ACCEPTED — fixed · **Found by:** Track 2

D-014 fixed the rule as "split by patient for LoDoPaB, **by volume for fastMRI**, never
by slice", on the assumption that a fastMRI volume *is* a patient. It is not. Reading
the attributes of all 199 extracted volumes:

| quantity | value |
|---|---|
| volumes in `singlecoil_val` | 199 |
| distinct `patient_id` | **96** |
| patients owning more than one volume | **96 of 96** |
| patients leaked if split by volume | **45 of 96 (47%)** |
| patients leaked under the patient-level split | **0** |

Every patient in the split has more than one volume, so the by-volume rule puts nearly
half of all patients on both sides of the calibration/test boundary. That is the exact
hazard D-014 was written to prevent, in the modality D-014 declared safe — and it fails
in the direction that flatters our own numbers, because a patient seen during
calibration makes the test set easier.

**Fix.** `caliber.data.splits` assigns the split from the de-identified `patient_id`
carried in each file's HDF5 attributes, deterministically and *operator-independently*
(D-022): a pure function of `(salt, patient_key)`, so no run, ordering, or acquisition
parameter can move a patient. `FastMRIDataset(split="auto")` uses it;
`verify_no_patient_straddles` is the assertion, and `split_summary` prints the counts
before a bank is produced. Resulting balance: 52/44 patients, 108/91 volumes.

A property worth having deliberately: because assignment is per key rather than a
shuffle, **growing the dataset does not move anybody**, so a bank built on 199 volumes
stays comparable with one built after the substrate grows.

*Affects:* D-014, D-022, P-06, T2.1, T2.3, every reported MRI coverage number.

---

## D-028 · Fixing D-024 reversed the direction of the naive-transfer coverage failure
**Date:** Week 1 · **Status:** ACCEPTED — recorded, not yet re-measured on a real solver · **Found by:** Track 2

Removing the `.real` cast (D-024) changed a smoke result, and the change is worth
recording rather than absorbing quietly.

| | `s_free` R=2 | `s_free` R=10 | `s_residual` |
|---|---|---|---|
| before (with `.real`) | 0.0827 | 0.0448 | 0.082 / 0.032 |
| after (complex) | 0.0326 | 0.0377 | **0.000000** |

Before the fix, the artefact `s_residual` dominated the combined score *and was larger
at the near operator*, so the score fell as acceleration rose. A threshold calibrated at
R=2 therefore under-covered at R=10, which is what `test_naive_transfer_loses_coverage_
under_operator_shift` asserted. With the projection kept complex, `s_free` rises with
acceleration — the physically correct direction, since a larger null space holds more
free content — and naive transfer now **over**-covers:

| calibration | deployment | coverage | gap |
|---|---|---|---|
| R=2 | R=2 | 0.947 | −0.047 |
| R=2 | R=10 | **1.000** | −0.100 |
| Mondrian | R=10 | 0.914 | −0.014 |

**Over-coverage here is not the safe side.** Coverage 1.0 means the transferred
threshold flags *every* region, so the flag carries no information at all — the same
vacuity D-020 records for the absolute label threshold, reached by a different route.
The guarantee promises the nominal level, not "at least".

What survives unchanged: naive transfer is materially miscalibrated under operator
shift, and the group-conditional backbone restores it (0.914 against a nominal 0.90).
The test now asserts the magnitude of the gap and the saturation, not a direction the
synthetic sampler does not produce.

**Scope, honestly.** This is the D-015/D-018 reference sampler, and nothing from it is
reportable. The direction under a *real* pretrained solver is unknown and must be
measured in Week 4, not inferred from here. What this does establish is that the
previously recorded direction was an artefact, so the README smoke table and the D-020
coverage figures should not be cited until they are regenerated.

*Affects:* D-018, D-020, D-024, README smoke table, T2.2, Week 4.

---

## D-029 · Solver algorithms are separated from the network that drives them
**Date:** Week 2 · **Status:** ACCEPTED · **Proposed by:** Track 2

The three back-ends (DPS, DDNM, FlowDPS) are implemented against a small `Denoiser`
interface — `predict_noise(x, t)` — rather than against `diffusers` directly. A
`DiffusersUNetDenoiser` adapter loads the real checkpoint; an `AnalyticDenoiser` provides
a closed-form stand-in.

*Alternative:* call `diffusers` inside each sampler, which is shorter.

*Reason:* three things become possible that otherwise are not. The sampling algorithms
are unit-testable in **1.6 s on CPU** with no checkpoint and no GPU — which matters
because the torch build on this machine is CPU-only and one real 100-step draw at 128²
costs about a minute, so a test suite that needed the real prior would not be run.
Swapping the prior (D-016: natural-image now, medical later) touches one adapter rather
than three samplers. And the determinism `solvers/base.py` promises can be *asserted*
bitwise, because a deterministic test double makes the comparison meaningful.

*Cost:* one more indirection, and the analytic denoiser is not an image prior — it tests
arithmetic and contracts, never reconstruction quality. Quality still needs the real
checkpoint and a human looking at pictures.

*Affects:* T2.2, T2.3, Week 10 ablation, D-016.

---

## D-030 · The flow-matching family uses diffusion-derived velocities, not a FlowDPS checkpoint
**Date:** Week 2 · **Status:** ACCEPTED — scope limitation, must appear in the write-up · **Proposed by:** Track 2

The Week 2 card asks for a second solver family "to establish solver-agnosticism early".
No public flow-matching checkpoint is available here and the project trains nothing
(Sec. 17). What is implemented is FlowDPS's **sampling dynamics** — a deterministic
rectified-flow ODE with per-step range replacement — driven by a velocity field derived
from the available diffusion checkpoint via the standard probability-flow correspondence
(matched SNR gives, for each flow time `s`, an index `t` and scale `c` with `x_t = c·x_s`;
`v = eps − x0`).

*What this is sufficient for:* the Week 2 deliverable and the Week 10 solver ablation.
Both need two families whose **dynamics genuinely differ** behind one interface —
stochastic ancestral sampling with per-step noise against deterministic ODE integration —
because that is the axis the ablation probes. Their posterior spread arises by different
mechanisms: FlowDPS's spread comes *entirely* from its initial noise.

*What it is not sufficient for:* claiming the project reproduced FlowDPS [26]. It did
not. The manuscript must say "flow-matching sampling dynamics on a diffusion-derived
velocity field", not "FlowDPS".

*Alternative rejected:* leaving the second family stubbed until a checkpoint appears —
which would push the solver-agnosticism check to Week 10, exactly what the card warns
against.

*Searched, Week 2 — the gap is real, not unexamined.* The reference implementation is
<https://github.com/FlowDPS-Inverse/FlowDPS>, which runs on **Stable Diffusion 3 / Flux**
latent flow-matching backbones. Neither is viable here on two independent counts: each is
well over 10 GB against 16 GB free on the working disk (P-14), and on CPU-only torch at
~0.6 s per 128² UNet forward (P-16) a backbone of that size is minutes per step, not per
draw. The Hub was also searched for a small unconditional flow-matching image prior that
would fit. The only two candidates are `Dinghuai/flow-matching-cifar10` (38.4 M params,
but **32x32** — it cannot reconstruct the 128²/256² slices the pipeline runs at) and
`sunnycloudhust/CelebA-flow-matching` (no model card, no parameter count, a bespoke
non-`diffusers` format, 12 downloads, no paper). Adopting the latter would trade a
documented approximation for an undocumented checkpoint of unknown training — worse
science, not better. **The deferral stands, and now stands on evidence.**

*How it closes:* a GPU with the disk to hold SD3, i.e. the same hardware decision P-16
forces for Weeks 3-4. If that machine appears, running true FlowDPS is a config change,
because the interface it would plug into is already built and tested.

*Affects:* T2.2, T2.5, Week 10 ablation, O7 manuscript wording, P-14, P-16.

---

## D-031 · Guidance strength is a fixed solver constant, never tuned per operator
**Date:** Week 2 · **Status:** ACCEPTED · **Proposed by:** Track 2

DPS's guidance step size is set once (`DEFAULT_STEP_SIZE = 5.0`, measured) and held fixed
across every operator in every curve.

*Reason:* Sec. 13 requires the acquisition operator to be the only quantity that varies
within a coverage curve. Re-tuning guidance strength at each acceleration would make
reconstruction quality depend on the operator through **two** paths at once, and the
coverage gap could no longer be attributed to the acquisition — the same class of
confound as D-028, where an artefact that varied with the operator masqueraded as a
finding.

*The value, measured* on the analytic denoiser at 32², R=4, 50 steps — data-consistency
error against step size:

| step size | 0.5 | 2 | **5** | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| dc error | 0.94 | 0.60 | **0.053** | 0.16 | 0.31 | 0.76 |

A clean minimum: too small and the soft constraint never bites, too large and it
overshoots the prior.

*Cost:* the value is **not transferable between priors** and must be re-measured when
medical weights land (D-016). Re-measuring is legitimate; re-measuring *per operator* is
not.

*Affects:* T2.2, T1.4 protocol, Weeks 4 and 10, D-028.

**Re-measured on the real UNet, Week 3 — the value 5.0 does not hold for this prior.** The
table above came from the analytic denoiser. On `ddpm-celebahq-256` (R=4, 128², 24 steps,
two items × two draws):

| ζ | 0.5 | 1 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| null share of error | 0.627 | 0.707 | 0.756 | 0.817 | 0.872 | 0.931 |
| data-consistency error | 0.454 | 0.363 | 0.305 | 0.238 | 0.182 | 0.126 |

No minimum in the range: the fit is still improving at 20, so the optimum is above it and
**unbracketed**. The analytic denoiser was never the prior, which is exactly the case this
entry's *Cost* line anticipated. `DEFAULT_STEP_SIZE` is left at 5.0 until the optimum is
bracketed — jointly with the step count, since strong guidance is what made DPS diverge at 8
steps (D-033 update) — and on more than one operator's worth of items. The decision itself is
unchanged: one value, fixed across every operator in a curve. Only the number is open.

---

## D-032 · Stage-2 decomposition diagnostics are a run artefact, not score-bank columns
**Date:** Week 2 · **Status:** ACCEPTED · **Proposed by:** Track 2

The range/null-space audit (`operators/decomposition.py`,
`scripts/stage2_decomposition.py`) writes its own JSON under `runs/stage2/`. It adds no
column to the score bank.

*Alternative:* six more columns — the energy, error and spread splits — on every bank
row. They are cheap to compute and every consumer would have them for free.

*Reason:* the schema is a cross-track contract (`score_bank_spec.md` §6), and its stated
rule is that a bank row carries **evidence about an item**: score components, error
magnitudes, provenance. These are diagnostics about *the pipeline* — whether the
projector is orthogonal, whether the solver converged — and they are per (operator,
solver), not per region. Putting them in the bank would mean every consumer joins past
six columns that are constant across every row of a run, and a schema addition that
Track 1 and Track 3 did not ask for.

*Cost:* an analysis that wants both has to read two files. Accepted: the audit is a
pre-flight check, and pre-flight checks that live inside the artefact they are checking
are worth less.

*Affects:* T2.2, T2.3, `score_bank_spec.md`, D-021.

---

## D-033 · The null-error share is a per-solver step-budget precondition on a bank
**Date:** Week 2 · **Status:** ACCEPTED · **Proposed by:** Track 2

Before a score bank is produced with a given (solver, step count), the Stage-2 audit must
show reconstruction error concentrated in the **null** component —
`null_error_share > 0.5`, and in practice far above it. A solver that fails this is not
producing hallucinations to be scored; it is producing an unconverged fit.

*Reason:* `s_free` and `err_free` both score the free component on the premise that what
a reconstruction gets wrong is what the operator could not see. That premise is a
*property of the solver's convergence*, not an identity, and until this week nothing
checked it. Measured (analytic denoiser, 32², R=4, null share of the error):

| DPS steps | 8 | 24 | **48** | 100 | 200 |
|---|---|---|---|---|---|
| null-error share | 0.20 | 0.50 | **0.99** | 0.99 | 0.99 |
| data-consistency error | 0.71 | 0.36 | **0.038** | 0.039 | 0.036 |

At 8 steps DPS's error is **range-dominated**: four fifths of it sits in the component
the measurements pin down. Scoring the free component of that output measures the step
count, not the prior's invention. DDNM and FlowDPS replace the range component
structurally and are at 0.983 and 0.968 by 8 steps, so the requirement binds per family —
which is the point. It is a *per-solver* precondition, not a global constant.

*Alternative:* gate it inside the runner and refuse to write the bank. Rejected for now:
the check needs ground truth, which is available under retrospective undersampling but
not in general, and a hard gate on a threshold measured on one denoiser at one image size
is more confidence than the number has earned.

*Cost:* DPS needs roughly 6× DDNM's step budget to be admissible, which on a CPU-only
machine (P-16) is most of the reason the Week 4 grid will run DDNM.

*Affects:* T2.2, T2.3, T1.4 protocol, Week 4 grid sizing, D-031, P-16.

**Re-measured on the real UNet, Week 3.** R=4, two items × two draws, at ζ=5.0:

| solver | resolution | steps | null share of error | data-consistency error |
|---|---|---|---|---|
| DDNM | 256² | 8 / 24 | 0.981 / 0.990 | 0.116 / 0.067 |
| DDNM | 128² | 8 / 24 | 0.969 / 0.986 | 0.118 / 0.068 |
| DPS | 128² | 8 | 0.003 | **116.7 — diverged** |
| DPS | 128² | 24 / 48 | 0.817 / 0.852 | 0.240 / 0.196 |

The gate's *direction* survives and DDNM's admissibility at 8 steps is confirmed on the real
prior, at both resolutions. DPS behaves worse than the stand-in predicted in one respect and
better in another: at 8 steps it **diverges** rather than merely under-converging, but from 24
steps it clears the gate — admissible at half the analytic denoiser's ~48 — and its remaining
gap to DDNM is mostly D-031's mis-set step size rather than the step count. DPS was measured at
128² only: at 256² it could not run on this machine at all (P-24).

---

## D-034 · The relative criterion is implemented and its tau curve measured; tau stays Track 1's
**Date:** Week 2 · **Status:** ACCEPTED — Track 2's half discharged, awaiting Track 1's tau
**Proposed by:** Track 2

`label_hallucinated(..., relative=True)` implements the criterion form agreed at the Week 1
sign-off (A1 / D21): `err_free / free_content > tau`. `scripts/criterion_sweep.py` reports
the positive rate against tau from a cached bank. **Track 2 does not pick tau.** The
criterion is O2 and belongs to Track 1; this supplies the curve they said they needed.

*Why now:* the sign-off deferred tau explicitly — "tau is set only after a real prior
lands (T2.2), and its positive rate is then reported, not targeted". A real prior has now
landed, so the deferral has expired. Until this week the column existed (`free_content`,
added Week 1) and the form was agreed, but nothing computed it — the only implemented
criterion was still the absolute one that P-02 shows is unusable.

*Measured*, on the 384-region DDNM / `ddpm-celebahq-256` / M4Raw bank:

| | |
|---|---|
| v0 absolute, `err_free > 0.10` | flags **100.0%** of regions — P-02, on the real prior |
| ratio `err_free / free_content` | min 1.057 · median 1.558 · max 2.759 |

| tau | 1.00 | 1.25 | **1.50** | **1.75** | **2.00** | 2.50 | 3.00 |
|---|---|---|---|---|---|---|---|
| positive rate | 100% | 92.4% | 60.2% | **25.5%** | **10.7%** | 0.8% | 0% |

The relative form **does** separate, where the absolute one does not: the rate moves
smoothly across 1.25–2.5 instead of pinning at 100%. A usable operating range is roughly
tau 1.75–2.0.

*Two cautions that belong with the number.* The ratio exceeds 1 in every region — the
reconstruction always puts more into the free component than the truth had there — which
is what a face prior on brain anatomy should do, and means tau is measuring a badly
mismatched prior as much as a criterion. And the median ratio rises with acceleration
(1.480 at R=4, 1.645 at R=8), so a single global tau is not operator-neutral; if that
trend survives a medical prior, tau has to be set per calibration group or the criterion
becomes a proxy for the operator.

*Cost:* the numbers come from a bank that is **not reportable** (D-016, D-015), so tau
read off it is provisional and must be re-measured when medical weights land.

*Affects:* O2, T1.1, T2.3, P-02, D-020, D21, Weeks 3-4.

---

### Track 1 ruling on tau — **RESOLVED**, full entry D37

**tau = 1.75 provisionally, global across operators, selected on the design split.**

**Global, not per-group — the opposite of this entry's suggested fallback.** Coverage
is conditional, `P(flag | hallucinated)`, so a base rate that varies across operators
does not by itself bias it. With a global tau, "hallucinated" means the same thing at
every operator and the positive rate is free to rise with acceleration — which it
*should*, since a larger null space permits more invention (D2, and §12.2's L2). With a
per-group tau tuned to equalise rates, "hallucinated" means something different at each
operator, and the headline coverage-vs-distance curve would compare incommensurable
quantities. That is the confound worth avoiding. The rising positive rate is a result to
report, not an artefact to normalise away.

**1.75 over 2.0 on calibration arithmetic.** D32 needs 9 positives per group to certify
at alpha=0.1. On the 384-region bank, 1.75 gives ~98 positives (~16 per group over ~6
groups); 2.0 gives ~41 (~7 per group) — below the floor, which would make Mondrian
unreportable everywhere. The label threshold and the calibration floor are coupled.

**Selected on the design split and frozen**, same rule as D28's weights: choosing tau on
the data you then calibrate against couples the label to the threshold.

**Provisional, and the caution is accepted in full** — a face prior on brain anatomy
(D-016, D-035). Re-measure with medical weights, against this pass/fail condition: with a
matched prior, *some* regions should show ratio < 1. If the ratio still exceeds 1
everywhere, the fault is not the prior and either the criterion or `free_content` needs
re-examining.

---

## D-035 · The prior's training domain is a third reportability gate
**Date:** Week 3 · **Status:** ACCEPTED · **Proposed by:** Track 2

A bank is reportable only if **all three** hold: the substrate is real, the sampler is a
generative prior rather than a reference, **and the prior was trained on the modality it
is reconstructing**. The third gate is new. `solvers/denoiser.py` maps checkpoint
basenames to domains, `ADMISSIBLE_DOMAINS` says which domain each modality accepts, and
the runner records `prior_domain` in every manifest.

*Reason:* D-018 gated on the first two and they are **not sufficient**. fastMRI is a real
reportable substrate (D-017) and `ddpm-celebahq-256` is a real generative prior, so the
first fastMRI run came out stamped `reportable: true` — a face model reconstructing knees,
which D-016 orders deliberately and which the T2.2 write-up lists as a reason nothing is
reportable. The code did not know what the prose knew. That is exactly the P-12 failure
D-018 was written to prevent, one level up: a caveat that survives only in whoever
remembers which checkpoint was configured.

*Fails closed.* An unregistered checkpoint is `"unknown"` and admissible nowhere. An
unknown checkpoint could be anything, and the cost of wrongly stamping a bank reportable
is a claim in a manuscript the data does not support. Making a prior reportable is then
one deliberate line in `PRIOR_DOMAINS` — visible in a diff, which is the point.

*Alternative rejected:* a config flag such as `reportable: false`. Rejected for the reason
D-018 gives — reportability is a property of the bank, derived from what actually
produced it, not an assertion someone remembered to write down.

*Cost:* the registry must be maintained. A new checkpoint is unreportable until someone
adds it, which will occasionally be briefly annoying and is the correct direction to fail.

*Affects:* D-018, D-016, P-12, P-18, T2.3, O7, every bank produced from Week 3 on.
## D-036 · A bare `data/` in .gitignore also ignored `src/caliber/data/`
**Date:** Week 2 · **Status:** FIXED · **Cause:** Track 1 · **Found by:** Track 1 (T1.5)

**Root cause of the two broken pushes.** `.gitignore` carried a bare `data/`.
A gitignore pattern with no slash before its final character matches at **any
depth**, so it ignored the *source package* `src/caliber/data/` as well as the
dataset directory. Files already tracked stayed tracked, so nothing looked
wrong — but every **new** file under `src/caliber/data/` was invisible to
`git add`.

That is exactly what happened to `src/caliber/data/splits.py`. It was never
committed, while `data/__init__.py`, `data/fastmri.py` and `tests/test_splits.py`
— all committed — import it. Because `caliber/__init__.py` imports the data
package eagerly, one silently-ignored file took down **every module in the
package**: operators, scoring, calibration, scorebank, runner. Two consecutive
pushes landed on a tree that could not be imported at all.

**Introduced by Track 1** in the D-025 consolidation, replacing
`data/*` + `!data/README.md` with `data/`. The original `data/*` contains an
internal slash and so was anchored to the repository root; the replacement was
not. The change looked like a simplification and was a scope widening.

**Fix:** `/data/` — the leading slash anchors it to the root. Verified both
ways: `src/caliber/data/splits.py` is no longer ignored, and `data/raw` still is.
A comment above the pattern now says the slash is load-bearing, since this is
precisely the kind of line that gets "tidied" back.

**Lesson worth keeping:** an ignore rule that is too broad fails *silently and
asymmetrically* — existing files keep working, new ones vanish — so it presents
as someone else forgetting to add a file rather than as a config error.

*Affects:* every module, T2.1, T2.2, D-025.

---

## D-037 · The bank's three free-component columns satisfy an exact identity
**Date:** Week 3 · **Status:** ACCEPTED · **Proposed by:** Track 1 (T1.1, Q6)

`runner.py` passes `components.ref_intensity` into `compute_region_truth`, so
`s_free`, `free_content` and `err_free` are three reductions of the same two
fields under one normalisation. Therefore, per region and exactly:

```
err_free² = s_free² + free_content² − 2 · s_free · free_content · cos θ
```

where `cos θ` is the angle between the reconstruction's hidden content and the
truth's. Verified at **0.00e+00 median and max relative residual** over the
4,608 regions of a `w01_smoke_phantom` bank. `caliber.scoring.attribution`
implements it; `scripts/q6_attribution.py` checks it before reading anything
else and refuses to continue if it fails.

**This is now a cheap integrity check on any bank, and it is worth running.** A
non-zero residual means the score components and the truth columns were scaled
by different reference intensities — a defect that is otherwise invisible, since
every column looks individually reasonable while every *ratio* between them is
in mixed units. It is also a constraint on future changes: the shared
`ref_intensity` is load-bearing and must not be refactored apart.

**Two results that matter outside Track 1:**

- *For Track 3 (T3.4):* `cos θ` has median **0.000** on the reference-sampler
  bank, which independently confirms §2's analytic finding that the placeholder
  harness's null-space fill is injected white noise carrying no image content.
  T3.4 derived it from the code; this measures it from the bank, and the two
  agree. The same column is the natural test of T3.4's prediction that a real
  learned prior will *stop* behaving this way — `cos θ` moving away from zero is
  precisely "the prior is inventing structure related to the image".
- *For Track 2 (D-034):* see D-038.

*Cost:* none — it is a property of what the runner already writes.
*Affects:* O2, T1.1, T2.3, T3.4, D-034, D21.

---

## D-038 · "ratio > 1 everywhere" says nothing about prior mismatch; and two asks
**Date:** Week 3 · **Status:** ACCEPTED (correction) · **OPEN** (asks) · **Proposed by:** Track 1

D-034 reported `err_free / free_content > 1` in every region of the M4Raw bank
and read it as "what a face prior on brain anatomy should do". Track 1's D37
repeated that reading and built a pass/fail check on it. **Both were unsound.**
By D-037's identity, with `cos θ = 0` the ratio is `sqrt(1 + u²)` for
`u = s_free/free_content`, which exceeds 1 for *any* `u > 0`. In general
`ratio > 1 ⟺ cos θ < u/2`. So the observation is what **any** prior whose hidden
content is uncorrelated with the truth's produces, matched or not — the
reference sampler, which is not a prior at all, does it too.

The check D-034 and D37 wanted is kept, restated on the quantity it is really
about: **with medical weights, some regions should show `cos θ ≥ u/2`**, and it
is only diagnostic where `u < 2`, since an over-filling prior cannot produce
`ratio < 1` at any alignment. `content_alignment()` and `fill_ratio()` report
both per region.

A useful side effect for τ: under orthogonal invention `τ = sqrt(1 + u²)`, so
**τ = 1.75 means the prior put 1.44× as much energy into the null space as the
truth had there**. Applied to D-034's quantiles, the celebahq prior's fill ratio
on M4Raw runs 0.34 – 2.57, median ≈ 1.20.

**Ask (a) — share the `w02_solver_smoke` bank.** It is the only real-prior output
in the project and exists only on Track 2's machine. Spec §3 states the intent:
*a bank is self-describing so another track can work from a copied directory*.
This is the first time that has been needed. 384 rows — tens of kilobytes.

**Ask (b) — one nullable column `s_range` = `‖P_R x̂‖` per region /
`ref_intensity`.** Same mechanics as A1: additive, nullable, backwards
compatible, no `SCHEMA_VERSION` bump (spec §6), one line beside the existing
`s_free` in `compute_score_components`. It is the deployment-available
denominator — the region's *measured* content, no ground truth — that D-037
shows the score needs, and without it the fix cannot be tested against anything
real.

*Cost:* (a) none; (b) one float32 column per row.
*Affects:* O2, T1.1, T1.4, T2.3, D-034, D21, D37.

---

## D-039 · Track 2's `splits.py` is canonical, and the split mapping itself is pinned
**Date:** Week 3 · **Status:** ACCEPTED — Track 1 to confirm no bank was built on the stand-in
**Proposed by:** Track 2

Two implementations of `caliber/data/splits.py` met when `main` was merged into `T2.3`.
Track 2's — which built the Week 3 bank — was kept. Track 1's stand-in, written while the
file was missing from `main` (D-036), was not. The *mapping* itself, which side each
patient lands on, is now a tested contract (`tests/test_split_golden.py`), not only its
properties.

*Why this is a decision rather than a merge detail:* the two files share a salt and both
pass all 21 split tests and the full suite, yet place **48 of the 96** real fastMRI
patients on opposite sides (P-22). Picking one is picking which banks stay comparable,
and D-022 requires the split to be identical across every run.

*Reason for Track 2's:* the stand-in's own docstring says to take Track 2's file when it
lands, and the 23,040-row Week 3 bank — 7.2 hours of CPU — was assigned under it. Nothing
in the repository shows a bank built under the stand-in.

*Cost, and the open half:* any bank Track 1 or Track 3 built with `split: auto` on `main`
between PR #21 and the Week 3 merge used the other assignment and is not comparable with
banks before or after. Run outputs are not committed, so git cannot show whether one
exists. **Track 1 to confirm.**

*How it changes deliberately:* bump `DEFAULT_SALT` and regenerate the pinned values in the
same commit, recorded here. Anything else that moves the mapping fails the golden test.

*Affects:* D-022, D-027, D-036, P-20, P-22, T2.3, every `split: auto` bank.

---

## D-040 · An interrupted run leaves a partial bank that readers refuse by default
**Date:** Week 3 · **Status:** ACCEPTED · **Proposed by:** Track 2

With `run.checkpoint_every: N`, the runner rewrites every finished row to
`scores.partial.parquet` and `manifest.partial.json` each N items, atomically. A finished
run writes the normal bank, marks its manifest `"complete": true`, and only then removes
the partial files. `read_bank` raises on a directory holding only a checkpoint unless
called with `allow_partial=True`.

*Reason:* P-24. At 256² this machine can kill the solver with a silent segfault, and the
Week 3 bank took 7.2 hours. The runner held every row in memory and wrote once at the
end, so a crash late in a run lost all of it, with no traceback and nothing on disk.

*Why refuse partial banks rather than just read them:* a partial bank is a perfectly
valid table. Nothing in its rows says the run stopped at item 40 of 90, and an analysis
over it would be quietly computed on a different — and non-randomly truncated — sample
from the one the config describes. That is the silent difference D-018 exists to prevent,
so recovering rows has to be a deliberate act.

*Why rewrite rather than append:* at this project's sizes a full rewrite is well under a
second, and it keeps the checkpoint a single complete table at every moment instead of a
set of fragments that a crash could leave inconsistent.

*Compatibility:* `complete` is a new manifest key. Every bank written before it lacks the
key and is read as complete, so no existing bank changes meaning. The schema of
`scores.parquet` is untouched.

*Resume:* added by D-042 — a restarted run of the same config now continues from the
checkpoint instead of recomputing from the first item.

*Alternative rejected:* setting a default `checkpoint_every` in `base.yaml`. It would
change the config hash, and therefore the run id, of every existing bank.

*Affects:* P-24, P-16, T2.3, `score_bank_spec.md`, every consumer of `read_bank`.

---

## D-041 · Item seeds come from a stable digest, not Python's salted `hash()`
**Date:** Week 3 · **Status:** ACCEPTED · **Proposed by:** Track 2

`caliber.runner.item_seed(base_seed, sample_id, slice_idx, operator_id)` returns
`base_seed + BLAKE2b("sample_id|slice_idx|operator_id") mod 10**6`. The runner and the
Stage-2 audit both call it; nothing derives a seed from `hash()` any more.

*Reason:* P-25. `hash()` on strings is salted per process, so the previous derivation
gave a different seed for the same item on every run, and no bank could be regenerated
from its config. A digest is the same on every process, platform and Python version.

*Why not set `PYTHONHASHSEED` instead:* it has to be in the environment before the
interpreter starts, so a script, a notebook, an IDE runner or a teammate's shell each
has to remember it — and forgetting is silent. A derivation that does not depend on
the environment cannot be forgotten.

*The separator is load-bearing:* without it, `("a1", 0)` and `("a", 10)` style inputs
could collide. Tested.

*Cost:* every bank produced from here on draws differently from every bank before it,
for the same config. That costs nothing that was not already lost — no earlier bank was
reproducible even against itself — and existing banks keep their recorded per-row seeds.
Changing the derivation again requires changing the pinned values in
`tests/test_reproducibility.py`, deliberately, and an entry here.

*Affects:* P-25, O7 (reproducibility package), T2.3, every bank, `stage2_decomposition.py`.

---

## D-042 · A restarted run resumes from its checkpoint — only for the same config
**Date:** Week 3 · **Status:** ACCEPTED · **Proposed by:** Track 2

When the runner starts and finds a checkpoint (D-040) whose config hash matches its own,
it loads the finished rows and skips those (item, operator) pairs. A checkpoint from a
different config makes the run refuse to start.

*Reason:* D-040 made a crash cost the tail of a run instead of all of it, but a restart
still recomputed everything. At 256² on this CPU that is hours per restart, and P-24 shows
restarts are not hypothetical.

*Why it is safe now and was not before:* a resumed bank is only correct if the pairs
computed after the restart are exactly what an uninterrupted run would have drawn. Until
D-041 each process drew different seeds, so a resumed bank would have been a splice of two
different random streams — valid-looking and not reproducible. With digest seeds it is
exact. Tested directly: crash a run part-way, restart it, and require the resulting bank to
equal one from a single uninterrupted run, and require the restart to call the sampler only
for the pairs the checkpoint lacked. That test fails against the previous runner.

*Why refuse a mismatched checkpoint rather than overwrite or merge it:* merging would put
two runs' rows in one bank with nothing to tell them apart; overwriting silently discards
someone's finished rows. Refusing costs one manual `rm` and cannot lose or corrupt data.
The run directory is named by config hash, so a mismatch in practice means `run.id` was
overridden.

*Automatic, not opt-in:* the alternative — recomputing and overwriting unless a flag says
otherwise — makes the destructive path the default.

*Affects:* D-040, D-041, P-24, P-16, T2.3, T2.4 onward.

---

## D-043 · A run that cannot fit in memory is refused before it starts
**Date:** Week 3 · **Status:** ACCEPTED · **Proposed by:** Track 2

`caliber.resources` estimates the peak commit charge of a run from its image shape and
draw count, compares it with what the machine has, and the runner **raises rather than
starts** when the estimate exceeds availability. `run.skip_memory_check: true` overrides.

*Reason:* P-24's third mitigation. Checkpointing (D-040) and resume (D-042) make a
memory death survivable; this makes the common case not happen. The failure it prevents is
particularly bad because it is often silent — a segfault with no traceback, hours in.

*The estimate is measured, not derived*, on this machine at `torch 2.13.0+cpu`:

    peak commit GB  ≈  max(2.8, 2.0 + 6.1e-5 × pixels × draws)

It reproduces eleven measured points within 0.4 GB, including both configurations that
actually failed (128²×8 segfaulted with 7.6 GB free; 256²×4 raised out-of-memory). A
backward pass (DPS) carries a flat +0.4 GB, measured at one size only.

*Refuse, not warn.* A warning at the start of a seven-hour job is read once and then
scrolls away, and the thing it warns about produces no error when it happens. Within 20%
of the limit the run proceeds with a warning, since that case depends on what other
applications do next.

*Overridable, deliberately:* the model is a fit to one machine, one checkpoint and one
torch build. On different hardware it will be wrong, and being wrong must cost a flag
rather than a week. `available_commit_gb()` returns `None` off Windows and the verdict is
then `"unknown"`, which never blocks — a check that cannot read the memory must not stop a
run.

*Consequence worth stating:* `w03_mri_fixed_operator.yaml` as written — 256², four draws,
~18 GB — is now **refused on this machine**, which is correct: it is the configuration
that raises out-of-memory today. The Week 3 bank exists because more memory was free when
it ran.

*Affects:* P-24, P-16, D-040, D-042, T2.3, every config's runnability.

---

## D-044 · Each posterior draw is seeded independently, so draws can be batched in any size
**Date:** Week 3 · **Status:** ACCEPTED · **Proposed by:** Track 2

Draw *i* of a sample is a function of ``(seed, i)`` alone —
``solvers/base.draw_seed`` — not of the batch it was computed in. Every pretrained
back-end takes ``max_draws_per_batch``: the draws are computed in batches of that size
and concatenated, and the result is **bit-identical** to computing them in one batch.

*Reason:* P-24. Peak memory is set by how many draws go through the UNet at once, and the
samplers put all of them through together, so `w03_mri_fixed_operator.yaml` (256², four
draws, ~18 GB) could not run on this machine at all. Chunking is the obvious fix and was
blocked by exactly one thing: draws shared a generator, so the noise a draw received
depended on how many others were requested alongside it. Batch size was a property of the
result.

*Measured:* the Week 3 config at 256², four draws, chunked to one — **6.0 GB instead of
~18 GB**, and it runs. The bank it produces is the bank a single batch would have
produced.

*Cost:* the draws differ from those of any bank produced before this entry, because the
noise each draw receives is now derived differently. That costs nothing that was not
already lost: no bank before D-041 was reproducible at all, and D-041 had already reset
the draws this week. Doing both resets in one week is deliberate.

*Not changed:* no committed config sets `max_draws_per_batch`, for the reason D-040 gives
about `checkpoint_every` — it would alter the config hash, and therefore the run id, of a
bank that already exists. Pass `--set experiment.solver.max_draws_per_batch=1` on a run
that needs it.

*Speed:* a smaller batch gives up some of the CPU's parallelism across draws. On this
machine that is a modest cost against not being able to run at all.

*Affects:* P-24, P-16, D-041, D-043, T2.3, Week 4 grid sizing.

---

## D-045 · Both of D-038's asks are answered: `s_range` shipped, and open-substrate banks are committed
**Date:** Week 3 · **Status:** ACCEPTED · **Proposed by:** Track 2 · **Answers:** D-038

**Ask (b), `s_range`.** `compute_score_components` now also returns
`‖P_R x̂‖` per region / `ref_intensity`, written as a nullable `s_range` column by the
same additive mechanism as A1 — no `SCHEMA_VERSION` bump, and every existing bank still
validates (spec §6). It is the range half of the split `s_free` takes the null half of,
under the same reduction and the same `ref`, so the two are directly comparable, which is
what D-037's identity needs. It is deliberately **not** part of `ScoreComponents.stack()`
and so cannot enter the score: it is a denominator Track 1 may use, not a fourth
component (D25). Tested for the property it exists for — that it needs no ground truth.

**Ask (a), the bank.** `banks/w02_solver_smoke/` holds the 384-row Week 2 bank: the
manifest, the operator registry and `scores.parquet`, 35 KB. `runs/` stays gitignored;
this is a deliberate exception with a rule, written down in `banks/README.md`:

* **only open substrates** — phantom and M4Raw. A bank is per-region scalars, not images,
  and M4Raw is CC-BY, so derived numbers may be redistributed with attribution.
* **never fastMRI-derived.** The NYU agreement forbids redistribution and whether that
  reaches a derived score bank is unanswered — T2.1 §7 flags it for confirmation before
  the O7 release. Until then the Week 3 fastMRI bank stays local and is copied by hand.
* **never `recons/`.** Cached posterior samples are images, far larger than the table, and
  the release rule is code, configs and score banks — never images.

*Why commit one at all:* spec §3 says a bank is self-describing so another track can work
from a copied directory. This is the first time that has been needed, and a file that only
exists on one laptop is not a shared interface. P-24 is the argument in miniature: that
laptop cannot currently reproduce this bank.

*Cost:* one float32 column per row, and 35 KB in the repository.

*Affects:* D-038, D-037, O2, T1.1, T2.3, `score_bank_spec.md`, O7 release scope.

---

## D-046 · Track 3 is unblocked on L: the weights were never needed, and the placeholder law is now measured against a real prior
**Date:** Week 3 · **Status:** ACCEPTED · **Proposed by:** Track 1 (T1.1) · **For:** Track 3 (T3.3, T3.4)

Three things Track 3 has been waiting on, settled. All of it reproducible with
`scripts/t1_component_lipschitz.py` and `scripts/q6_attribution.py` against
`banks/w02_solver_smoke`, which D-045 committed — no GPU, no dataset.

**1. `L` does not need the score weights.** T3.3 §3 reduces the Lipschitz assumption to
`L = Σ wᵢ Lᵢ` and has been held up on D28. D25 constrains the weights to the simplex, so
that is a **convex combination** and `L ≤ max Lᵢ` for *every* admissible weight vector.
The bound can be stated now; D28 decides only its tightness.

**Only the upper end bounds.** `min Lᵢ ≤ L` is false, and this bank shows why: components
that move in opposite directions cancel, and the combination is then less
operator-sensitive than any part of it — T3.4 §3's effect, here almost total (below).
Cancellation only helps a coverage-gap bound, so nothing is lost; but **a small combined
secant is not evidence of a small constant**, and a bound must not be read off one.

**A precondition worth knowing you depend on:** the components enter standardised,
`cᵢ/scaleᵢ`, and the reduction treats `scaleᵢ` as a constant. **D27** — scales frozen on
the calibration operator — is what makes it one. Re-estimating scales per deployment
operator makes `scaleᵢ` a function of `A`, and the per-component reduction stops being
valid. D27 was argued as protecting the coverage gap from being flattened; it is
load-bearing for T3.3 §3 as well, and should not be "simplified" by anyone.

**2. The per-component constants, measured on a prior for the first time.** R=4 → R=8,
`d_op = 0.3536` (null-space projection / ambient, D19 — and exactly `√(1/8)`, the
fraction of k-space lines whose sampled status changed, which is a clean independent
check on the implementation):

| component | signed shift (standardised) | `ℓᵢ` |
|---|---|---|
| `s_free` | +0.1738 | 0.4916 |
| `s_disagree` | +0.2021 | 0.5716 |
| `s_residual` | **−0.3878** | 1.0970 |

**These are lower bounds on `Lᵢ`, not estimates** — a mean-shift secant satisfies
`|E[c(A')] − E[c(A)]| ≤ E|c(A') − c(A)| ≤ Lᵢ d_op`, so it is a floor, useful for saying
how tight the final bound can ever be, and not a value to quote as `Lᵢ`.

Ceiling over the full simplex: **`L ≤ 1.10`**. With Track 1's D42 ruling that `s_residual`
carries no weight it **halves to `L ≤ 0.57`**, the two survivors move in the same direction
so their combination is genuinely bracketed in `[0.49, 0.57]` — 16% — and the remaining
simplex is one-dimensional.

**The cancellation, measured.** `s_residual` moves *opposite* to the other two: at equal
thirds the net standardised shift is **−0.004** against **+0.376** from `s_free` and
`s_disagree` together. This is T3.4 §3's cross-component cancellation, on a real prior for
the first time, and close to total. It is not an argument for keeping `s_residual`: the
near-zero is a small difference of large opposing terms, contingent on the weights, the
step count and a sign nothing guarantees, and it constrains the score's *mean* rather than
its distribution, which is what coverage depends on.

**3. T3.4's √null_frac law is broken by the real prior — in the direction T3.4 predicted.**
T3.4 §5 flagged this as "the most important caveat", expecting the law to fail once a
learned prior filled the null space, because that departure *is* the score's premise
showing up numerically. It fails, and the test is clean because the same three items are
reconstructed under both operators, so a paired comparison needs no model spectrum:
`free_content` on identical images gives the √null_frac scaling factor directly.

| quantity, R=8 / R=4 | value |
|---|---|
| √null_frac law, from `free_content` | 1.0816 |
| the same from the dimension count, independently | 1.0801 |
| **`s_free`, measured** | **1.1922** |
| **`s_disagree`, measured** | **1.2279** |

`s_free` grows **1.12× faster** than the law allows. The fill ratio `u = s_free/free_content`
rises in **149 of 192 paired regions** (sign test *p* = 6.9e-15) — under the law it would be
operator-independent. So §2's closed form is confirmed as the null hypothesis it was
offered as, and it is now rejected on real-prior data rather than awaited.

*What this means for T3.4:* the `√null_frac` result stands as a statement about the
placeholder harness and should be written up as such, with this as the measured departure.
The open item — deriving the cross-component correlation to turn `L` into a closed form —
is unaffected and still Track 3's, but it now has a real-prior target to hit rather than a
white-noise one.

*Still owed to Track 3 by Track 1:* T1.4's protocol, which gates T3.6. The ruling on P-21
follows separately.

*Cost:* two operators is one secant, not a curve — these are local ratios over one pair,
valid as the worst-observed-slope bound T3.3 §3 asks for over the tested range, and no
more. Week 4's grid gives the curve.

*Affects:* T3.3 §3, T3.4 §2/§5, T1.1, D25, D27, D28, D-033, D-045, L1, L2.

---

## D-047 · Config overrides are accepted silently, including typos and paths that do not exist
**Date:** Week 3 · **Status:** OPEN · **Found by:** Track 1 (T1.1) · **For:** Track 2

`load_config` accepts any `key=value` override without checking that the key resolves.
All three of these return an unchanged config and print nothing:

```
nonsense.key.that.does.not.exist=99      -> accepted silently
run.limitt=5                             -> accepted silently   (a typo for run.limit)
experiment.operators[0].acceleration=[4] -> accepted silently   (list indexing unsupported)
```

**Why it matters more than it looks.** Overrides are how every experiment is varied from a
base config, and a silently-ignored one means the run that executes is not the run that was
asked for — while the config hash, the manifest and the provenance all agree with each
other, because they record what ran. There is no trace to find afterwards. A mistyped
`run.limitt` costs 7 hours of reconstruction on the wrong item count (T2.3 §3) and looks
like a correct run.

This is P-19's shape exactly: no exception, no wrong answer, just a step that quietly did
not happen. It is also already live in the test suite — `tests/test_analyse_bank.py` and
`tests/test_checkpointing.py` pass overrides such as `run.skip_memory_check=true`, and a
typo in one of those would silently exercise different behaviour than the test name claims.

Found while writing a one-operator bank for `tests/test_component_lipschitz.py`: the
obvious way to build one is an override, and it produced a three-operator bank instead. The
test now filters an existing bank rather than configuring a smaller run, which works but
should not have been necessary.

*Suggested fix, Track 2's call:* raise on an override whose key path does not exist in the
resolved config. Adding list indexing is a separate and optional question — refusing
unknown keys is the part that prevents silent wrong runs.

*Affects:* every experiment config, T2.3, D-042 (restart-on-same-config), the test suite.

**Resolved by D-048.**

---

## D-048 · An override must name something the pipeline reads, or it is refused
**Date:** Week 4 · **Status:** ACCEPTED · **Proposed by:** Track 2 · **Answers:** D-047

`load_config` now checks every override key before the run starts. A key is accepted if:

1. it already exists in the resolved config; or
2. it is one of the runner's **declared optional keys** — `run.checkpoint_every`,
   `run.id`, `run.skip_memory_check`, `experiment.n_posterior` — which the runner reads but
   which are deliberately absent from `base.yaml`; or
3. it is a parameter the **configured** solver or dataset actually takes, checked against
   the class's constructor, after every override has been applied.

Anything else raises, naming the nearest real keys. List indexing (`operators[0]`) raises
with a pointer to overriding the whole list.

*Why not Track 1's simpler suggestion,* "raise if the key is not in the resolved config":
it would have refused three of Track 2's own legitimate overrides. `run.checkpoint_every`
and `run.skip_memory_check` are not in `base.yaml` on purpose — a default there would change
the config hash, and so the run id, of every existing bank (D-040) — and solver parameters
such as `max_draws_per_batch` are valid for one solver and not another. The check has to
know what is *valid*, not merely what is *present*.

*Two details that make it hold:*

* `DPSSampler` and `DDNMSampler` take `**kwargs` and forward them, so their own signatures
  understate what they accept. Parameters are collected across the class hierarchy.
* A catch-all `**_` — the reference sampler has one — is **not** read as "accepts
  anything". That is exactly how a typo slips through one level further down.

*A case D-047 did not list:* `experiment.solver.step_size=20` on a DDNM config. A real key,
for the wrong solver — DDNM has no guidance step. It used to be accepted and do nothing; it
is now refused, while the same key after `experiment.solver.name=dps` is accepted.

*Measured before the change:* the old loader accepted all of D-047's examples, and
`--set run.limitt=5` left `run.limit` at **30** — a seven-hour run of the wrong size.

*Verified not to break anything:* all 300 existing tests pass, including every override
they use; and all 10 distinct `--set` keys documented across `Documentation/`, `configs/`,
`scripts/` and the READMEs are still accepted.

*Guard against drift:* a test reads `runner.py`'s source for `run_cfg.get(...)` and
requires every such key to be declared, so a new optional key cannot be added to the runner
without being added here.

*Affects:* D-047, D-040, every experiment config, the test suite.

---

## D-049 · P-23 answered: the headline estimand is paired, and the crossed design does not divide the patient budget
**Date:** Week 4 · **Status:** ACCEPTED · **Proposed by:** Track 1 (T1.4) · **Answers:** P-23

P-23 concluded that fastMRI `singlecoil_val` cannot support a per-operator coverage claim at
any grid size, listed four options and left the choice to Track 1. **The measurement is
right, the conclusion is right about per-operator coverage, and it is the estimand that was
wrong.** Full working in `T1.4_experimental_design.md`; Track 1's entry is D45.

**Two corrections to the Week 4 sizing table, one of them pure arithmetic.**

**(a) The design is crossed, so the budget is not divided.** Retrospective undersampling
applies every mask to the *same* volume (Sec. 13). Verified on both banks in the repository:
every operator sees every item. `sizing.py` divided ~40 test patients by `k` operators and
concluded a six-operator grid reaches ±0.305; each operator in fact sees all ~40, so the
per-operator half-width is unchanged by grid size. **Adding operators costs compute, not
precision** — which reverses the script's headline advice. Now branched on `is_crossed()`
rather than assumed, because a design that reconstructed different patients at different
operators would make the division correct again.

**(b) The headline is a difference, and differences pair.** The central figure is coverage
*gap* against operator distance, not k coverages side by side. The same patient at both
operators makes it a within-patient contrast, and the patient's own difficulty — the dominant
variance term — cancels:

| between-patient SD `σ` | 0.00 | 0.05 | 0.10 | 0.20 |
|---|---|---|---|---|
| paired SE | 0.0130 | 0.0128 | 0.0130 | 0.0132 |
| unpaired SE | 0.0131 | 0.0181 | 0.0251 | 0.0385 |
| narrower by | 1.00× | 1.41× | 1.93× | 2.92× |

The paired SE is **flat in `σ`**. Cleanest demonstration, no substrate needed: coverage at an
operator minus coverage at the *same* operator is zero with no uncertainty, and the paired
estimator returns exactly that where differencing two independent bootstraps returns
SE ≈ 0.014 for a quantity that cannot vary.

**What Track 2 should run.** `scripts/sizing.py` now reports the crossed-design check, the
paired gap curve, the pairing factor, and paired sizing alongside the per-operator numbers.
On the Week 3 bank — which only exists on your machine (D-045), so this is not measured here:

```
PYTHONPATH=src python3 scripts/sizing.py runs/w03_mri_fixed_operator-fc534c8f4d898e1b \
    --calibration-operator mri.random1d.R2.c0.08.s0
```

**The pairing factor is a property of the substrate and is reported per bank, never assumed.**
On phantoms it is 1.0–1.25× because Shepp-Logan variants are nearly identical; real knee
anatomy across 30 patients is the opposite case, and the number that matters is the one your
bank gives.

**Adopted from P-23's menu:** option 2. Flag area becomes co-headline rather than a footnote —
your reasoning, reinforced by P-21's finding that the failure mode is efficiency rather than
validity.

**Not adopted:** option 1 (pooling with an operator term) is the regression form of the same
cancellation and is kept in reserve — the paired bootstrap is assumption-lighter and needs no
model of the operator effect. Option 4 (add `singlecoil_train`) stays the only lever on
contrast A and is unaffected by any of this.

*Cost:* nothing in the pipeline; it is a change of reported quantity and one script fix.
Contrast A really is capped at ±0.125 and this does not change it.

*Affects:* P-23, P-21, T1.4, T2.5, T2.6, T3.6, O3, Sec. 12, Sec. 15.2, Week 4 grid sizing.

---

## D-050 · The adjoint gate is size-aware, because a flat 1e-6 was a knife edge
**Date:** Week 4 · **Status:** ACCEPTED · **Found by:** Track 1 · **Fixes:** a non-portable test

`test_a_correct_adjoint_stays_far_below_the_tolerance` asserted
`adjoint_test() < 1e-6` at 32², 128² and 256². It passes on the machine it was written on
and **fails at 128² here** (1.265e-06).

**The adjoint is correct; the gate is wrong.** Measured at 128², a correct operator scatters
across accelerations and mask seeds:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| R=2 | 5.40e-07 | 5.37e-07 | 5.88e-07 |
| R=4 | **1.26e-06** | 6.10e-07 | 7.09e-07 |
| R=8 | 3.57e-07 | 8.41e-07 | 8.06e-07 |

The constant sits in the middle of the scatter, so whether it passes depends on the
acceleration, the mask seed and the platform's FFT summation order. There is no size trend
that would suggest a structural error — 256² scores 5.95e-07, *lower* than 128².

**Fix:** bound it by `sqrt(N)·eps_float32`, which is the size the error being measured
actually has — float32 summation over `N` terms. Still strict: the correct operator sits at
2–8% of the bound at every size, and at 256² the bound is 3.1e-5, some 30× below the ~1e-3
that the adjacent broken-adjoint test requires. So it separates the two cases by more than
the flat constant did, while no longer depending on the machine.

This is P-27's lesson one layer out. That entry made the *probe* deterministic; the
*tolerance* was still calibrated to one platform's rounding, which is the same class of
hidden dependency one level up.

*Alternative:* raise the constant to 2e-6. Rejected — it would be calibrated to this machine
instead of that one, and would silently loosen the 32² case by 20×.

*Affects:* T2.1, the runner's health gate, P-27.

---

## D-051 · A third split side, for the choices made by looking at scores
**Date:** Week 4 · **Status:** PROPOSED · **Proposed by:** Track 2 · **Unblocks:** T1.1

D28 (score component weights) and D37 (the criterion threshold) are both *selected* by
looking at scores. Until now there was nowhere to select them from: `SPLITS` was
`("calib", "test")`, so any selection had to come out of one of the two sides that the
coverage guarantee depends on being untouched.

Both failure modes are silent:

* **Selecting on calibration data** biases the conformal threshold. The calibration
  scores are no longer exchangeable with the test scores, and coverage degrades by an
  amount nothing in the pipeline reports.
* **Selecting on test data** biases the coverage number directly — the familiar one, and
  the one a reviewer will ask about first.

**Decision:** `split_for()` gains `design_fraction`, and patients below it land on a
third side, `"design"`. T1.1's weight selection and D37's threshold are fitted there and
nowhere else.

**It is carved out of the calibration range, not the test range.** That is the part worth
arguing about, and there are three reasons:

1. **The test side never moves.** For *any* `design_fraction`, a patient lands on `test`
   exactly when it would have under the two-way split. So a design split can be
   introduced after coverage has already been measured without invalidating the
   measurement, and two banks built with different design fractions remain comparable on
   the test side. `tests/test_design_split.py` asserts the set equality rather than
   trusting the arithmetic.
2. **The cost lands where it hurts least.** P-23 measured the binding constraint on the
   headline to be *test* patients: the paired gap half-width is ±0.121 against gaps of
   0.112 and 0.127. Calibration patients are not the scarce resource; test patients are.
3. **Nothing existing is disturbed.** `design_fraction=0.0` is the default and reproduces
   the two-way assignment exactly — same salt, same pins, same banks. The golden test
   (`test_split_golden.py`, including the real-fastMRI digest) passes unchanged, which is
   the evidence rather than the claim.

Asking for more design patients than there are calibration patients is refused with a
message naming `calib_fraction`, rather than quietly shrinking test.

**What Track 1 needs to pick.** The fraction itself is a protocol choice, not Track 2's.
T1.4 currently reads *"T1.1 is complete pending the weight selection in D28, which needs
the design split and therefore real data (C2)"* — the design split was the missing half of
that, and it was missing from **Track 2's** module, so the blocker was never only data.
Counted on the real 96-patient list at the default `calib_fraction=0.5`, with the test
side unchanged at 44 in every row:

| design_fraction | design | calib | test |
|---|---|---|---|
| 0.0 (today) | 0 | 52 | 44 |
| 0.10 | 8 | 44 | 44 |
| 0.15 | 19 | 33 | 44 |
| 0.20 | 25 | 27 | 44 |

**The draw is lumpy at this size** — 0.15 yields 19 rather than the nominal 14, and 0.10
yields 8 rather than 10. At 96 patients the fraction should be chosen by the count it
actually produces, which is why the measured table is here instead of the arithmetic.

*Alternative 1:* carve the design side out of the test range instead. Rejected — it
shrinks the one side P-23 identified as binding, and it would silently invalidate every
coverage number measured before the design side existed.

*Alternative 2:* carve proportionally from both. Rejected — it moves test patients, which
forfeits property (1) for no gain the project can currently use.

*Alternative 3:* select weights on calibration data and accept the bias. Rejected — the
bias is invisible and runs in the direction that flatters our own coverage, which P-19 and
P-23 have both already cost this project once each.

*Cost:* calibration patients, which makes the threshold slightly noisier. Unmeasured, and
it interacts with the `calib_fraction` rebalance offered at the end of P-23 — both draw on
the same pool, so they should be chosen together rather than one at a time.

*Affects:* T1.1, D28, D37, T2.1 §D-022, the fastMRI loader, Sec. 19.

---

## D-052 · Four of the five candidate operator distances are one measure on mask operators
**Date:** Week 4 · **Status:** ACCEPTED (measurement); conclusion revised by D-053 · **Measured by:** Track 2 · **For:** T1.3, T3.2, Week 6

> **Revised by D-053.** The measurement stands. The conclusion that comparing distances needs
> CT does not: the gap is directional, and the signed `null_space_growth` separates the
> operators on this MRI grid (ρ = −0.80 against the ascending naive gap on the full-data bank).

Choosing the distance axis for the Week 4 coverage-failure figure meant computing all
five candidates across the MRI grid. They are not five measures. Over **all 153 pairs**
of a 3-family × 6-acceleration grid at 256²:

| pair | agreement |
|---|---|
| `null_space_projection` vs `sampling_support` | identical to **1.1e-16** |
| `chordal_subspace` × √2 vs `sampling_support` | identical to **1.1e-16** |
| `operator_norm` vs `sampling_support` | within **9.1e-4**, Spearman ρ = 0.999915 |

**This is a property of the operator class, not of the implementations.** An MRI
undersampling operator is a diagonal projection in k-space. Which frequencies it keeps
*is* its range, so "how much do the sampling supports differ" and "how much do the null
spaces differ" are the same question asked twice, and the chordal distance between two
coordinate subspaces is that quantity over √2 by construction.

**Consequence, and it is a research one rather than a cleanup.** The MRI grid cannot
rank these measures against each other. T1.3 asks which distance best predicts the
coverage gap and T3.2 lists candidates to compare; on this operator class any ranking
among these four is numerical noise, and a Week 6 result that picked a winner here would
not survive a reviewer who noticed they are the same function. The comparison needs an
operator class where range and null space are not determined by a coordinate mask — CT
view subsampling (Week 7), where the projection is not diagonal.

**The fifth candidate could not be computed at all.** `spectral_divergence` raised for
every complex operator, which is every MRI operator (P-28, now fixed). So the one measure
that might have separated them was also the one nothing had ever run on this class.

**A second finding, for the grid design.** Distance from `mri.random1d.R2`:

| | R2 | R4 | R8 |
|---|---|---|---|
| `random1d` | 0.000 | 0.500 | 0.612 |
| `equispaced1d` | 0.644 | 0.656 | **0.650** |
| `variable_density2d` | 0.681 | 0.687 | 0.693 |

Changing the mask family **saturates** the distance: any family change costs ~0.65 at
once, after which acceleration contributes ~0.01 and not even monotonically — equispaced
R4 sits *above* R8. So the graded part of the axis is entirely within-family, and a
denser acceleration sweep across families buys near-duplicate points. This is why the
Week 4 config is 3 × 3 rather than the drafted 3 × 6, quite apart from the compute.

*Alternative:* report a single distance and drop the others. Rejected — the degeneracy
is the finding, and recording only one measure would hide it from Week 6.

*Affects:* T1.3, T3.2, T3.4 (L2 prefers `null_space_dim`), Week 6 Tool 2, Week 7 CT.

---

## D-053 · The gap is not symmetric in the operator pair, so the plotted axis is directional
**Date:** Week 4 · **Status:** ACCEPTED · **Proposed by:** Track 1 (T1.3) · **Answers:** D-052

D-052's measurement is right and its conclusion is too pessimistic by one step. Four of the
five registered distances agree to 1.1e-16 on mask operators because they are the same
function — and **all five are metrics**, while the quantity they are meant to predict is
not symmetric.

Measured on the phantom bank, naive transfer in both directions between the same pairs:

| calibrate → deploy | `null_space_projection` | gap | 95% CI |
|---|---|---|---|
| R2 → R8 | 0.6124 | −0.097 | [−0.134, −0.068] |
| **R8 → R2** | **0.6124** | **+0.902** | [+0.881, +0.921] |
| R4 → R8 | 0.3536 | −0.074 | [−0.094, −0.054] |
| **R8 → R4** | **0.3536** | **+0.234** | [+0.197, +0.270] |

Every x carries two y's of opposite sign with disjoint intervals. **The headline figure's
axis cannot be a metric.** The mechanism is P-21's with its direction named: deploying
where scores are higher over-flags (coverage gained, flag area exploded); deploying where
they are lower under-flags (coverage collapsed). The blueprint expects the second and the
grid has been exercising the first.

**`null_space_growth` is registered** — net ambient fraction newly hidden, signed. It
reproduces `null_space_projection` exactly (up to sign) on nested masks, so D19 is amended
rather than replaced, and `|growth| ≤ null_space_projection` always, so T3.3's bound is
untouched and merely loose in the benign direction. `distance_matrix` refuses it, since
mirroring an antisymmetric quantity would record half the pairs backwards.

**For the Week 4 figure.** From `random1d R2` at 256², the six cross-family deployment
operators:

| axis | spread over the 6 points | monotone in acceleration |
|---|---|---|
| `null_space_projection` | 0.0498 | no — `equispaced R4` > `R8` |
| `null_space_growth` | **0.6124** | yes, within every family |

**A prediction, and then its falsification against your own §7.1 — same day.** The signed
measure is *identical* across families at matched acceleration (0.0000 / 0.5000 / 0.6124 for
R2 / R4 / R8, all three families), because two masks keeping the same number of lines hide
as much as they reveal. That predicted equal coverage gaps at matched acceleration. **It is
wrong**, and your table already said so — I wrote the prediction before reading §7.1, which
was the wrong order:

| signed growth | gaps at that level | spread |
|---|---|---|
| +0.0000 | +0.038, **+0.142**, +0.070 | **0.104** |
| +0.5000 | −0.100, −0.098, +0.047 | **0.147** |
| +0.6124 | −0.100, −0.100, −0.092 | 0.008 |

Against a total gap range of 0.242, the family change carries ~60% of it. The worst point is
`equispaced1d R2` — your "one operator where naive transfer under-covers outright" — which
is the coherent-aliasing mechanism named as the candidate, so the direction of the
falsification is at least the expected one.

**What survives is the ordering, and it is worth having.** Spearman ρ of gap against axis
over your nine points:

| axis | ρ |
|---|---|
| `null_space_projection` | **+0.305**, and the sign is incoherent — gaps go *negative* as it grows |
| `null_space_growth` | **−0.751**, coherent: more newly hidden ⇒ more over-flagging |

**Revised: the figure takes two coordinates** — signed growth as the graded x-axis, mask
family as a categorical series. One line per family, not one curve through nine points.
D-052's conclusion that this operator class does not reduce to a single distance is
*upheld*; what it overstated is that the obstruction needs CT to see. Direction is
recoverable here; pattern is not reducible at all.

*Cost:* the measure is not injective — it collapses the family axis to zero — so it orders
the grid without separating it, and the family term has to be carried alongside.

*Affects:* T1.3, D19, T3.2, T3.3 §5, T1.4's figure, Week 4 grid analysis, Week 6 Tool 2.

---

## D-054 · `design_fraction = 0.10`; `calib_fraction` holds at 0.5 until the banks are rebuilt
**Date:** Week 4 · **Status:** ACCEPTED · **Proposed by:** Track 1 (T1.4) · **Answers:** D-051

D-051 is accepted as proposed, including the carve from the calibration range for the three
reasons given. The fraction was Track 1's and here it is, with the coupled second number.

**`design_fraction = 0.10`** — 8 patients on the real list. The argument is the flatness of
what it feeds: D42's weight sweep moves flag area by 0.015 across `w_free ∈ [0.3, 0.6]`, so
a noisy selection inside that region costs almost nothing. The design side needs to be
*separate*, not large. τ is the same case — D37 set it on calibration arithmetic, not by
fitting. And calibration can afford it: at τ=1.75 the positive rate is 0.401–0.772, leaving
thousands of positives per group against D32's floor of 9.

**`calib_fraction` stays 0.5, and the reason is comparability rather than statistics.**
Your P-23 lever (0.5 → 0.3, ±0.127 → ±0.111) is real and I am declining it for now:
**changing `calib_fraction` moves the test side** — 54 → 62 on a 96-key list — which is
precisely the property D-051 engineered for `design_fraction` and cannot provide for this
one. Taking it today makes every coverage number already measured incomparable with every
one measured after, for a gain that still misses ±0.10.

**Take it when the banks are being rebuilt anyway — the medical-prior swap (D-016).** That
is the one moment the re-baseline is free. Go to **0.4**, not 0.3, unless the
threshold-noise feedback you flagged has been measured by then; 0.3 trades a measured 0.016
against an unmeasured risk to the quantity the project certifies.

**One guard for the config rather than the prose:** `calib_fraction=0.3` with
`design_fraction=0.20` leaves **5** calibration patients, which neither number shows on its
own. Worth refusing, against D32's 9-positives-per-group floor rather than a patient count.

*Cost:* 8 calibration patients now; the paired half-width stays at ±0.127 until D-016.

*Affects:* T1.1, D28, D37, D-051, P-23, T2.1, the fastMRI loader.

---

## D-055 · The weight-selection rule is pre-registered; one run produces the number
**Date:** Week 4 · **Status:** ACCEPTED · **Proposed by:** Track 1 (T1.1) · **Closes:** D28's half of D-051

D-051 built the design split, D-054 set the fraction, and the selection rule is now written
down — **before** any bank with a design side exists, so it cannot be tuned to what the data
turns out to say. Track 1's entry is D49; `scoring/weights.py` and `scripts/select_weights.py`.

The rule in one line: sweep `w_free` on design rows, minimise **flagged area** at fixed α,
and among points within one cluster-bootstrap SE of the minimum take the largest `w_free`,
because `L = w·L_free + (1−w)·L_disagree` falls as `w_free` rises (D-046). Flat objective,
so the tie-break does the work and it is resolved toward the tighter bound rather than toward
noise.

**What Track 2 needs to run, and a plumbing gap worth knowing about first.**
`design_fraction` reached `splits.py` and the fastMRI loader, but **no config sets it, and the
phantom dataset cannot express it** — it takes a single fixed split string. So the three-way
split is reachable today for fastMRI only, as a *dataset* kwarg rather than a `run.` one:

```
caliber configs/experiments/w04_design_split.yaml
PYTHONPATH=src python3 scripts/select_weights.py runs/<run_id> \
    --calibration-operator mri.random1d.R2.c0.08.s0
```

**`keep_splits` is added to the fastMRI loader, and it is what makes that run affordable.**
The loader labelled every volume with its split while reconstructing all of them. For a side
that is 8 patients of 96 that discards ~92% of the work — at your measured 288 s per
item-operator, **597 reconstructions and 47.8 h** against **24 and 1.9 h**. It filters
*before* `limit_volumes`, which is the load-bearing ordering: limiting first returns 8
volumes of which roughly one is design.

It changes no patient's assignment. The split stays a pure function of (patient, fraction,
salt) per D-022/D-027; `keep_splits` chooses only what is worth reconstructing, and
`tests/test_split_filter.py` asserts a kept volume carries the label it would have carried
unfiltered. It is refused outright with an explicit `split`, where it would silently keep or
drop everything.

The config rather than overrides is deliberate: `max_draws_per_batch: 1` is mandatory
(P-24/D-044 — 4 draws at 256° needs ~18 GB and segfaults with no traceback) and
`checkpoint_every` is unset everywhere (D-040), so a two-hour run that dies loses everything.
Both belong in a file, not in four flags that D-048 will reject one character at a time.

One optional follow-up left for you: the phantom dataset takes a single fixed split string,
so the three-way path has no dataset-free smoke test.

*Cost:* one re-run of the Week 3 config. The test side does not move (D-051), so the coverage
numbers already measured stay comparable.

*Affects:* T1.1, D28, D-051, D-054, D-046, T3.3's constant.


---

## D-056 · Two failure modes, a corrected comparison rule, and Tool 2's first real evaluation
**Date:** Week 4 · **Status:** ACCEPTED · **Proposed by:** Track 1 (T1.4) · **See:** D50, D51, D52

Three results from closing out T1.4. The first is free and improves the Week 4 headline; the
third is a half-negative on the project's primary contribution.

**1. The grid has only been analysed in its benign direction.** Week 4 calibrates at
`random1d R2`, the least aggressive operator, and deploys upward. By D-053 that is the
direction where scores rise, so the threshold turns conservative and the failure shows up as
over-flagging — flag area 0.999, coverage 1.000. **Descending the grid is the other failure.**
On the phantom bank, calibrating at R8 and deploying to R2 gives naive coverage **0.000**
against Mondrian's 0.903. Coverage collapse — the failure §1 predicts, and nothing has been
pointed at it.

**This is a re-analysis of your existing bank, not a re-run.** The design is crossed (D-049),
so the calibration operator is an analysis-time choice — §17's premise. Both directions get
reported: the ascending half alone understates the risk, the descending half alone would be
cherry-picking.

**2. D-039's "compare flagged area, not power" needs a precondition, and I wrote it without
one.** Area is the right axis only while the conformal threshold actually pins the miss rate,
which requires calibration and deployment scores to be exchangeable — the thing operator
shift breaks. Caught by measurement: the nearest-operator arm posted the smallest flagged
area of any arm, at coverage **0.547**. An arm now competes on area only if its coverage
interval reaches nominal.

**3. Tool 2 has been evaluated for the first time, and it is half a win.**
`calibrate_weighted` appeared in no analysis path at all — implemented Week 2, never pointed
at a bank. `scripts/tool2_holdout.py` holds each operator out in turn. On the one interior
fold available: Tool 2 is **valid where the nearest-operator baseline is not** (that one
under-covers by 24 points), and flags **exactly as much area as Mondrian's unknown-group
fallback** — everything. It has earned validity and not efficiency.

The mechanism is not tuning. The oracle threshold is 0.8152; Tool 2 picks 0.5658, below the
held-out operator's *minimum* score. Bandwidth swept 0.02-0.8 moves it by 0.05 and never
changes the outcome, so **D33's open bandwidth question closes with "the bandwidth is not the
lever"**. A lower-tail quantile is set by the lowest-scoring regime in the mixture, and an
exponential kernel down-weights smoothly without ever excluding: at `h=0.08` the far regime
still holds 14% relative weight, enough to occupy the alpha=0.10 tail. Drive it to zero and
the ESS collapses, so D32 refuses to certify. **Tool 2's accuracy wants concentrated weight
and its certifiability wants spread weight, and on an ordered grid no `h` gives both.**

*For your grid:* Tool 2 cannot extrapolate, so a held-out **endpoint** operator tests nothing.
Q4 needs *interior* deployment operators, which your nine-operator grid has and the phantom
bank barely does.

*Cost:* one interior fold, a reference sampler, three operators. The direction of the
tail-domination argument should survive to nine operators, but the verdict is yours to
reproduce: `PYTHONPATH=src python3 scripts/tool2_holdout.py runs/<week4_run_id>`.

*Affects:* O3, T1.2, T1.4, D-039, D33, Q4, the Week 4 figure, Week 6 Tool 2.

---

## D-057 · Pre-run rulings: hold calib_fraction at 0.5, take design_fraction 0.10, keep all three families
**Date:** Week 5 · **Status:** ACCEPTED · **Proposed by:** Track 1 (T1.4) · **Answers:** Track 2's pre-run questions · **Amends:** D-054's reasoning

Full working in Track 1's D53.

**calib_fraction = 0.5.** Your table counts the test-side term only. Moving a patient from
calib to test also grows the threshold's own sampling noise, and they add in quadrature:
total half-width is **0.140 (0.5) against 0.137 (0.3)**, and **0.144 against 0.144** once the
design side is carved out. 0.3 buys 0.003, not 0.015, and none of the four settings reaches
±0.100. Re-slicing 96 patients cannot create precision. (Caveat: for the *paired* gap the
threshold noise partly cancels, so the real gain is in (0.003, 0.015] — still too small to
pay the comparability cost of moving the test side.)

**design_fraction = 0.10 — yes, in this run.** D49's rule is written and waiting on those
rows; `w_free` is the last unset number in the score. Free in a run that is happening anyway,
which makes `w04_design_split.yaml` redundant for this pass.

**Keep all three families.** Checked against your own §7.1: every cell below nominal coverage
is `random1d R2` (the calibration operator), `equispaced1d R2` (0.758),
`variable_density2d R2` (0.830) and `variable_density2d R4` (0.853). **Within `random1d`
alone every deployment cell is at coverage 1.000, so dropping the families leaves zero cells
supporting under-coverage** — the whole O3 evidence base in that grid is cross-family. It
would also drop the grid's lowest flag area (0.209), the only non-vacuous cell. If compute
forces a cut, cut accelerations: the within-family axis is graded and interpolable, the family
axis is categorical and is not.

*Affects:* T1.1, T1.4, D28, D-051, D-054, D-052, D-053, O3, Week 5 run.

---

## D-058 · Three fixes from the Week 5 review, two of them Track 1's bugs
**Date:** Week 5 · **Status:** ACCEPTED · **Proposed by:** Track 1 · **Found by:** Track 2
**See:** D54, D55, D56

**First, the result.** Calibrating at R8 and deploying down gives naive coverage **0.507**
at `variable_density2d R2` against group-conditional's **0.921**, and naive misses nominal
with the whole interval in **7 of 18 cells**. That is O3's coverage failure, demonstrated on
real data, and it exists because the grid was swept in the direction nobody had tried
(D-056). The ascending gap against `null_space_growth` reproduces at rho -0.80 against the
-0.75 measured on Week 4, which is the directional axis holding up on an independent run.

**1. `tool2_holdout.py` was in-sample, and it was my bug.** `split_col="__none__"` disabled
the split filter on all four arms. No row was reused, but the design is crossed, so the same
*patient* sat in calibration and evaluation — and `matched` calibrated on the very rows it
scored, which is why it landed on exactly 0.900. Fixed: calibration is the other operators'
`calib` rows, evaluation is the held-out operator's `test` rows, a single-split bank is
**refused** rather than silently reported, and the script now asserts the two frames are
disjoint on the clustering unit. Thank you for catching it; your re-run's finding stands —
Tool 2's verdict is unchanged and `nearest` goes 1/3 to 0/3, so the baseline was flattered.

This is the third instance of one failure in this project (P-19, P-23, now mine). The shape
is always that an in-sample number *looks right* — it lands on nominal — so nothing alerts
anyone.

**2. The weight selection should not have returned `w_free = 1.00`, and you were right to
distrust it.** 8 of 11 points were tied, and D49's tie-break walks to the boundary on a tied
sweep — which drops `s_disagree` from the score entirely. Eliminating a component is a
structural change and a tie cannot license one. The rule now checks whether the sweep
separated anything first: if more than half the points are tied, it returns D25's equal
weights flagged **NOT SELECTED** instead of a boundary value.

So **D28 closes on a bounded-cost default, not a fitted number**: equal weights over the live
components, with the cost against the unknown optimum bounded by D42's measured 0.015 of
flagged area. My D48 justification for `design_fraction = 0.10` was half wrong — flatness
bounds the *cost* of choosing wrong, it does not make the choice *informative*, and 8
patients cannot discriminate.

**3. The Windows crash was one of 31.** An audit of printable string literals found the same
defect in 13 files; the `<=` you hit was just the first one reached. All fixed by AST
position so docstrings kept their notation, and `tests/test_console_encoding.py` now enforces
it: printed literals ASCII, docstrings free, Markdown untouched. Worth knowing that this
class can only be caught on your machine, so the test is the substitute for that.

`tool2_holdout.py` line 227 also wrapped for ruff.

*Affects:* T1.1, T1.2, T1.4, D28, D48, D49, D52, O3, P-19, P-23.

---

## D-059 · The relative criterion gains a conjunctive absolute gate; and the Week 5 fallback ownership
**Date:** Week 5 · **Status:** ACCEPTED · **Proposed by:** Track 1 · **Found by:** Track 2
**See:** D57, D58, D59 · **Amends:** D21, D-034, D37

**The label defect is real and it is Track 1's.** Your 0.89-0.95 for low `free_content`
reproduces on the committed bank at 0.826, and the decomposition is worse than it looked:
`err_free` alone -- the *numerator* -- manages only 0.741, and
`spearman(ratio, free_content) = -0.520` against `spearman(ratio, err_free) = +0.450`. The
denominator has more influence on the label than the numerator does. A region with little
hidden content yields a large ratio from a negligible error, so the criterion labels smooth
tissue as hallucinated. `MIN_FREE_CONTENT` does not catch it: the smallest live value is
0.162, so the guard never fires -- the problem is the ratio's structure, not a near-zero
divide.

**Fix: the reported label is a conjunction**, `err_free/free_content > tau` **and**
`err_free > eps`. Neither gate suffices alone and they fail in opposite directions -- the
ratio is denominator-dominated, the absolute test is anatomy-confounded (D21's reason for
the ratio, and P-02). On the M4Raw bank at the 62nd percentile of `err_free`, the
denominator's AUROC falls 0.826 -> 0.696 while `s_free`'s rises 0.902 -> 0.974, at a
positive rate of 0.167. A *soft* floor fails -- it rescales the same structure and kills the
positive rate -- so the gate has to be independent.

`eps` is a **quantile on the design split, frozen**, never a constant: it is an error
magnitude and D-020 settled that those do not transfer across substrates.

**All of it is re-analysis, not re-reconstruction** -- D-002's dividend. Every coverage, FDR,
power and weight number needs recomputing, no GPU. **Q6 survives** the change (`s_free`
still invention-driven, AUC 0.900) with a smaller margin, which is a correction for the
write-up rather than a reversal.

**Your `s_disagree` observation needs re-measuring before it is acted on.** It beating the
combined score at every operator was measured against the defective label, and `s_disagree`
is the component *more* anti-correlated with anatomy (-0.155 vs `s_free`'s -0.035) -- so it
is the one the artefact most rewards. The equal-weights fix stands on its own grounds (D54);
the component ordering does not, yet.

**Fallback ownership: `calibration/` is right, they are Track 1's.** Methods are Track 1 per
Sec. 18, and your time is better spent on runs. They are also no longer contingencies: D52
measured Tool 2 underperforming, so the trigger has fired. Build order is **[7]
group-weighted first** -- weighting groups rather than points lets a group take weight zero,
which is D52's truncated-kernel lead in the form the register already named -- then [8]
doubly-robust. **[9] ACI is inapplicable** as the protocol stands: it updates the threshold
from observed coverage, which needs labels at deployment, which is exactly what is absent.
Carrying it as planned work would be a scope change in disguise.

**Seed floor: R4 is right as primary.** But the spread should grow with acceleration (R8
draws 12.5% of lines against R2's 50%), and the descending headline calibrates at R8, so add
R8 if it is cheap. Report per operator -- seeds are within-group variation (D-023), and the
groups are allowed to differ.

*Affects:* O2, O3, T1.1, T1.2, T1.4, D21, D37, D43, D52, D54, P-02, every label-dependent
number measured so far.

---

## D-060 · The gate goes on free_content, not err_free; and which direction collapses is solver-dependent
**Date:** Week 5 · **Status:** ACCEPTED · **Proposed by:** Track 1 · **Found by:** Track 2
**See:** D60, D61 · **Supersedes:** D57's gate · **Corrects:** D50, D-056

**Your 0.99-1.00 suspicion was right and it changes the fix.** By D-037's identity
`err_free -> s_free` as `free_content -> 0`, so where little is hidden the label's numerator
*is* the score. Measured by `free_content` tercile: `spearman(err_free, s_free)` is **+0.964**
in the bottom third against **+0.491** in the top. And my `err_free` floor selects exactly
those regions, so it removes the denominator artefact while amplifying the numerator one --
`s_free`'s AUROC goes to 0.973 at q62 and 0.985 at q75, which is the score being graded
against itself.

**Use the candidacy gate instead: `min_free_content` at q33 of `free_content` on the design
split.** Not a new parameter -- it is the one already in `label_hallucinated`, whose 1e-3
default is a guard that never fires (smallest live value 0.162).

| gate | pos rate | AUROC(-free_content) | AUROC(s_free) |
|---|---|---|---|
| ratio only | 0.255 | 0.826 | 0.902 |
| + err_free > q62 (D57) | 0.167 | 0.690 | **0.973** |
| + **free_content > q33** | 0.096 | **0.499** | 0.931 |

The candidacy gate reaches **exactly chance** on the denominator and leaves the score
believable. The reasoning is physical: a region where the operator hides almost nothing
cannot host a null-space hallucination, so it is not a candidate. **One gate, not two** --
adding the err_free floor on top changes nothing measurable.

**So the answer to your q question: q33, and on `free_content` rather than `err_free`.**
q33 rather than q50 on positive-rate grounds, given your 1.2% at `vd R2`.

**The direction flip is my error, and a narrower one than it looks.** I re-ran the phantom
bank under the candidacy gate and there the direction does *not* flip -- descending still
collapses. The reference sampler injects hallucination at a fixed scale so its scores order
purely by null-space size; a learned prior's depend on image content and mask pattern. So
**which direction fails is a property of the solver and substrate**, and D50 generalised from
phantoms to a claim about operator shift. D-053's mechanism survives -- the gap is not
symmetric and the axis carries the sign -- but D-056's assignment of which direction does
which does not, and your fastMRI result is the one that counts.

**Worth noting what this restores:** O3 as the blueprint wrote it. Naive transfer losing
coverage when deployed to a different operator, group-conditional restoring it, is what the
real-prior data shows once the label is fixed. Both P-21's over-flagging reading and my
descending-collapse reading were label artefacts. The claim never needed reframing.

*Affects:* O2, O3, T1.1, T1.4, D21, D37, D43, D50, D57, D-056, P-21, the midterm figures.

---

## D-061 · The label stops chasing anatomy-neutrality; stratify the analysis instead. Plus Week 8 scope
**Date:** Week 8 · **Status:** ACCEPTED · **Proposed by:** Track 1 · **Found by:** Track 2
**See:** D62, D63, D64 · **Supersedes:** D37's tau, D57's floor, D60's gate · **Answers:** P-31, T2.5 Sec. 4, T2.6 Sec. 4.3

**First, a retraction.** I validated D60's candidacy gate on a **pooled** AUROC of 0.499 and
called it chance. Per operator on the same bank it is **0.662 at R8** -- the pooled figure
was two opposite biases cancelling. Your 0.920 at `random1d R8` and 0.912 on CT are the
correct measurements and mine was the misleading one. Fourth time a pooled or in-sample
statistic in this project has returned the hoped-for number (P-19, P-23, D55, this).

**Neither of your two options works, and I tested both.**

| label | pooled | R4 | R8 |
|---|---|---|---|
| ratio only | 0.826 | 0.941 | 0.901 |
| ratio + candidacy q33 | 0.499 | thin | 0.662 |
| absolute err_free > q75 | 0.277 | **0.054** | 0.421 |

Dropping the division **inverts** the bias rather than removing it -- 0.054 at R4 means high
anatomy now predicts the label, which is the confound D21 added the ratio to fix. No function
of `err_free` and `free_content` alone is anatomy-neutral within an operator; both scale with
anatomical density. A CT-specific tau fixes neither modality's within-operator bias.

**A third route is closed too, informatively.** The principled label is the *invented* content,
`s_free*|sin theta|` by D-037, with no division. But `cos theta` has median 0.073 here, so
`corr(invented, s_free) = 0.9992` -- the label would be the score. That is what a prior
recovering none of the hidden content looks like (D-016). **No criterion can be validated
against this prior**, which bounds Week 8's claims whatever we pick.

**Ruling.** (1) Label: **absolute**, `err_free > eps`, `eps` a quantile of `err_free` on each
substrate's **own** design split, frozen. P-02 was a fixed constant; a quantile cannot flag
everything by construction. (2) Analysis: report every detection number **stratified by
`free_content` tercile, within operator** -- `stratified_auc` already does this. (3) Secondary
on MRI only: `require_annotation=True` against fastMRI+ boxes, the one anchor not defined by
our own arithmetic; unavailable on CT (P-03).

Do not encode a confound correction in the outcome definition; adjust for it in the analysis.
tau, the floor and the gate were three attempts at the same impossible thing. **This makes CT
readable and answers P-31 by the same change.**

**And the Tivnan result matters more than it looks.** kappa ~ 0 and mostly background means
the project has **no external validation of its label**. That is a stated scope limit for the
write-up, not a gap to fill quietly: the criterion is ours, in Sec. 4's terms, with the
annotation-restricted reading as the nearest outside anchor.

**Week 8 scope.** **Dose: no.** It changes the noise, not `A`, so `d_op` between two doses is
exactly **zero** -- two cells at the same x on the headline axis, with no sign to recover this
time. It also multiplies the `ct/{family}/v{views}/a{arc}` groups against D32's nine-positive
floor, and D22 already makes noise a group axis only if varied. Worth a **robustness check at
one fixed geometry** instead -- an appendix, and interesting precisely because `d_op = 0`
isolates noise from operator shift.

**CT distances: yes, but ~k+1 pairs, not the matrix.** CT is the only class in the project
where the five measures *can* differ -- D-052 showed they coincide on MRI because a mask
operator is diagonal. Compute: the calibration operator against each deployment operator
(the CT x-axis and T1.3's cross-modality item), plus one arc-change-at-fixed-views and one
views-change-at-fixed-arc, which are what test whether range and null space have come apart.
Include `spectral_divergence` on those pairs -- it never ran on MRI and CT is its first chance.

Good catch on P-34; that is D-021/D-023 enforced.

*Affects:* O2, O3, T1.1, T1.3, T1.4, T2.5, T2.6, D21, D37, D43, D57, D60, P-02, P-03, P-31,
every label-dependent number in both modalities.

---

## D-062 · Phase 2 gate: coverage claims stand, detection claims do not, and the DPS ablation is promoted
**Date:** Week 9 · **Status:** ACCEPTED · **Proposed by:** Track 1 (T1.6) · **See:** D67, D68, D69

The Phase 2 gate is in `T1.6_analysis_and_course_correction.md`. Three things need acting on.

**1. The central claim is demonstrated in both modalities and the secondary claim is not.**
O3 and O5 are statements about calibration behaviour under a *fixed* label, so the label's
defects change what "hallucination" means, not whether the comparison holds. Detection is a
claim about the score against a label that is approximately a function of the score --
T2.7's `err_free ~= sqrt(s_free^2 + free_content^2)` at Spearman 0.99 is D-037's identity at
`cos theta ~= 0`, so the 0.95-1.00 AUROCs measure the identity. **The midterm reports the
failure and its restoration, with intervals, and does not report detection quality as a
result.**

**2. The solver ablation must include DPS, and it is no longer an ablation.** Your two
findings differ in weight. The combination being worse in 7 of 9 operators by 0.002-0.014
AUROC is *inside* the split noise floor of 0.034-0.048 (one-sided sign test p = 0.090), so
it is "no evidence the combination helps". The dispersion result is the real one: within
0.001-0.012 AUROC of `s_disagree` at every operator, **because DDNM replaces the range
component structurally and has already confined the spread to the null space.** They are one
quantity computed twice. On this solver the decomposition cannot contribute to `s_disagree`.

D-033 put DPS's null-error share at 0.817-0.852 against DDNM's 0.99, so **DPS is the only
configuration where the decomposition can be tested at all.** Please schedule it ahead of
the rest of Phase 3.

**3. The MRI distance curve is not forced.** +0.976 on CT against +0.316 on MRI, because
MRI's worst failures are at matched acceleration across families where the scalar distance
is ~0. Plot MRI as one line per family; CT carries the curve claim.

**And your seed measurement retires my D59.** The floor is the patient split (0.034-0.048),
not the seed (0.007 and 0.001) -- no second seed sweep needed, and ~0.05 coverage is the bar
an ablation effect has to clear. Worth printing beside every ablation table.

*Also owed:* `tool2_holdout.py` on the CT grid, which has the most interior operators in the
project -- Tool 2 is dominated by the nearest-operator arm and fallback [7] (D66) has only
ever run where it reduces to that arm.

*Affects:* O3, O5, T1.2, T1.4, T1.6, T1.7, T2.5, T2.7, T2.8, D2, D52, D59, D66, Phase 3 scope.

---

## D-063 · The holdout harness runs on CT now; interiority was blind to the angular arc
**Date:** Week 9 · **Status:** ACCEPTED · **Proposed by:** Track 1 (T1.2/T1.4) · **See:** D71

Weighted recalibration on CT is Track 1's for Week 9. The bank is on your machine, so what I
could deliver is the harness being able to read it -- and it could not, for two reasons.

**It recomputed distances inline.** Every ordered pair, at minutes per pair on CT.
`--distances-csv` now reads `ct_distances.py`'s cache and only computes a pair that is
missing.

**And interiority came from `measurement_dim`, which cannot see an arc change.** Measured on
real CT operators at 32^2: `180deg/60v` and `90deg/60v` have the **same** measurement count
(1920), while the signed growth between them is **+0.5029** -- larger than halving the views
(+0.3370). So **narrowing the arc is the bigger operator shift and the old test scored it as
no change**, which would have classified every arc fold as an endpoint and dropped it from
Q4's comparison. Now defined on the signed axis: interior means some calibration operator
hides more and some hides less, so the weighting interpolates rather than extrapolates. The
MRI classification is unchanged, and that is a test rather than an assumption.

**What it unblocks.** **The CT grid is the only one in the project with enough interior
operators to tell Tool 2 and fallback [7] apart from the nearest-operator arm** -- where both
D52's and D66's verdicts have been stuck, the phantom grid having only two candidate groups.
Five arms, one command:

```
PYTHONPATH=src python3 scripts/tool2_holdout.py runs/<ct_run_id> \
    --distances-csv <ct_distances.csv>
```

Thanks for wiring the label gate into it -- I hit it immediately and the message told me
exactly which flag I wanted.

*Affects:* T1.2, T1.4, Q4, D52, D66, Week 9 arm A, the CT grid analysis.

---

## D-064 · The curve-collapse test, and one command to run it
**Date:** Week 9 · **Status:** ACCEPTED · **Proposed by:** Track 1 (T1.4) · **See:** D72, D70

Sec. 21 R5 is Track 1's and it is now a test rather than a reading of `fig5`. Pooled against
per-modality fits of gap on signed distance, with bootstrap slope intervals.

**The verdict is a magnitude test, not a significance test.** Collapse requires overlapping
slope intervals *and* the single curve to cost under **0.05 coverage** in RMSE -- your own
patient-split floor (D-062). The F-test is reported and does not decide: on a control with
slopes 0.3025 and 0.3029 it rejected at p = 0.031, because near-zero residuals make a trivial
intercept difference look decisive. Third time in this project a rejecting test has had to be
overruled by how much the difference is worth.

**Two caveats it prints rather than hides.** The modalities occupy different distance ranges,
so a *passing* test extrapolates one law into the other's range -- **a negative result is
strong, a positive one is weak**, and the summary says so when the ranges are disjoint. A
permutation test on modality labels is deliberately absent: with disjoint supports the labels
are not exchangeable.

**What I need from you, and the inputs already exist:**

```
PYTHONPATH=src python3 scripts/curve_collapse.py \
    --mri-coverage <mri_coverage.csv> --mri-distances <mri_distances.csv> \
    --ct-coverage  <ct_coverage.csv>  --ct-distances  <ct_distances.csv>
```

I did not run it on the real numbers on purpose: the MRI coverage table I hold predates the
current label, and a number from it could be mistaken for a result. Please run it on
`naive_transferred` for the claim and on `group_conditional` as the control -- a fix should
show no trend in either modality, so a collapse there is two flat lines agreeing rather than
a law.

*Affects:* O5, T1.4, T1.7, Sec. 21 R5, D-053, D-062, `fig5`.

## D-065 · Six CPU re-analyses are queued against your banks, and one GPU run is the priority
**Date:** Week 10 · **Status:** ACCEPTED · **Proposed by:** Track 1 · **See:** D73–D77, D68, D-064

Everything on Track 1 is now built and tested. What is left of mine is analysis of runs only
you can produce, so this is the whole ask in one place. All six are single commands; each has
fixtures where the answer was planted in **both** directions, so a negative comes back as a
negative.

**One needs the GPU, and it is the one that matters most.** The solver ablation (D68). Your
dispersion baseline ties `s_disagree` to within 0.001–0.012 AUROC, and the reason is
mechanical: DDNM replaces the range component structurally, so the spread is *already* in the
null space and projecting it there adds nothing. DPS leaves 0.817–0.852 of its error in the
null space against DDNM's 0.99, so it is the only configuration where the decomposition can
earn its keep. `scripts/solver_ablation.py` checks that premise first and refuses to proceed
quietly if the two solvers do not in fact differ.

**Five are CPU re-analysis of banks you already have:** `transfer_baseline.py` (D73, and
`runs/w05_analysis/dispersion.parquet` already exists), `tool2_holdout.py --exclude-nearest 1`
(D75), `label_comparison.py` (D74), `weight_sensitivity.py` (D76), and `curve_collapse.py`
(D-064, still outstanding).

**I fixed a bug of mine that was blocking the last of those.** `ct_distances.py` appended two
hard-coded CT pairs unconditionally, so pointing it at the MRI bank — which is how
`curve_collapse.py` gets its MRI distances, and which the manifest makes possible since
operators are rebuilt generically — wrote the real rows and then died on a `KeyError`, which
the surrounding `except ValueError` does not catch. Now filtered to what the bank holds, with
a note on stderr when a named pair is absent.

**One question back to you, and it affects D68's premise.** How were the 0.99 and 0.817–0.852
null-error shares computed? `P_N` is non-local, so the orthogonal decomposition does **not**
hold region by region: on the M4Raw bank `err_free² / err_total²` exceeds 1 in **97 of 384
regions**, up to 1.035. That is expected rather than a bank defect, but it means a mean of
per-region ratios is biased. `solver_ablation.py` takes it area-weighted per volume, where it
is the global MSE ratio and exact. If your figures came from per-region ratios the gap may
move, and the gap is what the ablation's rationale rests on.

**Why D73 is worth your time even before the DPS run.** The dispersion comparison has only
ever been held on detection AUROC, which D67 disqualified — so the number threatening Track
1's contribution was measured on the axis we already ruled out, and the axis the project
actually claims on had never been run against the baseline at all. D73 asks whose *threshold*
survives an operator change. It is a paired difference of transfer penalties, so each
detector's scale cancels, and the label's dependence on our score cancels too because it
moves both of CALIBER's arms together.

*Affects:* O3, O4, T1.1, T1.2, T1.4, T1.7, T2.6, D-053, D-061, D-062, D-064.

## D-066 · The DPS arm of the ablation: ζ = 5, 24 steps, the random1d family, its own bank
**Date:** Week 10 · **Status:** ACCEPTED · **Proposed by:** Track 2 · **See:** D-065, D68, D-004, D-031, D-033

The DPS bank (`configs/experiments/w10_mri_dps_gpu.yaml`) matches the DDNM bank of record in
everything except the solver: the same prior, 24 steps, 4 draws, 256², the same fastMRI volumes
and the same patient split. It runs as its own experiment and is merged afterwards (D-004).

*Guidance ζ = 5.0, the code default, not re-tuned.* D68's premise is the share Track 1 quoted,
0.817–0.852, which is DPS at ζ = 5 (D-033, Week 3). ζ is not neutral: at 24 steps the share
rises from 0.627 at ζ = 0.5 to 0.931 at ζ = 20 (D-031 re-measurement), so tuning DPS toward
its optimum would close most of the gap the ablation needs. The result is a statement about
DPS at ζ = 5.

*24 steps,* DDNM's budget: an equal budget is the fair comparison, and it is where 0.817 was
measured.

*random1d at R2 / R4 / R8 only.* DPS costs ~30 s/pair on the RTX 3050 (~2.4× DDNM), so the
nine-operator grid is ~15 h of GPU against ~5 h for these three. The ablation compares
detection per operator, never across operators (D60, retracted), so it needs the calibration
operator (R2) and a spread of accelerations, not every family.

*Premise, measured before the run* on 6 pairs, area-weighted per volume: DPS 0.44–0.50 at R2,
0.69–0.76 at R4, 0.85–0.90 at R8, against DDNM's 0.96–0.998 on the same pairs.

*Alternative:* all nine operators, for an operator set identical to the DDNM bank's.
*Reason:* ~10 more hours of GPU for operators the per-operator comparison does not need.
*Cost:* the ablation says nothing about the cross-family operators. And at R2, the
calibration operator, DPS sits at or below D-033's 0.5 admissibility line, so the widest
premise gap is also where DPS's fit is least converged. T2.9 states this beside every R2
number.
*Affects:* D68, O4, T2.9.

## Template

```
## D-0NN · <one-line decision>
**Date:** Week N · **Status:** PROPOSED | ACCEPTED | OPEN | SUPERSEDED
**Proposed by:** Track N

<what was decided>

*Alternative:* <what else was considered>
*Reason:* <why>
*Cost:* <what this gives up>
*Affects:* <Blueprint items>
```
