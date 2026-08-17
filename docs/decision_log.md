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
**Date:** Week 1 · **Status:** OPEN · **Proposed by:** Track 2

*Needs:* Week 2 decision (O2). If regions become anatomically derived,
`region_area` stops being constant and region-level rates become weighted,
which changes the risk definition itself.

*Affects:* O2, T1.2, T2.4, Week 2.

---

## D-010 · Working default distance is `null_space_projection` / `ambient`
**Date:** Week 1 · **Status:** OPEN · **Proposed by:** Track 2

All five Sec. 12.1 candidates are implemented. The primary measure is locked in
Week 3 by T1.3.

*Note:* normalisation stays configurable through Week 9, because which choice
makes the MRI and CT curves collapse is an empirical question (Sec. 21 R5).

*Affects:* T1.3, T3.2, Week 3, Week 9.

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
**Date:** Week 1 · **Status:** OPEN · **Raised by:** Track 2 · **Blocks:** Week 7

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

*Fix (Week 7, Track 2).* Replace FBP with a genuine pseudo-inverse for the
**decomposition** — conjugate gradient on the normal equations `AᵀA x = Aᵀ y`,
which converges to `A⁺y` in the least-squares sense and is a contraction. FBP
can remain the display reconstruction. Cost is a handful of matvecs per item,
which the score bank amortises anyway.

*Detection.* `ForwardOperator.null_projection_gain()` and the
`contraction_ok` field in `OperatorRegistry.check_health()`. Reported as a
runner warning rather than gated, so CT development is not blocked before the
fix lands.

*Affects:* O1, O2, L1, L2, D-006, T2.1, T2.5, T3.3, Sec. 13, Weeks 7-9.

---

## D-013 · TotalSegmentator masks do not align with LoDoPaB images
**Date:** Week 1 · **Status:** OPEN — needs a Week 1 answer · **Raised by:** Track 2

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

*Affects:* T2.2, T2.5, O7, Weeks 2, 10, 12.

---

## D-017 · fastMRI acquired by direct download, staged off-machine
**Date:** Week 1 · **Status:** ACCEPTED — one check outstanding · **Proposed by:** Track 2

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
