# Track 2 — Problems and Decisions

The single Track 2 record: what broke (**Part I**) and what was chosen (**Part II**).
Problems and decisions were previously kept in two files; they are merged here because
almost every entry in one cites an entry in the other, and keeping them apart meant two
places to update and two chances to let them drift.

Track 1 keeps its own equivalent in
[problems_and_decisions.md](problems_and_decisions.md). Track 2's *weekly work record*
is separate again — see [T2.1_data_and_operator_pipeline.md](T2.1_data_and_operator_pipeline.md),
which also carries the data-access provenance and the shared interface specs.

Ordering within Part I is deliberate: unresolved first, because those are the ones that
can still cost someone a week.

---

## Part I — Problems

What broke, how it showed up, and what was done about it. Every entry cross-references
the decision id in Part II where one exists, so the two halves cannot drift.

### Still open

#### P-02 - The v0 label threshold marks everything as hallucinated

**Blocks:** region FDR / power on any real substrate - **See:** D-020, D-003, D-009

At the v0 criterion `err_free > 0.10`, 100% of MRI regions on real M4Raw
k-space label as hallucinated. Region FDR is then identically zero and the flag
carries no information. CT shows the same symptom at `err_free > 0.05`.

Same symptom, different causes, and the difference took measuring to establish.
CT's is the P-01 projector defect. MRI's decomposition is sound - ratio 0.97,
all six operators pass the contraction check - so on the MRI side it is the
threshold that is wrong. Real brain anatomy carries far more null-space content
than a smooth phantom, so the whole `err_free` distribution shifts up: M4Raw's
1st percentile (0.116) already clears a threshold set at 0.10. The constant had
been calibrated against phantoms without anyone deciding that it had been.

**Where it stands:** open, needs the Week 2 O2 decision from Track 1. What can
be stated already is that the criterion cannot be a shared absolute constant -
it fails to transfer across modalities *and* across substrates within one
modality. It has to be relative: a quantile, a per-item normalisation, or a
definition where the positive rate is controlled rather than observed.

**Not a pipeline failure.** At `err_free > 0.25` the central phenomenon
reproduces on real k-space for the first time - naive transfer holds 0.944
coverage at the calibration operator and collapses to 0.217 at R=4 and 0.014 at
R=8, monotonicity rho = +0.829 against operator distance.

---

#### P-03 - TotalSegmentator masks do not annotate LoDoPaB images

**Blocks:** region-level CT evaluation - **See:** D-013

> **Decided in the Week 1 sign-off (Option A).** TotalSegmentator becomes the CT
> substrate. Track 1 does not object, and D20 removed the coupling that made this a
> cross-track question: annotations are an evaluation overlay, not the region definition.
> The remaining work is the Week 7 loader, so this stays listed until that is wired.

The plan listed LoDoPaB-CT as the CT substrate and TotalSegmentator as the
source of region labels for that substrate. Inspecting both showed the two
roles do not compose: TotalSegmentator's masks segment TotalSegmentator's own
volumes, a different cohort from LoDoPaB (which derives from LIDC-IDRI). There
is no subject correspondence, so LoDoPaB slices cannot be annotated this way at
all.

This was found by reading the datasets rather than by a failing test, which is
the only reason it surfaced before Week 7.

**Where it stands:** needs a kickoff decision. The recommended option is to
make TotalSegmentator the CT substrate outright - it ships clean images plus
117 aligned organ masks per subject, and since D-005 already means we
re-project ground truth with our own operator, no projection data is needed at
all. The decision also determines which ~49 GB of archives can be deleted, so
it is holding up disk as well as code.

---

#### P-04 - Randomised spectral estimation barely separates CT operators

**See:** D-011, D-007

The fitted singular-value decay exponent moves only 1.688 -> 1.676 between 180
and 20 views. A randomised range finder recovers the top of the spectrum, and
the top looks nearly identical across view counts; what actually distinguishes
these operators is the decay tail and the null space, which is exactly what the
estimator resolves worst.

Consequence: `spectral_divergence` is a weak distance candidate and should not
be selected without evidence it separates operators better than this. It also
reaches Track 3 - if the coverage-gap bound leans on a spectral-decay term, it
currently rests on an estimate of unestablished accuracy.

Related and open: CT `null_space_dim` is recorded as `ambient - measurements`,
a dimension-count bound rather than the true rank (D-007). Anything that scales
with null-space dimension is scaling with a proxy, and the write-up has to say
which.

---

#### P-05 - fastMRI access has external lead time, and a licensing question

**See:** D-015, D-017

fastMRI needs a signed NYU Data Sharing Agreement and an approved application.
That cannot be automated, it was the only item on the critical path with
external lead time, and it gates the Week 3 deliverable.

Two things came out of working around it, one settled and one not:

*Settled.* MRI development moved to M4Raw (CC-BY, no application, same `.h5`
layout with `kspace` and `reconstruction_rss` keys), so `FastMRIDataset` reads
it nearly unchanged. It carries pipeline work but no reported result - 0.3 T
low-field brain against fastMRI's 3 T knee, and no pathology annotations, so
the annotation-restricted evaluation cannot run on it.

**Update (Week 1 close-out).** The data has landed. `knee_singlecoil_val` was
downloaded directly and extracted to `data/raw/fastmri/singlecoil_val/`: **199 volumes,
7,135 slices, 19.4 GB**, all readable (every file opened and a real slice read, so a
truncated extraction would have surfaced - see P-08 for why that check exists). 155 of
the 199 volumes carry fastMRI+ annotations; 44 do not. The loader reads them unchanged.

`knee_singlecoil_test` was also downloaded, and is **unusable** - see P-13.

*Now settled too.* The licensing question is answered: the download links were
**NYU's own, emailed after approval** — not a Kaggle mirror and not third-party
redistribution. The clause that would have made this data unreportable is not
engaged. `singlecoil_val` is the reported MRI substrate, and M4Raw reverts to
being what D-015 scoped it as: a development stopgap, not the lawful fallback.

**This problem is closed.** See D-017. The one licensing clause still live is at
release, not now: §6 forbids redistributing images, so the O7 release ships code,
configs and score banks only.

---

#### P-06 - Patient-level splitting: wired for fastMRI, still open for LoDoPaB

**See:** D-014, D-022, **D-027**

**Update (Week 1 close-out).** Half of this is now fixed, and fixing it exposed that
the rule itself was wrong.

The rule said "split by patient for LoDoPaB, **by volume for fastMRI**, never by slice",
on the assumption that a fastMRI volume is a patient. Reading the attributes of all 199
extracted `singlecoil_val` volumes shows only **96 distinct `patient_id`** values, and
**every one of those 96 patients owns more than one volume**. Splitting by volume leaks
**45 of 96 patients (47%)** across the calibration/test boundary - the exact hazard
D-014 exists to prevent, in the modality D-014 declared safe.

`caliber.data.splits` now assigns from `patient_id`, deterministically and
operator-independently; `FastMRIDataset(split="auto")` uses it, and
`verify_no_patient_straddles` asserts it. Measured on the real data: 0 straddling
patients, 52/44 patients, 108/91 volumes. See D-027.

**Still open: LoDoPaB.** `LoDoPaBDataset` still indexes by file and slice offset and
does not read the patient CSVs. The hazard is live there until that is wired (Week 7).

---

#### P-06a - The original entry, kept for the record

**See:** D-014

LoDoPaB ships one patient ID per slice across 35,820 train / 3,522 validation /
3,553 test slices, spanning far fewer patients than slices. A slice-level
random calibration/test split therefore puts the same patient on both sides.
Adjacent slices are near-duplicates, so this inflates apparent coverage and
breaks the exchangeability the conformal guarantee rests on - in the direction
that flatters our own numbers, which is the worst way for an error to run.

The rule is decided: split by patient for LoDoPaB, by volume for fastMRI, never
by slice. But `LoDoPaBDataset` still indexes by file and slice offset and does
not read the patient CSVs yet. The hazard is live until that is wired (Week 7,
or earlier if CT moves up).

---

#### P-13 - fastMRI `singlecoil_test` cannot be used by this project at all

**See:** D-017, data_access.md section 1

`knee_singlecoil_test` (1.46 GB) was downloaded alongside the validation split. It
cannot be used, for two independent reasons, and both were checked rather than assumed:

```
keys:  ['ismrmrd_header', 'kspace', 'mask']
attrs: {'acceleration': 8, 'num_low_frequency': 15, ...}
kspace (37, 640, 368) complex64      mask (368,) bool
```

1. **The k-space is already undersampled** at a fixed R=8, and ships the mask it was
   undersampled with. The acquisition operator is the variable under study, so it has to
   be *chosen* and applied to fully-sampled k-space. An operator grid cannot be
   synthesised retrospectively from measurements that have already been decimated.
2. **There is no `reconstruction_esc`.** It is the challenge submission set, so it has no
   ground truth, and without ground truth there is no `err_total`, no `err_free`, and no
   region-level criterion to evaluate.

It is also not annotated: fastMRI+ covers the train and validation splits.

Compare the validation split, which has `kspace` **fully sampled**, plus
`reconstruction_esc`, `reconstruction_rss`, and `patient_id`.

**Consequence.** `singlecoil_val` is the entire reported MRI substrate. Its 96 patients
are the whole calibration *and* test population, which makes the per-patient split
(D-027) load-bearing rather than tidy, and it caps how thin `calib_group` can be sliced
before groups starve (D10, Tool 1). The train split was deliberately not downloaded
(72.7 GB) and nothing needs it: no config references it, and the project trains nothing.

---

#### P-14 - The disk is full, and the archives are the reason

> **Partly reclaimed.** `knee_singlecoil_val.tar` was deleted after a second full
> verification pass (199/199 volumes opened and read). Free space went 99 MB -> 16 GB,
> which is enough for the first score banks. The CT archives below are still reclaimable
> and still not needed.

**See:** P-10, D-017

Extracting `singlecoil_val` took the 215 GB volume to **99 MB free**. The 19.4 GB of
extracted `.h5` now coexists with the 16.0 GB `.tar` it came from, plus ~49 GB of CT
archives that D-005 and D-013 already establish are not needed.

Nothing is corrupted - the extraction completed and was verified - but there is no room
to write a score bank, which is the next thing T2.3 has to do. Reclaimable, in order of
how safe each one is:

| candidate | size | why it is safe |
|---|---|---|
| ~~`data/knee_singlecoil_val.tar`~~ | ~~16.0 GB~~ | **deleted** after re-verifying all 199 volumes — 16 GB reclaimed |
| `data/raw/fastmri/singlecoil_test/` | 1.4 GB | unusable, P-13 |
| `data/knee_singlecoil_test.tar` | 1.46 GB | unusable, P-13 |
| `data/archives/lodopab/observation_*.zip` | ~40 GB | not needed, D-005 |
| `data/archives/lodopab/ground_truth_train.zip` | ~9 GB | not needed, D-013 |

Not actioned unilaterally: the fastMRI links are time-limited, so deleting an archive is
not trivially reversible even though the extracted data is intact.

---

### Resolved

#### P-01 - The CT null projector amplifies instead of projecting

**Fixed** - **See:** D-012, D-006

CT `pinv` (filtered backprojection) is not the Moore-Penrose inverse, so
`P_N = I - A+A` built from it was not a contraction. Measured on real LoDoPaB
data, median `err_free / err_total` was 1.33; for a genuine orthogonal
projection that ratio cannot exceed 1. Worse than the constant was the trend:
`||P_N x|| / ||x||` climbed from 0.667 at 180 views to 1.738 at 30 views - the
projector degraded monotonically *in the same direction as operator distance*,
which could have made a CT coverage-gap-vs-distance curve come out monotone
for the wrong reason.

**Fix:** `CTProjectionOperator.pinv_cg` - conjugate gradient on the normal
equations `AᵀA x = Aᵀy`. Started from `x0 = 0`, CG never leaves
`row-space(A)` (both `AᵀA` and `b = Aᵀy` live there), so it heads within row
space towards the minimum-norm least-squares solution, `A⁺y`.

FBP stays exactly where it was, as the fast *display* reconstruction returned
by `pinv()` - solvers and score-bank baselines are unaffected. Only the
decomposition (`project_range` / `project_null`, and therefore `err_free` and
`s_free`) is repointed at CG, through a new `ForwardOperator.decomposition_pinv`
hook.

**Scoped deliberately to contraction, not idempotence.** Measured across the
Sec. 16 operator grid, `null_projection_gain()` clears `<= 1` reliably, often
from just 5-10 CG iterations - that gain, not idempotence, is what D-012 was
about, and `contraction_ok` is now a hard gate in the runner rather than a
warning. `projection_idempotence_error()` is a different, stricter property
that settles in the low tens-of-percent range at best and does not simply
improve with more iterations (applying a partially-converged solve twice, as
the idempotence test does, compounds error in a way the first solve's
iteration count doesn't control) - `exact_pinv` correctly stays False for CT
and the registry's idempotence gate stays bypassed for it, unchanged.

**A real numerical hazard, found and closed while building this.**
Unregularized CG on this rank-deficient system is not just slow past the
point of diminishing returns, it is unsafe: an early version of the fix
diverged by 6+ orders of magnitude beyond ~500 iterations on a 16-view
operator, once the Krylov recurrence exhausted the well-determined subspace
and further steps amplified rounding error in near-null directions instead of
signal. Fixed by tracking the best iterate by residual norm and stopping
outright on the first upward turn, rather than trusting a configured iteration
count to be safe. Iteration count and tolerance are now `cg_iterations` /
`cg_tol` constructor params (default 20 / 1e-6 - the contraction fix
saturates by ~20 iterations, and idempotence keeps wobbling non-monotonically
well past that point, so more iterations is cost without reliable benefit) -
recorded per-operator for
reproducibility, and deliberately excluded from `operator_id`.

**Re-measured on the real data the defect was found on.** Re-ran
`w07_ct_smoke.yaml` (real LoDoPaB validation slices, 128x128, 180/90/45 views,
1536 region rows) after the fix: median `err_free`/`err_total` is now
0.024-0.065 per operator (against the original 1.33), and zero of 1536 rows
violate `err_free <= err_total`. Also confirmed the CG iteration tuning
mattered in practice, not just on synthetic grids: the same run took 345s at
`cg_iterations=50` and 162s after tuning the default to 20, for the same
result. Refreshing P-02's specific numbers under the new decomposition is the
next CT item - that's Track 1's O2 threshold decision, not a pipeline fix.

**Lesson:** an approximation that looked like it would only loosen constants
(D-006) turned out to invert the inequality the free component is supposed to
satisfy. "Approximate" was not a safe thing to wave through - and the fix for
a *display* approximation did not require touching the display path at all,
once the decomposition was given its own inverse to call. A second lesson
showed up inside the fix itself: replacing an approximate method with an
iterative one traded a silent bias for a different failure mode (numerical
divergence) that needed its own guard.

---

#### P-07 - fastMRI+ boxes were rasterised on the wrong grid

**Fixed** - **See:** D-019

Placing the annotation boxes on the inverse-FFT grid put every annotation in
the wrong place. Knee k-space inverts to 640x368, but the boxes are drawn on
fastMRI's 320x320 `reconstruction_esc` reference image. Across the shipped file
- 16,154 boxes, 969 volumes - 100.0% survive a 256 centre crop under the
320x320 reading against 25.5% under the k-space reading, and
`max(x + width) = 319`, `max(y + height) = 320` pins the grid exactly.

**Why it is the worst kind of bug.** It had no symptom. A misplaced mask is
still a valid mask - right dtype, right shape, plausible area - so nothing
downstream objects. Regions would simply have been scored against annotations
describing different anatomy. The first real box tested rasterised to zero
pixels after the crop.

**Fix:** crop the image to the annotation grid first, then take an identical
second crop of image and mask to the working resolution, so the two agree by
construction rather than by coincidence. The grid is read from the volume when
present and falls back to `ANNOTATION_GRID` when absent. Guarded by
`test_boxes_are_placed_on_the_annotation_grid_not_the_kspace_grid` and by a
check against the real file.

**Left behind as a trade-off, not a bug:** at 256x256 all boxes survive the
crop; at 128x128 only 76.7% do. Whoever sets the MRI working resolution is
trading region FDR/power sample size against GPU time.

---

#### P-08 - A truncated archive sat recorded as fetched

**Fixed**

The M4Raw archive was recorded as downloaded at 0.8 GB of 2.8 GB. A truncated
download does not look like a failure: the file is present, it is large, and
its first bytes are a valid zip header. Only the central directory at the end
is missing, so nothing notices until an extraction is attempted - possibly
weeks later.

`scripts/fetch_data.py --verify` now checks every archive. The first version of
that check called `testzip()`, which CRC-checks by decompressing every member;
against 77 GB of archives it did not finish. Opening the zip is enough, because
the central directory is precisely what a truncated file has lost, and it costs
one seek. The deep check stayed available behind `--deep`, for corruption
rather than truncation.

Verifying all nine archives also turned up `observation_validation.zip`, which
the inventory had never listed - the recoverable total is ~55 GB, not ~49 GB.
Sizes in the inventory (`data_access.md` §6) are now measured rather than estimated.

---

#### P-09 - The Windows console mangled the error messages

**Fixed**

The Windows console is cp1252 and rendered em dashes and section signs as
replacement characters. That is a poor look anywhere and worse on an error path
whose entire job is to be read. Runtime output - runner, loaders,
`analyse_bank.py` - is ASCII. Documentation keeps its typography.

---

#### P-10 - The 6 GB laptop does not fit the data

**Worked around** - **See:** D-017

~30 GB free against a 7 GB knee singlecoil split plus a 30.1 GB
TotalSegmentator extraction. The fetch is staged elsewhere and only
`singlecoil_val` is copied in.

Two consequences are already handled in code, because archives staged through
another environment rarely arrive with the same nesting: volume discovery is
recursive (`data.fastmri.find_volumes`), and configs point at the split
directory rather than `data/raw/fastmri`, so train and val cannot be swept into
one index - the P-06 hazard in its fastMRI form.

---

#### P-11 - LoDoPaB's own sinograms make the operator grid impossible

**Accepted cost** - **See:** D-005

The shipped low-dose sinograms fix the acquisition at one geometry. The
operator grid is the entire object of study, so consuming them directly would
remove the independent variable. CT operators are built instead by
re-projecting LoDoPaB ground-truth images with our own registered
`CTProjectionOperator`.

The cost is real and belongs in Honest Scope: the CT forward model is ours, not
a scanner's. Filed here rather than only in the decision log because it is the
root of P-01 - our projector is also our problem.

---

#### P-12 - A score bank carries no mark of where it came from

**Fixed** - **See:** D-018

Three substrates and samplers are development-only by design: phantoms, M4Raw,
and the reference samplers. Each produces a completely valid score bank. A bank
is a table of floats, so the distinction survived only in whoever remembered
which config had been run - and over twelve weeks and three tracks reading each
other's banks, that is not a safe place to keep it. The failure would have been
silent, and in the direction that flatters the results.

`SliceDataset.reportable` is combined with `solver.family == "reference"` into a
`provenance` block written into every manifest. The runner prints
`NOT REPORTABLE` at run time and `analyse_bank.py` prints it again at analysis
time. A naming convention on `run_id` was rejected - conventions are not
enforced.

---

### Adding an entry

Number sequentially, state the symptom before the cause, and say plainly
whether it is fixed, worked around, or still open. If it is open, say what it
blocks and who has to decide. Link the decision-log id if the problem produced
a decision.

---

## Part II — Decisions

Required by task T1.5 and the Week 1 card: *establish a cross-track decision log and
adjudication process for discrepancies between empirical, methodological, and analytical
results* (Sec. 19).

**Adjudication process.** When two tracks disagree — a measured coverage gap
that contradicts the bound, a distance measure that behaves differently in
theory and in data — the disagreement is logged here as OPEN, discussed at the
weekly sync, and resolved by Track 1, which owns the central claim (T1.6). An
unexplained empirical finding is referred back to Track 3 for analytical
explanation rather than dropped (Sec. 19, cross-verification).

Every entry: date, decision, alternatives considered, reason, affected
Blueprint items.

---


Every entry: date, decision, alternatives considered, reason, affected Blueprint items.

### D-001 · Score bank stores components, not a combined score
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

### D-002 · Ground-truth labels are derived at analysis time
**Date:** Week 1 · **Status:** ACCEPTED · **Proposed by:** Track 2

The bank stores continuous per-region error magnitudes (`err_total`,
`err_free`) and annotation coverage, not boolean hallucination labels.

*Reason:* the O2 criterion is a Week 2 decision and will be revisited. A bank
of booleans would need regenerating on every revision.

*Affects:* O2, T2.3, Week 2 criterion definition.

---

### D-003 · `err_free` recorded separately from `err_total`
**Date:** Week 1 · **Status:** ACCEPTED · **Proposed by:** Track 2

A region can be wrong because the prior invented structure or because it
smoothed structure away. Only the first is hallucination in the Blueprint's
sense (Sec. 4). Restricting the error to the free component distinguishes them,
so both are stored.

*Open:* does the Week 2 criterion use `err_free`, `err_total`, or both? Current
default is `err_free`.

*Affects:* O2, Sec. 4, Week 2.

---

### D-004 · One solver per run, enforced in config and schema
**Date:** Week 1 · **Status:** ACCEPTED · **Proposed by:** Track 2

`experiment.solver` rejects a list, and score-bank validation reports any
`run_id` containing more than one `solver_id`.

*Reason:* Sec. 13 requires the solver held fixed within a coverage curve so the
operator is the only varying factor. Making the violation structurally
impossible is cheaper than catching it in review. The Week 10 solver ablation
runs one experiment per solver and merges afterwards.

*Affects:* Sec. 13, T1.4, Week 10 ablation.

---

### D-005 · CT uses our own projector, not LoDoPaB's supplied sinograms
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

### D-006 · CT range/null-space split is approximate, and says so
**Date:** Week 1 · **Status:** ACCEPTED · **Proposed by:** Track 2

CT `pinv` is filtered backprojection, not the Moore-Penrose inverse, so `P_R`
is not an exact projector. Operators carry `exact_pinv: false` and the health
check exempts them from the idempotence gate.

*Open for Track 3:* does L1/L2 need an exact projector, or does an approximate
one only loosen constants? If exactness is needed, the CT inverse must be
upgraded (CG on the normal equations) in Week 7.

*Affects:* L1, L2, T3.3, T3.4, Sec. 13, Week 7.

---

### D-007 · CT `null_space_dim` is a dimension-count bound
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

### D-008 · Single-coil MRI via root-sum-of-squares
**Date:** Week 1 · **Status:** ACCEPTED · **Proposed by:** Track 2

Multi-coil volumes are reduced by RSS; the studied operator is single-coil
Cartesian undersampling.

*Reason:* consistent with the Sec. 17 compute profile on a 6 GB GPU, and avoids
coil sensitivity estimation becoming a confound in the operator definition.

*Cost:* narrows the empirical claim. Belongs in the Experimental Setup.

*Affects:* Sec. 17, T2.1, T2.7.

---

### D-009 · Region definition v0 is a 16px patch grid
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

### D-010 · Working default distance is `null_space_projection` / `ambient`
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

### D-011 · Randomised spectral estimation barely separates CT operators
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

### D-012 · The FBP-based CT null projector amplifies; it must be replaced
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

#### The part that threatens the headline result

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

#### Fix, landed — scoped to the contraction property, not idempotence

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

### D-013 · TotalSegmentator masks do not align with LoDoPaB images
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

#### Option A — use TotalSegmentator as the CT substrate (recommended)

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

#### Option B — keep LoDoPaB and generate masks

Run the TotalSegmentator *model* (a separate pip package, not this dataset) over
LoDoPaB ground-truth images. Keeps LoDoPaB as substrate but adds an inference
dependency, GPU time we budgeted for reconstruction, and label noise from a
model rather than curated ground truth.

*Recommendation:* Option A. Decide at the Week 1 kickoff, since it determines
which ~49 GB of archives can be deleted and what the Week 7 loader targets.

*Affects:* O5, Sec. 11, Sec. 14, T2.1, T2.5, Weeks 7-9.

---

### D-014 · LoDoPaB must be split by patient, not by slice
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

### D-015 · M4Raw as an open stopgap while fastMRI access is pending
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

### D-016 · Natural-image diffusion prior wired before medical priors
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

### D-017 · fastMRI acquired by direct download, staged off-machine
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

### D-018 · Reportability is a property of the bank, not of whoever ran it
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

### D-019 · fastMRI+ boxes live on the 320x320 reference grid, not on k-space
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

### D-020 · The v0 label threshold is substrate-dependent, not just modality-dependent
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

#### Track 1 ruling — Week 1 sign-off · **Status: RESOLVED, with the diagnosis revised**

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

### D-021 · `calib_group` keeps its granularity and moves into the registry
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

### D-022 · Split is fixed per patient in the registry, and must be operator-independent
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

### D-023 · One calibration group per (family, acceleration); seeds are members
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

### D-024 · `NullSpaceNoiseSampler` leaks ~4% of its "hallucination" into the measured component
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

### D-025 · `data/README.md` removed; all data documentation consolidated into `data_access.md`
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

### D-026 · Reference `split_conformal_threshold` over-certifies thin groups
**Date:** Week 2 · **Status:** OPEN · **Raised by:** Track 1 (T1.2) · **Owner:** Track 2 · **Track 1 entry:** D32

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

---

### D-027 · fastMRI `singlecoil_val` has 199 volumes but only 96 patients; the split must be by `patient_id`
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

### D-028 · Fixing D-024 reversed the direction of the naive-transfer coverage failure
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

### Template

```
### D-0NN · <one-line decision>
**Date:** Week N · **Status:** PROPOSED | ACCEPTED | OPEN | SUPERSEDED
**Proposed by:** Track N

<what was decided>

*Alternative:* <what else was considered>
*Reason:* <why>
*Cost:* <what this gives up>
*Affects:* <Blueprint items>
```
