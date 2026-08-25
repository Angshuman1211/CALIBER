# Track 2 Problems Log

Problems hit while building the Track 2 pipeline: what broke, how it showed up,
and what was done about it. Kept separate from
[decision_log.md](decision_log.md), which records *choices*. This file records
*trouble* - including the trouble that is still open.

Track 1 keeps its own equivalent in
[../Documentation/problems_and_decisions.md](../Documentation/problems_and_decisions.md).
Every entry below cross-references the decision-log id where one exists, so the
two files do not drift.

The ordering is deliberate: unresolved first, because those are the ones that
can still cost someone a week.

---

## Still open

### P-01 - The CT null projector amplifies instead of projecting

**Blocks:** any reported CT coverage curve - **See:** D-012, D-006

CT `pinv` is filtered backprojection, not the Moore-Penrose inverse, so
`P_N = I - A+A` is not a contraction. Measured on real LoDoPaB data, median
`err_free / err_total` is 1.33; for a genuine orthogonal projection that ratio
cannot exceed 1.

The damaging part is not the constant, it is the trend. `||P_N x|| / ||x||`
climbs from 0.667 at 180 views to 1.738 at 30 views - the projector degrades
monotonically *in the same direction as operator distance*. So a CT
coverage-gap-vs-distance curve could come out monotone because our own
decomposition is falling apart, not because reliability is. The two effects are
confounded, and the Week 9 curve-collapse test cannot separate them.

**Where it stands:** detected, not fixed. `null_projection_gain()` and the
`contraction_ok` health field surface it as a runner warning rather than a hard
gate, so CT development is not blocked. The fix is conjugate gradient on the
normal equations for the *decomposition* (FBP can stay as the display
reconstruction), scheduled for Week 7. Until it lands, CT numbers are plumbing
only.

**Lesson:** an approximation that looked like it would only loosen constants
(D-006) turned out to invert the inequality the free component is supposed to
satisfy. "Approximate" was not a safe thing to wave through.

---

### P-02 - The v0 label threshold marks everything as hallucinated

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

### P-03 - TotalSegmentator masks do not annotate LoDoPaB images

**Blocks:** region-level CT evaluation - **See:** D-013

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

### P-04 - Randomised spectral estimation barely separates CT operators

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

### P-05 - fastMRI access has external lead time, and a licensing question

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

*Not settled.* The transfer was staged through Kaggle rather than the 6 GB
laptop. The NYU agreement forbids third-party redistribution, and it is not
established that staging through another account is compatible with the terms
signed. If it is not, the data is unusable for a reported result no matter how
well the pipeline works. This needs a real answer, not an assumption.

---

### P-06 - Patient-level splitting is decided but not yet wired

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

## Resolved

### P-07 - fastMRI+ boxes were rasterised on the wrong grid

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

### P-08 - A truncated archive sat recorded as fetched

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
Sizes in `data/README.md` are now measured rather than estimated.

---

### P-09 - The Windows console mangled the error messages

**Fixed**

The Windows console is cp1252 and rendered em dashes and section signs as
replacement characters. That is a poor look anywhere and worse on an error path
whose entire job is to be read. Runtime output - runner, loaders,
`analyse_bank.py` - is ASCII. Documentation keeps its typography.

---

### P-10 - The 6 GB laptop does not fit the data

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

### P-11 - LoDoPaB's own sinograms make the operator grid impossible

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

### P-12 - A score bank carries no mark of where it came from

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

## Adding an entry

Number sequentially, state the symptom before the cause, and say plainly
whether it is fixed, worked around, or still open. If it is open, say what it
blocks and who has to decide. Link the decision-log id if the problem produced
a decision.
