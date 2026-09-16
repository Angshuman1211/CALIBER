# Track 2 — Problems and Decisions

The Track 2 record: what broke (**Part I**, in full) and an index of what was chosen
(**Part II**). The decisions themselves stay in the cross-track
[decision_log.md](decision_log.md) — Track 1 rules some of those entries and links to
that file, so it is not Track 2's to absorb. The index here is what keeps the two
navigable as one.

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

#### P-24 - At 256² the solver dies with a silent segfault when the machine is short of memory

**Blocks:** reliable 256² runs of any length; DPS at 256² entirely - **See:** P-16, P-10, D-033

The Week 3 D-033 sweep on the real UNet ran DDNM at 256² cleanly, then every DPS
job printed its header, loaded the checkpoint, and **ended ten seconds later with
no traceback and no output file**. The sweep log said `exit=0` for each - which
was a bug in the logging, not a result: `exit=$?` was expanded after a `$(date)`
substitution on the same line, so it recorded `date`'s status. The real exit
code, captured properly, is **139: segmentation fault**.

Isolating it took three wrong turns worth recording, because each looked like an
answer:

- **Not the DPS algorithm.** DPS at 256² with one draw and two steps ran and
  produced a result.
- **Not batch size.** Batch 1 then crashed too.
- **Not a deterministic torch bug.** The identical batch-1 256² backward passed
  twice in a row, then crashed five times out of five minutes later - and
  finally even a **forward-only** pass at 256² segfaulted, which is the exact
  operation that had just run for 7.2 hours to build the Week 3 bank.

Nothing in the code had changed between the runs that worked and the runs that
did not. The machine had.

**Measured**, peak commit charge of one UNet pass, via `GetProcessMemoryInfo`:

| resolution | pass | peak working set | peak commit |
|---|---|---|---|
| 128² | forward, 1 image | 1.45 GB | 2.99 GB |
| 128² | forward + backward, 1 image | 1.58 GB | 3.42 GB |
| 128² | forward + backward, 2 images | 2.17 GB | 4.12 GB |
| 192² | forward, 1 image | 2.27 GB | 4.20 GB |
| 256² | forward, 1 image | 3.44 GB | **5.99 GB** |
| 256² | forward, 2 images | - | segfault |
| 256² | forward, 4 images | - | `RuntimeError: DefaultCPUAllocator: not enough memory` |

and at the time of the crashes, **5.1–5.3 GB of commit was free** out of a
~26 GB limit (15.7 GB RAM plus pagefile), with other applications holding the
rest. The single-image forward peak is linear in pixel count — about 2.0 GB of
fixed overhead plus 0.061 GB per thousand pixels, which predicts the 192² point
to within 0.04 GB.

*A correction to this entry's first version:* it estimated ~8 GB for a 256²
forward, because that pass would not run at the time. Measured once headroom
allowed, it is **6.0 GB**. The estimate was 2 GB high.

**What the batch-4 row settles.** The same forward pass that segfaults at
batch 2 raises an explicit out-of-memory error at batch 4. Allocation failure
is the cause; whether it surfaces as a Python exception or a segfault depends on
where in torch's native code the failing allocation happens. 256² succeeds when
other applications leave headroom and fails when they do not; 128² fits either
way (5 of 5).

**It bites the Week 3 config directly.** The samplers pass all `n_posterior`
draws through the UNet as one batch, and `w03_mri_fixed_operator.yaml` uses four.
That is the batch-4 row: **the Week 3 config cannot run on this machine with
today's memory load** — it completed a week earlier only because more was free.

**That lever is now built (D-042).** Seeding each draw from `(seed, index)` rather
than sharing one generator across the batch decoupled the result from the batch
size, and the back-ends take `max_draws_per_batch`. Measured: the Week 3 config at
256² with four draws needs **6.0 GB chunked to one draw, against ~18 GB batched**,
and it runs. The bank is bit-identical either way, which is the property that
makes the chunking a memory setting rather than an experimental choice.

**Why it is serious rather than inconvenient.** The runner writes the bank only
at the end. The Week 3 run took 7.2 hours; had the machine's memory filled at
hour 7, it would have died with no error, no traceback and no partial output,
and the log would have looked like a job that simply stopped. Nothing in the
pipeline would have said why.

**Where it stands: mitigated, Track 2's — all three are in** (D-038, D-040,
D-041), and D-042 removes the constraint for this config rather than only
reporting it: **the Week 3 config runs on this machine again** when its draws are
chunked. What remains is the underlying scarcity — a 256² pass needs ~6 GB
whatever the batching, so a long run still competes with whatever else is open.
In order of cost:

1. **Operational now:** close memory-heavy applications before a 256² run, and
   capture exit codes correctly - a 139 is a memory problem until proven
   otherwise.
2. **A preflight check** in the runner that compares free commit against an
   estimate for the configured resolution and batch, and refuses to start a job
   that cannot finish. **Done (D-041):** the estimate is
   `max(2.8, 2.0 + 6.1e-5 x pixels x draws)` GB, fitted to eleven measured points
   and accurate on all of them to within 0.4 GB; it calls both observed failures.
   Overridable with `run.skip_memory_check`, and silent where the available
   memory cannot be read.
3. **Incremental bank writes**, so a job killed at hour 7 keeps seven hours of
   rows. This is the one that turns a silent total loss into a partial result,
   and it also serves P-16, where every long job is a risk. **Done (D-038):**
   `run.checkpoint_every: N` rewrites every finished row to a partial bank each
   N items, atomically; `read_bank` refuses a partial bank unless asked. Tested
   by crashing a run part-way and recovering exactly the rows it finished — and
   that test fails against the previous runner.

**Resume is also in (D-040):** a restarted run of the same config continues from
its checkpoint, and the result is tested to equal an uninterrupted run exactly.
**The preflight check is in too (D-041):** the runner estimates the run's peak
commit charge, compares it with what the machine has, and refuses to start a run
that cannot finish - overridable, because the estimate is a fit to one machine.
All three mitigations are now built.

**No existing config sets `checkpoint_every`**, deliberately: adding it changes
the config hash and therefore the run id of a bank that already exists. Set it
with `--set run.checkpoint_every=10` on any long run, and in every new config.

This also bears on D-033: DPS was measured on the real UNet at **128²** instead,
because at 256² it cannot run here at all. Whether its step budget transfers to
256² is unmeasured.

---

#### P-23 - Two of the three Week 3 baselines were in-sample, and held out, 30 volumes cannot measure coverage at all

**The bug is fixed; the consequence is open** - **See:** P-19, P-16, P-21, D-022, D-027

**The bug.** P-19's fix to `analyse_bank.py` held out the *naive* threshold -
fitted on `split=calib`, scored on `split=test` - but still fitted the
matched-per-operator and group-conditional thresholds on the evaluation rows
themselves. Split conformal scored on its own calibration rows lands on nominal
by construction, which is why the T2.3 record reported both at **exactly
0.900 at every operator**. It also meant a held-out method was being compared
against two in-sample ones. That number was wrong and was written into the
Week 3 record as a result.

What gave it away was the exactness: held-out conformal coverage lands *near*
nominal, not *on* it, three operators in a row.

**Held out properly** (tau=3.0, calibration operator R=2, thresholds fitted on
13,824 `split=calib` rows from 18 volumes, scored on 9,216 `split=test` rows
from 12 volumes):

| method | R=2 | R=4 | R=8 |
|---|---|---|---|
| naive transferred | 0.914 | 0.915 | 0.955 |
| matched / group-conditional | 0.914 | 0.874 | **0.779** |

**Then the part that matters more.** A volume-level bootstrap (2,000 resamples)
puts 95% intervals on those numbers:

| method | operator | coverage | 95% CI |
|---|---|---|---|
| matched | R=2 | 0.914 | [0.000, 1.000] |
| matched | R=4 | 0.874 | [0.509, 0.985] |
| matched | R=8 | 0.779 | [0.457, 0.919] |
| naive | R=8 | 0.955 | [0.871, 0.989] |

**With 12 test volumes, per-operator coverage is essentially unmeasured.** The
bank cannot show that coverage holds - the in-sample 0.900 falsely suggested it
did - and it cannot show that it fails. At tau=3.0 the positives concentrate in
a handful of volumes, which is how R=2's interval spans the entire unit
interval. The bootstrap resamples volumes, and volumes from one patient are
correlated (D-027), so the true intervals are if anything *wider*.

**What stands and what does not.** P-21 stands: it rests on the label base rate
over all 23,040 regions and on flag area, which is a fraction of every region
rather than of the few positive ones, and neither is a small-sample coverage
estimate. Every *coverage* number in the Week 3 record does not stand as
evidence of anything, in either direction.

**Why this is open, and whose.** Scaling SE ~ 1/sqrt(n) from these intervals,
order of magnitude only:

| target CI half-width per operator | test volumes | items per operator (60% calib) | CPU hours per operator |
|---|---|---|---|
| ~0.10 | ~70 | ~170 | ~6.5 |
| ~0.05 | ~270 | ~700 | ~26 |

Two walls, not one. **Compute** (P-16): the ~0.05 row is ~26 CPU hours *per
operator*, so a Week 4 grid at useful precision is not a CPU job. **Data**, and
this one no hardware fixes: the unit of exchangeability is the patient, and
fastMRI `singlecoil_val` has **96**. At the default split that is ~44 test
patients, which caps precision near SE ~0.06 per operator no matter how long the
solver runs. The protocol (T1.4, Track 1) has to be designed against that
ceiling - pooled estimates, fewer and larger groups, or coverage reported with
intervals rather than as point values.

**The regression is pinned.** `test_every_threshold_is_fitted_on_calibration_
rows_and_scored_on_test_rows` spies on every fitting and scoring call and asserts
the split of the rows each receives. It was checked against the buggy version,
where it fails.

---

#### P-21 - The ground-truth label is itself operator-dependent, so the coverage-failure experiment cannot mean what it is supposed to mean

**Blocks:** the O3 reading of any coverage curve - **See:** D-034, D-028, D-020, P-02

On the Week 3 fastMRI bank (23,040 regions, DDNM, three accelerations), naive
transfer does **not** lose coverage. It gains it:

| tau | metric | R=2 (calib) | R=4 | R=8 |
|---|---|---|---|---|
| 1.75 | coverage | 0.851 | 1.000 | 1.000 |
| 1.75 | flag area | 0.676 | 0.999 | 0.990 |
| 3.00 | coverage | 0.914 | 0.915 | 0.955 |
| 3.00 | flag area | 0.244 | 0.502 | 0.668 |

The direction is robust - it holds at every tau tested (1.75, 2.5, 3.0, 4.0).
Coverage never collapses; **flag area explodes** instead. The certificate stays
valid and becomes useless, which is a failure of efficiency, not of validity.

**Why.** Both the score and the label move with the operator, in the same
direction:

| operator | label positive rate (tau=1.75) | median err_free/free_content | mean s_free |
|---|---|---|---|
| R=2 | 0.401 | 1.528 | 0.486 |
| R=4 | 0.681 | 2.296 | 0.847 |
| R=8 | 0.772 | 2.677 | 1.046 |

A threshold calibrated where positives are rare and scores are low, then
deployed where positives are common and scores are high, flags nearly
everything - and therefore catches nearly every positive. Coverage is achieved
trivially.

**This is not a calibration failure and not a pipeline failure.** It is a
property of the *criterion*. A coverage comparison across operators is only
interpretable if the quantity being covered means the same thing at each
operator, and at present it does not: the base rate of "hallucinated" nearly
doubles between R=2 and R=8. D-034 flagged this risk on M4Raw, where the trend
was mild (median 1.480 -> 1.645). On fastMRI knees with a face prior it is
severe (1.528 -> 2.677), and it has crossed from a caution into a blocker.

**Where it stands: open, and it is Track 1's to rule on (O2).** Track 2's part
is measured and delivered. The options, none of them Track 2's to choose:

1. **Set tau per calibration group** so the label base rate is matched across
   operators. Makes coverage comparable, but "hallucination" then means
   something slightly different at each acceleration, which has to be defended.
2. **Make flag area the headline** rather than coverage, with coverage reported
   alongside. Sec. 15.2 already requires flag area as a reported metric, and the
   over-flagging result above is a perfectly good demonstration of operator
   shift mattering - it is just not the *coverage* story O3 is written around.
3. **Redefine the criterion to be operator-invariant by construction.** Cleanest
   scientifically, most expensive, and it invalidates nothing already cached
   because the bank stores continuous quantities (D-002).

**Not a wasted run.** The bank is real, cached, and re-thresholdable without a
GPU (Sec. 17), so whichever option Track 1 picks is a re-analysis, not a
re-reconstruction.

**Read with P-23.** The coverage rows in the first table above are point
estimates from 12 test volumes, and P-23 shows intervals that wide cannot carry a
claim. The *diagnosis* does not depend on them: the label base rate
(0.401 -> 0.772) is measured over every region in the bank, and flag area is a
fraction of all regions, not of the few positive ones.

---

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

**Where it stands:** the relative criterion agreed at the Week 1 sign-off is now
**implemented and measured** (D-034), and what remains is Track 1's choice of
tau, not a missing component. Until Week 2 the form had been agreed and the
denominator column (`free_content`) written, but the only criterion anything
could actually apply was still the broken absolute one.

Measured on the 384-region real-prior bank, the relative form does what the
absolute one cannot - it separates:

| tau | 1.00 | 1.25 | 1.50 | 1.75 | 2.00 | 2.50 |
|---|---|---|---|---|---|---|
| positive rate | 100% | 92.4% | 60.2% | 25.5% | 10.7% | 0.8% |

against a flat 100% for `err_free > 0.10`. A usable operating range is roughly
tau 1.75-2.0. Two cautions travel with it: the ratio exceeds 1 in *every*
region, which is a face prior on brain anatomy showing through as much as it is
a criterion; and the median ratio rises with acceleration (1.480 at R=4, 1.645
at R=8), so a single global tau is not operator-neutral and may have to be set
per calibration group.

What was already established stands: the criterion cannot be a shared absolute
constant - it fails to transfer across modalities *and* across substrates within
one modality.

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

**See:** D-017, [T2.1 Appendix A.1](T2.1_data_and_operator_pipeline.md#appendix-a--data-access-and-provenance)

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

#### P-15 - A checkpoint's scheduler config is a 15-byte "Entry not found"

**See:** D-016, P-08

`checkpoints/ddpm-celebahq-256/scheduler/scheduler_config.json` is 15 bytes long and
contains the literal text `Entry not found`. The download failed and the error page was
saved with the filename it was asked for, so the directory *looks* complete: config,
weights, scheduler, all present.

```
-rw-r--r--  790        config.json
-rw-r--r--  454853117  diffusion_pytorch_model.bin
-rw-r--r--  15         scheduler/scheduler_config.json   <- "Entry not found"
```

The 454 MB weights are fine and load correctly (113.7M parameters). Only the scheduler
config is a stub, which is the worst case: had the *weights* failed this way, loading
would have raised immediately.

**This is P-08 again** — a failed fetch recorded as a success — in a different corner of
the project, which is what makes it worth its own entry rather than a footnote. The
lesson generalises: an artefact's presence is not evidence of its content, and the check
has to open the file.

**Not worked around, designed around.** The DDPM schedule (1000 steps, betas linear in
[1e-4, 0.02]) is written out explicitly in `solvers/denoiser.py` as named constants
rather than read from any config. The sampler therefore does not depend on that file at
all, and the numbers that scale every `x0` estimate are visible in source instead of
buried in JSON. `NoiseSchedule` is pinned against the closed form by
`test_predict_x0_inverts_the_forward_noising_exactly`, so a wrong schedule fails loudly
rather than producing plausible-looking but systematically mis-scaled reconstructions.

---

#### P-16 - No GPU: torch here is the CPU build

**See:** P-10, D-029, Sec. 17 risk R7

`torch.cuda.is_available()` is **False** — the installed torch is `2.13.0+cpu`, so the
6 GB RTX 3050 the compute budget was written around is not being used at all. Measured
cost of one UNet forward at 128²: **~0.6 s**, so:

| job | cost |
|---|---|
| one 100-step DDNM draw | ~1 min |
| one 100-step DPS draw | ~3 min (backward pass every step) |
| the Week 2 smoke run actually executed (6 items x 2 operators x 3 draws x 12 steps) | 119 s |

This does not block T2.2 — the back-ends work and produce banks — but it does bound
Week 3 and Week 4. A full operator grid at a useful step count is not affordable on CPU,
and the honest options are a GPU-enabled torch build, the college machine (RTX 4500 Ada,
24 GB), or a permanently reduced grid with the reduction stated in Experimental Setup
rather than left implicit.

The immediate mitigation is D-029: the samplers are testable against an analytic
denoiser, so correctness work does not wait on compute.

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
Sizes in the inventory ([T2.1 Appendix A](T2.1_data_and_operator_pipeline.md#appendix-a--data-access-and-provenance) §6) are now measured rather than estimated.

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

#### P-17 - Below its step budget, DPS's error is in the component it cannot be in

**Found and quantified** - **See:** D-033, D-031, P-16

The Week 2 Stage-2 audit was written to confirm something the scoring layer had
been assuming since Week 1: that reconstruction error lands in the null space.
`s_free` scores the free component, `err_free` labels it, and both are only
measuring invented structure if what the solver gets wrong is what the operator
could not see.

For DDNM and FlowDPS it holds immediately — both replace the range component of
every estimate, so at 8 steps the null share of their error is already 0.98 and
0.97. **DPS at 8 steps is 0.20.** Four fifths of its error sits in the measured
component: the sampler has not fitted the data yet, so scoring its free
component measures the step count rather than the prior.

| DPS steps | 8 | 24 | 48 | 100 | 200 |
|---|---|---|---|---|---|
| null share of error | 0.20 | 0.50 | **0.99** | 0.99 | 0.99 |
| data-consistency error | 0.71 | 0.36 | **0.038** | 0.039 | 0.036 |

Nothing was broken — DPS is a soft constraint and this is what a soft constraint
does before it converges. What was wrong was that no gate existed, so a DPS bank
at the smoke config's 24 steps would have looked exactly like a valid one. The
threshold is now written down as a precondition (D-033) and the audit script
warns when a run fails it.

Two consequences. The step budget is **per solver family**, not a global config
constant, so the two families are not comparable at equal `n_steps` — which the
Week 10 solver ablation has to hold fixed correctly or it will be measuring
convergence. And DPS needing ~6x DDNM's budget, on top of its backward pass per
step, is most of why the Week 4 grid will be DDNM (P-16).

---

#### P-18 - A face prior on knee MRI produced a bank stamped `reportable: true`

**Fixed** - **See:** D-035, D-018, D-016, P-12

The first fastMRI run of Week 3 wrote `"reportable": true` into its manifest.
The substrate was real (fastMRI, D-017 resolved) and the sampler was a real
generative prior (DDNM), so both of D-018's gates passed. The prior was
`ddpm-celebahq-256` - a **face model reconstructing knee anatomy**.

Nothing was wrong with the run. What was wrong is that the code did not know
what the prose already said: D-016 orders the natural-image prior deliberately
and the T2.2 write-up lists it as one of the reasons nothing is reportable, but
reportability was computed from substrate and solver family alone, and neither
of those notices the prior's training domain.

This is P-12 again, one level up. P-12 was "a score bank carries no mark of
where it came from"; D-018 fixed it for the substrate and the sampler and
stopped there. Two gates out of three passing produced a green light.

**The fix** adds the third gate (D-035): `PRIOR_DOMAINS` maps checkpoint
basenames to what they were trained on, `ADMISSIBLE_DOMAINS` says what each
modality accepts, and the runner refuses to call a bank reportable unless the
prior matches. It **fails closed** - an unregistered checkpoint is admissible
nowhere - and `prior_domain` is now recorded in every manifest, so existing
banks can be audited rather than re-run. The Week 3 run now says:

```
[caliber] NOT REPORTABLE   prior domain 'natural_image' is not appropriate for 'mri' (D-016, P-18)
```

**Worth stating plainly:** this was caught with one smoke run of two items,
because the missing `NOT REPORTABLE` line was conspicuous. Had it been noticed
three weeks later, every bank produced in between would have carried a false
flag into the write-up.

---

#### P-19 - `analyse_bank.py` had been broken for two weeks and nothing noticed

**Fixed** - **See:** D21, D26, D27, D-022, D-027

The script that produces the two Week 3 baselines raised `TypeError:
RegionTruth.__init__() missing 1 required positional argument: 'free_content'`
on every bank. `free_content` was added to `RegionTruth` when A1 / D21 landed in
Week 1; this caller was never updated.

It went unnoticed because **no test executed the script**. The unit tests
exercise the library the script calls, which is a different thing - every
function it uses was individually green.

Fixing the crash exposed two further defects that were never going to raise an
exception, and both would have biased the Week 3 baselines:

- **Calibration and evaluation shared rows.** The threshold was fitted on the
  calibration operator's rows and then evaluated on a frame still containing
  them, so coverage *at the calibration operator* - the reference point the
  whole naive-transfer story is read against - was in-sample and optimistic.
  It now calibrates on `split=calib` and evaluates on `split=test`, which is
  patient-disjoint by construction (D-022, D-027), and warns loudly when a bank
  carries only one split.
- **The score was an unstandardised `1,1,1` sum.** `score.py` says in its own
  docstring that the three components differ by an order of magnitude, so this
  reduces to `score = s_free` - the combination was nominally three-component
  and effectively one. It now uses T1.1's ratified standardised affine rule with
  scales frozen on the calibration operator (D26, D27), and prints the exact
  combination above every table.

**The real fix is the test.** `tests/test_analyse_bank.py` builds a bank on disk
and runs `main()` over it, which is what was missing. A structural test also
asserts that every `RegionTruth` field has a bank column, so the *next* field
added to the criterion cannot repeat this.

---

#### P-20 - An unanchored `data/` ignore rule shipped a `main` that could not be imported

**Fixed** - **See:** D-036 (Track 1's entry for the same bug), D-025 - reported by Track 3, found independently by Track 1

`main` raised `ModuleNotFoundError: No module named 'caliber.data.splits'` on
`import caliber.data`. Because `caliber/data/__init__.py` imports it too, the
whole package was unimportable and pytest collection failed for every test file.
Anyone cloning the repository got something that did not work at all.

`.gitignore` line 14 was `data/`, meant for the ~74 GB dataset directory at the
repository root. **A trailing slash does not anchor a pattern** - `data/` matches
a directory named `data` at *any* depth, so it also matched `src/caliber/data/`.
Five of the six files in that package had been committed before the rule reached
that form and stayed tracked; `splits.py` was added later, in the Week 1
close-out, and was swallowed silently. It never appeared in `git status`,
because ignored files do not.

**Local test suites passed the entire time**, because the file exists on the
machine that wrote it. That is exactly why it survived two weeks: nothing in a
local workflow can see it. It took someone checking out the repository
elsewhere - Track 3, working on T3.4 - to surface it.

**Whose it was, accurately.** Two failures had to coincide. The ignore rule was
widened by Track 1 in the D-025 consolidation, which replaced `data/*` plus
`!data/README.md` with a bare `data/`; the original contained an internal slash
and was therefore anchored, the replacement was not (Track 1 records this in
D-036 as self-inflicted). And Track 2 added `splits.py` after that change and
never checked that it had actually been committed. Neither alone would have
shipped a broken `main`. An earlier Track 2 note called this Track 2's bug
outright; that was half the story.

Track 1 hit the same failure independently, anchored the rule themselves, and
wrote a stand-in `splits.py` to keep working - which is what led to P-22.

**A second consequence of the same rule.** `ruff` also respects `.gitignore`, so
the linter had been skipping all six files in that package for as long as the
rule existed. Anchoring it revealed two pre-existing `E501`s in `fastmri.py`,
now wrapped.

**The fix** anchors the pattern to `/data/`, commits `splits.py` unchanged, and
carries a comment recording the trap. Verified on a clean `git archive` export
rather than the working copy that hid the bug: `import caliber.data` succeeds,
197 tests pass, `ruff check src tests scripts` is clean. Every `.py`, `.yaml`
and `.yml` under `src`, `tests`, `scripts` and `configs` was audited for the same
loss - `splits.py` was the only casualty.

**Lesson worth keeping:** a green local suite says nothing about what is in the
repository. The check that matters is an export of the committed tree, and
nothing in the project performed one until this.

---

#### P-22 - Two split implementations passed every test and disagreed on half the patients

**Fixed** - **See:** D-037, P-20, D-036, D-022, D-027

While `splits.py` was missing from `main` (P-20), Track 1 wrote a stand-in to the
same contract so their work could continue - a reasonable thing to do, and its
docstring said to take Track 2's file when it landed. It used the **same salt**,
passed **all 21 split tests** and the full suite. On the 96 real fastMRI patients
it put **48 on the opposite side** from the version that built the Week 3 bank.

The tests could not see it because they check what a correct split must
*satisfy* - deterministic, operator-independent, stable as patients are added,
balanced - and never *which side any patient lands on*. Two implementations can
satisfy every one of those properties and share almost no assignments. Banks
built from the two would have been silently non-comparable, which is exactly what
D-022's "identical across every run" requirement forbids.

Resolved at the merge by keeping Track 2's version (D-037), for the two reasons above.
**Anything Track 1 or Track 3 built from a `split: auto` bank on `main` between
PR #21 and the Week 3 merge used the other split**; nothing in git can show
whether that happened, because run outputs are not committed.

**The fix makes the mapping itself the contract.** `tests/test_split_golden.py`
pins the assignment of 40 fixed synthetic keys at two fractions, and the SHA-256
of the real 96-patient assignment - a digest, so no patient identifier is
committed and the NYU agreement is not engaged. The real-data test skips cleanly
where fastMRI is absent. Checked against Track 1's version: 3 of 5 fail, as they
should. Redrawing the split on purpose is still possible and now has to be
explicit: bump `DEFAULT_SALT`, regenerate the pins, record it.

---

#### P-25 - No bank the project had produced could be regenerated from its config

**Fixed going forward; not recoverable backward** - **See:** D-039

The runner's own docstring says a run is "fully determined by its config plus its
seed". It was not. Each (item, operator) seed came from

```python
seed + abs(hash((item.sample_id, item.slice_idx, desc.operator_id))) % 10**6
```

and Python salts `hash()` for strings **per process** unless `PYTHONHASHSEED` is
set. Nothing set it. Three invocations for the same item gave **77818, 549586 and
104573**. Every run therefore drew different noise and different posterior samples
for identical inputs. `scripts/stage2_decomposition.py` copied the same line.

**Why the test suite never saw it.** Every determinism test compared two runs
*inside one pytest process*, where the salt is shared, so the bug could not show.
Checkpointing's "does not change the bank" test (D-038) passed for the same reason.
A determinism test that runs in a single process cannot detect a per-process
source of non-determinism — the check has to cross a process boundary.

It was found reading `stage2_decomposition.py` line by line for an unrelated
cosmetic fix.

**Consequences, stated plainly.**

- **No existing bank can be regenerated**, including the 7.2-hour Week 3 bank. Its
  rows are still correct and interpretable - the `seed` column records the value
  each row was actually drawn with - but re-running its config produces different
  draws, not the same bank.
- The Week 10 reproducibility package (O7) was built on a false premise until now.
- Numbers quoted across two separate runs of "the same" config were never
  guaranteed to be comparable at the level of individual draws. Aggregates over many
  items are unaffected in expectation; single-item comparisons are not.

**The fix** (D-039) derives the seed from a BLAKE2b digest of
`sample_id|slice_idx|operator_id` - the approach `splits.py` already used correctly.
Pinned by `tests/test_reproducibility.py`, whose central test runs the pipeline in
**two separate processes with deliberately different `PYTHONHASHSEED` values** and
requires identical banks. It fails against the old derivation and passes against
the new one.

---

### Adding an entry

Number sequentially, state the symptom before the cause, and say plainly
whether it is fixed, worked around, or still open. If it is open, say what it
blocks and who has to decide. Link the decision-log id if the problem produced
a decision.

---

## Part II — Decisions index

The decisions themselves live in **[decision_log.md](decision_log.md)**, which is a
*cross-track* log, not Track 2's alone: D-021, D-022, D-023, D-025 and D-026 were ruled
by Track 1, and Track 1's own documents link to it. Copying the entries here would create
a second source of truth and two chances to drift, which is the problem this file exists
to avoid.

What lives here instead is the index — 42 entries, so a problem in Part I can be
traced to the decision that settled it without opening the log.

| ID | Decision | Week | Status |
|---|---|---|---|
| [D-001](decision_log.md) | Score bank stores components, not a combined score | Week 1 | ACCEPTED |
| [D-002](decision_log.md) | Ground-truth labels are derived at analysis time | Week 1 | ACCEPTED |
| [D-003](decision_log.md) | `err_free` recorded separately from `err_total` | Week 1 | ACCEPTED |
| [D-004](decision_log.md) | One solver per run, enforced in config and schema | Week 1 | ACCEPTED |
| [D-005](decision_log.md) | CT uses our own projector, not LoDoPaB's supplied sinograms | Week 1 | ACCEPTED |
| [D-006](decision_log.md) | CT range/null-space split is approximate, and says so | Week 1 | ACCEPTED |
| [D-007](decision_log.md) | CT `null_space_dim` is a dimension-count bound | Week 1 | OPEN |
| [D-008](decision_log.md) | Single-coil MRI via root-sum-of-squares | Week 1 | ACCEPTED |
| [D-009](decision_log.md) | Region definition v0 is a 16px patch grid | Week 1 | **RESOLVED — Week 1 sign-off** |
| [D-010](decision_log.md) | Working default distance is `null_space_projection` / `ambient` | Week 1 | **RESOLVED — locked as primary** |
| [D-011](decision_log.md) | Randomised spectral estimation barely separates CT operators | Week 1 | OPEN |
| [D-012](decision_log.md) | The FBP-based CT null projector amplifies; it must be replaced | Week 1 | RESOLVED — fixed |
| [D-013](decision_log.md) | TotalSegmentator masks do not align with LoDoPaB images | Week 1 | RESOLVED (Week 1 sign-off) — Option A adopted |
| [D-014](decision_log.md) | LoDoPaB must be split by patient, not by slice | Week 1 | ACCEPTED |
| [D-015](decision_log.md) | M4Raw as an open stopgap while fastMRI access is pending | Week 1 | ACCEPTED |
| [D-016](decision_log.md) | Natural-image diffusion prior wired before medical priors | Week 1 | ACCEPTED |
| [D-017](decision_log.md) | fastMRI acquired by direct download, staged off-machine | Week 1 | RESOLVED (Week 1 close-out) |
| [D-018](decision_log.md) | Reportability is a property of the bank, not of whoever ran it | Week 1 | ACCEPTED |
| [D-019](decision_log.md) | fastMRI+ boxes live on the 320x320 reference grid, not on k-space | Week 1 | ACCEPTED — fixed |
| [D-020](decision_log.md) | The v0 label threshold is substrate-dependent, not just modality-dependent | Week 1 | OPEN — needs the Week 2 O2 decision |
| [D-021](decision_log.md) | `calib_group` keeps its granularity and moves into the registry | Week 1 | RESOLVED |
| [D-022](decision_log.md) | Split is fixed per patient in the registry, and must be operator-independent | Week 1 | RESOLVED |
| [D-023](decision_log.md) | One calibration group per (family, acceleration); seeds are members | Week 1 | RESOLVED |
| [D-024](decision_log.md) | `NullSpaceNoiseSampler` leaks ~4% of its "hallucination" into the measured component | Week 2 | RESOLVED — fixed, and it moved a headline number |
| [D-025](decision_log.md) | `data/README.md` removed; all data documentation consolidated into `data_access.md` | Week 2 | DONE |
| [D-026](decision_log.md) | Reference `split_conformal_threshold` over-certifies thin groups | Week 2 | OPEN |
| [D-027](decision_log.md) | fastMRI `singlecoil_val` has 199 volumes but only 96 patients; the split must be by `patient_id` | Week 1 | ACCEPTED — fixed |
| [D-028](decision_log.md) | Fixing D-024 reversed the direction of the naive-transfer coverage failure | Week 1 | ACCEPTED — recorded, not yet re-measured on a real solver |
| [D-029](decision_log.md) | Solver algorithms are separated from the network that drives them | Week 2 | ACCEPTED |
| [D-030](decision_log.md) | The flow-matching family uses diffusion-derived velocities, not a FlowDPS checkpoint | Week 2 | ACCEPTED — scope limitation, must appear in the write-up |
| [D-031](decision_log.md) | Guidance strength is a fixed solver constant, never tuned per operator | Week 2 | ACCEPTED — **the value 5.0 is wrong for the real prior**; optimum above 20, unbracketed (Week 3) |
| [D-032](decision_log.md) | Stage-2 decomposition diagnostics are a run artefact, not score-bank columns | Week 2 | ACCEPTED |
| [D-033](decision_log.md) | The null-error share is a per-solver step-budget precondition on a bank | Week 2 | ACCEPTED — re-measured on the real UNet; DPS diverges at 8 steps, admissible from 24 (Week 3) |
| [D-034](decision_log.md) | Relative criterion implemented and its tau curve measured; tau stays Track 1's | Week 2 | ACCEPTED — Track 2's half discharged |
| [D-035](decision_log.md) | The prior's training domain is a third reportability gate | Week 3 | ACCEPTED |
| [D-036](decision_log.md) | A bare `data/` in .gitignore also ignored `src/caliber/data/` | Week 2 | FIXED — Track 1's entry; see P-20 |
| [D-037](decision_log.md) | Track 2's `splits.py` is canonical, and the split mapping itself is pinned | Week 3 | ACCEPTED — Track 1 to confirm no bank used the stand-in |
| [D-038](decision_log.md) | An interrupted run leaves a partial bank that readers refuse by default | Week 3 | ACCEPTED |
| [D-039](decision_log.md) | Item seeds come from a stable digest, not Python's salted `hash()` | Week 3 | ACCEPTED |
| [D-040](decision_log.md) | A restarted run resumes from its checkpoint — only for the same config | Week 3 | ACCEPTED |
| [D-041](decision_log.md) | A run that cannot fit in memory is refused before it starts | Week 3 | ACCEPTED |
| [D-042](decision_log.md) | Each posterior draw is seeded independently, so draws can be batched in any size | Week 3 | ACCEPTED |
