# Week 1 Kickoff — Interface Sign-Off Packet

**Owner:** Track 2 · **Required by:** the Week 1 card ("Kickoff integration
meeting — agree on score-bank schema and operator registry; establish a
cross-track decision log and adjudication process", T1.5, Sec. 19)

The Week 1 deliverable is *"shared interface spec v0 **agreed by all three
tracks**"*. Both specs are written and implemented; what is missing is the
agreement. This packet is the agenda: every question that needs a cross-track
answer, in one place, each with Track 2's recommendation and — more usefully —
what it costs to defer.

Read [score_bank_spec.md](score_bank_spec.md) and
[operator_registry_spec.md](operator_registry_spec.md) first. Decisions taken
here get written into [decision_log.md](decision_log.md) the same day.

---

## A. What is already settled

Accepted in Week 1 and implemented; raise now or treat as closed: D-001
(components not a combined score), D-002 (labels derived at analysis time),
D-004 (one solver per run, enforced), D-005 (our own CT projector), D-006 (CT
split is approximate and says so), D-008 (single-coil RSS), D-014 (split by
patient, never by slice), D-015 (M4Raw as stopgap), D-016 (natural-image prior
first).

Track 2's infrastructure is ready: operators for both modalities with exact
adjoints and health checks, all five distance measures, score-bank schema and
validator, region grid and scoring, evaluation harness, config system, runner,
66 tests. The CT path runs end to end on real LoDoPaB data.

---

## B. Sign-off item 1 — score-bank schema v0

Bumping `SCHEMA_VERSION` after Week 3 means migrating existing banks, so the
cost of these answers rises sharply once real production starts.

| # | Question | Track 2 recommendation | Cost of getting it wrong |
|---|---|---|---|
| B1 | **Region definition.** v0 is a 16px patch grid. Does the O2 risk definition need anatomically derived regions? | Keep the patch grid for v0; revisit in Week 2 with O2. | If regions become anatomical, `region_area` stops being constant and every region-level rate becomes weighted — that changes the risk definition itself, not just the table. **Track 1 must answer in Week 2.** |
| B2 | **`calib_group` granularity.** Currently modality/family/severity. | Keep, but move the definition out of `runner.calibration_group` into the registry so Track 1 owns it without touching the runner. | Too fine starves groups of calibration data (the declared Tool 1 trade-off); too coarse mixes regimes that genuinely differ. |
| B3 | **Calibration/test split.** Currently a column, assigned per run. | Fix split assignment per `sample_id` in the registry, so it is identical across every run and modality. | A per-run split makes two banks non-mergeable and silently breaks the Week 10 merge. Interacts with **D-014**: assignment must be by patient/volume. |
| B4 | **Multi-coil MRI.** v0 reduces coils by RSS (D-008). | Confirm and state it in Experimental Setup. | Not a defect, a narrowing: the studied operator is single-coil Cartesian undersampling. It needs to be written down, not discovered by a reader. |

## C. Sign-off item 2 — operator registry v0

| # | Question | Track 2 recommendation | Cost of getting it wrong |
|---|---|---|---|
| C1 | **Primary distance measure** (T1.3, locked Week 3). | `null_space_projection`, `ambient` normalisation. All five are implemented, so this is cheap to change until Week 3 and expensive after. | See **D-011**: `spectral_divergence` is a poor candidate on the evidence we have and should not be chosen without new evidence. |
| C2 | **`calib_group` as a registry field?** | Yes — same as B2. | Grouping is Track 1's concern; it should not require a Track 2 code change. |
| C3 | **Mask seeds.** Is a mask family at fixed acceleration one group or several? | One group per (family, acceleration); seeds are members of it. | Determines both the grouping and what "a mask-family shift" even means in the Week 4 grid. |
| C4 | **CT spectral estimation budget.** How many singular values does L2 need? | Cannot be answered by Track 2 alone — see **D-011**. | Track 3 needs this by Week 6, when L2 starts. |

## D. Decisions that need a call now

### D-013 — the only one that is blocking. **Needs a Week 1 answer.**

TotalSegmentator's masks segment TotalSegmentator's own CT volumes, not
LoDoPaB's; they are different cohorts, so LoDoPaB slices cannot be annotated
with them. Without region annotations the CT flag cannot be evaluated on
whether it fires *where it matters*, and Front II stops mirroring Front I.

**Track 2 recommends Option A:** make TotalSegmentator the CT substrate. Each
of its 1,228 subjects ships `ct.nii.gz` plus 117 aligned organ masks, which is
exactly the image-plus-region-annotation pairing MRI has. Because D-005 already
means we re-project ground-truth images with our own operator, no projection
data is needed at all.

Deciding this determines the Week 7 loader target and which ~49 GB of archives
can be deleted — and disk is currently the binding constraint (30 GB free, a
full TotalSegmentator extraction needs 30.1 GB).

### Deferrable, but state the default explicitly

- **D-003** — does the Week 2 criterion use `err_free`, `err_total`, or both?
  Default is `err_free`. **Interacts with D-012**: on CT, `err_free` is
  currently dominated by FBP error, so an `err_free`-only criterion is not yet
  meaningful in the second modality.
- **D-007** — CT `null_space_dim` is a dimension-count bound, not a rank.
  Track 3 must state what L2 requires before the scaling claim is written down,
  or the write-up has to say the claim is about a proxy.
- **D-009 / D-010** — resolved by B1 and C1 above.

## E. Findings referred to other tracks

Both were raised by Track 2 from measurements, and neither is Track 2's to
resolve. This is the adjudication process in Sec. 19 running for the first
time.

**D-011 → Track 3, needed by Week 6.** Randomised spectral estimation barely
separates CT operators: the fitted decay exponent moves 1.688 → 1.676 between
180 and 20 views. What distinguishes these operators is the decay tail and the
null space, which is what the estimator resolves worst. L2 relates the bound to
null-space dimension *and spectral decay*; if the decay term cannot be
estimated at this scale, L2 either rests on `null_space_dim` alone or must
declare its spectral term an estimate of unestablished accuracy.

**D-020 → Track 1, needed for the Week 2 O2 decision.** The v0 criterion
`err_free > 0.10` labels **100% of regions hallucinated** on real MRI k-space,
making region FDR identically zero and the flag vacuous. The projector is not
at fault here — MRI's `err_free`/`err_total` ratio is 0.97, within what an
orthogonal projection allows, unlike CT's 1.33. The threshold is simply
calibrated to phantoms, and real anatomy carries far more null-space content.
D-012 showed the constant does not transfer across modalities; this shows it
does not transfer across substrates within one modality. **The Week 2 criterion
cannot be an absolute constant** — it needs to be relative, or defined so the
positive rate is controlled rather than observed.

**D-012 → blocks Week 7, fix owned by Track 2.** The FBP-based CT null
projector amplifies (`‖P_N x‖/‖x‖` = 1.74 at 30 views, where an orthogonal
projection can never exceed 1), and it degrades **monotonically in the same
direction as operator distance**. A CT coverage-gap-vs-distance curve could
therefore come out monotone because our decomposition degrades, not because
reliability does. The two are currently confounded and the Week 9
curve-collapse test cannot separate them. Fix is a conjugate-gradient
pseudo-inverse for the decomposition, Week 7, before any CT coverage curve is
reported. This is also the empirical answer to what D-006 left open: the
approximation does not merely loosen constants, it inverts the inequality the
free component is supposed to satisfy.

## F. Status going into Week 2

| Item | State |
|---|---|
| Repo scaffolding | done — package, config-driven runner, CLI, 66 tests, CI-clean lint |
| Retrospective-undersampling pipeline | done — 3 MRI mask families, CT sparse-view and limited-angle, registry, health checks |
| fastMRI+ annotations | downloaded (969 files, 16,154 boxes) and parsed by the loader |
| fastMRI raw k-space | links obtained, download under way off-machine (D-017) |
| M4Raw stopgap | done — 240 volumes, pipeline verified end to end on real k-space |
| Solver back-ends | **Week 2** — DPS, DDNM, FlowDPS all raise `NotImplementedError`; interface is fixed |

**Week 2 entry condition:** nothing above blocks it. Solver wiring (T2.2) can
start against the natural-image prior per D-016, then move to medical weights.

---

## G. Decision record

**Track 1 signed off on all six of its items.** Full reasoning for each is in
`Documentation/problems_and_decisions.md` (D19–D24); cross-track entries updated in
`decision_log.md`. Note the two logs number differently — `D-0NN` here, `DNN` there.

| Item | Decision | Owner | Logged as |
|---|---|---|---|
| B1 region definition | **Confirmed as proposed** — patch grid is the risk unit; annotations are an evaluation overlay, never the region definition. `region_area` stays constant. Region size is a T1.4 ablation axis. | Track 1 | D-009 / **D20** |
| B2/C2 `calib_group` | **Confirmed as proposed** — granularity unchanged; definition moves into the operator registry. Noise is *not* a group axis in v1. | Track 1 | **D-021** / **D22** |
| B3 split assignment | **Confirmed, with an added requirement** — per `sample_id`, patient-level, in the registry. Must also be **operator-independent**: same patient, same side, every regime, or coverage differences are confounded with anatomy. | Track 1 + 2 | **D-022** / **D23** |
| B4 multi-coil RSS | Confirmed; Track 1 concurs — state the single-coil narrowing in Experimental Setup rather than leaving a reader to find it. | Track 2 | D-008 |
| C1 primary distance | **Locked: `null_space_projection` (Frobenius) / `ambient`.** Verified non-saturating and monotone. Ill-posedness normalisation deliberately kept *out* of the distance. Two actions follow — distance-even operator grids, and the `operator_norm` naming collision. | Track 1 | D-010 / **D19** |
| C3 mask seeds | **Confirmed as proposed** — one group per (family, acceleration); seeds are members. Defines the Week 4 axes, and makes seed variation the noise floor a real shift must clear. | Track 1 | **D-023** / **D24** |
| C4 spectral budget | Still Track 3's; unblocked by C1 being locked. | Track 3 | D-007, D-011 |
| D-013 CT substrate | Track 2's call; Track 1 does not object to Option A. D20 makes annotations an overlay rather than the region definition, which removes the region-definition coupling from this choice. | Track 2 | D-013 |

### Referred item D-020 — answered, with the diagnosis revised

**The threshold is not the problem, so do not re-tune it.** With `reference_nullspace`,
`err_free` structurally measures the *ground truth's own* null-space content rather
than anything the reconstruction invented — reproduced on synthetic images at r = 0.95.
The criterion's **form** is locked now (`err_free / free_content > τ`); **τ is set only
after a real prior lands (T2.2)**, and its positive rate is then reported, not targeted.
One cheap ask back: nullable bank column `free_content` (no schema bump). See **D21**.

### Still open after this sign-off

Track 1's remaining critical-path item is **the score combination rule** (T1.1 WI-3) —
`scoring/score.py` is still `v0-placeholder`, and nothing is reportable until it is real.
T3.3 §5 constrains it: the null-space-energy and disagreement terms likely need to be
*designed* Lipschitz in the D19 distance rather than checked afterwards.
