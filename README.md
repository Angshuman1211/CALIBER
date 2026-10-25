# CALIBER

Operator-shift-robust conformal hallucination flagging for generative-prior image reconstruction. Distribution-free reliability certificates that stay valid when the acquisition changes between calibration and deployment.

> Existing conformal reliability certificates for image reconstruction assume
> the acquisition operator is fixed. This project shows the guarantee breaks
> when the operator shifts between calibration and deployment, restores
> validity through group-conditional calibration and operator-distance-aware
> recalibration, demonstrates the result across two imaging modalities with
> structurally different operators, and derives a coverage-gap bound expressed
> in quantities computable from the operators themselves.

This is the **shared repository for all three tracks** (blueprint v3.0 §18). Track 1 owns methodology, calibration and scientific direction; Track 2 owns systems, data engineering and cross-domain validation; Track 3 owns the analytical foundations. Task branches (`T1.1`, `T2.1`, `T3.1`, …) merge to `main`.

The framework is a **solver-agnostic audit layer over pretrained priors** — it audits reconstruction models rather than training them, which is what makes it feasible on a single GPU.

### Three fronts, one claim (blueprint §8)

The investigation is one central claim supported on three fronts, not three separate problems. None can be removed without weakening the others.

| Front | Question it answers | Track |
|---|---|---|
| **I — Empirical core** (accelerated MRI) | Does the conformal guarantee actually fail under operator shift, and can it be restored? | 1 + 2 |
| **II — Cross-domain** (sparse-view and limited-angle CT) | Is the failure a property of the acquisition in general, or an artefact of one modality? | 2 |
| **III — Theory** (operator-distance coverage bound) | How much reliability must be lost, as a function of how far the operator has moved? | 3 |

### Scope, stated honestly

Precise scope-limiting is part of the contribution. The method assumes **known** linear/affine forward operators (true for retrospective MRI undersampling and for CT projection geometries, where the mask or view set is chosen). The guarantee is **exact** only in the discrete known-operator regime handled by group-conditional calibration; for continuous operator change the target is an explicit coverage-*gap bound*, not exact coverage. Operator shift changes the measurement process itself, so it is **not** a standard covariate shift — no new exact finite-sample guarantee is claimed for arbitrary shift. See `Documentation/problems_and_decisions.md` (D3, D6).

---

## Quick start

```bash
pip install -e ".[dev]"
pytest -q

# End-to-end on synthetic phantoms — no dataset access, no checkpoint needed
caliber configs/experiments/w01_smoke_phantom.yaml

# CPU-only analysis of the resulting bank
python scripts/analyse_bank.py runs/<run_id> \
    --calibration-operator mri.random1d.R2.c0.08.s0
```

The smoke run reproduces the project's central phenomenon in miniature. A
threshold calibrated at R=2 and transferred to more aggressive accelerations
loses coverage monotonically with operator distance, and group-conditional
calibration restores it:

| method | R=2 | R=4 | R=8 |
|---|---|---|---|
| naive transferred | 0.90 | 0.59 | 0.41 |
| matched per operator | 0.90 | 0.90 | 0.90 |
| group-conditional | 0.90 | 0.90 | 0.90 |

*(nominal 0.90; synthetic phantoms and a reference sampler — an interface
smoke test, not a result.)*

Note the price of restoration: flag area rises from 0.44 to 0.83. Valid but
over-conservative flags are of little practical use, which is exactly why
Sec. 15.2 makes flag area a reported metric alongside coverage.

## Data

The Front I substrate is accelerated MRI with raw k-space (fastMRI; M4Raw as an open stopgap while access clears), with fastMRI+ pathology annotations for region-level evaluation. The Front II substrate is CT from **TotalSegmentator** (D-013): ground-truth CT volumes with 117 aligned anatomical masks per subject, re-projected through the project's own parallel-beam operator (D-005). The operator is ours for the same reason the MRI masks are: angle subsampling and angular-range restriction must be applied, and varied, retrospectively. LoDoPaB-CT was the original plan, but it carries no region annotations, and TotalSegmentator's masks cannot be laid over its images, as they are different cohorts (P-03). Organ masks supply the region-level overlay (D20).

Nothing is committed. See [T2.1 Appendix A](Documentation/T2.1_data_and_operator_pipeline.md#appendix-a--data-access-and-provenance) for provenance, licensing and download.

---

## Structure

```
src/caliber/
  operators/    the known acquisition operator A, its range/null-space split,
                the shared operator registry, distance measures     Stage 2
  solvers/      pretrained diffusion and flow-matching back-ends behind one
                black-box posterior-sampler interface               Stage 1
  scoring/      region definition, operator-aware scoring, and the
                ground-truth hallucination criterion                Stage 3
                (Track 1 owns the score definition)
  scorebank/    the cached, shared data format all three tracks read
  calibration/  group-conditional and weighted recalibration        Stage 4
                (Track 1 owns this module)
  evaluation/   coverage, efficiency, detection, gap-vs-distance,   Stage 5
                paired estimands, transfer penalties, curve
                collapse, weight sensitivity
  data/         dataset access and preprocessing

configs/        version-controlled experiment definitions
Documentation/  all project documentation — see below
scripts/        CPU-only analysis entry points
tests/          correctness gates
```

### Documentation

One directory, all three tracks. Per-task records are prefixed with the blueprint §18 task ID.

| File | Owner | What |
|---|---|---|
| `problems_and_decisions.md` | Track 1 | Track 1's decision log — entries `D1`…`D78`, open questions, cross-track contracts |
| `decision_log.md` | cross-track | Cross-track decisions and adjudication — entries `D-001`…`D-065` |
| `week1_kickoff.md` | cross-track | Interface sign-off packet and its decision record |
| `score_bank_spec.md` · `operator_registry_spec.md` | Track 2 | The two shared interfaces |
| `T2_problems_and_decisions.md` | Track 2 | Track 2's problem log (`P-01`…`P-31`), unresolved first, plus an index of every decision |
| `T1.1_score_design.md` | Track 1 | Operator-aware score design |
| `T1.2_shift_robust_calibration.md` | Track 1 | The three calibration tools, what they control, and what they refuse |
| `T1.3_operator_distance_formulation.md` | Track 1 | The locked operator-distance measure and why |
| `T1.4_experimental_design.md` | Track 1 | The estimand, the paired intervals, and what the design can and cannot measure |
| `T1.5_cross_track_integration.md` | Track 1 | Shared interfaces, sign-off record, contracts, adjudication log |
| `T1.6_analysis_and_course_correction.md` | Track 1 | Phase gate reviews — all three fronts against the central claim, one section per checkpoint |
| `T1.7_manuscript_method_and_analysis.md` | Track 1 | The manuscript's method and analysis sections — what is claimed, what is not, and the gate on every number |
| `T1.7_preprint_draft.md` | Track 1 | The preprint cut — the narrowed paper: the failure, the fix, two modalities. Excludes Tools 2 and 3 and all detection claims (D77) |
| `T2.1_data_and_operator_pipeline.md` | Track 2 | Week 1 record — what was built, the numbers, honest limitations — plus dataset provenance, licensing and download (App. A) |
| `T2.2_solver_integration.md` | Track 2 | Week 2 record — DPS, DDNM and FlowDPS behind one interface; the Stage-2 split measured on real output |
| `T2.3_score_bank_and_baselines.md` | Track 2 | Week 3 record — the first fastMRI score bank, both baselines, and why neither can yet be read as a coverage result |
| `T2.4_operator_grid_and_harness.md` | Track 2 | Week 4 record — the 3 × 3 MRI operator grid, the evaluation harness, and the full 199-volume bank on the laptop GPU (§8 carries a correction box: its numbers predate the corrected label) |
| `T2.5_corrected_label_and_midterm_figures.md` | Track 2 | Weeks 5–6 record — every coverage number re-measured under D-060's candidacy gate, the 13-split and mask-seed noise floors, the posterior-dispersion baseline, the midterm figures |
| `T2.6_ct_onboarding_and_first_ct_bank.md` | Track 2 | Week 7 record — the TotalSegmentator CT substrate, three CT solver bugs fixed, the first CT bank (900 pairs), and why its coverage awaited a CT label |
| `T2.7_absolute_label_and_cross_modality_result.md` | Track 2 | Week 8 record — D-061's absolute label in every script, both banks re-measured, the CT coverage failure and its restoration, stratified detection, the CT operator distances |
| `T2.8_ct_harness_and_cross_modality_figures.md` | Track 2 | Week 9 record — the full evaluation harness on CT in both directions, and the coverage-gap-against-distance curves of both modalities on one axis |
| `T2.9_track1_week10_runs.md` | Track 2 | Week 10 record — Track 1's six queued runs: the baseline on the coverage axis, Tool 2 on a sparse CT grid, the direction label on CT, weight sensitivity, the curve-collapse test, and the DPS solver ablation |
| `T3.1`–`T3.5_*.md` | Track 3 | Formal review, distance candidates, L1 stability result, ill-posedness scaling, numerical verification |

> **Numbering warning.** The two logs number independently: `D12` (Track 1) and `D-012` (cross-track) are unrelated decisions. Always say which log.

---

## Work plan

Blueprint v3.0 (§18) splits the work into **three** concurrent, interdependent tracks, all working in this repository. They are not three sub-projects: the empirical core produces the phenomenon, the cross-domain front establishes that it is general, and the analytical front explains and bounds it — none can be removed without weakening the others (§8). The tracks integrate through two shared interfaces, the **score bank** and the **operator registry**, and adjudicate disagreements through a common decision log (§19).

> **Changed in v3.0.** The blueprint went from two tracks to three, and the coverage-gap bound moved **out of Track 1 into Track 3**. Track 1's T1.3 is consequently no longer "coverage-gap analysis" but the **operator-distance formulation** that Track 3 derives against. Commits and branches predating 2026-08-04 use the v2 numbering. See D14 in the decisions log.

### What exists today

| | |
|---|---|
| **Acquisition operators** | Both modalities behind one registry — 3 MRI mask families × CT sparse-view and limited-angle, exact adjoints, seeded and reproducible, with health checks |
| **Operator distance** | All five blueprint §12.1 candidates implemented; the primary measure locked and verified non-saturating across the acceleration sweep |
| **Hallucination score** | Three-component operator-aware score, cached as separable components; validated against *constructed* data-consistent hallucinations |
| **Calibration** | All three tools — naive baseline, group-conditional backbone, operator-distance weighted — with thin-group refusal and effective-sample-size tracking |
| **Analytical result** | Coverage gap bounded by the Kolmogorov distance of the label-conditional score law (tighter than the original TV statement), with a √d_op reduction to operator distance derived under stated assumptions; constants not yet measured |
| **Substrates** | fastMRI+ annotations parsed (16,154 boxes); M4Raw verified end to end on real k-space; TotalSegmentator CT wired as the CT substrate (100 subjects split by subject, organ annotations on every one) |
| **Score banks** | The full fastMRI knee validation set across the 3 × 3 MRI grid: 1,791 pairs, 458,496 regions, 6.4 h on the laptop's GPU. Mask-seed banks at `random1d` R4 and R8. **The first CT bank:** 100 TotalSegmentator subjects × 9 CT operators (sparse-view 180–30 views, limited-angle 150–90°), 900 pairs, ~16 h on the GPU. Development banks only: the prior is a face model (D-016) |
| **Coverage failure (O3)** | Measured under D-061's absolute label (`err_free` above its design-split q75), robust across 13 patient splits. On fastMRI, naive transfer from `random1d` R2 falls below 90% coverage at `equispaced1d` R2 (13/13 splits), `random1d` R8 (11/13) and `equispaced1d` R8 (10/13). Group-conditional calibration holds 0.884–0.901 wherever positives exist. Development values (T2.7) |
| **Second modality (O5)** | **The coverage failure reproduces on CT, graded by the shift.** Calibrated at 180 views, naive transfer's coverage falls as views thin (0.890 at 120 views to 0.693 at 30) and as the arc narrows (0.888 at 150° to 0.796 at 90°), below 90% on 13 of 13 splits at the severe end. Group-conditional calibration restores nominal at every CT operator (0.902–0.923), and D-053's signed distance orders CT's gaps at ρ = +0.976. On MRI the same distance orders them only weakly (ρ = +0.32 to +0.56), because MRI's worst failures sit at matched acceleration (T2.8). The curve-collapse test (D72) does not reject one curve, but it does not reject one for the group-conditional control either, so the pass is a lack of power, not collapse demonstrated (T2.9 §6). Under this prior the label cannot be validated as invention (D-061), a stated scope limit (T2.7) |
| **Baseline comparison** | Moved to the axis the project can claim on. Detection AUROC was disqualified by D67 (the label is near-determined by the score), so the baseline is compared on **coverage under operator shift**: whose threshold survives a change of operator, as a paired difference of transfer penalties (D73). **On CT the baseline transfers better.** Counting coverage lost in either direction, `s_disagree`'s threshold transfers as well or better than the CALIBER score at all 16 CT cells. The score loses up to 0.21 coverage going to sparse views and over-flags coming back. On MRI, against the literature's full-image posterior dispersion, neither threshold transfers: each detector over-flags in the direction where its own scale grows. Counted both ways, the dispersion is better at 9 cells to CALIBER's 4. The comparison as implemented counts over-coverage as good transfer, which on both modalities favours whichever detector over-flags (T2.9 §2, P-37) |
| **Hallucination criterion** | Four magnitude-based forms, all near-determined by the score. A fifth thresholds a **direction** instead: `cos θ` from the project's own identity, which is scale-free (D74). **On CT, against it the score ranks at chance** (AUC 0.50–0.58) and beats the anatomy control at 0 of 9 operators, against 9 of 9 under the magnitude label (T2.9 §4) |
| **Solver ablation (D68)** | A DPS bank (597 pairs) beside the DDNM one. DPS leaves 0.69 of its error in the null space against DDNM's 0.99, so it is where the decomposition has room. Under the magnitude label the score beats `s_disagree` by ≥ 0.05 AUC at all three DPS operators. **Under the direction label it does not beat it at any** (−0.034 to −0.002), so the positive is the label's identity, not detection. At R2, DPS fails D-033's admissibility gate on 196 of 199 volumes (T2.9 §1) |
| **Engineering** | 786 tests, config-driven runs, fixed seeds, self-describing score banks, CPU-only analysis path |

---

> **Two independent axes — don't read one as the other.** §18 (this section) is
> *who does what*: three tracks × seven tasks, addressed `Tx.y` = **track x,
> task y**. §20 (Timeline, below) is *when*: four phases, each ending at a
> review checkpoint. They are orthogonal — a task can span several phases, and
> a phase draws on tasks from all three tracks. The blueprint deliberately does
> **not** map tasks onto phases, so the Status column here reports the state of
> the work, never a scheduled phase.

### Track 1 — Methodology, Calibration and Scientific Direction

Owns the central claim, the design of the method, and the decisions that set direction at each stage.

| Task | Deliverable | Status |
|------|-------------|--------|
| **T1.1** Operator-aware score design | Definition and reference implementation of the hallucination score from the null-space / range decomposition | **Complete** — Q6 answered on a real prior (D43); `s_residual` ruled out (D42); `w_free`'s rule pre-registered (D49). The weights stay **equal by default** (D54), and D76's sweep converts that from a gap into a robustness claim by testing whether the result depends on them at all |
| **T1.2** Shift-robust calibration | Group-conditional and operator-distance weighted calibration converting scores into per-region flags | **Three tools + fallback [7]** — Tool 2 evaluated (D52): valid but no better than the conservative fallback, cause is tail domination not bandwidth. [7] (D66) fixes the location error. D75 adds `--exclude-nearest`: these grids are dense enough that the nearest operator is already nearly the matched one, so Tool 2's defeat may describe the grid rather than the method |
| **T1.3** Operator-distance formulation | Selection and definition of the operator-distance measure, applied consistently across both modalities | **Delivered** — D19, amended for direction (D47): the gap is not symmetric in the operator pair, so the plotted axis is signed. CT cross-check done: the signed measure orders the CT coverage gaps at ρ = +0.976 (T2.7) |
| **T1.4** Experimental design | Full protocol, operator grids, ablation matrix, and statistical-validation methodology for all tracks | **Delivered** — estimand (D45), operator grid (D50/D61), ablation matrix, stratified reporting (D62/D65), curve-collapse test (D72), and the coverage-axis baseline comparison (D73). Seed noise floor measured by Track 2 (T2.5 §7) |
| **T1.5** Cross-track integration | Specification of shared interfaces and score-bank formats; adjudication of results across the three fronts | **Week 1 sign-off delivered** (D19–D24); standing |
| **T1.6** Analysis and course correction | Interpretation of outcomes, diagnosis of failures, scope decisions at each checkpoint | **Phase 1 and Phase 2 gates delivered** — coverage claims stand, detection claims do not (D67); DPS ablation promoted (D68) |
| **T1.7** Technical authoring | Method and analysis sections of the manuscripts and the final report | **Method section and the preprint cut complete** — `T1.7_manuscript_method_and_analysis.md` and `T1.7_preprint_draft.md` (D77). The preprint leads with the failure, excludes Tools 2 and 3 and all detection claims, and promotes D67's circularity finding from a limitation to a result. One blocker: medical weights |

### Track 2 — Systems, Data Engineering and Cross-Domain Validation

Owns the reconstruction and evaluation infrastructure, and carries the study into the second modality. Produces the score banks the other two tracks consume.

| Task | Deliverable | Status |
|------|-------------|--------|
| **T2.1** Data and operator pipeline | Dataset access, preprocessing, and reproducible generation of acquisition operators across both modalities | **Delivered** — both modalities, seeded registry, health checks, annotations parsed |
| **T2.2** Solver integration | Configured, verified pretrained reconstruction back-ends behind the common interface | **Delivered** — DPS, DDNM and FlowDPS across two families, one config word apart; Stage-2 split measured on their output |
| **T2.3** Score-bank production | Cached reconstructions and non-conformity scores across the full operator grid | **Full MRI bank and first CT bank** — all 199 fastMRI validation volumes × 9 operators (458,496 regions), mask-seed banks, and 100 CT subjects × 9 operators (230,400 regions); checkpointed and resumable. Not reportable (face prior, D-016) |
| **T2.4** Evaluation harness | Coverage, efficiency, and region-level detection utilities with automated logging | **Delivered** — coverage, flag area, region FDR/power, ESS, coverage gap; intervals by patient (D-027); one shared label for every reporting script (D-061's absolute label by default); detection stratified by `free_content` beside an anatomy control; patient-split and mask-seed noise floors |
| **T2.5** Cross-domain experiments | Execution of the protocol in the second modality and the supporting appendix domain | **CT result obtained** — TotalSegmentator substrate (D-013), CT solver path fixed, 900-pair CT bank. Under D-061's label naive transfer fails on CT in proportion to the shift and group-conditional restores nominal (T2.7). Dose is not a grid axis (D-061) |
| **T2.6** Baseline reproduction | Published comparison methods reproduced under the shared protocol | **In progress** — naive-transferred, matched-per-operator and posterior dispersion run on the full MRI bank under D-060's label. Dispersion tracks `s_disagree` to within 0.012 AUROC on DDNM. The published Hallucination Index [22] is computed too, but on fastMRI it mostly flags background (a noisy ground truth) and does not agree with the project's label. On the coverage axis (D73), both modalities are run. On CT, `s_disagree`'s threshold transfers at least as well as the score's at every cell. On MRI, against the full-image dispersion (calibration-side draws regenerated, reproduction exact), neither transfers and the score's does not transfer better (T2.9 §2) |
| **T2.7** Figures and packaging | Visualisations, reproducibility package, and experimental-setup write-ups | **Midterm and cross-modality figures drawn** — MRI coverage by operator, gap against signed distance, coverage against flag area; CT coverage by operator; naive transfer's gap against distance for both modalities on one axis (`scripts/make_figures.py`, T2.8); reproducibility package not started |

### Track 3 — Analytical Foundations and Numerical Verification

Owns the derivation of the coverage-gap bound and its verification against measured behaviour. Works from the cached score banks, so it runs in parallel with reconstruction rather than behind it.

| Task | Deliverable | Status |
|------|-------------|--------|
| **T3.1** Formal review of the setting | Structured account of the non-exchangeable conformal framework and the assumptions it requires | **Complete** — verified against the source theorems; L3 scope decided |
| **T3.2** Operator-distance candidates | Comparative analysis of subspace, projection and spectral distance measures, with their properties | **Substantially complete** — five candidates reduced to three; saturation failure identified |
| **T3.3** Stability result (L1) | Derivation bounding the coverage gap by the operator distance in known quantities | **Fourth pass** — Step 1 stated on the Kolmogorov distance of the label-conditional law, with cluster-level proofs; Step 2 derived for the unconditional law and (under assumptions) the label-conditional one, √d_op rate; the label-conditional constants are unmeasured |
| **T3.4** Ill-posedness scaling (L2) | Extension relating the bound to null-space dimension and spectral decay | **First pass, revised twice** — √null_frac law holds for white-noise fill on sparse-view CT and mask operators, fails for limited-angle CT and a structured prior (D-046 on a real prior); the plateau and the 0.57 ceiling are not bounds |
| **T3.5** Numerical verification | Verification of derived results on small synthetic linear operators, requiring negligible compute | **Delivered** — repaired a TV/Kolmogorov error in T3.3, measured D, quantified the label-conditioning, clustering and solver-dependence mismatches, derived the label-conditional link; the bound is not yet useful on the toy harnesses |
| **T3.6** Bound-versus-measurement study | Comparison of predicted and observed coverage gaps using the cached score banks | Not started |
| **T3.7** Analytical authoring | Derivations, statements of assumptions, and the theoretical sections of the manuscripts | Not started |

Each task gets a branch (`T1.1`, `T2.1`, `T3.1`, …) and a record in `Documentation/`, merged to `main` by pull request.

---

## Timeline

Blueprint §20 groups the tasks into four term phases, sequenced so a valid, demonstrable result exists early and higher-risk elements come later — the coverage failure plus the group-conditional fix must be in hand by the midterm, before the weighted recalibration, the second modality, and the coverage-gap bound are attempted.

| Phase | Focus | Key artefacts | Checkpoint | Status |
|-------|-------|---------------|------------|--------|
| 1 | Setup, infrastructure and baselines | Operator pipeline and score bank in MRI; reproduced baselines; formal review of the analytical setting | Topic approval and feasibility | **Complete but for baselines** — see `Documentation/T1.6_analysis_and_course_correction.md` |
| 2 | Empirical core and guaranteed backbone | Operator-aware score; demonstrated coverage failure; group-conditional remedy; first analytical result; preprint draft | Midterm review | In progress — score and L1 delivered early |
| 3 | Generalisation and primary contribution | Second-modality score bank and results; weighted recalibration; coverage-gap-against-distance curves in both modalities; ill-posedness scaling | Interim results | **In progress** — CT bank produced (T2.6); CT coverage failure and restoration obtained, and CT operator distances computed (T2.7) |
| 4 | Consolidation and dissemination | Bound-versus-measurement study; ablations; manuscript submission; open-source release; final report | Final presentation and viva | Not started |

---

## Evaluation

Blueprint §15.

| Metric | What it measures |
|---|---|
| Empirical coverage vs nominal level | Whether the promised reliability actually holds |
| Flag area at fixed reliability | Efficiency — a valid but over-conservative flag is useless. Implemented literally by `area_at_matched_coverage`: coverage is **fixed** and area compared, so a detector cannot win by flagging almost nothing (D39's rule with D51's validity gate built in) |
| **Transfer penalty** | Coverage a detector loses when its threshold moves to another operator, as a paired difference. The axis on which the dispersion baseline is compared, since detection AUROC is disqualified by D67 (D73) |
| Region-level FDR and power | Quality of hallucination detection where annotations exist |
| **Coverage gap vs operator distance** | The headline relationship, reported in both modalities |
| Effective sample size under weighting | Honest accounting of the statistical cost of reweighting |
| Agreement between measured gap and derived bound | Whether the Track 3 result predicts observed behaviour |

Baselines: naive conformal transferred across operators (the failure) · conformal matched to each operator (upper-bound reference) · Mondrian (the backbone) · operator-distance weighted recalibration (the primary method) · doubly-robust and adaptive calibration (fallbacks) · posterior-sample dispersion and published hallucination indices (non-conformal context).

Headline experiment: calibrate at R = 4, deploy at R = 8 (plus intermediate factors), repeated across mask families and noise levels.

---

## Deliverables

Blueprint §22. Open-source audit package · reproducible **cross-modality** operator benchmark released with cached score banks · coverage-gap-vs-distance result in both modalities · operator-distance coverage bound verified against measurement · public preprint · paper at a fast-turnaround Scopus-indexed venue · extended Q1 journal version · final institutional report.
