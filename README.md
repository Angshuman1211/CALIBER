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

The Front I substrate is accelerated MRI with raw k-space (fastMRI; M4Raw as an open stopgap while access clears), with fastMRI+ pathology annotations for region-level evaluation. The Front II substrate (Phase 3) is a low-dose / sparse-view CT benchmark exposing **projection data** — LoDoPaB-CT — for the same reason: raw measurements permit angle subsampling, angular-range restriction, and dose reduction to be applied retrospectively. Region-level ground truth there comes from anatomical segmentation (TotalSegmentator, AbdomenCT-1K).

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
  evaluation/   coverage, efficiency, detection, gap-vs-distance    Stage 5
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
| `problems_and_decisions.md` | Track 1 | Track 1's decision log — entries `D1`…`D59`, open questions, cross-track contracts |
| `decision_log.md` | cross-track | Cross-track decisions and adjudication — entries `D-001`…`D-059` |
| `week1_kickoff.md` | cross-track | Interface sign-off packet and its decision record |
| `score_bank_spec.md` · `operator_registry_spec.md` | Track 2 | The two shared interfaces |
| `T2_problems_and_decisions.md` | Track 2 | Track 2's problem log (`P-01`…`P-27`), unresolved first, plus an index of every decision |
| `T1.1_score_design.md` | Track 1 | Operator-aware score design |
| `T1.2_shift_robust_calibration.md` | Track 1 | The three calibration tools, what they control, and what they refuse |
| `T1.3_operator_distance_formulation.md` | Track 1 | The locked operator-distance measure and why |
| `T1.4_experimental_design.md` | Track 1 | The estimand, the paired intervals, and what the design can and cannot measure |
| `T1.5_cross_track_integration.md` | Track 1 | Shared interfaces, sign-off record, contracts, adjudication log |
| `T1.6_analysis_and_course_correction.md` | Track 1 | Phase gate reviews — all three fronts against the central claim, one section per checkpoint |
| `T2.1_data_and_operator_pipeline.md` | Track 2 | Week 1 record — what was built, the numbers, honest limitations — plus dataset provenance, licensing and download (App. A) |
| `T2.2_solver_integration.md` | Track 2 | Week 2 record — DPS, DDNM and FlowDPS behind one interface; the Stage-2 split measured on real output |
| `T2.3_score_bank_and_baselines.md` | Track 2 | Week 3 record — the first fastMRI score bank, both baselines, and why neither can yet be read as a coverage result |
| `T3.1`–`T3.3_*.md` | Track 3 | Formal review, distance candidates, L1 stability result |

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
| **Analytical result** | Coverage gap bounded by a single distributional distance — a checked consequence of the non-exchangeable conformal analysis — with the reduction to operator distance drafted |
| **Substrates** | fastMRI+ annotations parsed (16,154 boxes); M4Raw verified end to end on real k-space; LoDoPaB CT running end to end |
| **Engineering** | 177 tests, config-driven runs, fixed seeds, self-describing score banks, CPU-only analysis path |

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
| **T1.1** Operator-aware score design | Definition and reference implementation of the hallucination score from the null-space / range decomposition | **Near complete** — Q6 answered on a real prior (D43); `s_residual` ruled out (D42); `w_free`'s selection rule pre-registered (D49), awaiting one run with a design split |
| **T1.2** Shift-robust calibration | Group-conditional and operator-distance weighted calibration converting scores into per-region flags | **Implemented; Tool 2 evaluated (D52)** — valid where the nearest-operator baseline is not, no better than the conservative fallback; the cause is tail domination, not bandwidth |
| **T1.3** Operator-distance formulation | Selection and definition of the operator-distance measure, applied consistently across both modalities | **Delivered** — D19, amended for direction (D47): the gap is not symmetric in the operator pair, so the plotted axis is signed. CT cross-check pending T2.5 |
| **T1.4** Experimental design | Full protocol, operator grids, ablation matrix, and statistical-validation methodology for all tracks | **Substantially delivered** — estimand (D45), operator grid (D50: both shift directions), ablation matrix. Seed noise floor unmeasured |
| **T1.5** Cross-track integration | Specification of shared interfaces and score-bank formats; adjudication of results across the three fronts | **Week 1 sign-off delivered** (D19–D24); standing |
| **T1.6** Analysis and course correction | Interpretation of outcomes, diagnosis of failures, scope decisions at each checkpoint | **Phase 1 gate delivered**; standing |
| **T1.7** Technical authoring | Method and analysis sections of the manuscripts and the final report | Not started |

### Track 2 — Systems, Data Engineering and Cross-Domain Validation

Owns the reconstruction and evaluation infrastructure, and carries the study into the second modality. Produces the score banks the other two tracks consume.

| Task | Deliverable | Status |
|------|-------------|--------|
| **T2.1** Data and operator pipeline | Dataset access, preprocessing, and reproducible generation of acquisition operators across both modalities | **Delivered** — both modalities, seeded registry, health checks, annotations parsed |
| **T2.2** Solver integration | Configured, verified pretrained reconstruction back-ends behind the common interface | **Delivered** — DPS, DDNM and FlowDPS across two families, one config word apart; Stage-2 split measured on their output |
| **T2.3** Score-bank production | Cached reconstructions and non-conformity scores across the full operator grid | **First real bank** — 23,040 fastMRI regions at three accelerations; not reportable (face prior, D-035) and too few test volumes to measure coverage (P-23) |
| **T2.4** Evaluation harness | Coverage, efficiency, and region-level detection utilities with automated logging | **Delivered** — coverage, flag area, region FDR/power, ESS, coverage gap |
| **T2.5** Cross-domain experiments | Execution of the protocol in the second modality and the supporting appendix domain | **In progress** — CT path runs end to end; projector fix (D-012) precedes reported curves |
| **T2.6** Baseline reproduction | Published comparison methods reproduced under the shared protocol | **First pass** — naive-transferred and matched-per-operator reproduced, held out; interpretation blocked on the criterion (P-21) and on sample size (P-23) |
| **T2.7** Figures and packaging | Visualisations, reproducibility package, and experimental-setup write-ups | Not started |

### Track 3 — Analytical Foundations and Numerical Verification

Owns the derivation of the coverage-gap bound and its verification against measured behaviour. Works from the cached score banks, so it runs in parallel with reconstruction rather than behind it.

| Task | Deliverable | Status |
|------|-------------|--------|
| **T3.1** Formal review of the setting | Structured account of the non-exchangeable conformal framework and the assumptions it requires | **Complete** — verified against the source theorems; L3 scope decided |
| **T3.2** Operator-distance candidates | Comparative analysis of subspace, projection and spectral distance measures, with their properties | **Substantially complete** — five candidates reduced to three; saturation failure identified |
| **T3.3** Stability result (L1) | Derivation bounding the coverage gap by the operator distance in known quantities | **First pass** — Step 1 rigorous; Step 2 sketched with two assumptions flagged, not hidden |
| **T3.4** Ill-posedness scaling (L2) | Extension relating the bound to null-space dimension and spectral decay | **First pass** — per-component breakdown done; rise-then-plateau explained as a cross-component cancellation effect, not a per-component law |
| **T3.5** Numerical verification | Verification of derived results on small synthetic linear operators, requiring negligible compute | Not started — checks T3.3's flagged assumption |
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
| 3 | Generalisation and primary contribution | Second-modality score bank and results; weighted recalibration; coverage-gap-against-distance curves in both modalities; ill-posedness scaling | Interim results | Not started |
| 4 | Consolidation and dissemination | Bound-versus-measurement study; ablations; manuscript submission; open-source release; final report | Final presentation and viva | Not started |

---

## Evaluation

Blueprint §15.

| Metric | What it measures |
|---|---|
| Empirical coverage vs nominal level | Whether the promised reliability actually holds |
| Flag area at fixed reliability | Efficiency — a valid but over-conservative flag is useless |
| Region-level FDR and power | Quality of hallucination detection where annotations exist |
| **Coverage gap vs operator distance** | The headline relationship, reported in both modalities |
| Effective sample size under weighting | Honest accounting of the statistical cost of reweighting |
| Agreement between measured gap and derived bound | Whether the Track 3 result predicts observed behaviour |

Baselines: naive conformal transferred across operators (the failure) · conformal matched to each operator (upper-bound reference) · Mondrian (the backbone) · operator-distance weighted recalibration (the primary method) · doubly-robust and adaptive calibration (fallbacks) · posterior-sample dispersion and published hallucination indices (non-conformal context).

Headline experiment: calibrate at R = 4, deploy at R = 8 (plus intermediate factors), repeated across mask families and noise levels.

---

## Deliverables

Blueprint §22. Open-source audit package · reproducible **cross-modality** operator benchmark released with cached score banks · coverage-gap-vs-distance result in both modalities · operator-distance coverage bound verified against measurement · public preprint · paper at a fast-turnaround Scopus-indexed venue · extended Q1 journal version · final institutional report.
