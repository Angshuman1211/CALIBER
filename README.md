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

Nothing is committed. See [Documentation/data_access.md](Documentation/data_access.md) for provenance, licensing and download.

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
| `problems_and_decisions.md` | Track 1 | Track 1's decision log — entries `D1`…`D24`, open questions, cross-track contracts |
| `decision_log.md` | cross-track | Cross-track decisions and adjudication — entries `D-001`…`D-023` |
| `week1_kickoff.md` | cross-track | Interface sign-off packet and its decision record |
| `score_bank_spec.md` · `operator_registry_spec.md` | Track 2 | The two shared interfaces |
| `data_access.md` | Track 2 | Dataset provenance, licensing, download |
| `T1.1_score_design.md` | Track 1 | Operator-aware score design |
| `T2_problems.md` | Track 2 | Track 2's problems log |
| `T3.1`–`T3.3` | Track 3 | Formal review, distance candidates, L1 stability result |

> **Numbering warning.** The two logs number independently: `D12` (Track 1) and `D-012` (cross-track) are unrelated decisions. Always say which log.

---

## Work plan

Blueprint v3.0 (§18) splits the work into **three** concurrent, interdependent tracks, all working in this repository. Cross-track dependencies are recorded as explicit contracts in `Documentation/problems_and_decisions.md`.

> **Changed in v3.0.** The blueprint went from two tracks to three, and the coverage-gap bound moved **out of Track 1 into Track 3**. Track 1's T1.3 is consequently no longer "coverage-gap analysis" but the **operator-distance formulation** that Track 3 derives against. Commits and branches predating 2026-08-04 use the v2 numbering. See D14 in the decisions log.

### Track 1 — Methodology, Calibration and Scientific Direction

Owns the central claim, the design of the method, and the decisions that set direction at each stage.

| Task | Deliverable | Status |
|------|-------------|--------|
| **T1.1** Operator-aware score design | Definition and reference implementation of the hallucination score from the null-space / range decomposition | **In progress** |
| **T1.2** Shift-robust calibration | Group-conditional and operator-distance weighted calibration converting scores into per-region flags | Not started |
| **T1.3** Operator-distance formulation | Selection and definition of the operator-distance measure, applied consistently across both modalities | **Delivered** — D19 |
| **T1.4** Experimental design | Full protocol, operator grids, ablation matrix, and statistical-validation methodology for all tracks | Not started |
| **T1.5** Cross-track integration | Specification of shared interfaces and score-bank formats; adjudication of results across the three fronts | Week 1 sign-off **delivered** (D19–D24); standing |
| **T1.6** Analysis and course correction | Interpretation of outcomes, diagnosis of failures, scope decisions at each checkpoint | Not started |
| **T1.7** Technical authoring | Method and analysis sections of the manuscripts and the final report | Not started |

Each task gets a branch (`T1.1`, `T1.2`, …) and a record in `Documentation/T1.<n>_*.md`, merged to `main`.

### Track 2 — Systems, Data Engineering and Cross-Domain Validation *(partner-owned)*

T2.1 data and operator pipeline · T2.2 solver integration · T2.3 score-bank production · T2.4 evaluation harness · T2.5 cross-domain experiments · T2.6 baseline reproduction · T2.7 figures and packaging.

### Track 3 — Analytical Foundations and Numerical Verification *(partner-owned)*

T3.1 formal review of the setting · T3.2 operator-distance candidates · T3.3 stability result (L1) · T3.4 ill-posedness scaling (L2) · T3.5 numerical verification · T3.6 bound-versus-measurement study · T3.7 analytical authoring.

---

## Timeline

Blueprint §20 groups the tasks into four term phases, sequenced so a valid, demonstrable result exists early and higher-risk elements come later — the coverage failure plus the group-conditional fix must be in hand by the midterm, before the weighted recalibration, the second modality, and the coverage-gap bound are attempted.

| Phase | Focus | Key artefacts | Checkpoint |
|-------|-------|---------------|------------|
| 1 | Setup, infrastructure and baselines | Operator pipeline and score bank in MRI; reproduced baselines; formal review of the analytical setting | Topic approval and feasibility |
| 2 | Empirical core and guaranteed backbone | Operator-aware score; demonstrated coverage failure; group-conditional remedy; first analytical result; preprint draft | Midterm review |
| 3 | Generalisation and primary contribution | Second-modality score bank and results; weighted recalibration; coverage-gap-against-distance curves in both modalities; ill-posedness scaling | Interim results |
| 4 | Consolidation and dissemination | Bound-versus-measurement study; ablations; manuscript submission; open-source release; final report | Final presentation and viva |

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
