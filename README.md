# CALIBER
**Calibrated Audit of Learned Image reconstruction Beyond Exchangeable Regimes**

Operator-shift-robust conformal hallucination flagging for generative-prior image reconstruction. Distribution-free reliability certificates that stay valid when the acquisition changes between calibration and deployment.

---

## What this is

Generative priors (diffusion, flow-matching) produce sharp accelerated-MRI reconstructions, but they invent detail that looks plausible and is not supported by the measurements — hallucination. PSNR and SSIM cannot reveal it: they are global, and they need the ground truth you do not have at deployment.

Conformal prediction can wrap any reconstruction model and emit per-region reliability flags with a provable error rate — but every existing conformal method for imaging assumes the **acquisition operator is fixed** between calibration and deployment. It is not. Change the sampling mask, the acceleration factor R, or the noise level, and exchangeability breaks: the method still reports 90% reliability while delivering less.

CALIBER shows that failure, fixes it, and measures what is left:

| | Role |
|---|---|
| **Group-conditional (Mondrian) calibration** | Guaranteed backbone — exact when the deployment operator is one of a known finite set |
| **Operator-distance-aware weighted recalibration** | Primary novel contribution — weights are *known*, since retrospective masks are chosen |
| **Operator-distance coverage-gap bound** | Theoretical stretch result, adapting non-exchangeable conformal prediction |

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

## Setup

```bash
conda create -n caliber python=3.11
conda activate caliber
pip install -r requirements.txt
```

Single modern GPU. No prior training required — all generative priors are pretrained and downloaded from upstream releases.

---

## Data

fastMRI is **not stored in this repository** and requires credentialed access — apply at [fastmri.med.nyu.edu](https://fastmri.med.nyu.edu/), then:

```bash
# fastMRI knee/brain raw k-space (after credentials are granted)
python scripts/download_fastmri.py

# fastMRI+ pathology annotations — region-level labels for flag evaluation
python scripts/download_fastmri_plus.py

# pretrained priors (FlowDPS primary; DPS / DDNM fallback and baselines)
python scripts/download_priors.py
```

Downloaded data lives in `data/` and pretrained weights in `checkpoints/`, both gitignored. Raw k-space is what makes this project possible: it allows the acquisition operator to be varied retrospectively in a controlled way.

The Front II substrate (Phase 3, Track 2) is a low-dose / sparse-view CT benchmark exposing **projection data** — LoDoPaB-CT — for the same reason: raw projections permit angle subsampling, angular-range restriction, and dose reduction to be applied retrospectively. Region-level ground truth there comes from anatomical segmentation (TotalSegmentator, AbdomenCT-1K).

---

## Structure

```
Project/
├── caliber/              # audit layer — operator decomposition, scoring, calibration
├── scripts/              # download, preprocessing, experiment entrypoints
├── configs/              # version-controlled experiment configs (all runs are config-driven)
├── Documentation/        # per-task records (T1.n_*.md) + problems_and_decisions.md
├── data/                 # gitignored — fastMRI, downloaded locally by each user
├── checkpoints/          # gitignored — pretrained priors
├── results/              # gitignored — logged metrics and tables
├── figures/              # output figures
└── Paper/                # gitignored — reference papers
```

---

## Work plan

Blueprint v3.0 (§18) splits the work into **three** concurrent, interdependent tracks. **This repository covers Track 1.** Tracks 2 and 3 are partner-owned; what Track 1 depends on from them, and what it owes them, is recorded as explicit contracts in `Documentation/problems_and_decisions.md`.

> **Changed in v3.0.** The blueprint went from two tracks to three, and the coverage-gap bound moved **out of Track 1 into Track 3**. Track 1's T1.3 is consequently no longer "coverage-gap analysis" but the **operator-distance formulation** that Track 3 derives against. Commits and branches predating 2026-08-04 use the v2 numbering. See D14 in the decisions log.

### Track 1 — Methodology, Calibration and Scientific Direction

Owns the central claim, the design of the method, and the decisions that set direction at each stage.

| Task | Deliverable | Status |
|------|-------------|--------|
| **T1.1** Operator-aware score design | Definition and reference implementation of the hallucination score from the null-space / range decomposition | **In progress** |
| **T1.2** Shift-robust calibration | Group-conditional and operator-distance weighted calibration converting scores into per-region flags | Not started |
| **T1.3** Operator-distance formulation | Selection and definition of the operator-distance measure, applied consistently across both modalities (resolves Q1) | Not started |
| **T1.4** Experimental design | Full protocol, operator grids, ablation matrix, and statistical-validation methodology for all tracks | Not started |
| **T1.5** Cross-track integration | Specification of shared interfaces and score-bank formats; adjudication of results across the three fronts | Not started |
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
