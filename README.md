# CALIBER
**Calibrated Audit of Learned Image reconstruction Beyond Exchangeable Regimes**

Operator-shift-robust conformal hallucination flagging for accelerated-MRI reconstruction. Distribution-free reliability certificates that stay valid when the acquisition changes between calibration and deployment.

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

### Scope, stated honestly

Precise scope-limiting is part of the contribution. The method assumes **known** linear/affine forward operators (true for retrospective MRI undersampling, where the mask is chosen). The guarantee is **exact** only in the discrete known-operator regime handled by group-conditional calibration; for continuous operator change the target is an explicit coverage-*gap bound*, not exact coverage. Operator shift changes the measurement process itself, so it is **not** a standard covariate shift — no new exact finite-sample guarantee is claimed for arbitrary shift. See `Documentation/problems_and_decisions.md` (D3, D6).

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

The blueprint (§15) splits the work into two concurrent, interdependent tracks. **This repository covers Track 1.** Track 2 is partner-owned; what Track 1 depends on from it is recorded as explicit contracts in `Documentation/problems_and_decisions.md`.

### Track 1 — Methodology, Calibration and Theoretical Guarantees

| Task | Deliverable | Status |
|------|-------------|--------|
| **T1.1** Operator-aware score design | Definition and reference implementation of the hallucination score from the null-space / range decomposition | **In progress** |
| **T1.2** Shift-robust calibration | Group-conditional and operator-distance weighted calibration converting scores into per-region flags | Not started |
| **T1.3** Coverage-gap analysis | Operator-distance coverage-gap treatment and its validation in the known-operator regime | Not started |
| **T1.4** Experimental design | Protocol, ablation matrix, and statistical-validation methodology | Not started |
| **T1.5** Analysis and iteration | Interpretation of results and refinement of score and calibration | Not started |
| **T1.6** Technical authoring | Methodology and analysis sections of the manuscript and report | Not started |

Each task gets a branch (`T1.1`, `T1.2`, …) and a record in `Documentation/T1.<n>_*.md`, merged to `main`.

### Track 2 — Systems, Data Engineering and Empirical Validation *(partner-owned)*

T2.1 literature and method survey · T2.2 data and operator pipeline · T2.3 baseline reproduction · T2.4 evaluation harness · T2.5 experiment execution · T2.6 figures and packaging.

---

## Timeline

Blueprint §16 groups the tasks into four term phases, sequenced so a valid, demonstrable result exists early and higher-risk elements come later — the coverage failure plus the group-conditional fix must be in hand by the midterm, before the weighted recalibration and coverage-gap bound are attempted.

| Phase | Focus | Checkpoint |
|-------|-------|------------|
| 1 | Setup, survey, baseline reproduction | Topic approval and feasibility |
| 2 | Empirical gap + guaranteed backbone | Midterm review |
| 3 | Primary contribution and analysis | Interim results |
| 4 | Consolidation and dissemination | Final presentation and viva |

---

## Evaluation

| Metric | What it measures |
|---|---|
| Empirical coverage vs nominal level | Whether the promised reliability actually holds |
| Prediction-set size / flag area | Efficiency — a valid but over-conservative flag is useless |
| Flag FDR and power | Region-level quality of hallucination detection |
| **Coverage gap vs operator distance** | The headline relationship — reliability lost as the acquisition moves |
| Effective sample size under weighting | Honest accounting of the cost of reweighting |

Headline experiment: calibrate at R = 4, deploy at R = 8 (plus intermediate factors), repeated across mask families and noise levels.

---

## Deliverables

Open-source audit package · reproducible fastMRI operator benchmark · coverage-gap-vs-distance result · public preprint · paper at a fast-turnaround Scopus-indexed venue · extended Q1 journal version · final institutional report.
