# CALIBER — Problems & Decisions Log

Running log of every non-obvious design decision and every problem encountered during implementation. Update this file immediately when a new issue arises or a design choice is made. Entries are numbered sequentially; the originating task (blueprint §15 work plan — T1.1, T1.2, …) is noted in brackets.

This repository covers **Track 1** — methodology, calibration and theoretical guarantees. Track 2 (systems, data engineering, empirical validation) is partner-owned; cross-track dependencies are recorded here as *contracts*, not as work items.

---

## Pre-Development Decisions (Blueprint v2.0, July 2026)

### D1 — Audit pretrained priors; train nothing [Blueprint]
The expensive step — training a large generative prior — has already been done and released by others (FlowDPS, DPS, DDNM). Decision: CALIBER is a reliability layer placed *on top of* existing pretrained solvers, treated as black-box posterior samplers behind a uniform interface. Reason: this is the single choice that makes the project feasible on one GPU within one academic term, and it makes the contribution solver-agnostic — any conforming solver can be wrapped unchanged. Alternative rejected: training a task-specific MRI prior — burns the entire term's compute on a component that is not the contribution.

### D2 — Score on the null-space / range decomposition, not the whole image [Blueprint]
Because the forward operator A is known, any reconstruction splits into a **range (measured)** component that the data pins down and a **null-space (free)** component the operator cannot see. Every hallucination must live in the free component; the measured component is constrained by the data and cannot be freely altered. Decision: direct the hallucination score at the free component rather than treating the image uniformly. This is what "operator-aware" means in this project. Consequence: the method requires a **linear or affine** forward operator (D6) — true for accelerated MRI, not true in general.

### D3 — Group-conditional (Mondrian) calibration is the guaranteed backbone, and it ships first [Blueprint]
Three calibration tools are in scope, in increasing ambition: Mondrian (exact for a known finite operator set), operator-distance weighted recalibration (Tibshirani et al. 2019), and a coverage-gap bound (Barber–Candès–Ramdas–Tibshirani 2023). Decision: lead with Mondrian, which restores exact within-group coverage with no assumptions about the shift, and ship it by the Phase-2 midterm. Reason: risk-ordering — the project must own a valid, demonstrable result before attempting the harder corrections. Trade-off accepted: each acquisition regime needs enough calibration data; thin groups give noisy thresholds (see D10, and the group-weighted CP fallback).

### D4 — Operator-distance-aware weighted recalibration is the primary novel contribution [Blueprint]
Reweights calibration examples by how much the known operator change moves the reconstruction. Decision: this is the headline method, not Mondrian. Reason: retrospective MRI masks are **chosen**, so the required weights are known in advance rather than estimated — the best-justified setting for weighted conformal prediction, and the reason this correction is defensible here when it would be shaky elsewhere. Trade-off: reweighting reduces the effective calibration sample size, which the analysis must report rather than hide (hence ESS is a first-class metric).

### D5 — The coverage-gap bound is scoped as an extension, so the primary outcome is never blocked [Blueprint]
Adapting the non-exchangeable conformal analysis to relate lost coverage to a concrete *operator* distance is the standout theoretical result, but combining data-dependent weights with that analysis needs care and the bound must be re-derived for the known-operator setting rather than assumed. Decision: ship the empirical result plus the guaranteed backbone first; treat the bound as a Phase-3 reach item. Risk register explicitly permits the project to succeed without it.

### D6 — Operator shift is NOT a covariate shift; claim accordingly [Blueprint]
A covariate shift moves the input distribution while holding the input→output relationship fixed. Changing the acquisition operator changes the **measurement process itself**. Decision: state this explicitly and do not claim a new exact finite-sample guarantee for arbitrary operator shift. What is claimed: exact coverage in the discrete known-operator regime (Mondrian); an explicit coverage-*gap bound*, not exact coverage, for continuous operator change; known linear/affine operators throughout. Reason: reviewers in this area reward precise scope and penalise over-claiming — the honesty is load-bearing, not decorative. This is the single most important framing constraint in the project; check every abstract, method section, and commit message against it.

### D7 — Flow-matching prior primary, diffusion fallback; contribution is family-independent [Blueprint]
Primary back-end is a pretrained flow-matching solver (FlowDPS); pretrained diffusion solvers (DPS, DDNM) serve as fallback and as baselines. Decision recorded as a *risk mitigation*: if the flow-matching prior underperforms on MRI, fall back to diffusion — the contribution is the audit layer and does not depend on the generative family. Consequence for design: the solver interface must be genuinely uniform from day one, or this fallback is not actually available when needed.

### D8 — fastMRI as the primary benchmark [Blueprint]
Two reasons, both necessary. (a) It is the shared standard in this literature, so results are directly comparable to prior work. (b) Its **raw k-space** allows retrospective undersampling — the acquisition operator can be varied in a controlled way across mask families, acceleration factors, and noise levels, which is precisely the independent variable this project studies. fastMRI+ pathology annotations supply region-level labels for measuring whether flags fire in clinically meaningful areas. InverseBench is a secondary cross-check against an established inverse-problem benchmark. Alternative rejected: image-domain-only MRI datasets — no k-space means no controllable operator, which would remove the study variable entirely.

### D9 — Risk-controlling calibration, not plain split-conformal coverage [Blueprint]
The deliverable is a per-region flag with a controlled rate of **missed hallucinations**, not a marginal interval around a pixel value. Decision: build on risk-controlling prediction sets (Bates et al. 2021) and conformal risk control (Angelopoulos et al. 2024), which let the controlled quantity be a chosen error measure rather than only coverage. The risk register also names "cast the target as risk control" as the mitigation if weighted recalibration under-covers.

### D10 — Start single-coil, curated fastMRI subsets [Blueprint]
Mitigation for compute and data-access constraints. Decision: begin with single-coil data and curated subsets; audit pretrained priors only. Multi-coil is an extension, not a Phase-1 requirement. Note the interaction with D3: high-acceleration regimes are exactly where per-regime calibration data is scarcest, so subset curation must not starve the thin groups.

### D11 — Preprint early to establish priority [Blueprint]
Risk register flags an active competing group in conformal MRI uncertainty. Decision: release a preprint at Phase 2 rather than waiting for the full result, and differentiate explicitly on three axes — **operator** shift (not anatomical shift), coverage restoration **across acceleration factors**, and the **flow-matching** prior. Publication path: preprint → fast-turnaround Scopus-indexed signal-processing venue → extended Q1 journal version, since journal review timelines exceed the project window.

### D12 — Data and weights are never committed; scripts + credentialed download instead [infrastructure]
fastMRI requires an approved NYU application and raw k-space volumes run 1–10 GB per file, against GitHub's 100 MB hard limit. Decision: `data/` and `checkpoints/` are gitignored; each collaborator obtains their own fastMRI credentials and runs `scripts/download_fastmri.py`. Reproducibility is carried by version-controlled configs and fixed seeds, not by committed artefacts. Alternative rejected: shared cloud drive — no versioning, links rot, not reproducible for external reviewers.

### D13 — `Project/` is the git repository; blueprints stay outside it [infrastructure]
Matches the layout used in MIRAGE, Vestige and Argus: the outer folder holds the blueprint `.docx`/`.pdf` and background PDFs and is *not* version-controlled; the inner `Project/` is the repo that gets pushed to `github.com/Ganglet/CALIBER.git`. Reason: blueprints are binary, revised in Word, and not source.

---

## Open Questions

### Q1 — "Operator distance" is not yet defined [OPEN, owned by T1.2/T1.3]
The coverage-gap-vs-operator-distance plot is the paper's central figure, and the x-axis of that figure is currently undefined. The blueprint gestures at two candidates — how much the sampling masks differ, and how much larger the null space becomes — without committing. This choice determines the weighting scheme (D4) *and* the form of the bound (D5), so it cannot be deferred to T1.3. Candidates to evaluate: symmetric difference of mask supports; a subspace distance between null spaces (e.g. principal angles / projector distance ‖P₁ − P₂‖); a divergence between the induced score distributions. Requirement: whatever is chosen must be *computable without ground truth* and monotone in something a radiographer would recognise as "a bigger change to the scan." The T1.1 projector utilities (WI-2 there) are the substrate for the subspace-distance candidates.

### Q2 — Region definition for the flags [OPEN, owned by T1.1]
"Per-region" is used throughout the blueprint without fixing what a region is — pixel, patch, superpixel, or annotated anatomical/pathological structure. It determines the multiplicity correction, the flag-area metric, and how fastMRI+ annotations are matched to flags. Note the tension: fastMRI+ labels are lesion-level bounding boxes, so evaluation naturally lives at that scale even if calibration runs on patches.

### Q3 — How many solver draws per reconstruction [OPEN, cost-driving, owned by T1.1]
The score combines null-space structure with cross-sample disagreement, so it needs several solver draws per measurement set. Each draw is a full diffusion/flow solve. Draws × calibration-set size × number of operator regimes is the dominant compute cost of the whole project and should be budgeted during T1.1, not discovered during T1.5.

---

## Track 1 Task Log

Per-task records live in `T1.<n>_*.md`. Decisions and problems arising during a task are logged below as `T1.<n>-D<k>`.

### T1.1 — Operator-aware score design
See `T1.1_score_design.md`. Branch `T1.1`.

_No entries yet — implementation not started._

### T1.2 — Shift-robust calibration
_Not started._

### T1.3 — Coverage-gap analysis
_Not started._

### T1.4 — Experimental design
_Not started._

### T1.5 — Analysis and iteration
_Not started._

### T1.6 — Technical authoring
_Not started._

---

## Cross-Track Contracts

Track 2 is partner-owned. Anything Track 1 needs from it is recorded here so the interface is explicit and dated, rather than discovered at integration.

| # | Track 1 needs | From | Status |
|---|---|---|---|
| **C1** | Operator pipeline: masks by family × acceleration factor × noise, seeded and reproducible, with the regime label carried through to calibration | T2.2 | Open |
| **C2** | Pretrained solver back-end behind a uniform interface — given measurements + operator, return N draws | T2.3 | Open |
| **C3** | Evaluation harness: coverage, flag area, region-level detection, automated logging | T2.4 | Open |
| **C4** | fastMRI+ annotations mapped onto the region definition agreed in Q2 | T2.2 | Open |

**Why C1's last clause matters:** group-conditional calibration (D3) groups *by operator*, so the regime label has to survive from the data loader all the way to the calibrator. If that is discovered late it means touching every stage of the partner's pipeline.
