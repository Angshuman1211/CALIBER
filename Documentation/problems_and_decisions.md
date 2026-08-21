# CALIBER — Problems & Decisions Log

Running log of every non-obvious design decision and every problem encountered during implementation. Update this file immediately when a new issue arises or a design choice is made. Entries are numbered sequentially; the originating task (blueprint v3.0 §18 work plan — T1.1, T1.2, …) is noted in brackets.

This repository covers **Track 1** — methodology, calibration and scientific direction. Tracks 2 (systems, data engineering, cross-domain validation) and 3 (analytical foundations and numerical verification) are partner-owned; cross-track dependencies are recorded here as *contracts*, not as work items.

> **Numbering warning.** Entries D1–D13 were written against blueprint v2.0, when the project had two tracks and Track 1 owned the coverage-gap bound. Blueprint v3.0 (August 2026) restructured this. The v2 entries are left **verbatim as a historical record** and are not retro-edited; where v3.0 supersedes them, D14 onwards says so explicitly. Section references inside D1–D13 are v2 section numbers.

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

## Blueprint v3.0 Revisions (August 2026)

### D14 — Three tracks, and the coverage-gap bound leaves Track 1 [Blueprint v3.0 §18]
v3.0 splits the work across **three** tracks rather than two: Track 1 methodology/calibration/scientific direction, Track 2 systems/data engineering/cross-domain validation, Track 3 analytical foundations and numerical verification. The consequence for this repository is a reallocation, not a rename: **the coverage-gap bound (D5) is now Track 3's** (T3.3 stability result, T3.4 ill-posedness scaling, T3.6 bound-versus-measurement). Track 1's T1.3 changes from "coverage-gap analysis" to **"operator-distance formulation"**, and Track 1 gains T1.5 cross-track integration and T1.6 analysis/course correction, with authoring pushed to T1.7. Track 2 is also renumbered: the operator pipeline is now **T2.1** (was T2.2) and the solver back-end **T2.2** (was T2.3). Every task reference in commits or branches predating 2026-08-04 uses the old numbering and must be read accordingly.

**Why this matters beyond bookkeeping:** Track 1 no longer derives the bound, but it still *defines the quantity the bound is stated in*. Q1 is therefore no longer a question Track 1 can defer to whoever gets there first; it is a Track 1 deliverable that Track 3 is blocked on. See D16.

### D15 — Second modality added as a first-class front, not an appendix [Blueprint v3.0 §8, §11]
v2.0 was MRI-only. v3.0 promotes **sparse-view and limited-angle CT** to Front II: the same failure and the same remedy, reproduced where the forward operator has a structurally different form while staying linear and known. Natural-image restoration drops to a low-cost robustness appendix (declared optional and cut first); non-linear scientific imaging is deferred. The substantive claim this buys is stronger and more specific than "it also works on CT": that **a single operator-distance measure, defined consistently across operator families, predicts the loss of coverage in both modalities**, so the two coverage-gap curves collapse onto a shared trend once normalised for ill-posedness. That collapse is the cross-domain contribution.

**Consequence for Track 1:** the operator-distance measure chosen in T1.3 must be definable for both a k-space mask and a projection-angle set. This eliminates any candidate phrased purely in terms of MRI sampling patterns (for example, raw symmetric difference of mask supports) unless it generalises. Subspace/projector distances on `P_null` do generalise, and are therefore favoured going in.

### D16 — Operator distance is a Track 1 deliverable and the Track 1 → Track 3 interface [T1.3]
Following D14, Q1 is owned by **T1.3 alone** (v2 recorded it as T1.2/T1.3, shared). It is the x-axis of the paper's central figure, it determines the weighting scheme (D4), and Track 3 cannot state a bound in "quantities computable from the operators alone" until the quantity exists. Sequencing: T1.3 selects and defines, Track 3's T3.2 analyses the candidates' properties, T3.3 derives against the selection. If T1.3 slips, Track 3 slips with it.

### D17 — Flow-matching demoted from novelty claim to solver choice; record the cost [Blueprint v3.0 §6.1, D7]
The gap analysis (`Operator-Shift-Robust … Gap Analysis`, July 2026) ranked this project top because **two** cells were verified empty as of mid-2026: operator-shift-robust conformal coverage, *and* flow-matching × conformal anything. v3.0 keeps the first and drops the second: D7 makes FlowDPS primary with diffusion as fallback and declares the contribution family-independent, and the §6.1 prior-work table no longer lists the generative family as a differentiator at all.

Decision recorded as-is — family-independence is the right *risk* posture, since the audit layer genuinely does not depend on the prior. But the cost is real and should be stated rather than discovered at review: the project now competes on one empty cell instead of two, against a group (Wen / Ahmad / Schniter) whose pipeline is a conditional normalizing flow plus E2E-VarNet, from which the flow-matching angle would have separated it cleanly. **Mitigation:** keep FlowDPS as the primary back-end in the headline results even though the claim does not rest on it, so the differentiator remains available in the write-up at zero extra cost.

### D18 — WI-2 was wrongly recorded as blocked on Track 2 [T1.1]
`T1.1_score_design.md` listed the range/null-space projectors as blocked on the Track 2 operator pipeline. They are not. For Cartesian MRI undersampling `A = M·F`, so `P_range = F* M F` and `P_null = I − F* M F` are computable from the mask alone with two FFTs. Track 2 supplies masks at scale with regime labels attached; it does not supply the decomposition. Corrected 2026-08-21. The same correction partially unblocks WI-4 and WI-5 (build against a mock C2-conforming sampler; validate the null-space term against synthetic `P_null·δ` injections, which are exactly data-consistent by construction and therefore hallucinations by this project's own definition). Details in the task doc.

**General lesson for the contracts table:** a cross-track contract is a dependency on *scale, labels, or artefacts*, not on mathematics. Anything derivable from a definition Track 1 already owns is not blocked, and marking it blocked costs a phase.

---

## Open Questions

### Q1 — "Operator distance" is not yet defined [OPEN, owned by T1.3 — see D16]
The coverage-gap-vs-operator-distance plot is the paper's central figure, and the x-axis of that figure is currently undefined. This choice determines the weighting scheme (D4) *and* the form of the bound (D5, now Track 3's), and Track 3 is blocked on it. Blueprint v3.0 §12.1 widens the candidate list to five: null-space projection distance; subspace and principal-angle distances (geodesic, chordal, projection); operator and pseudo-inverse norms; sampling-support difference; spectral divergence in singular-value decay or conditioning.

Three requirements, all binding:
1. **Computable without ground truth** — it is a function of the two operators only.
2. **Monotone in something a radiographer would recognise** as "a bigger change to the scan."
3. **Defined identically for both modalities** (D15) — it must accept a k-space mask *and* a projection-angle set. This is new in v3.0 and it prunes the list: sampling-support difference is the natural MRI-only candidate and does not obviously generalise; the projector and principal-angle distances do, since `P_null` exists for any linear A.

The T1.1 projector utilities (WI-2) are the substrate for the subspace-distance candidates, and following D18 they are unblocked, so this question is answerable now rather than after Track 2 delivers.

### Q2 — Region definition for the flags [OPEN, owned by T1.1]
"Per-region" is used throughout the blueprint without fixing what a region is — pixel, patch, superpixel, or annotated anatomical/pathological structure. It determines the multiplicity correction, the flag-area metric, and how fastMRI+ annotations are matched to flags. Note the tension: fastMRI+ labels are lesion-level bounding boxes, so evaluation naturally lives at that scale even if calibration runs on patches.

### Q3 — How many solver draws per reconstruction [OPEN, cost-driving, owned by T1.1]
The score combines null-space structure with cross-sample disagreement, so it needs several solver draws per measurement set. Each draw is a full diffusion/flow solve. Draws × calibration-set size × number of operator regimes is the dominant compute cost of the whole project and should be budgeted during T1.1, not discovered during T1.6. Following D18 the question splits: the *statistical* half (draws needed before the disagreement term stabilises) is measurable now on a mock sampler; the *cost* half (seconds per draw) waits on T2.2. Multiply at integration.

### Q4 — What experiment shows weighted recalibration beating the Mondrian backbone? [OPEN, owned by T1.4]
D3 makes Mondrian the guaranteed backbone and D4 makes operator-distance weighted recalibration the headline contribution. But Mondrian is *exact* in the known-finite-operator regime and weighted CP is only approximate there (D6), so a reviewer will reasonably ask why the approximate method is the contribution. The background research reached the same conclusion independently, calling group-conditional CP "a serious competitor to the whole merger, and arguably the honest primary method."

The answer exists and is defensible — weighting buys what Mondrian structurally cannot: (a) **thin groups**, where per-regime calibration data is too scarce for a stable threshold, exactly the high-acceleration regimes D10 warns about; and (b) **continuous operator change**, where there is no finite group to calibrate within. But neither case is currently an explicit line in the blueprint's experimental protocol (§16), which means the paper's headline method has no experiment that demonstrates its advantage over its own baseline. T1.4 must name that experiment: sweep calibration-set size per regime down to where Mondrian's thresholds become noisy and weighting wins, and include off-grid deployment operators that no group covers. Report both, not just the coverage curve.

### Q5 — Scoop recheck is overdue [OPEN, owned by T1.6, standing]
The two "verified empty" cells (operator-shift-robust conformal coverage; flow-matching × conformal) were verified as of **mid-2026**, and the research documents explicitly flag that absence of evidence is not proof, naming VOILA (Fridovich-Keil), Schniter/Ahmad and Sulam as the groups to monitor monthly across cs.CV, eess.IV and stat.ME. D11 already mandates an early preprint for priority. The sweep has not been repeated since the research was written. Repeat it before further design work, and again at write-up.

---

## Track 1 Task Log

Per-task records live in `T1.<n>_*.md`. Decisions and problems arising during a task are logged below as `T1.<n>-D<k>`. Task names follow blueprint **v3.0 §18** (see D14 for the change from v2).

### T1.1 — Operator-aware score design
See `T1.1_score_design.md`. Branch `T1.1`.

_No implementation entries yet. Design correction D18 (WI-2 unblocked) logged 2026-08-21._

### T1.2 — Shift-robust calibration
_Not started._

### T1.3 — Operator-distance formulation
_Not started. Owns Q1; Track 3 is blocked on it (D16)._

### T1.4 — Experimental design
_Not started. Owns Q4._

### T1.5 — Cross-track integration
_Not started. New in v3.0._

### T1.6 — Analysis and course correction
_Not started. New in v3.0. Owns Q5._

### T1.7 — Technical authoring
_Not started._

---

## Cross-Track Contracts

Tracks 2 and 3 are partner-owned. Anything Track 1 needs from them, and anything they need from Track 1, is recorded here so the interface is explicit and dated rather than discovered at integration. Task numbers are **v3.0** (D14).

A contract is a dependency on *scale, labels, or artefacts*. It is not a dependency on mathematics Track 1 already owns — see D18 for what that error cost.

### What Track 1 needs

| # | Track 1 needs | From | Status |
|---|---|---|---|
| **C1** | Operator pipeline: masks by family × acceleration factor × noise, seeded and reproducible, with the regime label carried through to calibration | T2.1 | Open |
| **C2** | Pretrained solver back-end behind a uniform interface — given measurements + operator, return N draws | T2.2 | Open |
| **C3** | Evaluation harness: coverage, flag area, region-level detection, automated logging | T2.4 | Open |
| **C4** | fastMRI+ annotations mapped onto the region definition agreed in Q2 | T2.1 | Open |
| **C5** | CT operator pipeline: projection-angle sets, angular-range restrictions and dose levels, same regime-label contract as C1 | T2.5 | Open, Phase 3 |
| **C6** | The derived coverage-gap bound in evaluable form, so measured gap can be compared against it | T3.3 / T3.6 | Open, Phase 4 |

**Why C1's last clause matters:** group-conditional calibration (D3) groups *by operator*, so the regime label has to survive from the data loader all the way to the calibrator. If that is discovered late it means touching every stage of the partner's pipeline.

**Why C2 does not block the score:** the interface is the contract, not the implementation. Build against a mock that honours it (D18).

### What Track 1 owes

| # | Track 1 supplies | To | Status |
|---|---|---|---|
| **S1** | The operator-distance definition (Q1 / T1.3) — the quantity the bound must be stated in | T3.2, T3.3 | Open, blocking |
| **S2** | The region definition (Q2 / T1.1 WI-1) — fixes the multiplicity correction, the flag-area metric, and how annotations are matched | T2.1 (C4), T2.4 | Open, blocking |
| **S3** | Score-bank format and the non-conformity score specification | T2.3, T1.5 | Open |
| **S4** | Experimental protocol, operator grids and ablation matrix | T2.5, T2.6, T3.6 | Open |

S1 and S2 are both **blocking for partners** and both currently open questions. They are the two things to close first.
