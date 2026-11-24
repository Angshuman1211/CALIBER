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

## Week 1 Sign-Off Rulings (August 2026)

These six close the blocking items in `Documentation/week1_kickoff.md` §G. Cross-track IDs in Track 2's
`Documentation/decision_log.md` are noted per entry — **`D-0NN` there, `DNN` here; they are different logs.**

### D19 — Primary operator distance is the Frobenius null-space projector distance [T1.3, resolves Q1 / S1; Track 2 D-010]
**Ruling.** The primary measure is `null_space_projection` — $\|P_{N}(A_\text{cal}) - P_{N}(A_\text{dep})\|_F$ — with `ambient` normalisation. Locked for Week 3. Ill-posedness normalisation is **not** folded into the distance.

**Why this one.** It satisfies all three Q1 requirements: computable from the two operators alone; defined identically for a k-space mask and a projection-angle set, because $P_N$ exists for any linear $A$ (D15); and interpretable — under `ambient` normalisation on Cartesian MRI it reduces to $\sqrt{\text{fraction of k-space lines whose sampled/unsampled status changed}}$, which is exactly "how much more of the scan is now unmeasured." It is also the aggregate form Track 3's T3.2 §5 recommends, so this ruling ratifies a convergence rather than choosing between the tracks.

**Verified, not assumed** (128², `random1d`, calibrate at R=2): the measure is graded and monotone across R = 2, 3, 4, 6, 8, 12, 16 → 0.000, 0.405, 0.500, 0.580, 0.612, 0.644, 0.650. T3.2 §3's saturation failure does not occur, because Track 2 implemented the **Frobenius** form, not the operator-norm form that pins at 1 under dimension mismatch.

**Rejected, with reasons.**
- *Operator-norm form of family (A)* — saturates at exactly 1 whenever the null-space dimensions differ, which is the typical case across an acceleration sweep (T3.2 §3). **Not currently implemented; do not add it** under a name that invites confusion with the primary.
- *`sampling_support`* — verified numerically **identical** to the primary in MRI (both 0.612372 at R=2→8), because MRI's operator is diagonal in the Fourier basis so the two coincide exactly (T3.2 §2). In CT it is only a heuristic. Retained as a free MRI-only diagnostic; never the headline.
- *`chordal_subspace`* — the same number divided by $\sqrt2$. Alias, not an alternative; no separate experiment.
- *`spectral_divergence`* — measures severity of ill-posedness, not direction of change, so it belongs to L2 (T3.2 §1); D-011 also shows it barely separates CT operators.

**Two consequences that must be acted on, not just noted.**
1. **The axis compresses.** The measure is a square root, so R=2→3 moves it 0.405 while R=8→16 moves it 0.037. The headline plot's most severe points bunch into a sliver. **T1.4 must design operator grids to spread *distance* evenly, not acceleration evenly**, and the interpretable secondary axis is $d^2$ (the plain fraction of dimensions whose measured status changed).
2. **Naming collision.** `operator_norm` in `operators/distance.py` is $\|A-B\|_F$ — family (B) — *not* the family-(A) operator-norm T3.2 warns against. Two different things with confusable names; document at the call site.

**Why ill-posedness normalisation stays out.** §11.2's collapse claim and §12.2's L2 both involve ill-posedness, and folding it into $d$ would double-count: L1's bound is gap $\le 2ML\cdot d_\text{op}$, and L2's whole job is to express $L$ and $M$ in terms of null-space dimension and spectral decay. If $d$ already absorbs that, L2's scaling becomes unidentifiable. So $d$ stays a pure geometric property of the operator pair, and ill-posedness enters at plot time as a separate, **named** factor. The Week 9 collapse test then has something to test.

### D20 — The region is a fixed patch grid; annotations are an overlay, not the unit [T1.1 WI-1, resolves Q2 / S2; Track 2 D-009]
**Ruling.** The risk unit stays the non-overlapping square patch grid, 16 px on the 320×320 reference grid. Annotations (fastMRI+ boxes, CT organ masks) are an **evaluation overlay**, never the region definition. `region_area` therefore stays constant and B1's weighted-rate problem does not arise.

**Why.** Three reasons, in order of weight.
1. **Commensurability.** §11 requires one region definition in both modalities. MRI ships lesion bounding boxes; CT ships 117 organ segmentations. Those are not the same kind of object, and no single anatomical unit spans them. A geometric grid does.
2. **Hallucination does not respect anatomy.** Invented structure lives in the null space, which is a property of the operator, not of the organ. An annotation marks where a hallucination would *matter clinically*, not where it *is*. Conflating the two would make the risk definition itself depend on annotation availability — and CT's annotations come from a different cohort entirely (D-013).
3. **Unweighted rates.** Equal-area regions keep every region-level rate a plain proportion. Anatomical regions would make each rate area-weighted, which changes the meaning of the controlled risk, not merely its computation.

**How C4 is satisfied:** intersect flagged patches with annotation masks. `ann_coverage` already carries exactly this, and `label_hallucinated(require_annotation=True)` already implements the stricter "fires where it matters" reading. Report both readings; the annotation-restricted one is the secondary.

**Open sub-item handed to T1.4:** region size (16 px) is a declared ablation axis, not a settled constant. It trades detection resolution against per-region calibration sample size, and both directions have a cost. Sweep it.

### D21 — Do not recalibrate the label threshold yet; the current measurement is a solver artefact [T1.1, answers Track 2 D-020]
**Ruling.** The Week 2 criterion is **not** a re-tuned constant, and the threshold is **not** set from the M4Raw run. Lock the criterion's *form* now; set its threshold only once a real generative prior is wired (T2.2).

**Why — the 100%-positive rate is not primarily a threshold problem.** `w02_m4raw_smoke` uses `reference_nullspace`, whose draws are $A^{+}y + P_N(\text{texture})$. Because $A^{+}y$ carries **zero** null-space content by construction, $P_N(\overline{x} - x_\text{true})$ is dominated by $-P_N(x_\text{true})$. So on this sampler `err_free` is measuring **the true image's own hidden content**, not what the reconstruction invented. Real anatomy has a lot of it; a smooth phantom has little. That is the whole phantom-vs-M4Raw gap.

**Verified on synthetic images alone** (128², R=4, reference sampler, 16 px grid), which reproduces Track 2's substrate split without touching M4Raw:

| true image | mean `err_free` | $\|P_N x_\text{true}\|$/ref | labelled at 0.10 |
|---|---|---|---|
| smooth (phantom-like) | 0.0715 | 0.0771 | 14.1% |
| textured (anatomy-like) | 0.1878 | 0.2046 | **100%** |

Across regions, `err_free` vs $\|P_N x_\text{true}\|$: **r = 0.95**, mean ratio 0.92. This also explains the `err_free`/`err_total` ratio Track 2 observed moving 0.78 → 0.97: as the truth's null-space content grows, essentially *all* of a prior-free reconstruction's error becomes free-component error.

**Consequence:** no constant — absolute *or* relative-to-global-intensity — can fix this while the sampler has no prior, because the quantity being thresholded is a property of the anatomy. Note `err_free` is *already* divided by `ref_intensity` in `labels.py`, so "make it relative" has in effect already been tried.

**Criterion form v1 (locked now).** Normalise by the region's own free-component content rather than by global intensity:

$$\text{halluc}(r) \iff \frac{\texttt{err\_free}(r)}{\texttt{free\_content}(r) + \epsilon} > \tau, \qquad \texttt{free\_content}(r) = \|P_N x_\text{true}\|_r / \texttt{ref}$$

Read plainly: *what fraction of this region's hidden content did the reconstruction get wrong* — and $>1$ means it invented more than was ever there. Using $x_\text{true}$ is legitimate here; a **label** may use ground truth, only the **score** may not. This quantity is dimensionless, substrate-independent, and modality-independent by construction, which is what D-012 and D-020 jointly demand.

**Ask to Track 2 (cheap, no schema bump):** add one nullable column `free_content` = $\|P_N x_\text{true}\|$ per region / `ref_intensity`. Per score-bank spec §6 an added nullable column is backwards-compatible. With it, the criterion is re-derivable from cached values forever (Rule 2), and this ruling costs zero GPU seconds.

**Diagnostic value of the v1 form:** on `reference_nullspace` the ratio sits near 0.92 everywhere, i.e. it correctly reports *"this sampler does not invent — it simply fails to fill the null space."* A criterion that says something true about a stub sampler is a criterion worth trusting on a real one.

**τ is set later, and its positive rate is reported, not targeted.** Once T2.2 lands, choose τ once on held-out data and report the induced positive rate per substrate as a **measured property**. Do not tune τ to hit a target rate — that would make the positive rate definitional and hollow out region-level FDR and power.

### D22 — `calib_group` keeps its granularity and moves into the registry [T1.2, answers B2/C2; Track 2 D-021]
**Ruling.** Granularity is unchanged — `mri/{family}/R{acceleration}`, `ct/{family}/v{n_views}` — and the definition **moves out of `runner.calibration_group` into the operator registry** as a field, so Track 1 can re-group without a Track 2 code change. Agreed with Track 2's recommendation on both halves.

**Noise is not a group axis in v1.** Adding it multiplies the group count, and D10 already warns that high-acceleration regimes are where per-regime calibration data is scarcest — precisely where Tool 1's declared trade-off bites hardest. If T1.4's protocol varies noise as a studied axis, it *becomes* a group axis and the registry field must reflect it. Recorded here so that stays a decision rather than a silent default.

### D23 — Split is fixed per patient in the registry, and must be operator-independent [T1.2, answers B3; Track 2 D-022]
**Ruling.** Split assignment is deterministic per `sample_id`, made at **patient/volume** level (D-014), stored in the registry, and identical across every run, every operator, and both modalities. Never assigned per run.

Track 2's two reasons stand: per-run splits make two banks non-mergeable and silently break the Week 10 merge, and slice-level splits leak between calibration and test through adjacent slices.

**Track 1 adds a third, which is the binding one.** The split must be **operator-independent**: the same patient must fall on the same side in *every* regime. Otherwise group-conditional calibration compares different patient populations across regimes, and a coverage difference between R=4 and R=8 is confounded with anatomy rather than attributable to acquisition. That is the §13 controlled-comparison assumption applied to the split, and violating it would undermine the headline claim rather than merely inconvenience the merge.

### D24 — One calibration group per (family, acceleration); seeds are members [T1.2, answers C3; Track 2 D-023]
**Ruling.** A mask family at a fixed acceleration is **one** group. Different `mask_seed` values are members of it, not groups of their own.

**Why.** Seeds drawn from the same family at the same acceleration are exchangeable draws from a single acquisition regime — and that exchangeability is exactly what licenses using them as calibration data for that regime. Splitting by seed would discard the only within-regime exchangeability the design actually has, while starving each group (D10, Tool 1's trade-off).

**This also defines the Week 4 grid's axes,** which is the part that matters beyond grouping: a *mask-family shift* means changing family at fixed acceleration; an *acceleration shift* means changing R at fixed family. Both are between-group moves. Seed variation is within-group and is therefore the natural noise floor against which a real shift must be judged — **T1.4 should report it as such**, since a coverage change smaller than seed-to-seed variation is not a finding.

---

## T1.1 WI-3 / WI-5 — Score Combination (August 2026)

### D25 — The combination is affine in the components, by requirement of the theory track [T1.1 WI-3]
**Ruling.** `score(r) = Σᵢ wᵢ · sᵢ(r) / scaleᵢ`, with non-negative weights. Not a ratio, not a gate, not a product.

**Why the form is constrained rather than chosen.** T3.3 §3's assumption (ii) needs the score to be Lipschitz in the D19 operator distance. Any non-negative weighted sum of Lipschitz components is Lipschitz with `L = Σ wᵢLᵢ` — the constant follows immediately. A ratio such as `s_disagree / s_free`, which is otherwise attractive ("what fraction of the hidden content is unstable?"), is **not** Lipschitz: it diverges as the denominator vanishes, which is exactly what happens in empty background regions. Choosing it would leave T3.3 unable to close L1 for reasons that have nothing to do with the operator.

This is T3.3 §5's request acted on rather than deferred — it asked that the smoothness property be *designed in* rather than checked afterwards.

**Non-negativity is load-bearing too:** it keeps the score monotone in each piece of evidence, so more unexplained free-component structure can never *lower* a score. A negative weight would let two kinds of evidence cancel into a confident-looking zero.

**Handed to T3.3.** Each component is Lipschitz in `d_op` **with the reconstruction held fixed** — e.g. `‖P_N^A x̂ − P_N^B x̂‖ ≤ ‖P_N^A − P_N^B‖_F ‖x̂‖ = d_op‖x̂‖`. But `x̂` itself depends on the operator, through the solver, which is a black box (D1). So the honest statement available to L1 is Lipschitz-conditional-on-the-reconstruction; the remaining dependence flows through solver stability and cannot be bounded analytically here. T3.3's assumption (ii) should be split accordingly rather than assumed whole.

### D26 — Standardise the components before weighting [T1.1 WI-3]
**Ruling.** Each component is divided by a fixed reference scale before weighting. Scales are a robust quantile (default 0.75) pooled over regions.

**Why.** The three components differ by roughly an order of magnitude, so the v0 placeholder's `1,1,1` default was in practice *"score = s_free"* — the other two contributed noise, not evidence. Standardising is what makes the weights mean what they say, and what makes the T1.4 ablation interpretable at all.

**Why the 0.75 quantile and not the median:** medical images are background-heavy. A large fraction of regions is near-empty, which drags the median of every component toward zero and makes the division numerically unstable. A mid-high quantile sits inside the anatomy while staying insensitive to the extreme tail. A floor guards the degenerate single-draw case, where `s_disagree` is identically zero and would otherwise produce NaNs that propagate silently into the bank.

### D27 — Scales come from the calibration operator and are never re-estimated [T1.1 WI-3, protects the headline result]
**Ruling.** `ScoreScales` is built once, from calibration-operator data, and applied unchanged to every deployment operator.

**Why this is not a detail.** Re-estimating the scales per deployment operator would map each operator's score distribution onto a common spread — which is *precisely* the difference the coverage-gap experiment exists to measure. It would make the naive-transfer baseline look artificially healthy and shrink the headline curve toward zero. The project would then be measuring its own normalisation.

Pinned by a regression test (`tests/test_score_combination.py::test_scales_are_frozen_not_re_estimated_per_operator`) that asserts the frozen-scale score drift between R=2 and R=8 exceeds the re-fitted one. It is a behavioural test, not a style check.

### D28 — Weights are selected on a design split and frozen before calibration [T1.1 WI-3]
**Ruling.** Non-negative weights summing to 1, selected on a **design split disjoint from both calibration and test**, then frozen.

**Why.** Conformal calibration chooses a *threshold*, not the weights. Fitting weights on calibration data would leak the calibration set into the score definition and quietly void the exchangeability the guarantee rests on. Three-way splitting is the standard remedy and costs only sample size. Until the design split exists, the default is the uninformative `1/3, 1/3, 1/3` on standardised components.

### D29 — Null-space projections must stay complex; `.real` breaks data consistency [T1.1, referred to Track 2 as D-024]
**Ruling.** Never take `.real` of a null-space projection in this codebase. Carry the complex value; take magnitudes only at the point of measurement.

**Measured.** A random 1-D Cartesian mask is **not conjugate-symmetric**, so `P_N` of a real image genuinely has an imaginary part — here **33%** of its norm. Injecting the full complex `P_N·δ` leaves the measurements unchanged to 9.6e-8 relative (float noise). Injecting only its real part changes them by **4.1% relative** — the perturbation is no longer confined to the null space, and the construction stops being a hallucination by this project's own definition (D2).

Found while building the WI-4 injection test, which failed its own premise until this was fixed. That failure was the test working correctly.

**Referred to Track 2 (D-024):** `solvers/reference.py` applies the same `.real` cast to `P_N(texture)` in `NullSpaceNoiseSampler`. Its draws therefore perturb the *measured* component by a few percent, so it is not the "faithful caricature" its docstring describes: `s_residual` will be non-zero as a pure artefact, and any smoke number taken from it — the README table, D-020's coverage figures — carries that confound. Not urgent (nothing from that sampler is reportable, D-018), but it should be fixed before the sampler is used to characterise anything.

---

## Phase 1 Gate (August 2026)

Full assessment in `T1.6_analysis_and_course_correction.md` (Phase 1 Gate section).

### D30 — Scoop recheck done; the targeted gap is open, but one group is a single experiment away [T1.6, resolves Q5's current instance]
**The gap stands.** Nothing found calibrates conformal prediction at one acquisition operator, deploys at another, and restores coverage. The §6 landscape table is still accurate.

**Two adjacent results have closed in, and one changes the urgency.**

*arXiv 2601.13236* (Jan 2026) applies conformalized quantile regression to accelerated MRI across **acceleration factors 2–10** and reports that predictors *calibrated per acceleration rate* hold coverage at their matching rate. That is CALIBER's matched-operator upper-bound baseline, already published, by a group that therefore has the substrate, the sweep and the conformal machinery assembled. The mismatched cell — this project's entire claim — is one experiment away for them. **D11's early preprint moves from prudent to urgent.**

*CHEM* (arXiv 2512.09806, Dec 2025) is a conformal, distribution-free hallucination-localisation metric whose null-space hallucination map decomposes the reconstruction by SVD of the forward operator. It compares against ground truth, so it is an evaluation metric, not a deployment score — nearer CALIBER's `err_free` label than its `s_free` score. Independent convergence on the D21 criterion's form; cite it as corroboration.

*Wen, Ahmad & Schniter, Minimax Multi-Target CP* (TMLR, Nov 2025) — adjacent, different problem. The group remains the one to monitor.

**Sourcing caveat:** abstract- and HTML-level reads. CHEM's PDF exceeded the fetch limit, so its Null-HM description comes from the abstract and summary text. Verify against the method section before citing specifics.

**Q5 stays standing** — repeat at write-up, per its original terms.

### D31 — Novelty rests on operator shift alone; the decomposition is not a contribution [T1.6, framing constraint]
Following D30: "conformal + distribution-free hallucination localisation" (CHEM) and "null-space/range decomposition of the forward operator" (CHEM's Null-HM, and the fundamental-limits line before it) are both **published under a fixed operator**. Neither is available to claim.

**What CALIBER contributes is the operator-shift axis and nothing else on this list:** the demonstrated coverage failure under acquisition change, its restoration by group-conditional and operator-distance-weighted calibration, the cross-modality collapse, and the operator-distance bound. The operator-aware score is *machinery in service of that*, not a result.

**How to apply:** in the manuscript, the score section is method, not contribution. Cite CHEM at the point the decomposition is introduced rather than after a reviewer finds it. D2 remains a correct design decision; it is simply not a novel one, and the earlier framing in D2 and `T1.1_score_design.md` reads more proprietary than the literature now permits. This is the D6/D17 discipline — claim narrowly, on the record — applied to the score.

---

## T1.2 — Shift-Robust Calibration (September 2026)

Full record in `T1.2_shift_robust_calibration.md`.

### D32 — Control the region-level miss rate; floor the quantile; refuse thin groups [T1.2]
**Ruling.** The controlled quantity is the **region-level miss rate** — among genuinely hallucinated regions, the fraction the flag fails to raise. Risk control (D9), not marginal coverage. A region is flagged when `score >= tau`, and calibration takes the `k`-th smallest score among that group's hallucinated calibration regions with **`k = floor(alpha*(n+1))`**.

**Why flooring, not ceiling.** For exchangeable scores the new positive's rank is uniform on `1..n+1`, so `P(miss) = k/(n+1) <= alpha` exactly. Rounding *up* breaks this on thin groups: at `n=1, alpha=0.1` a ceiling gives `k=1`, so `tau` is that single score and the true miss rate is **1/2 by symmetry** — five times the promise, reported as certified. Measured at 0.5 in `test_a_ceiling_rule_would_over_promise_on_one_positive`.

**Thin groups refuse rather than certify.** With `n` positives the smallest achievable bound is `1/(n+1)`, so level `alpha` needs `n+1 >= 1/alpha` — **9 positives at `alpha=0.1`**. Below that, no threshold is valid and the calibrator returns `-inf` (flag everything: zero misses by construction, visibly useless) with `certifiable=False`. This makes D3's declared "thin groups give noisy thresholds" trade-off *checkable* rather than merely acknowledged, and `CalibrationReport.reportable` gates the whole run on it.

**Every fallback path is conservative downward.** No positives, an over-heavy test atom, an unknown group at apply time — all flag everything. Under-flagging would silently assert that nothing in an uncalibrated regime is hallucinated.

**Defect found in the code this supersedes — referred to Track 2 as D-026.** `calibration/__init__.py`'s reference `split_conformal_threshold` has a docstring that says the `ceil((n+1)(1-alpha))`-th smallest score while the code takes the `ceil((n+1)*alpha)`-th. The code is nearer correct, but its ceiling certifies groups it should refuse. It is marked not-for-reported-results, so nothing published is affected; it should still be fixed before someone reuses it.

### D33 — Weights are an exponential kernel in operator distance, and the result is approximate [T1.2, D4's mechanism]
**Ruling.** `w_i = exp(-d_op(A_i, A_dep) / h)` with `d_op` from D19 and `h` defaulting to the median non-zero distance. The lower-tail weighted quantile follows Tibshirani et al. (2019) with the test point contributing an atom at `-inf`.

**Why a kernel and not a cut-off.** A hard radius is a Mondrian group with extra steps, and it discards the near-neighbours carrying most of the usable information about a nearby operator. The exponential decays smoothly, so a deployment operator sitting *between* two calibration regimes borrows from both — precisely the case Tool 1 structurally cannot serve, and therefore the case that justifies Tool 2 existing at all.

**Why the median bandwidth.** Scale-free, and it adapts to whatever operator grid the experiment uses. A fixed `h` in absolute units would silently change the method's behaviour when the grid changes — the same confound T1.4's grids are designed to avoid. To be re-chosen on the design split once real banks exist.

**Honest scope, stated at the call site.** Tibshirani's guarantee holds when the weights are the true likelihood ratio between deployment and calibration. A kernel in operator distance **is not that ratio**, and operator shift is a measurement-process change rather than a covariate shift (D6). So Tool 2 carries **no exact finite-sample guarantee**. Tool 1 holds the exact statement, Tool 2 is the reach result, and the residual gap is what T3.3's bound quantifies. ESS is reported on every weighted calibration, and gates certifiability, so the cost of reweighting is visible rather than inferred.

### D34 — The naive baseline must name the operator it calibrated at [T1.2]
**Ruling.** `calibrate_naive` takes an explicit `calibration_group`.

**Why this is not an API nicety.** §16's first bullet is *calibrate at one acquisition setting and apply at others*. The first implementation calibrated on whatever rows it received; a test measured the consequence of pooling every regime instead — the miss rate on the deployment regime fell from **0.76 to 0.19**, making naive transfer look four times healthier than it is. The baseline exists to expose the coverage collapse, so understating it would have quietly weakened the paper's central empirical claim while every test still passed. Pinned by `test_pooling_every_regime_understates_the_naive_failure`.

### D35 — `s_residual` is structurally zero for data-consistent solvers; exclude degenerate components rather than standardising them [T1.1, found via D-028]
**The finding.** Track 2's D-028 recorded `s_residual = 0.000000` after the D-024 fix. That is not a small number, it is a **structural** one. The reference sampler builds draws as `A⁺y + P_N(z)`, so `A·x̂ − y = 0` identically. Measured: residual `1.1e-7` (float noise) for a data-consistent sampler against `6.7e-3` when the measured component is perturbed — five orders of magnitude.

**Why this generalises past the mock.** **DDNM enforces hard data consistency by construction**, and it is one of the project's back-ends. For any such solver the third score component carries no information at all, so D25's three-component design is a *two*-component design, and T1.4's ablation will show a flat residual row that means "structurally absent", not "tried and unhelpful".

**The bug this exposed in D26.** Standardisation rescales every component to O(1). Dividing a structurally-zero component by its own tiny scale therefore **amplifies float noise to the magnitude of real signal**: measured scale `9.1e-8` against `3.5e-2` for `s_free`, standardising to mean 0.95 — indistinguishable from a live component — and then collecting a full 1/3 of the weight budget. Correlation between the score with and without the residual term was 0.944, so the noise was measurably degrading the ranking.

**Ruling.** `ScoreScales.from_components` marks any component whose scale falls below `1e-6` as **degenerate**; `combine_components` drops those and renormalises the remaining weights. The threshold is absolute because the components are already normalised by reference intensity and so share an O(1) scale — anything at 1e-6 or below is FFT round-trip noise. Renormalising rather than merely zeroing keeps the score on a comparable scale across solvers where the residual is live and solvers where it is dead. Degeneracy is decided once, from the calibration operator, and carried unchanged to deployment — the same freezing rule as D27, for the same reason. All three degenerate raises rather than silently returning zeros.

**`s_residual` is retained, as a guard rather than a detector.** A non-zero residual means the solver has drifted off the data-consistent set — under-converged, or not enforcing consistency at all — which is worth catching. What it cannot be counted on for is detection power. **T1.4 must report per-component detection per solver**, since the residual row's meaning changes with the back-end.

### D36 — Anchor ignore patterns; a bare `data/` matched the source package too [T1.5, self-inflicted]
The D-025 consolidation replaced `data/*` + `!data/README.md` with a bare `data/`. `data/*` has an internal slash and is anchored to the repository root; `data/` has none and therefore matches **any directory of that name at any depth** — including `src/caliber/data/`.

Consequence: `src/caliber/data/splits.py` could never be committed, while three committed files import it. `caliber/__init__.py` imports the data package eagerly, so the whole package failed to import — two consecutive Track 2 pushes landed unusable, and it read as their omission rather than a Track 1 config error.

Fixed to `/data/`, verified in both directions, with a comment marking the slash as load-bearing. Recorded cross-track as D-029.

**How to apply:** an over-broad ignore rule fails *silently and asymmetrically* — tracked files keep working while new ones vanish — so it disguises itself as someone forgetting `git add`. When a partner's push arrives with a missing module, check `git check-ignore -v` on the path before assuming they forgot it.

### D37 — tau is global across operators, chosen on a design split; provisionally 1.75 [T1.1/O2, answers D-034]
Track 2 discharged their half of D-034 — the relative criterion is implemented and its tau curve measured — and left the value to Track 1. Three rulings.

**1. tau is global. Never per calibration group.** D-034 cautions that the median ratio rises with acceleration (1.480 at R=4, 1.645 at R=8) and suggests a per-group tau if that survives a medical prior. **Take the opposite branch**, because the reasoning inverts on inspection:

- Coverage is *conditional*: `P(flag | hallucinated)`. A base rate that differs across operators does not mechanically bias it, so a rising positive rate is not by itself a confound.
- With a **global** tau, "hallucinated" means the same thing at every operator. The positive rate is then free to vary — and *should*, because a larger null space genuinely permits more invention. That is the project's own premise (D2) and exactly what §12.2's L2 formalises: an operator that hides more permits weaker guarantees.
- With a **per-group** tau tuned to equalise positive rates, "hallucinated" means something different at each operator. Coverage at R=8 would then be measured against a different ground truth than coverage at R=2, and the headline coverage-vs-distance curve would compare incommensurable quantities. **That** is the confound, and it is the one that would be fatal.

So the rising positive rate is a **result to report, not an artefact to normalise away** — arguably independent evidence for the premise. This is D23's operator-independence principle applied to the label rather than the split: the definition must not move with the operator.

**2. tau is selected on the design split, then frozen** — the same rule and the same reason as D28's weights. Picking tau by looking at the data you then calibrate on couples the label to the threshold and leaks the calibration set into the ground truth. Design split → freeze → apply unchanged to calibration and test.

**3. Provisional value: tau = 1.75**, for development only, explicitly not reportable.

Chosen over 2.0 on a calibration-arithmetic argument rather than taste. On the 384-region bank, tau=1.75 gives 25.5% positives (~98 regions) and tau=2.0 gives 10.7% (~41). D32 requires **9 positives per group** to certify at alpha=0.1. Spread across ~6 operator groups that is ~16 each at 1.75 — certifiable — against ~7 each at 2.0, which is *below the floor and would make Mondrian unreportable everywhere*. The label threshold and the calibration floor are coupled, and 2.0 breaks the coupling.

**Not reportable, and the reason matters.** The bank is DDNM with `ddpm-celebahq-256` — a **face** prior on brain anatomy (D-016, D-035). Every region has ratio > 1, i.e. the reconstruction always puts more into the free component than the truth had there, which is what a badly mismatched prior should do. tau read off this bank measures the mismatch as much as the criterion. Re-measure when medical weights land.

**A falsifiable check for that re-measurement.** With a *matched* medical prior, some regions should come back with ratio < 1 — places where the prior reconstructed the hidden content better than a shrug. If the ratio still exceeds 1 everywhere with medical weights, the fault is not the prior and either the criterion or `free_content` needs re-examining. Stating it now so the re-measurement has a pass/fail condition rather than a vibe.

**Superseded in part by D40.** The "some regions should show ratio < 1" condition is right, but it was stated without its precondition, and the reading of "ratio > 1 everywhere" as evidence of a mismatched prior is wrong. See D40.

---

### D38 — The bank's three free-component columns satisfy an exact identity, and Q6 is answered through it [T1.1, Q6]
`runner.py:184` passes `components.ref_intensity` into `compute_region_truth`, so `s_free`, `free_content` and `err_free` share one normalisation. All three are region-wise RMS reductions of the same posterior mean through the same projector. Writing `a = P_N x̂` and `b = P_N x_true` over one region:

```
err_free² = s_free² + free_content² − 2 · s_free · free_content · cos θ
```

exactly — not approximately, and not as a modelling assumption. **Verified at 0.00e+00 median and maximum relative residual across the 4,608 regions of a runner-produced bank**, which also makes it a cheap integrity check on any future bank: a non-zero residual means the columns were normalised differently and every ratio computed from them compares quantities in different units. `identity_residual()` runs it, and `q6_attribution.py` refuses to proceed when it fails.

Three things follow, all from the algebra rather than from data:

**1. The relative criterion is monotone in one ratio, and it is not `s_free`.** With `u = s_free / free_content`, the label `err_free/free_content > τ` becomes `u² + 1 − 2u·cos θ > τ²`. At fixed alignment the label is monotone in `u`. So the statistic matched to the label is **`s_free / free_content`** — and `free_content` needs ground truth. A flag built on `s_free` alone is mismatched to its own label by exactly as much as `free_content` varies across regions, which is Q6's mechanism, derived rather than suspected. Measured on the local phantom bank: the oracle ratio flags **0.717 of the area against `s_free`'s 0.843 at identical coverage** — a 15% reduction for the same guarantee.

**2. `cos θ` is recoverable per region, and it is the quantity Q6 actually wants.** `cos θ → 1` means the prior *recovered* the hidden anatomy, so content in the free component is reconstruction and a flag on it is a false positive by Sec. 4's own definition; `cos θ → 0` means invention. On the phantom bank `cos θ` has median **0.000**, which independently confirms T3.4's analytic finding that the reference sampler's null-space fill is injected white noise orthogonal to image content.

**3. The fix is a denominator that exists at deployment.** `‖P_R x̂‖` — the region's *measured* content — needs no ground truth, and `s_free / s_range` is then "how much did the prior invent relative to what it actually saw", dimensionless and operator-aware, the exact mirror of the label's `err_free / free_content`. Whether it is a good substitute is an empirical question (does `free_content` track `s_range`?) and is **ask A2**. Until that column exists the oracle ratio is scored alongside the raw component, clearly named, as the ceiling.

*Cost:* the identity holds only while both sides share `ref_intensity`. That coupling is now load-bearing and must not be "tidied" — the same class of hazard as D36's ignore pattern.

---

### D39 — Component ablations report flagged area at fixed alpha, never power [T1.1/T1.4, refines Q6's requirement]
Q6 asks T1.4's ablation to report per-component FDR and power. **Power is the wrong axis and reporting it would mislead.** A conformal threshold pins the miss rate at `α`, so every candidate score has power `≈ 1 − α` whatever it measures — the number compares calibrators, not scores.

Measured, on the local phantom bank at `α=0.1`, five candidates from an excellent one to a structurally dead one:

| candidate | power | flag area |
|---|---|---|
| oracle `s_free/free_content` | 0.9022 | **0.717** |
| combination | 0.9022 | 0.842 |
| `s_free` | 0.9022 | 0.843 |
| `s_disagree` | 0.9022 | 0.873 |
| `s_residual` (degenerate, D35) | 0.9022 | 0.949 |

Power is identical to four decimal places across all five. Flag area separates them cleanly, and it is Sec. 15.2's own second metric ("flag area at fixed reliability"). FDR is reported too and moves with area, as it must at fixed coverage.

**So T1.4's ablation matrix reports flagged area at fixed `α` as the comparison, with power and miss rate alongside as the check that calibration held.** Group-conditional thresholds, not a single pooled one, so coverage is equalised within each regime and the areas are like for like — D34's lesson applied to an ablation.

---

### D40 — "ratio > 1 in every region" is forced by orthogonal invention, not evidence of a mismatched prior [T1.1/O2, corrects part of D37]
D-034 and D37 both read the M4Raw bank's `err_free/free_content > 1` in **every** region as what "a face prior on brain anatomy should do". The identity says otherwise: with `cos θ = 0` the ratio is `sqrt(1 + u²)`, which exceeds 1 for **any** `u > 0`. In general

```
ratio > 1   ⟺   cos θ < u / 2
```

So a ratio above 1 everywhere is exactly what *any* prior whose hidden content is uncorrelated with the truth's produces, well-matched or not. It carries no information about domain mismatch, and D37's inference from it was unsound. The local phantom bank makes the point concretely: the reference sampler is not a mismatched prior at all — it is not a prior — and its ratio also exceeds 1 in every region, at a median `u` of only 0.167.

**Two corrections to D37's falsifiable check, which is otherwise kept:**

- It has an unstated precondition: `u < 2`. A prior that over-fills the null space cannot produce `ratio < 1` at any alignment, so the check proves nothing where `u > 2`. `fill_ratio()` reports the fraction of regions in that regime; the check applies to the rest.
- It should be stated on the quantity it is really about: **with medical weights, some regions should show `cos θ ≥ u/2`** — the prior recovering the hidden content rather than replacing it. That is the same condition, measured directly instead of inferred, and `content_alignment()` computes it per region.

**A reading of τ that is physical rather than arithmetic.** D37 chose τ = 1.75 on positive-rate and calibration-floor arithmetic. Under orthogonal invention, τ = `sqrt(1 + u²)`, so **τ = 1.75 means "the prior put 1.44× as much energy into the null space as the truth had there"**, and τ = 2.0 means 1.73×. That is a statement a radiographer could be asked about, and it gives the medical re-measurement a target with units. Applying it to D-034's reported quantiles (and assuming `cos θ ≈ 0`, which the diagnostic will confirm or refute), the celebahq prior's fill ratio on M4Raw runs from 0.34 to 2.57 with a median of about **1.20** — so most regions sit below the `u = 2` line and the check is diagnostic there.

---

### A2 — Two asks to Track 2, both cheap [T1.5, for Q6]
**(a) Share the `w02_solver_smoke` bank.** The Week 2 bank (`runs/w02_solver_smoke-56bc92fa6a76906d/`, 384 rows) is the only real-prior output in the project and it exists only on Track 2's machine. Score-bank spec §3 states the design intent — *"a bank is self-describing, so Track 1 can calibrate from a copied directory with no access to the machine that produced it"* — and Q6 is the first time that has been needed. The parquet and manifest are tens of kilobytes. Committing dev banks under a path of Track 2's choosing, or sending the directory, both work.

**(b) One nullable column `s_range` = ‖P_R x̂‖ per region / `ref_intensity`.** Same shape as A1, same backwards-compatible mechanics (additive, nullable, no `SCHEMA_VERSION` bump, spec §6), and one line in `compute_score_components` beside the existing `s_free`. It is the deployment-available denominator D38 identifies: the region's measured content, needing no ground truth. Without it the per-region normalisation cannot be tested against anything real, and `s_free` stays mismatched to its own label.

---

### D41 — The weights are not needed for a valid Lipschitz constant; D25's simplex bounds it [T1.1, frees T3.3]
Track 3 has been unable to state a value for T3.3's Lipschitz constant because §3 reduces it to `L = Σ wᵢ Lᵢ` and the weights are Track 1's, still unset (D28). **They are not needed.** D25 constrains the weights to the simplex — non-negative, summing to one — so `L` is a *convex combination* of the per-component constants and therefore

```
L  ≤  max_i L_i     for every admissible weight vector
```

T3.3 can state a valid bound today. D28 decides only where `L` lands below that ceiling, i.e. how tight the bound is, never whether it holds.

**Only the upper end bounds, and the reason is a finding rather than a technicality.** The companion `min_i Lᵢ ≤ L` looks natural and is **false**: components can move in opposite directions under the same operator change, and then they cancel and the combination is *less* operator-sensitive than any of its parts. That is T3.4 §3's cross-component cancellation, and on the real bank it is nearly total — `s_residual` falls by 0.388 (standardised) where `s_free` and `s_disagree` together rise by 0.376, leaving an equal-thirds net shift of **−0.004**. Cancellation only ever helps a coverage-gap bound, so T3.3 loses nothing; but it does mean **a small combined secant is not evidence of a small constant**, and a bound must not be read off one. `tests/test_component_lipschitz.py` pins both halves — the bound that holds over 200 random weightings, and the cancellation that breaks the other one.

**D27 is a precondition of that reduction, not statistical hygiene.** Components enter the combination standardised, `cᵢ/scaleᵢ`, and the step above treats `scaleᵢ` as a constant. D27 is what makes it one — scales frozen on the calibration operator. Re-estimate them per deployment operator and `scaleᵢ` becomes a function of `A`, the standardised component stops inheriting its component's Lipschitz constant, and the per-component reduction silently stops being valid. D27 was argued as protecting the coverage gap from being shrunk away; it turns out to be load-bearing for the theory too, and anyone "simplifying" it breaks T3.3 §3.

Measured on the real-prior bank (`banks/w02_solver_smoke`, DDNM + `ddpm-celebahq-256` on M4Raw), `scripts/t1_component_lipschitz.py`, R=4 → R=8 at `d_op = 0.3536`:

| component | signed shift (standardised) | `ℓᵢ` |
|---|---|---|
| `s_free` | +0.1738 | 0.4916 |
| `s_disagree` | +0.2021 | 0.5716 |
| `s_residual` | **−0.3878** | **1.0970** |

**These are the first per-component sensitivities measured against a prior that decided what to put in the null space** rather than against injected noise. Ceiling over the full simplex: `L ≤ 1.10`.

**They are lower bounds on `Lᵢ`, not estimates of it.** If `cᵢ(A, ·)` is `Lᵢ`-Lipschitz then `|E[cᵢ(A')] − E[cᵢ(A)]| ≤ E|cᵢ(A') − cᵢ(A)| ≤ Lᵢ·d_op`, so `ℓᵢ ≤ Lᵢ`, with equality only if every region moves by the same amount in the same direction. What that buys T3.3 is a **floor**: a component whose mean already moves this much cannot have a small constant, which bounds how tight the final result can ever be. It does not license quoting `ℓᵢ` as `Lᵢ`, and this document does not.

---

### D42 — `s_residual` carries no weight; it is a bank-validity diagnostic, not a score component [T1.1/D28, answers part of Q6]
**The ruling:** the score is `s_free` + `s_disagree`. `s_residual` stays computed, stays in the bank, and stays out of the score.

The argument is from the project's own decisions, not from fitting a weight to 384 rows:

**1. With a consistency-enforcing solver it measures step count, not invention.** D-033 states this precisely — a solver whose error is not concentrated in the null component "is not producing hallucinations to be scored; it is producing an unconverged fit" — and uses the residual as the *gate* that detects it. Using one quantity as both the admissibility gate on a bank and as evidence about a region inside it double-counts a solver property as a property of the image.

**2. It is the most operator-sensitive component, by more than 2×** (D41's table), and T3.3's bound is governed by `max_i Lᵢ`. Dropping it therefore **halves the ceiling on `L`, from 1.097 to 0.572**, which is a direct improvement to the headline theoretical guarantee and costs nothing.

*Note what this argument is not.* `s_residual`'s shift is also **opposite in sign** to the other two, so at equal thirds the three nearly cancel (net −0.004). It would be wrong to say including it makes the combined score more operator-sensitive — measured on the mean, it makes it much less. But that near-zero is a small difference of large opposing terms: it depends on the weights, on the step count, and on a sign that no argument guarantees, and it says nothing about the score's *distribution*, which is what conformal coverage actually depends on. A bound built on that cancellation would be a bound built on a coincidence. The ceiling argument above needs no such assumption, which is why it is the one used.

**3. It is anti-informative on the one real-prior bank that exists.** AUC 0.334 against the relative label — *below* chance, so high residual predicts a region is **less** likely hallucinated. Equal thirds therefore scores 0.828 AUC / 0.549 flag area against `s_free` alone at 0.890 / 0.505: the default combination is worse than one of its own components.

**And D35's guard does not catch this.** D35 excludes components at the float-noise floor, measured at 9.1e-8 on the analytic denoiser. On a real UNet at finite steps DDNM's data-consistency error is 0.07–0.12 (D-033's re-measurement), so `s_residual` here is 0.0558 — **three orders of magnitude above the degeneracy floor, and therefore fully live**. It would take a third of the weight budget silently. *`s_residual` is not degenerate on real banks; it is small, alive and unhelpful,* which is a different failure and needs a different rule. D35's guard stays as it is — it catches the structurally-zero case correctly — and this is the separate ruling for the anti-informative one.

**What this frees.** With `s_residual` out, the ceiling on `L` drops from **1.097 to 0.572**, the two surviving components move in the *same* direction so their weighted combination is genuinely bracketed by `[0.49, 0.57]` — a spread of 16% — and the remaining weight simplex is one-dimensional (`w_free + w_disagree = 1`). T3.3 can quote a constant good to 16% without waiting for D28 at all.

**Not a weight selection.** `w_free` itself remains unset. The sweep below is reported whole, as a diagnostic, precisely so that no one reads an argmin off three items:

| `w_free` | 0.0 | 0.2 | 0.4 | 0.5 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|
| AUC | 0.876 | 0.893 | 0.904 | 0.907 | 0.907 | 0.907 | 0.902 |
| flag area | 0.484 | 0.453 | **0.438** | 0.443 | 0.451 | 0.492 | 0.505 |
| `L` | 0.572 | 0.556 | 0.540 | 0.532 | 0.524 | 0.508 | **0.492** |

Every two-component mix beats equal thirds on detection (0.828 AUC / 0.549 flag area), and the curve is flat across 0.3–0.6. The `L` column is a genuine convex combination here because both surviving components shift in the same direction. Note the two axes disagree at the ends — detection prefers a middling `w_free`, the bound prefers `w_free = 1` — so D28's selection on the design split has a real trade-off to resolve, and now knows what it is.

---

### D43 — Q6 answered on a real prior: `s_free` is invention-driven, not an anatomy meter [T1.1, resolves Q6's empirical half]
Ran `scripts/q6_attribution.py` against `banks/w02_solver_smoke` — real M4Raw k-space, real pretrained prior, 384 regions, τ = 1.75 (D37), 98 positives.

| | AUC | detail-matched | over anatomy control | ρ(·, anatomy) | verdict |
|---|---|---|---|---|---|
| `s_free` | 0.890 | 0.933 | **+0.715** | **−0.035** | invention-driven |
| `s_disagree` | 0.866 | 0.848 | +0.631 | −0.155 | invention-driven |
| `s_residual` | 0.334 | 0.490 | +0.273 | 0.090 | uninformative |
| oracle `s_free/free_content` | 0.974 | 0.940 | +0.723 | −0.618 | invention-driven |

**Q6's feared outcome does not occur.** `s_free`'s correlation with the ground truth's own null-space content is −0.035 — indistinguishable from zero — and its separation *strengthens* when anatomical content is held fixed. It is not detecting detail. The score does not reduce to a posterior-spread detector, and D31's reading stands: the contribution rests on operator shift, with the decomposition earning its place as method.

**`cos θ` median 0.073, maximum 0.269, never above 0.5.** The prior's hidden content is essentially orthogonal to the truth's across every region: it invents rather than recovers. That also settles the assumption I used to read D-034's quantiles — `cos θ ≈ 0` is measured, not assumed, at least for this prior.

**Two cautions, both real.** The bank is 3 items × 2 operators × 64 regions, all `split=calib`, so nothing here is held out and 3 patients is not an anatomical sample — every number above is in-sample. And it is a face prior on brain MRI (D-016, D-035), so the *magnitude* of invention is not a property of a deployable system. **The verdict is nonetheless the one Q6 asked for**, because the question is about what `s_free` responds to, and a mismatched prior inventing freely is if anything the easier case in which to mistake anatomy for invention — it is the direction that would have produced a false pass, and the test passed on the other axis (ρ ≈ 0) instead.

**Still open:** the deployable version. `s_range` shipped (D-045) but this bank predates it, so `s_free/s_range` could not be scored. The oracle ratio halves the flagged area — 0.247 against 0.505 — and drops FDR from 0.53 to 0.04, so the prize is large. It needs one bank regenerated with the new column.

---

### D44 — Config overrides are silently ignored when the key does not resolve; referred to Track 2 as D-047 [T1.5]
`load_config` accepts `run.limitt=5` — a typo for `run.limit` — and any other non-resolving key, changes nothing, and says nothing. Found while trying to build a one-operator bank for `tests/test_component_lipschitz.py`; the override was ignored and produced the full three-operator grid.

Recorded here because it is the same shape as the two findings that have already cost this project weeks (D29's `.real` cast, P-19's broken script): **no exception, no wrong answer, just a step that quietly did not happen**, with the manifest and config hash agreeing with each other afterwards because they record what ran rather than what was asked for. Referred out rather than fixed — `config.py` is Track 2's.

---

### D45 — The headline estimand is a paired within-patient difference; the design is crossed, so the patient budget is not divided [T1.4, answers P-23]
Track 2's P-23 concluded that fastMRI `singlecoil_val` "cannot support a per-operator coverage claim at any grid size", handed Track 1 four options, and stopped. **The conclusion is right about per-operator coverage and wrong about the project's claim.** Full working in `T1.4_experimental_design.md`; three rulings.

**1. Three contrasts, and the headline is not the one P-23 sized.** (A) coverage at one operator vs nominal — capped at ±0.125 by the patient count, unrescuable, kept as descriptive with its interval. (B) coverage at the deployment operator minus at the calibration operator — the headline figure's y-axis. (C) coverage under two calibrations at one operator — the central claim, "the remedy restores what naive transfer loses". A varies nothing and cancels nothing; B cancels the patient; C cancels the patient *and* the operator. P-23 sized A.

**2. The design is fully crossed, so the budget is not divided.** Retrospective undersampling applies every mask to the same volume (Sec. 13) — verified on both banks in the repo, every operator sees every item. P-23's grid table divides ~40 test patients by `k` and concludes six operators give ±0.305; each operator in fact sees all ~40, so the per-operator half-width stays put whatever the grid holds. **Adding operators costs compute, not precision.** `scripts/sizing.py` fixed; `is_crossed()` checks it per bank rather than asserting it, since a split design would make the division correct again.

**3. B and C are paired, and the patient's difficulty cancels.** `Var(A−B) = Var(A)+Var(B)−2Cov(A,B)`, and the dominant term in each variance is how hard that patient's anatomy is — shared, so it cancels. Measured on a controlled sweep, with `σ` the between-patient SD:

| `σ` | 0.00 | 0.05 | 0.10 | 0.20 |
|---|---|---|---|---|
| paired SE | 0.0130 | 0.0128 | 0.0130 | 0.0132 |
| unpaired SE | 0.0131 | 0.0181 | 0.0251 | 0.0385 |
| narrower by | 1.00× | 1.41× | 1.93× | **2.92×** |

**The paired SE is flat** — it does not grow with heterogeneity at all, because heterogeneity is what the pairing removes.

**The gain is a property of the substrate, so it is reported per bank and never assumed.** On phantoms it is 1.0–1.25×, correctly, because Shepp-Logan variants are nearly identical. On the Week 3 fastMRI bank it is **unmeasured here** — that bank is not redistributable and stays on Track 2's machine; it is one `sizing.py` command for them. I am not claiming the gain, I am claiming the estimand.

**The demonstration that needs no substrate:** coverage at an operator minus coverage at the *same* operator is zero with no uncertainty. The paired estimator returns exactly that; differencing two independent bootstraps returns SE ≈ 0.014 for a quantity that cannot vary. That is the error P-23's grid table was propagating, isolated.

**What this does not fix, stated so the reprieve is not over-read:** contrast A is still capped; this is not more data; clustering is still by volume not patient (D-027), so every interval is optimistic by ~2×; and nothing becomes reportable, since the prior is still a face model (D-016).

**Adopted from P-23's own menu:** option 2 — flag area becomes co-headline rather than a footnote, for exactly the reason Track 2 gave, and reinforced by P-21's finding that the failure mode under operator shift is efficiency rather than validity.

---

### D46 — The adjoint gate was calibrated to one machine's rounding; fixed as D-050 [T1.5]
`test_a_correct_adjoint_stays_far_below_the_tolerance` asserted a flat `1e-6` and failed here at 128² (1.265e-06). The adjoint is correct — a correct operator scatters 3.6e-07 to 1.3e-06 across accelerations and mask seeds at that size, and 256² scores *lower* than 128², so there is no structural trend. The constant sat in the middle of the scatter. Bounded by `sqrt(N)·eps_float32` instead, which is the size the float32 summation error actually has; the correct operator now sits at 2–8% of the bound at every size and the broken-adjoint case is still separated by 30×.

Recorded on this side because it is the third instance of the same shape in this project — a number calibrated to one environment and read as a property of the code (D29's `.real` cast, D36's ignore pattern, now a tolerance). P-27 had already made the *probe* deterministic; the *tolerance* was still machine-specific, one layer out.

---

### D47 — The coverage gap is not a symmetric function of the operator pair; the plotted axis must be directional [T1.3, amends D19, answers D-052]
D-052 found that four of the five registered distances are the *same function* on MRI mask operators (agreeing to 1.1e-16) and that a mask-family change saturates all of them at ~0.65, non-monotonically in acceleration. Its conclusion was that the MRI grid cannot rank distance measures and the comparison must wait for CT (Week 7).

**The missing observation is that all five are metrics, and the thing being predicted is not symmetric.** Measured on the phantom bank, naive transfer between the same operator pairs in both directions:

| calibrate → deploy | `null_space_projection` | coverage gap | 95% CI |
|---|---|---|---|
| R2 → R8 | 0.6124 | −0.097 | [−0.134, −0.068] |
| **R8 → R2** | **0.6124** | **+0.902** | [+0.881, +0.921] |
| R2 → R4 | 0.5000 | −0.097 | [−0.134, −0.068] |
| **R4 → R2** | **0.5000** | **+0.902** | [+0.883, +0.920] |
| R4 → R8 | 0.3536 | −0.074 | [−0.094, −0.054] |
| **R8 → R4** | **0.3536** | **+0.234** | [+0.197, +0.270] |

Every distance value carries **two** gaps, of opposite sign, with intervals nowhere near each other. **A coverage-gap-against-distance curve plotted on a symmetric axis is not a function — it is two branches**, and the headline figure as currently specified would plot both and fit a line through them.

The mechanism is P-21's, now with its direction named: scores rise with acceleration, so a threshold calibrated where scores are low and deployed where they are high flags everything (over-coverage, exploding flag area), and the reverse flags almost nothing (coverage collapse). **The blueprint expects the second failure and the grid has been exercising the first.**

**Ruling: `null_space_growth`**, registered alongside the others — the net fraction of the ambient space newly hidden, signed, with a signed square root to sit on D19's amplitude scale. Three properties, all tested:

- **Nested masks reproduce D19 exactly, up to sign.** Where deployment samples a subset of calibration, nothing is newly revealed and `|growth|` *is* `null_space_projection`. So D19 is amended, not replaced, and its interpretation ("the fraction of k-space lines whose sampled status changed") survives on the graded within-family axis.
- **It de-saturates the cross-family grid by 12×.** From `random1d R2` at 256², the six cross-family deployment operators span 0.0498 on the symmetric axis (and are non-monotone — `equispaced R4` sits above `R8`) against 0.6124 on the signed one, monotone in acceleration within every family.
- **`|growth| ≤ null_space_projection` always**, so T3.3's Lipschitz bound is untouched: a bound proved against the metric still bounds a gap plotted against this, merely loosely in the benign direction. This is T1.3's "plotted need not equal proved" caution returning in a new form, with the same resolution T3.3 §5 gave it.

**The limitation, stated rather than left to be found.** Two masks keeping the same *number* of lines in different places give exactly zero — so the measure collapses the family axis entirely and is not injective on the grid. Operator shift on this class has two components: how much is newly hidden (graded, directional) and whether the pattern changed (categorical). This carries only the first, and the family change should be a covariate rather than a coordinate.

**That makes a falsifiable prediction the Week 4 bank can test today:** at matched acceleration, all three families should show the *same* coverage gap. If they do not, the gap depends on *which* lines, not only how many, and the axis needs a term this measure lacks. Either outcome is a result, and **it does not need CT** — which is the part of D-052's conclusion this overturns.

**Tested against T2.4 §7.1 the same day. The prediction is FALSIFIED, and the axis is kept anyway.** The data already existed; I should have read the Week 4 table before writing the prediction rather than after.

| signed growth | gaps at that level | spread |
|---|---|---|
| +0.0000 (matched acceleration, family changed) | +0.038, **+0.142**, +0.070 | **0.104** |
| +0.5000 | −0.100, −0.098, +0.047 | **0.147** |
| +0.6124 | −0.100, −0.100, −0.092 | 0.008 |

Predicted ~0 spread within each level; measured up to 0.147, against a total gap range of 0.242. **The family change is not a nuisance covariate — it carries roughly 60% of the gap's range.** The worst case is `equispaced1d R2` (+0.142, the only operator where naive transfer under-covers outright), which is the coherent-aliasing mechanism I named as the candidate failure, so the falsification is at least the expected one.

**What survives, and it is the substantive half.** Spearman ρ of the gap against each axis over the nine grid points:

| axis | ρ | sign coherent? |
|---|---|---|
| `null_space_projection` | **+0.305** | no — gaps go *negative* as the symmetric distance grows |
| `null_space_growth` | **−0.751** | yes — more newly hidden ⇒ more over-flagging ⇒ gap falls |

So the directional amendment is validated as an *ordering* (0.31 → 0.75, and the symmetric axis does not even have the right sign), and refuted as a *sufficient* axis. **Revised ruling: the figure needs two coordinates** — signed growth as the graded x-axis, and mask family as a categorical series rather than a position on it. One line per family, not one curve through all nine points. D-052 was right that this operator class cannot reduce to a single distance; it is wrong only in that the obstruction is direction plus pattern, and direction is recoverable without CT.

---

### D48 — `design_fraction = 0.10`; `calib_fraction` stays at 0.5 until banks are rebuilt anyway [T1.4, answers D-051]
Track 2 built the third split side and asked Track 1 for the fraction. Two numbers, and they are coupled because both draw on the same 96 patients.

**`design_fraction = 0.10`** — 8 patients on the real list. **The argument is the flatness of the objective it feeds.** D42's weight sweep is flat across `w_free ∈ [0.3, 0.6]`: flag area 0.453, 0.438, 0.443, 0.451, a spread of 0.015. A noisy selection inside a flat region costs almost nothing, so the design side does not need to be large — it needs to be *separate*. τ is in the same position: D37 chose it on calibration arithmetic rather than by fitting, so it needs the positive-rate curve, which is well determined from any decent sample. And the calibration side can afford it with room to spare: at τ=1.75 the fastMRI positive rate is 0.401–0.772, so 44 calibration patients × 256 regions leaves thousands of positives per group against D32's floor of **9**.

**`calib_fraction` stays at 0.5, and the reason is not statistical.** P-23's lever (0.5 → 0.3, buying ±0.127 → ±0.111 on the paired gap) is real and I am declining it *for now*, because **changing `calib_fraction` moves the test side** — verified, 54 → 62 test patients on a 96-key list — which is exactly the property D-051 took care to give `design_fraction` and cannot give this one. Taking the lever today would make every coverage number already measured incomparable with every number measured after it, for a gain that still does not reach ±0.10.

**When to take it: bundle it with the medical-prior regeneration (D-016).** That is the one moment when every bank is being rebuilt anyway, so the re-baseline is free. Track 2 should take `calib_fraction` down then — to 0.4 rather than 0.3, unless the threshold-noise feedback they flagged has been measured by then, since 0.3 trades a measured 0.016 improvement against an unmeasured risk to the thing the project certifies.

**One guard worth adding to the config, not the prose:** `calib_fraction=0.3` with `design_fraction=0.20` leaves **5** calibration patients on a 96-key list. The composition is not obvious from either number alone, and the floor that matters is D32's 9 positives per group, not a patient count.

---

### D49 — The weight-selection rule, pre-registered; the number itself waits on one run [T1.1/D28]
D28 has been the last unset number in the score definition since Week 3. D-051 built the design split and D48 set its fraction, so the rule can now be written — and it is written **before** any bank with a design side exists, so it cannot be chosen to suit what the data says. Same discipline as D37's τ and Q6's verdict thresholds. `scoring/weights.py`, 11 gates.

**The simplex is one-dimensional.** D42 took `s_residual` out, so selecting the score means selecting one scalar: `w_free + w_disagree = 1`.

**The rule.**
1. Sweep `w_free` over a fixed grid, on **design** rows only — refused outright if the bank has none, since selecting on calibration rows biases the threshold and selecting on test rows biases the coverage number, and both fail silently (D-051).
2. At each point, calibrate group-conditionally at α and record **flagged area** (D39: power is pinned by the calibrator at ≈ 1−α, so it compares calibrators rather than scores — now asserted as a guard in the suite, which holds the sweep's miss rate inside 0.02).
3. Take the minimum, then every point within **one cluster-bootstrap SE** of it. The objective is flat — D42 measured 0.015 of movement across `w_free ∈ [0.3, 0.6]`, inside the noise of eight design patients — so the raw argmin of a flat curve selects noise.
4. **Break the tie toward the larger `w_free`.** D41 gives `L = w·L_free + (1−w)·L_disagree` with `L_free = 0.4916 < L_disagree = 0.5716`, so `L` falls monotonically as `w_free` rises. Within a statistically indistinguishable set, take the weight that hands Track 3 the tighter constant. This is the step that needed a *ruling* rather than a measurement, and it is principled rather than a second pass over the same data.

**Endpoints are reported, not avoided.** A selection at `w_free = 1` means cross-sample disagreement earned no weight — the mirror of D2's honest-negative and a substantive finding about the score's composition. `at_endpoint` says so and the script prints it.

**What is claimed and what is not.** The design rows are both calibrated and evaluated on, so the *level* of flagged area is optimistically biased. The bias is common to every point, so the *comparison* — all the rule uses — is unbiased. No level from this sweep is quotable.

**What still blocks the number, and it is smaller than it looked.** Not "Track 2 must run something" — the gap was in the plumbing. `design_fraction` reached `splits.py` and the fastMRI loader but **no config sets it and the phantom dataset cannot express it** (it takes one fixed split string). So it is reachable today for fastMRI only, as a dataset kwarg:

```
caliber configs/experiments/w04_design_split.yaml
PYTHONPATH=src python3 scripts/select_weights.py runs/<run_id> \
    --calibration-operator mri.random1d.R2.c0.08.s0
```

**And the naive version of that run costs 25x what it needs to.** The loader *labelled*
every volume with its split while reconstructing all of them — right for calib and test,
ruinous for a side that is 8 patients of 96, since ~92% of the work is then discarded. At
T2.3's measured 288 s per item-operator:

| | reconstructions | measured | quiet floor |
|---|---|---|---|
| full dataset, as the loader stood | 597 | **47.8 h** | 23.2 h |
| `keep_splits: design`, 8 volumes x 3 operators | 24 | **1.9 h** | 0.9 h |

`keep_splits` is added to the fastMRI loader and **filters before `limit_volumes`**, which
is the ordering that matters: limiting first would return 8 volumes of which ~1 is design.
It changes no patient's assignment — the split stays a pure function of (patient, fraction,
salt) per D-022/D-027, and `tests/test_split_filter.py` asserts a kept volume carries the
label it would have carried unfiltered, rather than trusting that it does.

Delivered as a config rather than four `--set` overrides: `design_fraction` is a *dataset*
kwarg rather than a `run.` one, `max_draws_per_batch: 1` is mandatory (4 draws at 256°
needs ~18 GB and segfaults silently, P-24/D-044), and since D-048 a mistyped override is
refused outright. Four chances to get a key wrong is three too many.

**And the number selected from that run is still a development value.** The prior is a face model (D-016, D-035), so the score's definition is only *pinned* by a selection on a reportable bank. The rule is what is being delivered here; the constant it produces is provisional twice over — once for the prior and once for eight patients.

---

### D50 — Operator shift has two failure modes; the project has only ever analysed the benign one [T1.4, improves the headline at zero compute]
The Week 4 grid calibrates at `random1d R2` — the *least* aggressive operator — and deploys to R4 and R8. By D47 that is the direction in which scores rise, so the transferred threshold becomes conservative and the failure appears as over-flagging. Measured on the phantom bank, both directions, α=0.10:

| | naive coverage | naive flag area | Mondrian coverage | Mondrian area |
|---|---|---|---|---|
| R2 → R4 | 1.000 | **1.000** | 0.902 | 0.873 |
| R2 → R8 | 1.000 | **1.000** | 0.902 | 0.832 |
| **R8 → R2** | **0.000** | 0.000 | 0.903 | 0.820 |
| **R8 → R4** | 0.668 | 0.600 | 0.902 | 0.874 |

**Ascending the grid breaks efficiency; descending it breaks validity.** Deploying to a less-hidden operator collapses coverage from 0.90 to 0.00 — the failure §1 predicts and the grid has never been pointed at. Mondrian restores coverage to ~0.902 in *every* cell, both directions, which is the blueprint's claim demonstrated rather than asserted.

**This is a re-analysis, not a re-run.** The Week 4 bank is fully crossed (D45), so the calibration operator is an analysis-time choice — §17's whole premise. Zero GPU.

**Both get reported, and that is the point.** Reporting only the ascending direction understates the risk; reporting only the descending one would be cherry-picking. The honest claim is stronger than the blueprint's: operator shift has two distinct failure modes, we characterise both, and the group-conditional remedy fixes both. A protocol change in a clinic can go either way.

*To T1.4's grid:* every operator pair is swept in both directions, and the figure carries the sign (D47).

---

### D51 — D39's "compare area, not power" has a precondition, and across-regime comparisons violate it [T1.4, corrects D39]
D39 ruled that component and method comparisons are made on flagged area because a conformal threshold pins the miss rate at α. **That holds only while the calibration scores are exchangeable with the deployment scores** — which operator shift breaks by construction, and which is the entire object of study.

Caught by running the Tool 2 comparison: the nearest-operator arm posted the **smallest flagged area of any arm** while averaging coverage **0.547**. Flagging almost nothing is cheap once you have stopped catching anything, and a ranking on area alone would have declared it the winner.

**Corrected rule.** An arm competes on flagged area only if it is **valid** — the upper end of its coverage interval reaches nominal. An invalid arm's small area is reported as the symptom it is, not as efficiency. D39 stands where it was argued, *within* one calibration regime comparing score components; it does not transfer across regimes.

---

### D52 — Tool 2's first real evaluation: valid where the obvious baseline is not, and no better than the conservative fallback [T1.2/T1.4, answers Q4 on a development bank]
`calibrate_weighted` — the project's primary contribution (§9) — appeared in **no** analysis path until now. Implemented in Week 2, unit-tested on synthetic banks, never pointed at a bank. `scripts/tool2_holdout.py` is Q4's experiment: hold each operator out in turn and compare four ways of thresholding it.

On the phantom bank's one interior fold (held out R=4; the endpoint folds ask the weighting to extrapolate and are reported separately):

| arm | coverage | flag area | valid? |
|---|---|---|---|
| `matched` (oracle, unavailable at deployment) | 0.902 | 0.874 | yes |
| `mondrian` (no group for the held-out operator) | 1.000 | 1.000 | yes, vacuously |
| `nearest` (closest operator's threshold) | 0.666 | 0.599 | **no** |
| **`weighted` (Tool 2)** | 1.000 | 1.000 | yes, vacuously |

**The honest reading: Tool 2 is valid where the obvious alternative is not, and buys nothing over the conservative fallback.** Half a win. `nearest` is what a sensible person does without Tool 2 and it under-covers by 24 points, so the weighting is not pointless — but it flags every region, exactly as Mondrian's unknown-group fallback does, so it has not earned its place on the efficiency axis.

**The mechanism, and it is not a tuning problem.** The oracle threshold is 0.8152; Tool 2 picks 0.5658, *below the held-out operator's minimum score* of 0.6459. Swept the bandwidth over 0.02–0.8:

| `h` | 0.02 | 0.08 | 0.2 | 0.427 | 0.8 |
|---|---|---|---|---|---|
| threshold | −inf | 0.528 | 0.575 | 0.566 | 0.563 |
| coverage / area | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

Nothing moves. **A lower-tail quantile is set by the lowest-scoring regime in the mixture**, and an exponential kernel down-weights smoothly without ever excluding — at `h=0.08` the far regime still holds 14% relative weight, and at α=0.10 that is enough to occupy the whole tail. Drive the weight to zero and the ESS collapses, at which point D32 refuses to certify and the threshold returns −inf.

**So Tool 2's accuracy wants concentrated weight and its certifiability wants spread weight, and on an ordered grid there is no `h` that satisfies both.** That is a real tension in the primary contribution, stated plainly, and it closes D33's open bandwidth question with "the bandwidth is not the lever".

**Two leads, neither implemented here.** A **truncated** kernel — hard-exclude beyond a distance cut, keeping ESS from what remains instead of inheriting a contaminating tail. Or **interpolate the threshold rather than pooling the scores**: each regime's own quantile plus the known distance, which would land near the oracle, at the cost of the conformal guarantee (D6 already declines an exact guarantee for arbitrary shift, so the cost may be smaller than it sounds).

**Caveats that matter.** One interior fold, a reference sampler, three operators. The real Week 4 grid has nine operators and many interior folds, where the tail-domination argument predicts the same outcome but the bandwidth has more room. One command on their bank settles it.

---

### D53 — Pre-run rulings for the full MRI run: hold `calib_fraction`, take the design side, keep all three families [T1.4, amends D48's reasoning]
Track 2 asked for three calls before the full run. Two confirm D48; one needed its arithmetic redone, because their new numbers made my earlier premise false.

**1. `calib_fraction` stays 0.5 — and the reason has changed.** D48 declined the 0.3 lever partly because "the gain still misses ±0.10". Their refined numbers reach it (±0.115 → ±0.100), so that premise is gone. The ruling survives on a better argument: **their table counts only the test-side term.** Moving a patient from calib to test also grows the conformal threshold's own sampling noise, and the two add in quadrature. Taking the calibration unit to be the patient (D-027) and `Var ≈ α(1−α)/(n+2)`:

| setting | calib / test | test half | threshold half | **total** |
|---|---|---|---|---|
| 0.5, design 0 (today) | 52 / 44 | 0.115 | 0.080 | **0.140** |
| 0.5, design 0.10 | 44 / 44 | 0.115 | 0.087 | **0.144** |
| 0.3, design 0 | 38 / 58 | 0.100 | 0.093 | **0.137** |
| 0.3, design 0.10 | 30 / 58 | 0.100 | 0.104 | **0.144** |

**The total is flat.** 0.3 buys 0.003, not 0.015 — and with the design side carved out it ends at 0.144, *worse* than today. None of the four reaches ±0.100. Re-slicing 96 patients cannot create precision; only adding patients can (P-23's option 4).

*Honest limit on this calculation:* treating all positives in a patient as one exchangeable unit is the conservative end, and for the **paired** gap the threshold noise partially cancels, since one threshold moves both arms together. So the true gain from 0.3 is somewhere in (0.003, 0.015] rather than exactly 0.003. It is still small enough that the comparability cost of moving the test side is not worth paying.

**2. `design_fraction = 0.10` — yes, take it in this run.** D49's selection rule is written and waiting on exactly these rows, and `w_free` is the last unset number in the score definition. Since a full run is happening anyway, the design rows come free in the same pass, which makes `configs/experiments/w04_design_split.yaml` unnecessary for this case — it stays for a standalone top-up. Cost 0.004 of total half-width, against closing D28.

**3. Keep all three mask families.** Their reasoning — the families sit at ~0.65 so add no new points (D-052) — is the premise D47 refuted: distance is not the axis. Checked against their own §7.1 table, every grid cell whose coverage falls below nominal:

| cell | coverage | flag area | |
|---|---|---|---|
| `random1d R2` | 0.862 | 0.694 | the calibration operator itself |
| `equispaced1d R2` | **0.758** | 0.630 | cross-family |
| `variable_density2d R2` | 0.830 | 0.209 | cross-family |
| `variable_density2d R4` | 0.853 | 0.531 | cross-family |

**Within `random1d` alone, every deployment cell sits at coverage 1.000.** Dropping the other two families leaves **zero** cells supporting under-coverage — the entire O3 evidence base in the grid is cross-family. It would also drop the lowest flag area in the grid (`variable_density2d R2`, 0.209), the only cell where the flag is useful rather than vacuous.

*If compute forces a cut, cut accelerations, not families.* The within-family axis is graded and interpolable (D47's signed measure is monotone along it); the family axis is categorical and cannot be interpolated at all.

---

### D54 — The weight selection declines to choose when the sweep cannot separate anything; and D48's justification was half wrong [T1.1/D28, amends D49]
Track 2 ran D49's rule on the Week 5 bank: it returned **`w_free = 1.00`** — an endpoint, dropping `s_disagree` from the score entirely — and they correctly flagged it as the tie-break rather than a result. 8 of 11 sweep points were tied with the minimum, and step 4 walks toward the tighter Lipschitz bound, which on a tied sweep means walking to the boundary.

**Eliminating a component is a structural change to the score, and a tie cannot license one.** D49 said "endpoints are reported, not avoided", which is right when the data *chooses* an endpoint and wrong when the tie-break *arrives* at one.

**Amendment.** Before the tie-break, check whether the sweep separated anything: if more than half its points are tied with the minimum, the selection has no information, and the rule returns D25's stated default — **equal weights over the live components** — flagged `discriminated=False`, with a note that says NOT SELECTED. The tie-break still applies when the tied set is small, which is what it was for. The size of the tied set is the right instrument and more direct than comparing the sweep's range to its noise, which I tried first and which let the pathological case through.

**And I have to own the reasoning error in D48.** I set `design_fraction = 0.10` on the argument that D42's objective is flat across `w_free ∈ [0.3, 0.6]`, so "a noisy selection inside a flat region costs almost nothing". That is true about the **cost** of choosing wrong and false about the **informativeness** of the choice: when the standard error is the size of the objective's whole range, the selection is noise, and no amount of tie-breaking rescues it. Flatness justifies *not caring much* about the answer; it does not justify *claiming* one. Eight patients cannot discriminate here and a design side large enough to do so is not affordable on 96 patients.

**So D28 closes on a bounded-cost default rather than a fitted number**, which is the honest form: the weights are equal over the live components, the cost of that choice against the unknown optimum is bounded by D42's measured 0.015 of flagged area, and `w_free` is re-selected if a substrate ever makes the objective discriminating. That is a weaker claim than "selected on a design split" and it is the one the data supports.

---

### D55 — `tool2_holdout.py` calibrated and evaluated on the same patients; my bug, and the third of its kind here [T1.4, found by Track 2]
The first version passed `split_col="__none__"` to all four arms, disabling the split filter. No single *row* was reused — the pool excludes the held-out operator — but **the design is crossed (D45), so the same patient appeared in calibration and evaluation**, and the patient is the exchangeable unit (D-027). `matched` was worse: it calibrated on the held-out operator's own rows and scored those same rows, which is why it landed on exactly 0.900 — the in-sample artefact P-23 is named for, reproduced by me two weeks after reading it.

**Fixed:** calibration is now the other operators' `split=calib` rows, evaluation is the held-out operator's `split=test` rows, and the script **refuses a single-split bank** rather than quietly reporting in-sample numbers (`--allow-in-sample` to override, for development). It also warns if any cluster appears on both sides, which is the check that would have caught this.

**What it changed, from Track 2's re-run:** Tool 2's verdict is unchanged (valid on 2 of 3 interior folds), `nearest` drops from 1/3 to 0/3 valid, and `matched` stops sitting exactly on nominal. So the conclusion in D52 survives, and the baseline it was compared against was flattered.

**Third instance of one failure.** P-19 (the script nobody executed), P-23 (two of three baselines in-sample), now this. The common shape is that an in-sample number *looks right* — it lands on nominal, which is the value you were hoping for — so nothing alerts you. The guard that generalises is the one now in the script: assert the two frames are disjoint on the clustering unit, rather than trusting that the filtering was done.

---

### D56 — Printed strings must be ASCII; docstrings need not be [T1.5, found by Track 2]
`select_weights.py` crashed with `UnicodeEncodeError` on Track 2's Windows console printing a `<=`. Development is on macOS where stdout is UTF-8; every bank is produced on Windows where a legacy console is cp1252. An **audit found 31 printable string literals across 13 files** with the same defect — the crash was the first one anyone happened to hit, not the only one.

Fixed mechanically, by AST position so docstrings were untouched, and the rule is now enforced by `tests/test_console_encoding.py`: non-docstring string literals must be ASCII, docstrings may keep mathematical notation (`||P_N x||`, `theta`, `alpha`), and Markdown is unaffected. Both forms of the check are kept — ASCII, which is the stricter rule, and cp1252-encodability, which is the failure that actually happened.

---

### D57 — The relative criterion needs a conjunctive absolute gate; the ratio is driven more by its denominator than its numerator [T1.1/O2, amends D21 and D37]
Track 2 found that on the 44 test patients, **low `free_content` alone** ranks the relative label at AUROC **0.89-0.95** — better than any score in the project — and asked whether that is what the criterion is meant to do. It is not. Reproduced on the real-prior bank and decomposed:

| ranker -> relative label | AUROC |
|---|---|
| `-free_content` (low anatomy) | **0.826** |
| `err_free` (the numerator alone) | **0.741** |
| `s_free` | 0.902 |
| combined score | 0.907 |

`spearman(ratio, free_content) = -0.520` against `spearman(ratio, err_free) = +0.450`. **The denominator has more influence on the label than the numerator does.** A region with little hidden content yields a large ratio from an error far too small to matter, so the criterion systematically labels smooth tissue as hallucinated, and anything correlated with low anatomical density scores well against it. `MIN_FREE_CONTENT = 1e-3` does not help: the smallest live `free_content` is 0.162, so the guard never fires — the problem is the ratio's structure across the live range, not division by near-zero.

**This is my defect.** D21 introduced the ratio to stop `err_free` being confounded by anatomy density, and it over-corrected into the opposite confound. The two gates fail in opposite directions:

- **ratio alone** — denominator-dominated (this finding);
- **absolute alone** — anatomy-confounded, which is D21's original reason and P-02's symptom (`err_free > 0.10` flags every region).

**Ruling: the reported label is the conjunction.** `err_free / free_content > τ` **and** `err_free > ε`. Implemented as `label_hallucinated(..., relative=True, absolute_floor=ε)`; passing a floor to the absolute form is refused, since that would be two thresholds on one quantity.

Measured on the M4Raw bank at the 62nd percentile of `err_free` (ε = 0.403):

| | ratio only | conjunction |
|---|---|---|
| positive rate | 0.255 | 0.167 |
| AUROC(`-free_content`) | 0.826 | **0.696** |
| AUROC(`s_free`) | 0.902 | **0.974** |

The denominator's grip falls toward chance, the score's rises, and the positive rate stays well clear of D32's nine-per-group floor.

**ε is a quantile on the design split, never a constant.** It is an error magnitude, and D-020 established that those transfer across neither modalities nor substrates. Same discipline as D37's τ and D28's weights: set on the design split, frozen, re-set per substrate.

**A soft floor does not work, and I tried it first.** `err_free / (free_content + c) > τ` *raises* the denominator's AUROC to 0.878 at c=0.05 and kills the positive rate entirely by c=0.10. It rescales the same ratio structure instead of adding an independent gate.

**What this invalidates, and what it does not.** Every label-dependent number measured so far is affected — coverage, FDR, power, the weight sweep, Q6. All of it is **re-analysis, not re-reconstruction**, which is precisely what D-002 bought by storing continuous quantities. **Q6's verdict survives**: under the conjunction `s_free` is still invention-driven (AUC 0.900, detail-matched 0.868), though its excess over the anatomy control falls from +0.715 to +0.468 — so part of D43's margin *was* the denominator effect, and that correction belongs in the write-up.

**And it reframes Track 2's AUROC observation.** They report `s_disagree` alone beating the combined score at every operator. That comparison is against the defective label, and `s_disagree` is the component *more* anti-correlated with anatomy (rho -0.155 against `s_free`'s -0.035), so it is the component most rewarded by the artefact. The ordering must be re-measured under the conjunction before anything is concluded from it.

---

### D58 — The Week 5 fallbacks: Track 1 owns them, [7] is the one to build, and [9] is inapplicable [T1.2, answers Track 2's ownership question]
The cards say Track 2, `calibration/` says Track 1. **`calibration/` is right** — these are methods, and §18 gives methodology to Track 1. Track 2's time is better spent on runs, which need the GPU and the data.

**They are no longer contingencies.** The risk register listed them as fallbacks if Tool 2 underperformed. D52 measured Tool 2 as valid on 2 of 3 interior folds and no better than the conservative fallback, so the trigger condition has fired.

**Priority, from D52's mechanism** — the lower-tail quantile is set by the lowest-scoring regime, and a smooth kernel down-weights without ever excluding:

- **[7] group-weighted — build this one.** Weighting whole groups rather than points allows a group to take weight *zero*, which is exactly the truncated kernel D52 proposed as its lead. It is the fallback that addresses the measured failure, already named in the register.
- **[8] doubly-robust — second.** Needs a model of the shift, which the project does not have; worth attempting only if [7] does not close the gap.
- **[9] ACI — inapplicable as the protocol stands, and worth saying so rather than carrying it as planned work.** Adaptive conformal inference updates the threshold from *observed* coverage, which requires labels at deployment. The deployment setting has none — that is the whole premise. It becomes applicable only under a different protocol where a clinician adjudicates a sampled fraction of flags, which is a scope change, not a fallback.

Code goes in `src/caliber/calibration/fallbacks.py`, same module Track 1 already owns.

---

### D59 — The seed noise floor: R4 is right, and report it per operator [T1.4, answers Track 2]
R4 was the operator I had in mind and it is the right primary — mid-grid, and already running.

**But the spread almost certainly grows with acceleration,** so one floor measured at R4 will understate the ablations that matter most. A `random1d` mask at R=2 draws 50% of lines and at R=8 draws 12.5%, so two seeds differ far more at R8 — and the descending-direction headline (D50) *calibrates* at R8. If a second sweep is cheap, R8 is the one that bounds the headline; if not, R4 stands with the caveat stated.

**Report it per operator, not as one global number.** Seeds are members of a calibration group rather than groups of their own (D-023), so this is within-group variation and it is allowed to differ between groups.

---

### D60 — The gate belongs on `free_content`, not `err_free`; my D57 fixed one artefact and amplified another [T1.1/O2, supersedes D57's gate]
Track 2 asked which quantile to use for D57's absolute floor, and flagged that AUROC reaches 0.99-1.00 at the R2 operators where positives are scarce, suspecting that "where true free content is small, `s_free` might be close to measuring the label's own numerator." **They are right, it is derivable from D38, and it means my gate was the wrong one.**

**The circularity, measured.** By D38's identity, `err_free² = s_free² + free_content² - 2·s_free·free_content·cos θ`, so `err_free -> s_free` as `free_content -> 0`. On the real-prior bank, `spearman(err_free, s_free)` by `free_content` tercile:

| tercile | spearman | mean abs difference |
|---|---|---|
| low `free_content` | **+0.964** | 0.048 |
| mid | +0.960 | 0.068 |
| high | +0.491 | 0.135 |

Where little is hidden, **the label's numerator is the score**. Grading `s_free` against such a label is grading it against itself.

**And D57's gate selects exactly those regions.** Gating on large `err_free` at small `free_content` picks the circular cases, which is why it inflates rather than corrects:

| gate | pos rate | AUROC(`-free_content`) | AUROC(`s_free`) |
|---|---|---|---|
| ratio only | 0.255 | 0.826 | 0.902 |
| + `err_free` > q62 (**D57**) | 0.167 | 0.690 | **0.973** |
| + `err_free` > q75 | 0.117 | 0.615 | **0.985** |
| + **`free_content` > q33** | 0.096 | **0.499** | 0.931 |
| + `free_content` > q50 | 0.039 | 0.372 | 0.919 |

The candidacy gate takes the denominator's AUROC to **0.499 — exactly chance** — and leaves the score's at a believable 0.931. D57's gate never reaches chance and pushes the score's into the range Track 2 correctly distrusted.

**Ruling: `min_free_content` at the 33rd percentile of `free_content` on the design split.** Not a new parameter — it is the one already in `label_hallucinated`, whose default of 1e-3 is a numerical guard that never fires, since the smallest live value is 0.162. The reasoning is physical: **a region where the operator hides almost nothing cannot host a null-space hallucination, so it is not a candidate**, and asking the relative question there is what manufactures both artefacts.

**One gate, not two.** Adding D57's `err_free` floor on top of the candidacy gate changes nothing measurable (0.039 / 0.372 / 0.919 either way), so it is superseded rather than kept alongside. It stays available and is documented as not recommended.

**q33 rather than q50**, on positive-rate grounds: 0.096 leaves ~2,200 positives on the 23k bank (~245 per group against D32's floor of 9), where q50's 0.039 leaves ~100 per group and Track 2 already measured 1.2% at `vd R2`. The artefact is already fully removed at q33.

---

### D61 — Which direction collapses is solver-dependent; D50 over-generalised from a reference sampler [T1.4, corrects D50]
D50 ruled that ascending the grid breaks efficiency and descending breaks validity. Under the corrected label on fastMRI, Track 2 measured the **opposite**: descending only over-flags (0 cells fail at any q), and **ascending collapses** — naive reaching 0.298 at `equispaced R2`, with 7-9 of 9 ascending cells missing nominal, consistently across all three q. Group-conditional restores 0.80-0.94.

**Re-checked on the phantom bank under the candidacy gate, and there the direction does *not* flip** — descending still collapses to 0.000. So D50 was not wrong about its own data; it was wrong to generalise it. The reference sampler injects hallucination at a fixed scale, so its score distributions order purely by null-space size; a learned prior's depend on image content and on the mask *pattern*. **Which direction fails is a property of the solver and substrate, not of operator shift.**

**What survives, and what does not.** D47's mechanism survives intact: the gap is not symmetric in the operator pair, one direction over-flags and the other under-covers, and the plotted axis must carry the sign. What does not survive is D50's *assignment* of which direction does which, and the phantom numbers must not be quoted as though they were the project's result.

**And the blueprint's original expectation is restored.** O3 as written — naive transfer loses coverage when deployed to a different operator, group-conditional restores it — is what the real-prior data shows once the label is fixed. Both the over-flagging reading (P-21) and my descending-collapse reading were label artefacts. The claim did not need reframing; the label needed fixing.

*Consequence:* the midterm figures are redone against the candidacy-gated label, in the ascending direction, with the descending cells reported as the over-flagging case.

---

### D62 — No function of `(err_free, free_content)` is anatomy-neutral; stop fixing it in the label and stratify the analysis [T1.1/O2, supersedes the gate in D60, answers P-31 and the CT label]
Track 2 reports that on CT the label marks **88%** of candidate regions as hallucinated with AUROC 0.912 on "how little a region hides" against 0.527 on the actual error, and that on MRI the denominator problem **persists within operators** — 0.920 at `random1d R8` (P-31). They offered two options: a CT-specific τ, or a criterion that does not divide by `free_content`.

**Neither, and first I have to retract how D60 was validated.** I checked the candidacy gate on a **pooled** AUROC and got 0.499 — "exactly chance". Measured per operator on the same bank:

| label | pooled | R4 | R8 |
|---|---|---|---|
| ratio only | 0.826 | 0.941 | 0.901 |
| ratio + candidacy q33 (D60) | **0.499** | thin | **0.662** |
| absolute `err_free` > q75 | 0.277 | **0.054** | 0.421 |

**The 0.499 was two opposite biases cancelling, not neutrality.** The gate reduced the within-operator artefact and did not remove it. That is the fourth time in this project a pooled or in-sample statistic has reported the number someone was hoping for (P-19, P-23, D55, now this), and I validated a criterion on one after writing the entry that warns about the others.

**And the second option does not work either — I tested it.** An absolute label inverts the bias rather than removing it: AUROC(`-free_content`) of 0.277 pooled and **0.054 at R4** means *high* anatomy now predicts the label, which is exactly the confound D21 introduced the ratio to fix. The ratio over-corrects, the absolute form under-corrects, and the gated ratio sits in between. **No function of `err_free` and `free_content` alone is anatomy-neutral within an operator**, because both quantities scale with anatomical density.

**A third route is also closed, and for an informative reason.** The principled label is the *invented* content — the part of the reconstruction's hidden content orthogonal to the truth's, which by D38 is `s_free·|sin θ|` and involves no division. On this bank `cos θ` has median 0.073, so `sin θ ≈ 0.997` and **`corr(invented content, s_free) = 0.9992`**. The label would be the score. That is not a defect in the criterion: it is what a prior that recovers *none* of the hidden content looks like (D-016, a face prior on anatomy). With a matched prior `cos θ` would be substantially positive and the quantity would discriminate. **So no criterion can be validated against this prior**, and that bounds what Week 8 may claim whatever is chosen.

**Ruling: the label stops trying to be anatomy-neutral, and the analysis controls anatomy instead.**

1. **Label:** absolute, `err_free > ε`, with `ε` a **quantile of `err_free` on each substrate's own design split**, frozen. Simple, monotone in the thing it names, and no denominator. P-02's failure was a fixed *constant* (`0.10` flagged every region); a quantile cannot do that by construction.
2. **Analysis:** every detection number is reported **stratified by `free_content` tercile**, within operator. The machinery exists — `stratified_auc` and `attribute_components` already do it, and they are what caught Q6.
3. **Secondary reading on MRI only:** `require_annotation=True` against fastMRI+ boxes. That is the one anchor not defined by our own error arithmetic. It is unavailable on CT — TotalSegmentator does not annotate LoDoPaB (P-03) — so it cannot be the primary.

This is the standard move and I should have reached it sooner: **do not encode a confound correction in the outcome definition; adjust for it in the analysis.** A divisor in the label makes the label depend on the confound. Stratification leaves the label clean and puts the control where it can be checked.

*Consequence:* τ and the candidacy gate are both retired from the reported criterion. D37's τ, D57's floor and D60's gate were three attempts at the same impossible thing. CT becomes readable, and P-31 is answered for MRI by the same change.

*And the project now has no external validation of its label.* Track 2 reports the published Hallucination Index (Tivnan 2024) disagrees at kappa ~ 0 and mostly flags background. That was the independent check. Its absence is a **stated scope limit for the write-up**, not a gap to be quietly filled: the criterion is ours, it is defined in §4's terms, and the annotation-restricted reading is the nearest thing to an outside anchor.

---

### D63 — CT dose is not a grid axis [T1.4, answers Week 8 scope]
**No.** Three reasons, in order of how badly they bite.

1. **Dose changes the noise, not the operator.** `A` is unchanged between two photon counts, so the null space is identical and **`d_op` between two doses is exactly zero**. Two distinct cells would sit at the same x on the headline coverage-gap-against-distance axis, which is the same non-functional axis problem D47 found for direction — and this time with no sign to recover.
2. **It multiplies calibration groups against D32's floor.** CT groups are now `ct/{family}/v{views}/a{arc}` (P-34); a dose axis multiplies that again, and the floor is nine positives per group.
3. **D22 already settled the form:** noise is a group axis *if* it is varied. Varying it here buys a confound rather than a result, because the project's claim is about the acquisition geometry.

*If there is appetite for it:* a **robustness check at one fixed geometry** — does the flag survive a dose change at constant `A`? — which is a short appendix, not a grid dimension. Worth doing precisely because `d_op = 0` there: it isolates noise sensitivity from operator shift.

---

### D64 — CT signed distances are needed, but only a handful of pairs [T1.3, answers Week 8 scope]
**Yes, still needed, and they are the point of the CT side for T1.3** — but not the full matrix.

D-052 found four of five distance measures coincide to 1.1e-16 on MRI masks, because a mask operator is diagonal in k-space so "which frequencies it keeps" *is* its range. CT view subsampling is not diagonal, so **CT is the only operator class in the project where those measures can differ at all**. That makes a small number of CT pairs worth more than the whole MRI matrix for this question.

Minimum set, at minutes per pair:
- the CT calibration operator against **each** deployment operator (k−1 pairs) — T1.3's cross-modality checklist item, and the x-axis for the CT curves;
- **one pair that changes arc at fixed views**, and one that changes views at fixed arc — the two that test whether range and null space have come apart from a coordinate mask, which is D-052's open question.

That is roughly k+1 pairs rather than k(k−1)/2. Report `null_space_projection` and `null_space_growth` for them, plus `spectral_divergence` on the same pairs if it runs — it is the measure D-052 could never compute on MRI, and CT is its first chance.

---

### D65 — D62 implemented, and the criterion has a usable band rather than a fix [T1.1/T1.4, amends D62]
D62 moved the anatomy correction out of the label and into the analysis. Both halves are now built — `absolute_threshold_from_design` and `evaluation/stratified.py`, 8 gates — and building them turned up two things that change what D62 may claim.

**1. My first `anatomy_auc` was one-directional and flattered the result.** I scored the artefact as `AUC(-free_content)`, which assumes the confound runs low-anatomy. That holds for the *ratio* label and is backwards for the *absolute* one, where it runs high-anatomy — so the measure reported `1 - truth` and the verdict came back "the score beats the anatomy artefact in every stratum by at least +0.329". Direction-agnostic (`max(auc, 1-auc)`), the same bank and the same label give **FAILS, worst margin -0.408**. Fixed, and it is the fifth time in this project a statistic has returned the hoped-for number because of how it was framed.

**2. The honest reading of the corrected table, per operator:**

| operator | positives | score AUC by stratum | anatomy AUC by stratum |
|---|---|---|---|
| `random1d R8` | 79 | **0.991 / 0.990 / 0.804** | 0.648 / 0.660 / 0.786 |
| `random1d R4` | 17 (16 in q3) | — / — / 0.583 | — / — / **0.991** |

At R8 the score wins decisively in every stratum. At R4 it loses badly — but the reason is visible: the absolute label puts **16 of 17 positives in the top anatomy tercile**, so within that tercile anatomy predicts the label at 0.991 and there is nothing left for a score to explain.

**Why, from D38's identity.** `err_free² = s_free² + free_content² − 2·s_free·free_content·cos θ`. At **low** `u = s_free/free_content` the invention term is small and `err_free → free_content`: the label *is* anatomy. At **high** `u` the truth term is small and `err_free → s_free`: the label *is* the score, which is the circularity Track 2 found (`corr = 0.9992`). **The criterion is an anatomy meter at one end and a mirror of the score at the other, and is only meaningful in a middle band of `u`.** That is a property of the quantity, not of any threshold placed on it, which is why D37's τ, D57's floor, D60's gate and now D62's ε all failed in the same shape.

**So D62 stands as the right form and is amended in what it promises.** The absolute label plus stratified analysis is the only arrangement that keeps the confound visible instead of buried — but it does not make every cell interpretable. **Report `u` per operator alongside the detection table, and read only the cells where `u` sits in the band.** `R8` qualifies on this bank; `R4` does not, and its numbers should be shown as uninterpretable rather than as a failure of the score.

**What this does not resolve, and I want it on the record before the write-up.** Both ends of the band are consequences of D-016: a prior that recovers none of the hidden content (`cos θ` median 0.073) pushes every region toward the circular end, and a prior that invents little pushes them toward the anatomy end. **The criterion cannot be validated, and the detection numbers cannot be read, until the prior matches the modality.** That is one task, it is Track 2's, and it now gates the label as well as the reportability of every number.

*Implementation notes:* `absolute_threshold_from_design` refuses a bank with no design side rather than falling back (D-051's two silent biases). `detection_by_content` cuts strata **within** operator and offers no cross-operator row at all, so D62's retraction cannot recur through it — pinned by a regression test that builds two operators with opposite biases and checks the report refuses to average them.

---

### D66 — Fallback [7] built: it fixes Tool 2's location error and the phantom grid cannot test it [T1.2, delivers D58's first item]
`calibration/fallbacks.py`, 14 gates. Two changes from Tool 2, each addressing a measured defect rather than a hypothetical one.

**1. Hard truncation.** Weight is per group and a group may receive *exactly* zero. D52's mechanism is that the alpha-quantile is a lower-tail statistic, so it is set by the lowest-scoring regime in the mixture, and an exponential kernel down-weights smoothly without ever excluding — at `h = 0.08` the far regime still held 14% relative weight, enough to occupy the alpha = 0.10 tail. Removing it outright is the one thing the kernel cannot do. **Tool 2 is the special case with nothing truncated**, so [7] generalises it rather than competing with it (tested: equal groups, no truncation, thresholds agree to 0.15).

**2. Equal mass per group.** Tool 2 weights every *region*, so a group with ten times the regions contributes ten times the mass and a large distant regime outvotes a small near one on size alone. Here each retained group contributes a fixed share, divided among its own regions. **No bandwidth reproduces this**, which is what makes [7] a different method rather than a retuning. Live on CT's `ct/{family}/v{views}/a{arc}` groups after P-34; dormant on the MRI grid where every operator sees every item.

**The selection rule resolves D52's tension instead of restating it.** D52 ended at "accuracy wants concentrated weight, certifiability wants spread weight, and no `h` gives both". But D32 says exactly how much is enough — nine positives at alpha = 0.1 — so `nearest_certifiable_set` adds groups in distance order and stops at the first set that certifies. That is the **tightest** calibration set that can still carry a guarantee, which answers both requirements at once. It reports failure honestly when every group is retained and the result still will not certify.

**Measured on the interior fold where Tool 2 failed:**

| arm | threshold | coverage | flag area |
|---|---|---|---|
| oracle (unavailable at deployment) | 0.8152 | 0.902 | 0.874 |
| Mondrian fallback | 0.5298 | 1.000 | 1.000 |
| Tool 2 | 0.5658 | 1.000 | 1.000 |
| **[7] nearest-certifiable** | **0.8926** | 0.666 | 0.599 |

**[7] fixes the location error** — 0.8926 against the oracle's 0.8152, where Tool 2 sat at 0.5658, below the deployment operator's minimum score. It got there by truncating the far group entirely, which is exactly the designed behaviour.

**And it over-shoots, so by D51 it is invalid on this fold.** Coverage 0.666 against nominal 0.90. It retained only the nearest group, whose scores are *higher* than the deployment operator's, so the threshold is too high — which means on this fold [7] reduces exactly to D52's `nearest` arm (identical numbers), because retaining one group *is* using that group's threshold.

**The honest limitation: two candidate groups cannot test [7].** Its content is choosing *how many* groups to retain, and with two there is nothing between "nearest" (under-covers) and "all" (vacuous). The phantom bank has two; the real nine-operator grid has eight candidates per fold, which is where the rule has somewhere to stop. **So [7] is built, correct, reduced to the right special cases, and unevaluated.** One command on Track 2's grid settles it: `tool2_holdout.py` with the arm added.

**One cost to state, because it is not obvious.** Truncation makes the implied weight function discontinuous in operator distance, so T3.3's Lipschitz argument for the smooth kernel does not transfer. **A bound proved for Tool 2 does not cover [7]** — which is the reason [7] is a fallback rather than a replacement, and it belongs in any write-up that reports it.

*Found while building it:* a scaling bug of mine. Passing `1/n_groups` as the test-point weight against group masses normalised to sum to one made the test atom's own probability 0.33 against alpha = 0.10, so the `-inf` atom satisfied the quantile condition alone and every threshold came back `-inf` — presenting as "the method refuses to certify" rather than as a scaling error, and the diagnostic note said so. Fixed, the note now distinguishes the two causes, and there is a regression test.

---

### D67 — Coverage claims survive the label defects; detection claims do not [T1.6, Phase 2 gate]
The separation that decides what the midterm may claim, and it had not been stated anywhere.

**Coverage survives.** O3 and O5 are statements about calibration *behaviour* under a fixed label: a threshold transferred across operators loses coverage, a per-group threshold does not. Both arms are scored against the same label, so its defects change what "hallucination" *means*, not whether the comparison is valid. MRI: naive catches 0.328 [0.151, 0.492] at `variable_density2d R2`, missing nominal with the whole interval in 4 of 8 cells; CT: graded, 0.890 → 0.693 as views thin, below nominal on 13 of 13 splits at four operators. Group-conditional restores in both (MRI 0.895 mean, CT 0.902–0.923).

**Detection does not survive.** It is a claim about the score's merit against a label that is approximately a function of the score. T2.7 measured `err_free ≈ sqrt(s_free² + free_content²)` to Spearman 0.99 — D38's identity at `cos θ ≈ 0` — so the stratified AUROCs of 0.95–1.00 measure the identity. D65 established this on one bank; it now holds on both modalities.

**So the midterm reports the coverage failure and its restoration, with intervals, and does not report detection quality as a result.** Where detection numbers appear they are labelled as a consequence of the identity under a mismatched prior. This is a narrowing of the claim, not a weakening of it: the part that survives is the part the blueprint's O3 and O5 actually assert.

---

### D68 — The decomposition can only be tested on DPS; the solver ablation is promoted from ablation to load-bearing [T1.6/T1.4, Phase 2 gate]
Two of Track 2's measurements say the operator decomposition is not earning its place on detection, and they say different things.

**The combination result is inside the noise.** The equal-weight score is worse than its best single component in 7 of 9 operators, by 0.002–0.014 AUROC, against a patient-split noise floor of 0.034–0.048 in coverage units. One-sided sign test *p* = 0.090. Honest statement: **no evidence the combination helps** — not that it hurts.

**The dispersion result is the serious one, and its mechanism is the finding.** Posterior dispersion ranks hallucinations within **0.001–0.012 AUROC of `s_disagree`** at every operator. That is not two methods agreeing: **DDNM replaces the range component structurally, so the posterior spread is already confined to the null space.** Dispersion and `s_disagree` are one quantity computed two ways, and on this solver the decomposition *cannot* contribute to `s_disagree` because the solver has already performed it.

**Consequence.** The decomposition is testable only on a solver that does not structurally confine spread to the null space. D-033 measured DPS's null-error share at 0.817–0.852 at 24–48 steps against DDNM's 0.99. **So the solver ablation must include DPS, and it stops being an ablation: it is the only configuration in which the decomposition can earn its keep.** Run on DDNM alone it measures the solver's design rather than the score's.

This is scheduled ahead of everything else in Phase 3, because it is the single experiment that could change what the project is able to claim about its own score.

---

### D69 — The MRI distance curve is not forced; CT carries the curve claim [T1.6/T1.4, Phase 2 gate]
The headline coverage-gap-against-distance curve works on CT and does not on MRI: ρ = **+0.976** against **+0.316** (from `random1d R2`; +0.556 from R8). The reason is D47's, now confirmed on the full grid — MRI's worst failures sit at matched acceleration across mask families, where the scalar distance is ≈ 0, so a single curve would be a line fitted through two clusters.

**Ruling: MRI is plotted as one line per mask family, and the curve claim is carried by CT.** This is the two-coordinate reading of D47 applied to the figure rather than to the axis definition, and it is also the more honest presentation: the modality where the distance is a genuine scalar gets the scalar axis.

**And the noise floor is the patient split, not the seed** — 0.034–0.048 against 0.007 at R4 and 0.001 at R8. **This retires D59's emphasis**, which asked for a second seed sweep at R8 on the expectation that seed variation grows with acceleration. It does not; it is negligible at both. The bar an ablation effect must clear is ~0.05 coverage, and it belongs printed beside every ablation table.

---

### D70 — The strong form of the cross-domain claim fails; state the weaker one [T1.7/T1.4, §21 R5]
Surfaced by drafting §0 of the manuscript. The cross-domain claim's strongest form is that the two modalities' coverage-gap curves **collapse onto one curve** against operator distance — which would mean the gap is a function of how far the operator moved and not of the modality, i.e. "the failure is a property of the acquisition in general".

**The evidence already says the strong form fails.** The gap is ordered by signed distance on CT (ρ = +0.976) and barely on MRI (+0.316), because MRI's worst failures sit at matched acceleration across mask families where the scalar distance is ≈ 0 (D47, D69). There is no single curve for the two to collapse onto, so the curve-collapse test — which is Track 1's per the Week 9 card — is answered in the negative before being formally run.

**Ruling: claim the weaker form, which is still the cross-domain result.** *The same failure and the same fix appear in both modalities, and the ordering by distance is demonstrated where the distance is a genuine scalar.* That is what §0 asserts. Claiming collapse would be claiming something the project has measured to be false, and a reviewer with `fig5` in front of them would see it immediately.

**Why this is not a retreat.** The two modalities exercise structurally different operator classes, and that is itself measured rather than asserted: four of the five candidate distances coincide to 1.1e-16 on MRI because a mask operator is diagonal in k-space, and they genuinely differ on CT. A failure that reproduces across *that* difference is the cross-domain claim worth having. The scalar axis working on one of the two is a finding about the axis, which D47 and D69 already own.

*Still owed by Track 1 on CT:* weighted recalibration and fallback [7], both unstarted. The CT grid has the most interior operators in the project, so it is the only place Tool 2 and [7] can be distinguished from the nearest-operator arm — and the bank is on Track 2's machine, so it is a request rather than a run.

---

### D71 — The leave-one-out harness could not run on CT, for two reasons neither of which announces itself [T1.2/T1.4, Week 9]
Weighted recalibration on CT is Track 1's per the Week 9 card. The CT bank lives on Track 2's machine, so the deliverable here is the harness being *able* to run on it — and it could not, for two reasons.

**1. It recomputed every ordered pair's distance inline.** On CT one pair costs minutes at 256², and a leave-one-out sweep needs every ordered pair of the grid. `ct_distances.py` already caches them, so `--distances-csv` now reads that cache, falling back to computing a pair only when it is absent. A CPU re-analysis stays a CPU re-analysis.

**2. Interiority was decided from `measurement_dim`, which is blind to an angular-arc change.** This matters more, and it is the kind of defect that produces a quiet wrong answer rather than an error. Measured on real CT operators at 32²:

| operator | `measurement_dim` | signed growth from `180°/60v` |
|---|---|---|
| `180°/60v` | 1920 | — |
| `90°/60v` | **1920** | **+0.5029** |
| `180°/30v` | 960 | +0.3370 |

**Narrowing the arc at a fixed view count changes nothing `measurement_dim` can see, and it is the larger operator shift of the two** — bigger than halving the views. Under the old test every arc fold would have been classified as an *endpoint* and dropped from Q4's comparison, and the arc folds are a large part of what CT is in the project for.

**Fixed by defining interiority on the signed axis**: a deployment operator is interior when some calibration operator hides more than it does and some hides less, so the retained regimes bracket it and the weighting can interpolate rather than extrapolate (D52's constraint). That is the right test in both modalities and it is measured from the same quantity the figure plots.

**The MRI classification is unchanged** by the generalisation — R4 interior between R2 and R8, endpoints still endpoints — which is asserted as a test rather than assumed, because a generalisation that moves the case it generalises from is a different change.

**What this unblocks, and what is still owed.** `tool2_holdout.py` now runs on a CT bank with five arms — matched, Mondrian, nearest, Tool 2, and fallback [7] — reading cached distances. **CT is the only grid in the project with enough interior operators to distinguish Tool 2 and [7] from the nearest-operator arm**, which is where D52's and D66's verdicts have both been stuck. The bank is not local, so this is one command for Track 2:

```
PYTHONPATH=src python3 scripts/tool2_holdout.py runs/<ct_run_id> \
    --distances-csv <ct_distances.csv>
```

---

### D72 — The curve-collapse test is built, and its verdict rests on magnitude rather than significance [T1.4, §21 R5, delivers D70's test]
D70 ruled that the cross-domain claim's strong form fails and the weak form is what the manuscript asserts. That was a reading of a figure. §21 R5 asks for the *test*, and a reviewer will ask for the statistic, so here it is: `evaluation/collapse.py` and `scripts/curve_collapse.py`, 12 gates.

**What it compares.** Two nested fits of coverage gap on signed operator distance — **pooled** (one slope and intercept for both modalities, the strong form) against **separate** (one each, the weak form) — with bootstrap intervals on each modality's slope.

**The design decision, and it came from a control that failed.** My first version let an F-test decide. On a control with slopes **0.3025 and 0.3029** and overlapping intervals, it returned **p = 0.031** and declared no collapse. The F-test asks whether the fits differ *at all*, and it is hypersensitive where residuals are small: `RSS_separate` was near zero, so a negligible intercept difference filled the denominator.

**So the verdict rests on magnitude against D69's noise floor.** Collapse requires the slope intervals to overlap *and* the single curve to cost less than **0.05 coverage** in RMSE — the patient split's own spread. The F-test is still reported, labelled as not decisive. This is the same significance-versus-magnitude distinction I have had to apply to Track 2's component ordering (D68) and to my own stratified verdict (D65); it is the third time in this project that a test which *rejects* has had to be overruled by a measurement of how much the difference is worth.

**Validated on both controls:** equal slopes collapse (+0.0020 against the 0.05 floor) despite the F-test rejecting; a 25× slope difference does not (+0.1322, intervals disjoint).

**Two caveats the result carries rather than hides.**
- **The modalities occupy different distance ranges**, so a passing test would extrapolate one modality's law into the other's range. The summary prints a CAUTION when the ranges are disjoint, and the asymmetry is stated for the write-up: **a negative result here is strong, a positive one is weak.**
- **A permutation test on modality labels is deliberately not offered.** With non-overlapping distance supports the labels are not exchangeable, so the permutation null would be answering a different question.

**I also wrote the F survival function out rather than importing scipy** — one function is a heavyweight import on the analysis path, and scipy has broken a development install in this project. That makes it a liability unless checked, so it is verified against **seven published critical values to within 5e-3**, plus monotonicity and the `f = 0 → p = 1` edge case.

**Not run on the real numbers, deliberately.** It needs the two coverage tables and the two distance CSVs; the MRI table I hold locally predates the current label, and producing a number from it risks that number being quoted. One command for Track 2, and the inputs are artefacts that already exist:

```
PYTHONPATH=src python3 scripts/curve_collapse.py \
    --mri-coverage <mri_coverage.csv> --mri-distances <mri_distances.csv> \
    --ct-coverage  <ct_coverage.csv>  --ct-distances  <ct_distances.csv>
```

Run it on `naive_transferred` for the claim, and on `group_conditional` as the control: a fix should show **no trend in either modality**, so a collapse there would be two flat lines agreeing, which is not evidence of a law.

---

### D73 — The dispersion baseline is compared on the coverage axis, not the detection axis [T1.4, answers the D68 risk from the other side]
D68 named the risk that sinks Track 1's own contribution: posterior dispersion ranks hallucinations within **0.001–0.012 AUROC** of `s_disagree` at every operator, so the decomposition appears to buy nothing. The planned answer was the DPS ablation, which needs Track 2's GPU.

**There is a second answer, and it needs no GPU: the comparison was being held on the wrong axis.** `scripts/dispersion_baseline.py evaluate` reports AUROC and AP and nothing else — a pure detection comparison. D67 had already ruled detection numbers unreportable on this bank, because the label is near-determined by the score. So the number that threatens the contribution was measured on the axis the project disqualified, while the axis the project actually claims — coverage under operator shift — had never been run against the baseline at all. `scripts/coverage_by_patient.py` takes no `--score` flag; it is hardwired to CALIBER's score.

**The estimand.** Absolute coverage is not comparable between detectors: different scales, different thresholds, different flag areas. The *loss from transferring* is, because it is within-detector:

    penalty(d) = coverage(d, threshold fitted at the deployment operator)
               - coverage(d, threshold fitted elsewhere)
    delta      = penalty(baseline) - penalty(CALIBER)

Each detector is measured against itself first, and only the two self-comparisons are compared. `delta > 0` is the decomposition buying **transferability** rather than detection power — which is the project's thesis, and is exactly what dispersion has no mechanism for, since nothing about full-image spread tracks what the operator changed.

**This also survives the label's defects, and for a structural reason rather than by hope.** Both detectors are scored against the same label, so a label that is partly a function of CALIBER's score inflates CALIBER's coverage *in both of its arms equally* and cancels in the penalty. That is an argument about the estimand; it is not a repair of the label and does not license reading the AUROCs again.

**I shipped this with a defective interval and the null caught it.** The first version resampled the test patients and held the thresholds fixed. But a split-conformal threshold is a single order statistic of a few dozen positives, and its variation is the *larger* term — so the interval measured patient variation, ignored threshold variation, and read as tight. Measured on a 10-calibration-patient fixture with **nothing drifting**: `|delta|` averaged **0.0386** and reached **0.1277**, against the 0.02 floor I had pre-registered. It would have called threshold noise a CALIBER win most of the time. This is the seventh instance of the project's recurring failure — a statistic returning the hoped-for number — and the fourth of them mine.

**The fix is to resample the calibration side too and refit every threshold per replicate.** `_tau_replicates` calls `split_threshold` rather than reimplementing it, so the flooring rule and the `-inf` for an uncertifiable group are identical to the point estimate's. Validated both ways:

| fixture | result |
|---|---|
| null, 40 fixtures, nothing drifts | interval covers zero **40/40**, **zero** directional verdicts |
| 1.5-separation drift, 20 patients | detected **16/16** |
| 0.75-separation drift | 10/16 at 20 patients, **15/16** at 40 |
| interval width | 0.29 at 20 patients → 0.20 at 40 → 0.14 at 80 |

**Two consequences for how this is read, both of which weaken what can be claimed.**
- **A single detector's penalty is not evidence of transfer failure.** With no drift it is centred on zero but reaches 0.05 at 10 calibration patients (0.014 at 100). Only the difference between two detectors on the same patients is reportable.
- **Equivalence now requires the whole interval inside the floor, not the point estimate.** `abs(delta) <= floor` would have called the null "the two transfer alike" off an estimate of 0.015 with an interval of ±0.15. At ~40 patients and a 0.02 floor that condition is unreachable, so the honest verdict on this bank will usually be **`undetermined`**, and `floor_for_equivalence` reports what it would take to leave it. The project cannot establish detector equivalence on the coverage axis at this sample size; it can detect a difference of ~0.2 coverage points.

`evaluation/transfer.py` + `scripts/transfer_baseline.py`, 26 gates. The GPU step is already done — the 0.001–0.012 numbers prove `dispersion.parquet` exists — so this is one CPU command on Track 2's machine:

```
PYTHONPATH=src python3 scripts/transfer_baseline.py runs/<run_id>_by_patient \
    --low mri.random1d.R2.c0.08.s0 --high mri.random1d.R8.c0.08.s0 \
    --dispersion runs/<analysis>/dispersion.parquet
```

**Not run on the real bank, deliberately** — the local bank is calibration-only with 3 samples, and the guards correctly refuse it.

### D74 — The hallucination label thresholds a direction, not a magnitude [T1.1, attacks D67's cause]
D67 ruled detection claims unreportable because `err_free ≈ sqrt(s_free² + free_content²)` at Spearman 0.99 — the project's own identity at `cos θ ≈ 0` — so the 0.95–1.00 AUROCs measure the identity. Four label forms were tried and **all four thresholded a magnitude**: `err_free`, the ratio, the ratio under a candidacy gate, `err_free` under a design quantile. Each inherited the same tautology, which is why moving the threshold never fixed it.

`cos θ` is the other coordinate of the same decomposition and it is **scale-free**: it says which way the invented content points, not how much there is. `attribution.py` has inverted the identity to recover it since Q6; no label had ever used it. D74's label is `cos θ < kappa`, with `kappa` a design-split quantile (0.25, pre-registered).

**Why this is the right coordinate, physically.** A region where the prior recovered the hidden anatomy and one where it invented structure can carry *identical* `err_free`. Only the direction separates them, and the project's own definition (Sec. 4) says the first is reconstruction and a flag firing on it is a false positive.

**Measured on the w02 M4Raw bank** (real prior, 384 regions — a smoke bank, not a reported one):

| quantity | value |
|---|---|
| identity residual, max relative | **0.000e+00** — `cos θ` exactly recoverable |
| `cos θ` spread | sd **0.0892**, range [−0.338, +0.269], median 0.073 |
| ρ(`err_free`, `hypot(s_free, free_content)`) | **+0.971** ← what a magnitude label inherits |
| ρ(`cos θ`, `s_free`) | **−0.435** ← what the direction label inherits |

**The honest claim is that the dependence is moderate rather than algebraic, not that it is absent.** Nothing built from `(s_free, free_content, err_free)` can be independent of a score built from `s_free`, because the identity ties all three. What changes is that the score *can now fail* against this label, where against a magnitude label it could not.

**And on the preliminary evidence it partly does fail.** Per operator, direction-agnostic (`max(auc, 1-auc)`, D65's rule), against the `free_content` anatomy control:

| operator | label | score | anatomy | excess |
|---|---|---|---|---|
| R4 | magnitude | 0.572 | 0.946 | **−0.374** |
| R4 | alignment | 0.618 | 0.770 | **−0.152** |
| R8 | magnitude | 0.881 | 0.579 | +0.302 |
| R8 | alignment | 0.615 | 0.515 | **+0.100** |

So the direction label **removes the saturation** — AUCs fall from 0.88 to a believable 0.62 — and **halves the anatomy deficit at R4**, but does not by itself rescue a detection claim: at the milder operator the anatomy control still wins. Two caveats: R4 carries only 16–17 positives, barely above D32's nine-positive floor, so those AUCs are very noisy; and my first reading of this table was **pooled**, which gave `+0.097` and hid the sign reversal between operators — D60's retracted error, reproduced in the same session as the module built to avoid it.

**One trap, encoded as a test rather than a comment.** A label like `s_free · sin θ` — invented *mass* rather than direction — looks more physical and restores the tautology intact: `sin θ ≈ 1` wherever `cos θ ≈ 0`, which is nearly everywhere on this bank, so the product collapses back to `s_free` (ρ > 0.95 measured). Do not "improve" the label this way.

Undefined-`cos θ` regions are labelled **negative and counted**, never dropped: dropping them would make the label's support depend on score magnitude, which is the same confound by another route. `label_comparison.py` refuses any bank failing the identity, because `cos θ` is meaningless there.

`attribution.py` + `evaluation/bank.py` (label `alignment`) + `scripts/label_comparison.py`, 17 gates. **D-061's absolute label remains the reported one** until this is run on a reportable bank.

### D75 — Tool 2's defeat by the nearest-operator arm may be a property of the grid, not the method [T1.2/T1.4]
Every grid run so far reports Tool 2 as valid but flagging no less area than `nearest`, recorded as a negative result. That comparison has a feature of the **grid** built into it: these grids are dense, so each held-out operator has a calibration operator close enough that borrowing its threshold wholesale is nearly right. Weighting cannot beat `nearest` when `nearest` is already nearly `matched`, and that says more about the operator grid than about the method.

`--exclude-nearest N` drops the N closest calibration operators along with the held-out one, posing the deployment case Tool 2 was argued for (§9, D4): an operator with no near neighbour to borrow from. The `nearest` arm is recomputed over the survivors, so both arms face the same handicap.

**The trap, and it is why interiority had to move inside the fold loop.** Removing a neighbour can leave a previously bracketed operator at the end of the retained signed axis, where the weighting can only extrapolate (D52). Counting such a fold as interior would credit Tool 2 with an interpolation it cannot perform. `is_interior` now takes the retained list rather than the whole grid, and the dropped operators leave the calibration pool as well — otherwise the weighting would still be borrowing from the operator the fold removed.

A re-analysis of an existing bank; no new GPU. Both readings should be reported: Tool 2 buys nothing on a dense grid, and the open question is whether it buys anything on a sparse one. The harness now says so itself when `nearest` wins and the flag is unset.

### D76 — "The weights could not be selected" and "the result depends on the weights" are different claims [T1.1, converts D54]
D54 recorded that 8 design patients cannot separate any weighting from any other, so `WeightSelection` reports `discriminated=False` and the project ships equal weights as a stated default (D25). That reads as a gap: a score whose weights were never chosen.

But only the first claim was ever established. If the coverage result holds at **every** weighting on the sweep, the inability to select one costs the project nothing — and that is a stronger position than a selected weight, which would invite the question of what the result would have been at another.

`weight_sensitivity` sweeps `w_free` across D28's grid and recomputes the quantity the paper leads with — the paired within-patient coverage restoration — at each point. The verdict has **two** conditions, not one: the restoration must hold at every weighting, *and* the spread across the grid must be smaller than one weighting's own interval. A flat curve is evidence of insensitivity only if a single point is determined to better than the curve's spread; without that second condition it is a curve nobody can see the shape of, which is D72's lesson pointed the other way.

The label is held fixed and **asserted** to be, per weight: every label form is a function of cached truth columns (and, for D74, of `s_free`), none of which move with the weighting. A label that moved would make each row answer a different question.

**Also delivered: flag area at matched coverage** (`area_at_matched_coverage`). D39 ruled that detectors compete on area because a conformal threshold pins the miss rate; D51 added that an area means nothing unless the arm is valid, after the `nearest` arm posted the project's smallest area at coverage 0.547. Fixing *coverage* and comparing area closes that loop by construction — the degenerate detector cannot win, because reaching the target is what it is charged for, and a constant score has to flag the whole image to do it. Independent of any AUROC, so it stays available under D67: a statement about operating points, not about ranking.

`evaluation/sensitivity.py` + `evaluation/transfer.py` + `scripts/weight_sensitivity.py`, 16 gates including a negative control — a bank where `s_disagree` is pure noise must **not** be reported as weight-insensitive.

---

### D77 — The preprint cut excludes both novel tools and leads with the failure [T1.7, delivers §23's first publication]
`T1.7_preprint_draft.md`. The cut is narrower than the blueprint's §23 sketch, in three specific ways, each because the excluded thing is either not ready or not true.

**Out: Tool 2.** Valid, and flags no less area than the nearest-operator arm on every grid run (D71). D75 raises the real possibility that this is a property of dense grids rather than of the method, and that experiment has not run. **A preprint is the wrong place to debut an unresolved negative about your own primary contribution** — it fixes the framing in the literature before the question is settled, and if `--exclude-nearest` later vindicates the weighting, the preprint has already conceded it.

**Out: Tool 3.** T3.6 does not exist and the √null_frac law was refuted at p = 6.9e-15. §23 always planned a separate analytical paper; this just confirms the split was right.

**Out: every detection claim**, per D67.

**What is left is a complete contribution**: a demonstrated failure, a demonstrated fix, two modalities with structurally different operators, and the failure graded by a quantity computable from the operators alone. Every number in it is one Track 2 has already measured and reported — CT 0.890→0.693 along views and 0.888→0.796 along arc with 13/13-split failures at the severe end; group-conditional restoring 0.902–0.923 on CT and 0.884–0.901 on MRI; ρ = +0.976 for the signed distance against +0.762 symmetric. Nothing is invented and nothing waits on a run.

**The title leads with the negative.** "Conformal reliability certificates ... do not survive a change of acquisition operator." The measured result *is* the failure; the fix is textbook once the failure is established, and claiming the fix as the contribution invites the correct objection that group-conditional CP is Vovk et al. Leading with the failure also makes the paper's novelty defensible at a workshop-tier venue, which is where this realistically lands.

**§6.4 is promoted from limitation to result, and it is the part most likely to outlive the rest.** D67's finding — that a magnitude-based hallucination criterion becomes a function of the score under a mismatched prior, via the identity, at Spearman 0.99 — is a caution about how *the field* evaluates these detectors, not just about our bank. It comes with the identity that predicts when it happens, the four criteria that all inherit it, the anatomy control that beats the score (0.946 against 0.572), the pooling error that reads as "exactly chance" (0.499 pooled, 0.662 within operator), and D74's direction-based alternative. A reader can apply all of that without using anything else in the paper.

**One hard blocker remains: medical weights (D-016).** The draft states at §6's head that every number is development output and why. The structure of the results should survive the prior change; the magnitudes will not.

---

### D78 — The solver ablation's analysis is written ahead of the run, and its premise has to be measured at image level [T1.6/T1.4, prepares D68]
D68 promoted the DPS ablation from ablation to load-bearing: it is the only configuration in which the decomposition can earn its keep, because DDNM replaces the range component structurally and has already confined the posterior spread to the null space. Writing the analysis now means the run is one command rather than a session's work, and it means the verdict rules are **pre-registered** rather than chosen after seeing the numbers — which matters more here than usual, since this is the experiment the project wants a particular answer from.

`scripts/solver_ablation.py`, 13 gates, with fixtures that plant the answer **both** ways: a bank where the low-share solver's score beats its baseline, and one where it ties. A verdict that cannot come back negative is not a verdict, and the negative branch is the outcome this project should expect.

**Three sections in a fixed order, and the first can stop the other two.** The premise — the null-error share per solver — is checked before any detection number is computed, and if the two solvers do not differ by at least 0.05 the script says `PREMISE ABSENT` on stderr: D68's rationale would simply not hold on those banks, and reading sections 2 and 3 would be reading an experiment that was never set up.

**The finding that came out of building it: the null-error share cannot be computed per region.** `P_N` is a **non-local** operator, so projecting and then restricting to a region is not the same as restricting and then projecting — energy crosses region boundaries and the orthogonal decomposition does not hold region by region. Measured on the M4Raw bank, `err_free² / err_total²` exceeds 1 in **97 of 384 regions**, reaching 1.035. This is not a bank defect and not a normalisation error; the identity residual on the same bank is exactly 0. It is simply the wrong place to take the ratio.

So the script takes it as an **area-weighted energy ratio over each volume**, where it is the global MSE ratio and exact, and prints the per-region out-of-range count as a diagnostic rather than hiding it — a *large* excess there would indicate a genuine normalisation problem of the kind `identity_residual` exists to catch.

**This has a consequence for D68 itself.** The 0.99-against-0.817–0.852 figures the ablation's rationale rests on came from Track 2. If they were computed as a mean of per-region ratios they carry a bias, and the gap is exactly the quantity that decides whether the ablation has a premise. Asked in cross-track D-065.

**One more caveat the verdict carries rather than drops:** an excess measured against a magnitude label can be the identity rather than a detector (D67), so the script says so in its output and points at `label_comparison.py`. The ablation can establish that the decomposition beats its baseline; it cannot by itself establish that either of them detects invention.

---

### D79 — The medical prior was never licence-blocked; the preprint's one hard blocker is lifted [T1.6/T1.7, corrects D-016]
D-016 has deferred the medical prior since **Week 1**, and the stated reason was that Song et al.'s MRI checkpoint is fastMRI-derived, so NYU's data agreement plausibly reached the weights. **That is false.** Their training sets are LIDC (lung CT, 320×320), LDCT (head/chest/abdomen CT, 512×512) and **BraTS 2021 (brain MRI, 240×240)**. fastMRI is cited only for the *undersampling mask pattern*, which is where the confusion almost certainly started. Correction written into D-016.

**Why this is the most consequential entry in weeks.** D-016 gates every reportable number in the project, the `reportable` flag on every bank, D67's criterion-validity argument, and T1.7's preprint — where it is the *only* hard blocker. A nine-week deferral rested on a misreading of one citation.

**And the better substrate was already in the repository.** The MRI prior is brain; the fastMRI bank is knee, so BraTS is cross-anatomy there. But `banks/w02_solver_smoke/` is **M4Raw — brain MRI on real k-space, verified end to end**. BraTS → M4Raw is an anatomically matched medical prior, which is a stronger claim than anything the project has had, and it needs no new substrate.

**What it does not fix, and this matters for how the result is framed.**
- **Resolution.** 240×240 / 320×320 / 512×512 against the project's reduced-resolution 2D grid. The region definition (D20) lives on that grid, so changing working resolution moves the region count and every per-region quantity with it — including the design-split quantiles that set the label. Every gate is re-valued, not carried over.
- **The criterion is still circular** (D67). A medical prior should *raise* `cos θ` — its invented content should be less orthogonal to real anatomy than a face model's — which would weaken the `err ≈ hypot(s, c)` collapse. That is a prediction, and D74's `label_comparison.py` is the test. **It is written before the run, deliberately**, because this is a prediction the project wants to come true.
- **The licence question moved rather than vanished.** The weights carry **no stated licence**: no LICENSE file, Google Drive hosting. Fine for a cited preprint; a real question for the O7 release, and it should be raised before packaging rather than at it.

**Song et al.'s checkpoints turned out to be the wrong artefact, and the better one was elsewhere.** Their files are **JAX/Flax** msgpack (`step`/`optimizer`/`target`), not PyTorch, and NCSN++ **VE** score models against this project's **VP/DDPM** epsilon samplers — two conversions deep, either of which silently mis-scales. That download was abandoned mid-transfer once the format was clear.

**What is wired instead: SegGuidedDiff's unconditional models** (Konz et al., MICCAI 2024) — `ddim-breast_mri-256` (Duke Breast MRI) and `ddim-ct_organ_large-256` (CT-ORG, neck to pelvis). `diffusers` `UNet2DModel`, `in_channels=1`, 113.7 M parameters, trained on `DDPMScheduler` defaults so the schedule is this project's exactly. **Zero new loader code**; the schedule was verified end to end rather than assumed (table in D-016's resolution). Track 2's Week 2 search that found "no open substitute" was two years stale.

**Two things improve beyond simply having a medical prior.** The models are single-channel, so `DiffusersUNetDenoiser`'s greyscale-to-RGB replication becomes a no-op and one stated narrowing of the method disappears. And **CT-ORG is neck-to-pelvis organ CT against TotalSegmentator's whole body** — a genuinely close anatomical match, where LIDC would have been lung-only. CT is also where the project's results are strongest (ρ = +0.976, 13/13-split failures), so the best-matched prior sits on the strongest result.

Treat downloaded weights as untrusted input: a checkpoint can be a pickle, so `torch.load` executes arbitrary code. These are `.safetensors`, which is the format that exists to avoid that.

---

### D80 — Anatomy gates claims, not banks [T1.6, answering T2.10's question]
Track 2 registered Song's BraTS prior as `mri_brain` rather than `mri`, parked it in `AWAITING_RULING`, and asked Track 1 whether an anatomy-matched prior makes a bank reportable. Their framing is right and is adopted: **domains carry anatomy**, for every medical prior, because a bare `mri` would stamp a brain prior reportable on fastMRI *knees*.

**The ruling: a prior is admissible for a modality when it is a medical prior of that modality, whatever anatomy it saw. Anatomy mismatch gates which claims the bank can carry, not whether the bank is reportable.**

This is **D67's coverage/detection split applied to the prior instead of to the label**:
- **Coverage claims survive cross-anatomy.** Both arms are scored against the same label on the same reconstructions. A prior that suits the anatomy badly changes what gets labelled; it does not change whether a transferred threshold loses coverage relative to an in-group one. That comparison is the project's central claim and it holds.
- **Detection claims do not.** The criterion's validity rests on `cos θ`, and anatomy mismatch is precisely the mechanism that drives `cos θ` toward zero — the face-prior pathology reappearing in the one quantity the label depends on. T2.9 §4 measures it: at chance on CT, 0 of 9 operators beating the anatomy control.

**Why not block the bank.** Blocking would have blocked the *coverage* result too, and that is the one the project can actually make. The detection result is already gone for the same underlying reason, so refusing the bank buys no protection and costs the headline claim. `AWAITING_RULING` is now empty and kept only so the next prior from an unanticipated anatomy parks there rather than being admitted by whoever adds it.

**Not changed:** D-015's separate question of whether M4Raw is reportable as a *substrate*. That is a property of the dataset, not of the prior. An anatomy-matched prior does not launder a substrate.

**Five priors are now registered anatomy-qualified** across two independent families — Song VE/NCSN++ (`mri_brain`, `ct_chest`, `ct_head_chest_abdomen`) and SegGuidedDiff diffusers (`mri_breast`, `ct_neck_to_pelvis`). Keeping both is deliberate: it makes "does the result depend on which medical prior?" an ablation rather than an assumption.

### D81 — Two defects in my Week 10 scripts, both found by Track 2, both fixed [T1.4]
Track 2 ran the six queued analyses and overturned two of my results by hand. Both defects are mine and both are in the same family as entries I had already written warnings about.

**1. D73's penalty is signed, so over-coverage counted as transferring well.** `penalty = coverage(in-group) − coverage(transferred)`. A transferred threshold that **over-flags** has coverage ≈ 1.000, hence a *negative* penalty, hence reads as better than a threshold that is merely right. My `interpretable` gate checked `reaches_nominal` and `trivial` on the **in-group** arm only; the transferred arm's flag area never entered the verdict. Track 2 measured the cost: on CT, CALIBER's transferred arm flags ≥ 99.7% of all regions at 6 of 8 operators, and my script called that "CALIBER transfers better at 7 of 8". **This is D51's rule — an area means nothing until validity is reached — reproduced on the arm D51 was not looking at, in a module whose docstring cites D51.**

Fixed in three parts, because the obvious fix is insufficient:
- The comparison is now on **`|penalty|`**, taken **inside** the bootstrap per replicate before differencing. Taking it outside would give an interval for the signed difference wearing the absolute one's label.
- `|penalty|` alone does **not** suffice, and a test caught it: over-coverage can only move coverage from ~0.90 to at most 1.000, so it is charged ~0.10 however badly it flags, while under-covering can cost the full 0.90. A detector can still win by flagging everything.
- So a detector whose transferred arm over-flags **cannot be declared the winner**: the verdict returns `winner_over_flags`, which counts as undetermined rather than as either side's win, and points at `area_at_matched_coverage` — the estimand that compares at fixed coverage, where over-flagging is charged properly.

**2. `solver_ablation.py` valued one label on the pooled design rows of both banks.** Pooled ε = 0.856 against 0.906 (DDNM) and 0.726 (DPS), so each arm was scored at a positive rate that was not its own. A label gate is a magnitude, and **D-020 is this project's own finding that magnitudes transfer across neither modalities nor substrates** — two solvers are two substrates for this purpose. Now one label per bank, from that bank's own design split, with the two printed side by side and a note when they differ.

**And the verdict it printed was unsound.** It said "D68 ANSWERED, POSITIVE" off the magnitude label, with the D67 caveat relegated to a closing line. By D38's identity that label is approximately a function of `s_free`; the score contains `s_free` and `s_disagree` does not, so the excess *is* the identity. A positive is now withheld unless the label is D74's direction label. **A negative still stands on any label** — if the score cannot win where the label favours it by construction, that is decisive — and the asymmetry is deliberate.

**I also withdraw the question I put to Track 2 in D-065** about how the D-033 null-error shares were computed. They are computed at image level per item, exactly as I later established by reading `operators/decomposition.py` myself. The question should never have been asked; the answer was in the repository. Their definition is also better than the one I wrote: its denominator is `‖P_N δ‖² + ‖P_R δ‖²` rather than `‖δ‖²`, so it stays meaningful when the projector is only approximately orthogonal — the CT case, D-012.

---

### D82 — M4Raw is reportable for coverage claims and not for detection [T1.6, scopes D-015, follows D80]
D-015 made M4Raw a development substrate. Its stated reason, in `fastmri.py`, is worth quoting exactly because the scope of the ruling follows from it:

> `reportable = False`. M4Raw is 0.3 T brain against fastMRI's 3 T knee and carries **no pathology annotations**, so it cannot close the annotation-linked evaluation (Sec. 14).

**The consequence clause is annotation-linked.** Field strength is named as a way the substrate differs; the stated *harm* is that Sec. 14's annotation-linked evaluation cannot be closed. Coverage under operator shift consumes **no annotations**: the reported label is `err_free > eps` valued on the design split, and `require_annotation` is an optional secondary reading that the coverage claim does not use. So the gate's own rationale does not reach the coverage claim.

**Ruling: M4Raw carries coverage claims and not detection claims.** This is D80's structure applied to the substrate instead of the prior — *the defect gates which claims the bank can carry, not whether the bank exists*. The two rulings are now one principle with two instances, rather than a rule and an exception.

**Why this is worth ruling rather than routing around.** The MRI medical-prior result — naive transfer at 0.000–0.159 against group-conditional's 0.80–0.91, below nominal on every one of 13 patient splits in both directions — is the strongest demonstration of the central claim anywhere in the project. Dropping it would also drop Front II and objective O5 to a limitations paragraph, and the blueprint promises two modalities. Routing around the gate for CT (D-070) was the right unblock; it was never a reason to abandon MRI.

**Implementation: not a boolean flip.** Setting `reportable = True` would let a *detection* number from M4Raw be stamped reportable, which is exactly what this ruling forbids — and this project has made that error before (P-18: the first fastMRI bank stamped `reportable: true` under a face prior, because the gate could not see the thing that mattered). Following D-018's own logic, that reportability is a property of the bank so it travels in the manifest, the flag becomes scoped:

```python
reportable = False                            # the unqualified claim
reportable_claims = frozenset({"coverage"})   # D82: what this substrate carries
```

The harness then stamps "reportable for coverage" and refuses a detection number from that bank by construction rather than by memory. The same field expresses D80's cross-anatomy case — CT + LIDC is chest-prior on whole-body anatomy, so coverage-reportable and detection-not — so one mechanism serves both rulings.

**Two limits that travel with the claim, and must appear in the write-up.**
- **0.3 T is a generalisation limit, not a validity one.** The comparison is internally valid at 0.3 T: both arms, same label, same reconstructions. What field strength limits is whether the *magnitude* transfers to 3 T, which is untested.
- **"13 of 13 splits" is not 13 independent confirmations.** Thirty patients resampled 13 ways is a robustness check against split choice — which is what it was designed for — not 13 replications. Phrase it as "on every one of 13 patient splits" and never in a way that implies independence. The same caveat applies to the CT 13/13.

**External precedent, found while clearing the prior art (D-070):** Shor & Bronstein (arXiv 2610.26558, Oct 2026) use **M4Raw as the evaluation target** for a reconstruction-risk method, with fastMRI brain as source. A 2026 paper treating M4Raw as a legitimate evaluation substrate is not a licence, but it is evidence that the substrate objection is about scope rather than admissibility.

---

### D83 — Phase 4's frozen choices are code, and two of the frozen choices were measured wrong first [T1-A, T1-D, implements D-070 Sec. 2]

**The freeze is a module, not a table.** `src/caliber/evaluation/prereg.py` holds every tunable choice behind T1-A..E as a module constant with no setter, and the analyses import them rather than taking them as arguments. D-070 Sec. 2 asked for these to be fixed before the CT bank finished; a Markdown table does not achieve that, because nothing stops it being edited in the same commit as the result it governs. `prereg.fingerprint()` stamps a short hash of the whole freeze onto every result row, so a table produced under a different freeze is visible without reading git history. Five of this project's eight statistics traps were choices made after seeing a number, which is the entire reason for the mechanism.

**Two of the frozen choices were then checked, and one was wrong.** Both corrections are recorded here because both changed what gets reported.

**1. The permutation null shuffles operator labels, not the distance column.** Shuffling the distance column i.i.d. destroys the antisymmetry of `null_space_growth` and the triangle structure of the matrix, giving a null world in which no sign rule could work. Measured over 60 independent noise fixtures at 500 permutations each: label permutation rejects at **0.050** at the 5% level and **0.100** at the 10% level — calibrated to three decimal places — while the i.i.d. column shuffle rejects at **0.083**. The inflation is about 1.2 Monte-Carlo standard errors at 60 trials, so it is suggestive rather than decisive; the label permutation is used because its exchangeability argument is correct, not because the alternative was proved broken. Power is intact: the planted-perfect fixture returns p = 0.002 at J = +1.000.

**2. The closed-form Beta law is the wrong reference for a measured spread, and the first version of this module had the sign of the error backwards.** `Beta(n+1-k, k)` is the law of coverage against an *infinite* test set. A split measures coverage on a finite one, so

```
Var(measured) ~= Var_Beta(threshold) + p(1-p)/n_test
```

Simulation through the project's own split structure confirms this to within ~4% across a fiftyfold range of sizes (0.0368 vs 0.0385 predicted at 8 positives/patient; 0.0118 vs 0.0122 at 80; 0.0054 vs 0.0055 at 400). At the project's balanced splits the measured spread runs about **1.36x the closed-form Beta SD**, so comparing the two directly reports a *correctly calibrated* method as over-dispersed — and would invite a repair to something with nothing wrong with it.

The effect this module was originally written to handle — that the 13 splits re-cut the same 30 patients, so their coverages are positively correlated and the sample variance is biased low — is **real, in the predicted direction, and about fifteen times smaller**: 0.0366 against 0.0374 at matched split sizes, a 2.2% deflation. It is reported because it is true, not because it changes a conclusion.

**3. The spread check is underpowered at 13 splits, and says so.** The sample SD of `n` split coverages carries relative noise about `1/sqrt(2(n-1))`, so the reference interval spans **0.62x to 1.49x** the correct SD at 13 splits. The check therefore cannot distinguish correct calibration from a 36% error — the size of the very correction above. More splits barely help: 120 of them still leave a 1.27x band. A passing check is reported as `consistent_but_underpowered` rather than `consistent_with_exchangeability`, because a verdict of "consistent" from an interval that wide is low power and not evidence. **T1-D's worst-operator bound is the solid half of T1-D; the Beta-law spread comparison can only catch gross mis-calibration.**

**What the safety rule reports, and the comparator that makes it meaningful.** The primary endpoint is the **false-safe rate** — of the ordered pairs the rule green-lights, the fraction that under-cover — reported against the **trivial rule**, whose false-safe rate equals the base rate of unsafe pairs. If 10% of pairs under-cover, a 10% false-safe rate is what saying nothing achieves. It is reported beside the **green-light fraction**, because a rule that admits two pairs in seventy-two can reach a perfect false-safe rate and be useless; this is coverage and area again, on a different axis. The significance test runs on Youden's J instead, because J's denominators are counts of the measured outcomes and so are invariant under the permutation, whereas the false-safe rate conditions on a green-lit set that changes size under every replicate.

**Three refusals built in rather than left to judgement.** A reliably *reversed* rule is significant two-sided and returns `no_better_than_trivial`, not a result. An incomplete pair grid returns **no p-value at all** rather than one computed on a shrinking sample per replicate. And the worst-cell bound reports three separate numbers — the observed minimum (biased low), the pointwise bound (anti-conservative, since that cell was selected for being lowest), and the **Bonferroni simultaneous bound**, which is the only one a deployment certificate can cite.

**Also found, and relevant to T1-A's reading.** The three limited-angle operators and `ct.sparse_view.v120.a180` have **identical null-space dimension** (64000 at 320^2), because dimension counting sees views times detectors and not the arc. Any distance that reduces to a dimension count predicts zero difference across four of the nine operators whose ill-posedness differs sharply, so the limited-angle cells are where the probe-based signed measure is actually tested. The first computed pair agrees: `a120 -> a150` growth = **-0.1247**, correctly negative for a wider arc revealing more.

**Cost note for the plan.** The signed distance matrix is a property of the operators alone — no bank, no prior, no reconstruction — so it was started before T2-A exists. It is also **~15 hours of CPU**: 771 s per (pair, measure) at 320^2, 36 unordered pairs by 2 measures, with the reverse direction filled by antisymmetry rather than computed. It is resumable and contends with nothing on the GPU. This was not a scheduled item in D-070 and should have been.

**Status:** 47 new tests, full suite 603 passed / 0 failed / 5 skipped. No result from any of this — T1-A and T1-D need T2-A's bank for their coverage side.

---

### D84 — The signed distance matrix is a perfect total order on the CT grid, and that scopes T1-B's baseline [T1-0, amends D83's freeze]

**T1-0 is finished:** 36 unordered pairs x 2 measures over the 9 CT operators at 320^2, 72 of 72 computed, in `runs/t1a_ct_distances/distances.csv`. Operator-only -- no bank, no prior, no reconstruction -- so it was computed while T2-A was still unrun. Total cost ~11 h CPU at 180-770 s per computation, the reverse direction filled by antisymmetry rather than computed. **The grid matches `w12_ct_lidc_gpu.yaml` exactly**: same nine operators, same 320^2, so the matrix is valid for the bank T1-A will read.

**The signed measure induces a single consistent ill-posedness ordering, with zero violations in 72 ordered pairs:**

| | operator | index |
|---|---|---|
| 1 | `sparse_view.v180.a180` | +0.426 |
| 2 | `sparse_view.v120.a180` | +0.216 |
| 3 | `limited_angle.v120.a150` | +0.169 |
| 4 | `limited_angle.v120.a120` | +0.111 |
| 5 | `limited_angle.v120.a90` | +0.030 |
| 6 | `sparse_view.v90.a180` | -0.024 |
| 7 | `sparse_view.v60.a180` | -0.206 |
| 8 | `sparse_view.v45.a180` | -0.316 |
| 9 | `sparse_view.v30.a180` | -0.405 |

Every one of the 72 signed distances agrees in sign with this ranking. Three things follow, and one of them changes the pre-registration.

**1. The sign rule has a plain-language reading.** "Safe iff the signed distance is at most zero" is exactly "safe iff you deploy on an operator that is no more ill-posed than the one you calibrated on", because on this grid the signed distance reduces to a one-dimensional index. That is the form a deployment rule wants.

**2. The green-light fraction is exactly 50%, forced by antisymmetry, and is not evidence of anything.** For any antisymmetric measure, exactly half of the ordered pairs are negative: 36 of 72. So D83's `green_light_fraction >= 0.10` floor cannot bind on this grid and must not be reported as a check that passed. The rule's selectivity is not a free parameter here -- which is good, there is nothing to tune, but it means the floor is doing no work and saying otherwise would be padding.

**3. The limited-angle family interleaves with the sparse-view family, and that is where the distance earns its keep.** The three limited-angle operators sit *between* `v120` and `v90` on the common scale. No acquisition parameter can produce that ordering: all three limited-angle operators have 120 views, and all six sparse-view operators span 180 degrees, so **no single scalar knob orders a limited-angle operator against a sparse-view one**. Combined with the fact that those four 120-view operators share a null-space dimension of 64000, the cross-family cells are the only place where both parameter-style baselines are either undefined or degenerate.

**Consequence: T1-B's baseline was underspecified, and it is amended before any coverage number exists.** `GAP_PREDICTION_BASELINES` was `("constant", "acquisition_parameter")`. The acquisition parameter is a *strong* baseline within a family -- which is the whole force of the "any monotone function of view count orders the gaps" objection -- and **undefined across families**. Leaving that unstated would have let the comparison be settled after seeing which cells the distance won. So:

- the baselines are now `("constant", "acquisition_parameter", "null_space_dimension")`, with `BASELINE_SCOPE` recording where each one is defined;
- `null_space_dimension` is added because it is the only parameter-style baseline defined on the cross-family cells, **and it is degenerate there** -- which is the point, not a flaw to hide: a baseline that cannot separate four of nine operators is a weak one, and saying so is better than quoting a flattering margin against it;
- within-family and cross-family results are reported **separately and always both** (`GAP_PREDICTION_REPORT_BY_FAMILY_SCOPE`), because pooling lets 36 pairs with a strong baseline average against 36 where it does not exist.

**On the legitimacy of amending a freeze.** An amendment is only honest while no number it governs exists. This one was derived from the forward operators alone with T2-A unrun, and it is recorded in `prereg.AMENDMENTS` with its date and reason rather than silently edited, so `fingerprint()` moves visibly (`f2ee5cef4f66` -> `4495741687bf`) and any table stamped with the old hash is identifiable. A test asserts every amendment is dated at or after the freeze and carries a reason.

**The symmetric measure cannot ground the rule.** All 36 `null_space_projection` values are positive, range 0.295 to 0.574. It carries magnitude and no direction, so it is reported beside the signed measure and cannot substitute for it. Worth stating because D-052 found four of five measures coincided on MRI masks; on CT they do not -- for the pair `a120 -> a150`, signed growth is **-0.1247** while the symmetric distance is **+0.4773**. The two disagree about whether that transfer is downhill, which is exactly the information the signed measure adds.

**T1-A is claim-compatible with T2-A, checked rather than assumed.** T1-A and T1-D report coverage and nothing else, so `scripts/t1a_safety_rule.py` now calls Track 2's `require_claim(provenance, "coverage")` and refuses a bank that does not carry it. Under T2-B's rules, TotalSegmentator with the LIDC chest prior carries **coverage only** -- cross-anatomy, so D80 grants coverage and refuses detection -- which is exactly the claim the analyses need. A test pins the agreement, so if the admissibility rules move it fails then rather than on the day the bank lands. The same check shows the old face-prior M4Raw bank carries **nothing** (the prior is not admissible for MRI), and that M4Raw under the BraTS prior carries coverage: D82 works end to end, but banks written before the field existed read as carrying nothing until `scripts/restamp_bank.py` is run on them.

**Status:** 56 Track 1 tests. No result -- the coverage side of T1-A, T1-B and T1-D is still T2-A's bank.

---

### D85 — The LIDC control passes on the registry split once its interval includes the calibration draw [rules P-44, T2.11 Sec. 1e; corrects D83's own gate]

Track 2 built T2-A's bank (900 pairs, 19 Nov 20:18), applied the pre-registered matched-operator control, got **FAIL**, read nothing downstream, and escalated the reading to Track 1. That was the right call and the right order. This is the ruling.

**The ruling: the control PASSES, on the registry split, at 9 of 9 operators — not because the control was redefined, but because the interval it was tested against omitted the larger of its two variance terms.**

**What the control did.** Per operator, the group-conditional arm was calibrated on the 41 calibration subjects, tested on the 51 test subjects, and compared against a 95% interval that — in Track 2's own words — "resamples test subjects only. It does not move the 41 calibration subjects, and they are where most of this spread comes from." The interval contained nominal at 4 of 9 operators, so the rule failed.

**Why that interval is the wrong one, and why this is not a post-hoc repair.** The requirement that a conformal coverage interval must move the calibration side is **not** being introduced now to rescue a failing control. It was frozen in D83 on **18 November**, before this bank finished on **19 November**, as `REFIT_THRESHOLD_PER_REPLICATE = True`, with the reason stated: *"the threshold is a single order statistic of a few dozen positives, so its own variability is the larger term, and ignoring it understated the interval by roughly 5x."* D-070 Sec. 2's pre-registration says the same thing in its own words — *"recomputed inside every bootstrap resample... Not held fixed, that is D73's defect."*

So **the pre-registration contradicts itself**: Sec. 1c's control specifies a test-side-only interval, and Sec. 2 forbids exactly that for every coverage interval in the project. The conflict is resolved in favour of Sec. 2, which is the general rule, carries a documented reason, and was frozen first. The dates are checkable in git.

**With the calibration-draw term restored, every cell contains nominal.** Taking each operator's reported test-only interval, recovering its test-side SD, and adding the calibration-draw term in quadrature (41 calibration subjects as the unit, `sd = sqrt(0.9 x 0.1 / 41) = 0.047`):

| operator | coverage | test-only 95% | contains 0.90 | with calibration draw | contains 0.90 |
|---|---|---|---|---|---|
| sparse-view 180 | 0.804 | [0.707, 0.913] | yes | [0.666, 0.942] | yes |
| sparse-view 120 | 0.795 | [0.670, 0.905] | yes | [0.646, 0.944] | yes |
| sparse-view 90 | 0.809 | [0.710, 0.899] | **no** | [0.677, 0.941] | **yes** |
| sparse-view 60 | 0.833 | [0.752, 0.906] | yes | [0.713, 0.953] | yes |
| sparse-view 45 | 0.837 | [0.778, 0.891] | **no** | [0.729, 0.945] | **yes** |
| sparse-view 30 | 0.847 | [0.802, 0.890] | **no** | [0.745, 0.949] | **yes** |
| limited-angle 150 | 0.850 | [0.781, 0.913] | yes | [0.737, 0.963] | yes |
| limited-angle 120 | 0.856 | [0.827, 0.885] | **no** | [0.760, 0.952] | **yes** |
| limited-angle 90 | 0.847 | [0.820, 0.877] | **no** | [0.751, 0.943] | **yes** |

**4 of 9 becomes 9 of 9.** Two independent checks agree that the bank is fine. The registry split sits **1.29 SD below** the 13-split mean using Track 2's own measured across-split SD of 0.051, which is about the 10th percentile — and the lowest of 13 draws is expected at the 7th. "The registry split is the lowest of 13 at four operators" is what the lowest of 13 looks like, not evidence of a defect. Separately, the measured spread implies **~35 effective units** behind 16,400 calibration regions, which is the subject count: exactly what it should be.

**This ruling is weaker than Track 2's reading 2 in one way and stronger in another.** Weaker: it does not claim the control "passes at 0.897", because that number comes from a form of the control chosen after the first form failed, and the paper would have to say so. Stronger: it does not require it. The registry split stays the reporting split, and no number has to be re-expressed as an across-split quantity.

**Conditions, and one of them can still overturn this.**

1. **Track 2 recomputes the control's interval properly** — cluster bootstrap over subjects on *both* sides, threshold refitted inside every replicate, as `caliber.evaluation.reliability.clustered_coverage` now does. The quadrature table above is a back-of-envelope with `sd_cal` **estimated from the subject count rather than measured**, and it is not the final answer. **If the correct bootstrap still excludes nominal at any operator, this ruling reverts to reading 1 and the bank is not read.**
2. **The registry split's value is reported in the paper**, with its rank among the 13 and the across-split mean beside it. Not replaced by the mean, not omitted.
3. **The paper states that the control's first interval omitted the calibration draw**, and that the rule requiring otherwise was pre-registered on 18 November, before the bank finished. A reader who suspects a post-hoc rescue is entitled to the dates.
4. **The unexplained spread travels as a limitation.** This bank's across-split SD is 0.051 against the face-prior bank's 0.034 on the same partition. Two candidate causes were measured and ruled out. It is T1-D's first row and it is an open question, not a solved one.
5. **A single-split coverage is still not a per-cell "measured validity" for T1-A** — not because the registry split is bad, but because any single-split cell coverage on this bank has an SD of about 0.05, so a hard "valid / invalid" call at nominal is a coin-flip near the boundary. T1-A's "actually unsafe" label must therefore be taken from an **interval** that moves both sides, not a point estimate. This is a change to T1-A's frozen label definition and is recorded as such below.

**D83's own gate had the mirror-image defect, and it is fixed.** `scripts/t1a_safety_rule.py` compared each matched cell against `beta_reference(len(calib))` — the raw **region** count. On this bank that is 16,400 regions, giving a 98% band of `[0.889, 0.911]`, and **it would have failed all nine cells**, for the same reason Track 2's interval failed five: the region is not the independent unit. The control is now a cluster bootstrap over subjects with the threshold refitted per replicate; the Beta band is still printed, labelled as the independence-assumed idealisation, beside a **design effect** so the gap is visible rather than implicit. On T2-A's geometry the design effect is of order 100. A test pins design effect ~1 on independent singletons and >10 on clustered fixtures, and another pins that resampling both sides gives a wider interval than resampling the test side alone — which is this whole decision in one assertion.

Writing `design_effect` also produced a smaller correction worth recording, because it is the same mistake a third time: the first definition divided the observed SD by the **Beta SD alone**, and so reported a design effect of 2.6 on a fixture built with no clustering at all. The missing factor was the test-set term, which contributes an equal share at balanced split sizes. The reference is now `sqrt(Var_Beta(n_cal) + p(1-p)/n_test)`, and independent singletons come back at 1.3.

**Amendment to T1-A's frozen label (condition 5).** `UNSAFE_TOLERANCE = 0.0` applied to a point estimate was frozen on 18 November on the stated grounds that it is the *conservative* orientation. On a bank whose per-cell coverage SD is 0.05 that is no longer true in the way intended — it makes the label a coin flip for any cell within a standard error of nominal, which is most of them. The label becomes: **a cell is actually unsafe iff the upper end of its cluster-bootstrap interval is below nominal.** That is still the conservative direction (it takes more evidence to call a cell unsafe, so it cannot inflate the rule's apparent accuracy by manufacturing easy negatives), it uses the interval the project's own rules already require, and it is fixed before any ordered-pair coverage has been computed. Recorded in `prereg.AMENDMENTS` with this reasoning.

**Status:** 60 Track 1 tests; full suite 655 passed / 0 failed / 5 skipped before this change. Still no result: T1-A's coverage side needs the per-pair intervals, which is the first thing to run now that the bank exists.

---

### D86 - T2-A1 fails D85's reverting clause, and the clause was the mis-specified part [rules P-44 round 2; corrects D85 conditions 1 and 2]

Track 2 implemented D85 condition 1 to the letter: cluster bootstrap over subjects on both
sides, threshold refitted per replicate, 4,000 replicates, percentile interval. **It excludes
nominal at 3 of 9** (limited-angle 120 deg 0.856 [0.805, 0.897]; limited-angle 90 deg 0.847
[0.801, 0.888]; sparse-view 30 0.847 [0.785, 0.899]), confirmed at two further seeds and
10,000 replicates. By D85's own words -- *"if the correct bootstrap still excludes nominal at
any operator, this ruling reverts to reading 1 and the bank is not read"* -- the bank is not
read. Track 2 ran nothing downstream and reported it. That was exactly right.

**The ruling: the bank IS read. D85's condition 1 was the defective part of D85, not T2-A's
bank.** Reversing one's own reverting clause is the shape of a post-hoc rescue, so what
follows is organised around keeping the load off any rule that was not already public.

**What was actually wrong with condition 1.** Two things, both mine.

1. *A pooled SD applied cell by cell.* D85's quadrature used one `sd_cal = sqrt(0.9 x 0.1 / 41)
   = 0.047` for all nine operators -- the fully-clustered bound, with the subject as the unit.
   The real calibration-draw SD is operator-specific. Track 2 measured it: mean 0.051 (my
   0.047 was right on the mean, which is why the table looked convincing) but **0.024 to 0.066
   across operators**, and smallest at exactly the three cells nearest nominal. "4 of 9 becomes
   9 of 9" was an average masquerading as nine cell-level predictions.
2. *A 9-cell pointwise gate.* Condition 1 fails the bank if **any one** of nine intervals
   misses nominal. Under a sound bank that gate fires with probability between 0.025 (cells
   perfectly correlated) and 0.204 (independent); the cells share one calibration draw, so the
   truth is nearer the low end, but it is several times the 2.5% the phrase "the interval
   contains nominal" suggests.

**The gate is anti-correlated with the defect it claims to detect.** On the registry split all
nine cells sit below nominal, by 0.044 to 0.105. The gate fires on the three *closest* to
nominal and passes the five *furthest* from it:

| operator | below nominal | across-draw sd | T2-A1 |
|---|---|---|---|
| sparse-view 120 | 0.105 | 0.066 | pass |
| sparse-view 180 | 0.096 | 0.058 | pass |
| sparse-view 90 | 0.091 | 0.066 | pass |
| sparse-view 60 | 0.067 | 0.062 | pass |
| sparse-view 45 | 0.063 | 0.050 | pass |
| sparse-view 30 | 0.053 | 0.040 | **fail** |
| limited-angle 90 | 0.053 | 0.024 | **fail** |
| limited-angle 150 | 0.050 | 0.061 | pass |
| limited-angle 120 | 0.044 | 0.030 | **fail** |

Each cell's test is individually correct -- a 0.105 shortfall against an SD of 0.066 *is* less
surprising than 0.044 against 0.030. What does not follow is the conclusion. Which cells fire
is set by each operator's score sensitivity, and that is a property of the operator, not
evidence about the bank.

**What the ruling rests on instead: Track 2's own Sec. 1e table, with no rule reinterpreted.**
Per operator, the mean over 13 patient splits is 0.898, 0.901, 0.895, 0.894, 0.892, 0.895,
0.897, 0.899, 0.905 -- nominal at every one of the nine, the widest miss 0.008. A bank with an
operator-specific defect cannot produce that; limited-angle 90 deg, the worst T2-A1 cell, has
the *highest* across-draw mean of the nine. And the registry split is one ordinary low draw:
its shortfall against each operator's own across-draw mean is a near-constant **-1.38 SD**
(spread of the nine standardised positions: 0.48), with `corr(shortfall, across-draw sd) =
+0.61` -- the signature of one draw displacing every cell together, each by its own
sensitivity. **P(the lowest of 13 draws is at least this low) = 0.68.** Finding one split this
low among 13 is the expected outcome, not an anomaly.

**Provenance, stated honestly, including where it is weak.** D85 claimed the refit rule was
frozen on 18 Nov and "checkable in git". It is better than that: `git log -S` puts *"resample
the calibration side too and refit every threshold per replicate"* in `problems_and_decisions.md`
at commit `c06c51f`, **25 October 2026**, three and a half weeks before the bank. D85's
attribution of it to D-070 Sec. 2 was wrong -- that text is D73's -- but the substance is
committed and early.

The multiplicity rule is **not** in that position, and committing it has not fixed that.
`reliability.py` and D83 landed in `1e94d37`, **20 Nov 16:20** -- a day after the bank
finished and hours after T2-A1's result was known. Git now confirms the rule's *content* but
can say nothing about its 18 Nov date, which rests on my word and on the file's own
`FROZEN_ON`. A reader who suspects the correction was reached for once the gate fired cannot
be answered from the history. **This ruling therefore does not rest on it**, and that is why
it is built on Track 2's 13-split table instead, which needs no rule of mine at all. For the record the
correction would not settle the matter anyway: rescaling the three percentile intervals to the
one-sided Bonferroni level over nine cells clears limited-angle 120 deg (0.909) and sparse-view
30 (0.914) but leaves limited-angle 90 deg at **0.9001** -- on the line. Unresolved, not passed.
Limited-angle 90 deg is reported as unresolved on the registry split.

**What this costs Track 1, which is the point of the next two paragraphs.**

**Condition 2 of D85 is reversed for the label.** D85 said "the registry split stays the
reporting split, and no number has to be re-expressed as an across-split quantity". For
descriptive coverage tables that stands. For **T1-A's "actually unsafe" label it is wrong**, and
T2-A1 is what shows why: the draw's displacement is *common to the cells*, so it moves all 72
ordered pairs the same way and cannot average out over them. A label read from one draw is
biased one way by a known amount -- which is not what "conservative" was bought for. The label
is now taken from the across-split mean (`prereg.UNSAFE_LABEL_SOURCE = "across_split_mean"`,
minimum 8 splits). So Track 2 does need the split-averaged tables, for the reason Track 2's
reading 2 did not give.

**And that amendment flatters Track 1's primary endpoint, so it carries a cost.** The registry
split depresses every pair by about 0.07, which manufactures unsafe cells among the green-lit
pairs and *inflates* the false-safe rate; removing it lowers the headline number. A change in
that direction does not get to be silent. `UNSAFE_LABEL_SOURCE_SENSITIVITY` now requires the
single-draw false-safe rate to be reported beside the primary one, every time, so a reader sees
which way the choice cut. Pre-registration fingerprint `2d5f00f0761f` -> `6248be6039f4`.

**What the matched arm does not contaminate.** `scripts/t1a_safety_rule.py` skips `a == b`, so
the diagonal is not among T1-A's 72 ordered pairs and the three failing cells inject no false
safes into the endpoint. The converse also holds and is worth stating in the paper: the matched
diagonal is a validity check, not evidence for the sign rule, because the conformal guarantee
makes it come out right whatever the rule does.

**The spread is no longer unexplained, and D-071 is back as the suspect.** Track 2 reports
`rho(tau, coverage) = -0.93 to -0.99` over the 13 splits and tau varying 3x more than on the
face-prior bank: the spread is the *threshold*, not per-subject coverage. Sec. 1e ruled D-071
out on two measurements -- positive concentration, and per-subject coverage at a **pooled**
threshold -- neither of which can see the tau channel. Per-volume min-max (D-071) makes the
score scale subject-dependent, so the calibration half's composition moves tau directly. The
discriminating variable is **dynamic range, not anatomy**: the mean per-volume HU range of the
calibration half. It also predicts the contrast already observed, since the face-prior bank
maps a fixed HU window (D-013) and so has a subject-independent score scale (sd 0.034 against
0.051). Named as the T2-A2 test; untested.

**Implemented, not just decided.** An amendment nothing imports is prose, so the freeze is
wired: `safety.LABEL_SOURCE_COLUMNS` maps each source to its columns, `safety_rule` reads
`prereg.UNSAFE_LABEL_SOURCE`, and a frame lacking the frozen source's columns **raises rather
than falling back** -- the one outcome that must not happen is a table computed on the
registry split and read as the primary endpoint. `reliability.draw_marginal_coverage` forms
the mean with the overlap taken out (`n_eff = n / (1 + (n-1) rho)`, `rho` the mean pairwise
calibration overlap, which is the correlation and *not* the Jaccard figure the existing
spread check uses); at 13 re-cuts of 100 subjects taking 41 each, `n_eff` is 2.6 and the
honest standard error is **2.2x** the naive one. The quantile is frozen normal, with the
Student inflation (3.3x at 1.6 df) reported beside it: compounding a modelled `rho` with a t
on 1.6 df would shrink the unsafe class until the false-safe rate was near-vacuously zero,
which is the flattering direction again.

The runner makes its **own** re-cuts from the bank's subject ids rather than waiting on
Track 2, so T1-A runs the moment the bank lands; the registry partition is split 0 and the
design subjects are held out of both sides everywhere. One difference from Track 2's
13-split run, in the conservative direction: theirs re-values the label's epsilon per split,
this holds the registry epsilon fixed, so the spread here misses that component, the interval
is narrower, and *more* pairs are called unsafe.

**A defect the end-to-end run caught, which no unit test would have.** The first sensitivity
table scored each arm on its own usable rows. A pair can be uncertifiable on the registry
split and fine averaged over re-cuts, so the arms came out over 1 and 2 pairs and their
false-safe rates sat on **different denominators** -- the difference would have been partly a
change of sample dressed as a change of label. The arms are now restricted to the pairs every
arm can score, with `n_dropped` reporting what that cost.

**Status:** pre-registration amended twice (`2d5f00f0761f` -> `b63c76138528`), 84 Track 1
tests passing, and the runner verified end to end on the MRI fixture bank. T1-A's coverage
side is unblocked and waits only on the bank.

### D87 - The identity violations do concentrate in the cells that failed T2-A1, and the reading still stands [rules T2.11 Sec. 1g; discharges D86's live condition]

D86 named one measurement a live condition that could still overturn it: the per-operator
cross-tab of the 1,744 regions with `|cos theta| > 1`. The words were that concentration in
the three cells that failed T2-A1 "is the pattern that sends the bank back". **They
concentrate.** So this entry owes either that reversal or the reason the condition was the
wrong one. It is the second, and the condition was mine.

**The concentration is real; it should not be softened.** Per operator: la 90 deg 714,
sv 30 329, la 150 deg 230, sv 180 176, la 120 deg 156, sv 90 76, sv 45 63, sv 120 and sv 60
none. The three failing cells average 400 violations against the six passing cells' 91 --
**4.4x**. An exact one-sided rank-sum over the nine cells puts the failing three at W = 8
against an expected 15: **p = 0.048** (4 of the 84 ways to choose 3 of 9 are at least this
extreme). Track 2's "69% of the mass" overstates it slightly -- la 90 deg alone is 41% of all
violations, and setting it aside leaves the failing three holding 485 against the passing
six's 545 -- but the effect survives that, and p = 0.048 is the number to quote.

**The exoneration offered does not do the work claimed, and the reason matters.** Two figures
came with it: only 221 of the 1,744 are positives (0.23% of positives), and dropping every
violating region moves matched coverage by at most 0.0014. The second is close to the first
restated. Those 221 imply about 10,700 positives per operator, so at la 90 deg (about 90
violating positives) the *arithmetic ceiling* on dropping them is +0.0072 upward and -0.0013
downward. The shortfall being explained is 0.053. **The test's entire range is 14% of the
quantity it was pointed at**; it could not have convicted the bank and it cannot acquit it.
It is a bound, not evidence, and belongs in Sec. 1g as one.

There is a further reason not to lean on it. `|cos theta| > 1` is a **threshold on a
continuous numerical error**, so the 1,744 are its visible tail, not the affected population.
Dropping them removes the regions where the inconsistency crossed 1 and keeps every region
where it reached 0.99. A CG solve that converges worse at limited angle converges worse
there everywhere, not only where it broke the identity.

**What settles it: the defect is split-invariant, the failure is split-specific.** The
reconstructions and the scores are computed once. Re-cutting calibration and deployment
re-partitions subject ids; it does not recompute a single reconstruction. So whatever score
corruption the CG tolerance causes is **identical in all 13 splits**. D86's load sits on
Sec. 1e: per-operator 13-split means 0.892 to 0.905, nominal contained at all nine, widest
miss 0.008 -- and la 90 deg, the worst T2-A1 cell and the holder of 41% of the violations,
carries the *highest* across-draw mean of the nine. A defect present in every split cannot
explain a shortfall that appears in one split and vanishes in the mean of thirteen. The
concentration survives as a fact about which operators are numerically hardest. It is not
evidence about the bank's coverage.

**A second check, from Track 2's own transfer table.** sv 120 is the out-of-order operator,
hardest to cover from almost every row. It has **zero** violations, as does sv 60. The most
anomalous cell in the transfer structure is one of the two numerically cleanest, so the
defect does not explain that structure either.

**Where the concentration does bite, and it is not a coverage question.** It is largely a
family effect: la holds 1,100 over three cells, sv 644 over six. Limited angle is more
ill-posed, so CG converges worse -- *and* limited angle has the largest reconstruction error
and the greatest score sensitivity. One cause, both symptoms. That is a confound for **T1-B**,
which predicts the transfer gap from operator distance: if solver quality co-varies with
ill-posedness, part of the distance-gap slope is the solver rather than the geometry. Within
la the violations do not order the failures -- la 150 deg has more violations than la 120 deg
and passed -- which is mild evidence for two separate channels, on n = 3.

**My error in D86, the third one.** "If they concentrate, the bank goes back" named a
correlation without naming the mechanism it would have to act through, and the mechanism is
ruled out by the split invariance above. A condition that cannot distinguish "these operators
are harder" from "these numbers are wrong" is not a test. Conditions in this log should name
the channel, not the pattern.

**Still owed, and what would overturn this.** The **CG residual per operator** -- not the
violation count -- is the measurement that settles whether the limited-angle scores are
numerically sound. The `label_comparison.py` guard stays unchanged until it exists: a
share-based threshold chosen after seeing 0.48% is the same shape as reversing a reverting
clause, and it should be set from the residual, not from the share it has to admit. If the
residual at limited angle moves `err_free` by an appreciable fraction of epsilon, the
*detection* content at those operators is in question -- not the coverage -- and D80 already
refuses that independently.

#### Both of my Sec. 1f predictions failed, and one failure should close its question

- **Fixed epsilon does not shrink the 13-split spread** (0.051 either way). That kills the
  explanation D86 offered for Track 2's bootstrap SEs running 0.73 to 0.92 of the empirical
  across-split sds. The residual gap is unexplained; I should stop attributing it.
- **The calibration half's mean per-volume HU range does not order tau** (rho -0.05 to +0.36,
  median +0.08). Within a subject a wider range does go with lower positive scores (-0.19 to
  -0.44), so D-071's normalisation does reach the scores -- the half's *mean* is simply the
  wrong aggregate for a *quantile*.

The next step is **not** a third composition variable. `tau` is the k-th smallest positive
score over about 41 subjects' worth of regions, and an order statistic of a subsample varies
whether or not any composition variable drives it. **Test that null before fishing further:**
draw 13 subject-level subsamples of the observed size from the pooled positives, recompute
`tau` in each, and compare the simulated sd of `tau` with the observed one. If the null
reproduces it there is nothing to explain, D-071 is exonerated, and the face/CT contrast
(sd 0.034 vs 0.051) becomes tail shape -- which the same simulation checks on both banks. If
the null undershoots, a composition channel exists and the predictor to try is the **lower
tail** of the per-volume range distribution, not its mean, because a low quantile is set by
the widest-range subjects. That direction is named here, before the run, so that if it works
it is a prediction rather than a shape found afterwards.

#### The artefact contract, which was the other thing asked for

T1-A could not read Sec. 1g's table. Now it can, and the plumbing carries three refusals
rather than conveniences:

- `caliber.data.splits.PROBE_SALTS` and `calib_sets_for_salts` -- the 13-salt re-cut design in
  one place instead of three literal copies. Membership is **reconstructed** from the salt, so
  a per-split table need not carry it and cannot disagree with it. A salt the protocol does
  not name is refused, because an unplaceable draw's share of the overlap cannot be measured.
- `reliability.pair_coverage_from_splits` -- their two CSVs to the one ordered-pair frame the
  label reads, with both label sources on it. Measured overlap **0.413**, so n_eff = 2.2 and
  the standard error is **2.44x the naive one**. Their `split_sd / sqrt(13)` is the naive one,
  so the interval is rebuilt here and never read: accepting a stored one would narrow it, and
  a narrower interval calls more pairs unsafe. The merge is outer, so a pair in one table only
  keeps its row instead of shrinking the false-safe denominator.
- `t1a_safety_rule.py --pair-coverage-dir` -- both arms come from that one artefact rather
  than mixing its across-split arm with this script's own bootstrap for the registry arm.
  Otherwise the sensitivity table would compare pipelines as well as label sources, which is
  D86's different-denominators defect in a subtler dress. The split fractions are checked
  against the bank's own `split` column, since the overlap is reconstructed from them and the
  wrong ones would silently report another design's overlap. This script's bootstrap is kept
  beside theirs under `_local` names, as a free cross-implementation check.

**Status:** 699 tests passing (was 683). Verified end to end by running Track 2's own
`pair_coverage_by_split.py` on a synthetic 120-subject CT-shaped bank and reading its output
with `t1a_safety_rule.py`; the two implementations' registry coverage agreed to 0.0000 on
every pair. Apache-2.0 for the repository is agreed; NOTICE must carry the TotalSegmentator
(CC BY 4.0) and LIDC-IDRI (CC BY 3.0) attributions, since both licences require them.

### D88 - T1-B's response variable was never frozen; it is frozen now, with its contamination stated [T1-B; completes D83 and D84's freeze]

D83 froze the distance. D84 froze the baselines and their scope. `GAP_PREDICTION_FORM`
froze the functional form. **Nobody froze what the gap is.** The one quantity T1-B is
*about* -- the response -- was the one left selectable, and T2-A3's cell table now exists,
so it had to be closed before the first fit rather than after.

The hole is not academic. Both of these are defensible readings of "the gap":

- `NOMINAL - coverage(a -> b)` -- the shortfall against the promise;
- `coverage(b -> b) - coverage(a -> b)` -- the shortfall against what that deployment
  operator reaches when it calibrates on itself.

They give different fits, and picking the one the distance predicts better after seeing
both is exactly the forking path `prereg.py` exists to close.

**The ruling: `GAP_DEFINITION = "matched_minus_transfer"`.** T1-B's claim is that distance
predicts *how much coverage you lose by having calibrated somewhere else*, and what you
would otherwise have had is the matched cell at the deployment operator. Subtracting it
removes the bank's own per-operator level -- a property of the bank, not of the shift --
from the response instead of leaving it in. It also vanishes identically at zero distance,
which anchors a monotone fit at the origin. `nominal_minus_transfer` is reported beside it
every time as the operational reading, since that is what a deployment certificate cares
about.

**A property of the difference worth stating, because it is a check on D86.** D86 moved
T1-A's label to the across-split mean because one calibration draw displaces every cell
together -- a bias common to all 72 pairs, which therefore does not average out across
them. A *gap* is a difference of two cells, so a displacement common to both cancels to
first order. Track 2's own numbers put the registry displacement at a median -0.061 on
transfer cells and a comparable amount on matched ones. **Prediction, recorded before the
fit: the registry arm and the across-split arm will agree far more closely for the gap than
they do for the level.** If they do not, the displacement is not common-mode, and D86's
central argument -- not merely D88's choice -- needs revisiting. The sensitivity table
therefore tests D86 as well as D88, which is why it runs on both arms crossed with both
definitions, four combinations, every time.

**`GAP_ERROR_METRIC = "mae"`, chosen so it is *not* the method's own objective.** PAVA
minimises squared error. Grading the fit by RMSE would score the method on the criterion it
optimises while scoring the baselines on someone else's. MAE also stops the single extreme
pair -- the worst gap is near 0.5 -- from setting the comparison by itself. RMSE is reported
alongside.

**`GAP_CLUSTER_UNIT = "operator"`.** There are 72 ordered pairs and nine operators, each
appearing in 16 pairs. Every interval resamples operators and weights a pair by the
*product* of its two endpoints' draw counts, which is the dyadic form: a pair is an
observation about two operators, so a replicate drawing one of them twice should count it
twice. Resampling pairs instead would report an interval for a sample size the design does
not have, which is D86's error in a new costume.

**Both protocols, and both *directions* of the cross-family fit, reported separately.** The
work plan asks to "fit on sparse-view only, predict limited-angle, **then reverse**". The
first implementation here aggregated over folds and so averaged the two directions. That
was wrong, and the correction is the most informative thing in this entry -- see below.

#### What this declaration is worth, stated honestly

`GAP_DECLARATION_KNEW` records the six summary figures from T2.11 Sec. 1g that had already
been reported to Track 1 when the above was named. So this is a freeze **before the
per-pair gaps**, not before any knowledge of them, and the difference matters. The last
entry is the costly one: *sv 120 is out of order, hardest to cover from almost every row*
-- a monotonicity violation, already known. Two consequences bind:

1. The sv 120 anomaly **cannot be reported as something the fit discovered**. It was known
   first, and it is in the pre-registration as known.
2. **No sv-120 term may be added** to the model, under any name.

A freeze that does not state its own contamination is a claim of a freeze, not one.

#### Three findings from the implementation, all before a real gap existed

Verified on the real nine-operator grid (`runs/t1a_ct_distances/`) with synthetic responses,
so the grid is real and only the gaps are invented.

1. **The cross-family protocol is radically asymmetric, and averaging the directions hides
   it.** Fitting on the 30 within-sparse-view pairs and predicting limited-angle and the
   span pairs gives MAE about 0.005 to 0.010 with **0 per cent extrapolation**. Fitting on
   the 6 within-limited-angle pairs and predicting sparse-view gives MAE about 0.05 with
   **78 to 93 per cent of the held-out points outside the training range**. Outside that
   range a monotone fit is clamped, so in that direction it *is* a constant predictor, and
   any apparent skill is the clamp. `extrapolated` is therefore reported on every row and a
   warning fires above one half. Six within-family pairs cannot support a curve, and the
   honest form of the limited-angle direction may be "not estimable" rather than a number.
2. **`null_space_dimension` collapses to exactly the constant predictor** on any fold
   trained inside limited angle, because all three limited-angle operators share a
   null-space dimension of 64000. Beating it there is beating a constant twice, not beating
   two baselines, and the verdict table must not be read as though it were. D84 predicted
   the degeneracy; it is now visible in the output rather than in a footnote.
3. **The machinery can say no.** Given gaps generated from the view-count knob alone, the
   distance is reported as predictive on **none** of the four protocol x scope cells, and
   the acquisition parameter is correctly left unbeaten within family. A comparison that
   cannot decline is not evidence, so this is pinned as a test rather than checked once.

One smaller thing, recorded because it looked like a bug and was not: on one slice the
null-space baseline's MAE equalled the constant's to ten decimal places. Six observations
differed in prediction by a constant offset, three with positive residuals and three with
negative, so the absolute-error changes cancelled exactly. Coincidence, confirmed by
computation rather than assumed.

**Status:** 737 tests passing, 31 of them T1-B's; `src/caliber/evaluation/gap.py` and
`scripts/t1b_gap_prediction.py`. Fingerprint `b63c76138528` -> `fe192e610104`, amendments
now five. On branch `T1-B`. **No result**: the response needs T2-A3's cell table, which is
on Track 2's disk. The runner reads that artefact directly, so T1-B is one command away
from a number the moment it arrives.

### D89 - T1-A's rule is refuted on the CT bank, and the refutation is an identity, not a surprise [rules T1-A; refutes D83's direction; corrects the gate]

**The result, first.** Track 2 ran `t1a_safety_rule.py --pair-coverage-dir` unmodified on
their tables. Pooled false-safe rate 0.361 [0.09, 0.67] against a trivial rate of 0.319:
`no_better_than_trivial`, Youden's J = -0.10, permutation p = 0.81. No tolerance between 0
and 0.05 beats trivial. By family cell the rule is worse than trivial everywhere it is
defined: sv->sv 0.333 vs 0.167, la->sv 1.000 vs 0.889, la->la 0.667 vs 0.333. The
pre-registered rule is reported as failed.

**It did not fail by being uninformative. It failed by being backwards.** Within every
family cell the gap *falls* with the signed distance: rho -0.72, -0.70, -0.70 and -1.00.
Pooled it is -0.09. On the face-prior bank the same measure gave +0.976. A rule with the
sign inverted performs at or just below chance, which is exactly what J = -0.10 is, so the
two findings are one finding.

**Why, derived rather than guessed.** Coverage of the positives at `b` is the fraction
whose score clears the threshold they were given, so with `F_b` the CDF of those scores:

    coverage(a->b) = 1 - F_b(tau_a)        coverage(b->b) = 1 - F_b(tau_b)
    gap            = F_b(tau_a) - F_b(tau_b)

`F_b` is non-decreasing, so `sign(gap) == sign(tau_a - tau_b)` **exactly, at every ordered
pair**. Verified at 72 of 72 on a synthetic grid and pinned as a test. The threshold rises
with ill-posedness, because the score tracks reconstruction error and an ill-posed operator
has more of it; and the signed distance is positive precisely when `b` is the more
ill-posed operator, i.e. when `tau_b > tau_a`. So a positive distance *forces* a negative
gap. The frozen direction is backwards by arithmetic, not by bad luck, and no bank would
have shown otherwise.

**The conceptual error in D83, in one sentence.** It conflated where hallucinations are
*likely* with where they are *hard to catch*. Deploying on a more ill-posed operator does
give more hallucination -- and also larger scores, which clear an inherited threshold more
easily. The two effects run in opposite directions and the rule bet on the weaker one.

**The obvious fix is barred, and barring it was the point.** Having derived the identity,
the temptation is to enter `tau_a - tau_b` as a predictor. It would beat the distance and
every baseline, and the win would be arithmetic: its sign agreement with the gap is 100% by
theorem, not by measurement. It is the endpoint in other units. `prereg.GAP_BARRED_PREDICTORS`
now names it and two aliases, and the bar was written before anyone proposed them, which is
the only time such a bar is worth anything. This would have been the eighth instance of the
failure mode this project keeps hitting: a statistic that returns the hoped-for number.
(Corrected in D91: the entry above counted seven instances, and the tally had already
reached seven at the bootstrap-interval defect. The miscount is left visible rather than
edited away, because the tally is the only thing that makes the pattern legible.)

**What the identity is good for instead: a free check on the bank.** Agreement below 100%
cannot happen if the recorded thresholds and coverages came from the same scores, so a
disagreeing pair means the bank contradicts the definition of coverage. That is a defect no
coverage result survives, and it costs one groupby --
`gap.gap_sign_from_thresholds(ordered_pairs)`. It is strictly stronger than the matched
control, which only asks whether the diagonal sits at nominal. A check that can only find
faults cannot flatter the hypothesis, which is why this one is allowed and the predictor is
not. **Track 2 should run it on their `ordered_pairs.csv` and report the agreement rate.**

**What the reversal does not license.** Flipping the frozen direction and reporting T1-B
confirmed. That is one free bit fitted to the data that refuted the hypothesis, on the same
nine operators, and one free bit on nine units is not a result. The implementation now fits
both directions because the frozen one, applied to a reversed truth, returns the training
mean -- so before this change a perfect reversal and pure noise produced identical output,
which is the wrong answer for the wrong reason. The decreasing fit is a diagnostic in the
`reversal` frame, flagged `pre_registered = False`, and is barred from `verdict`. Pinned by
a test that the opposite-direction fit cannot turn a withheld claim into a reported one.

**The out-of-sample test, stated before the number exists.** If this account is right, the
sign of the measured relation is set by how `tau` orders the operators and nothing else --
the identity fixes the gap's relation to `tau`, and fixes nothing about its relation to the
distance. So the face-prior bank's +0.976 requires the *opposite* threshold ordering: `tau`
must *fall* as the face operators hide more of the image. That bank's own
`ordered_pairs.csv` already carries the thresholds, so the check is one groupby and no new
bank pass. Recorded as `prereg.GAP_FACE_BANK_PREDICTION`. **If `tau` rises there too, this
account is wrong and the difference between the two banks is unexplained -- and must then be
reported as unexplained, not patched.**

**A fourth thing, which is about the distance measure rather than the rule.** Every family
cell is between -0.70 and -1.00 while the pooled correlation is -0.09. For the pool to be
near zero the between-family component must be strongly positive, cancelling the within-cell
relation: a Simpson reversal, and a sign that `null_space_growth` does not order the
operators the way `tau` does across families. The mechanism is visible in the grid -- all
three limited-angle operators share `null_space_dim = 64000`, yet sparse-view 120 is the
operator that is hardest to cover from almost every row. The measure counts *how much* of the
image is unseen and not *how badly conditioned* what is seen is, and those come apart between
the families. This is evidence on Q1's fifth candidate (spectral divergence in singular-value
decay or conditioning) and it is the first evidence from a bank rather than from operators
alone. It is **not** licence to swap the frozen measure for T1-B; it is a reason to compute
the conditioning-based measure and pre-register it for the next bank.

**The gate, corrected, with the audit it needs.** Track 2's run failed the matched control
at limited-angle 90 deg -- 98% interval [0.795, 0.896] against nominal 0.90, short by 0.004
at the upper end -- and they proceeded with `--ignore-control`, which labels the output
diagnostic. The gate was wrong, for a reason provable from its own docstring: it applied
`CONTROL_LEVEL = 0.98` *per cell* to nine cells, which leaves a family-wise spurious-failure
rate of `1 - 0.98**9 = 16.6%`, not 2%. The docstring had already identified multiplicity as
the thing to correct for ("a 95% band would fail one cell in twenty") and then under-solved
it by a factor of nine. The per-cell level is now Bonferroni-derived from the number of
checkable cells (99.778% at nine), so the stated level means what it says.

This is the fourth time in this sequence that a stopping condition of mine has moved after
it fired, and that pattern is the shape of a post-hoc rescue whatever the reasoning behind
any single instance. So, explicitly:

- The correction makes the gate **harder to fail**, which is the self-interested direction.
- The uncorrected band and its own pass/fail stay in the output on every row
  (`coverage_lo_uncorrected`, `coverage_hi_uncorrected`, `status_uncorrected`), and the
  runner prints which cells the two levels disagree about.
- **The gate cannot reach any number the rule reports.** The pair table is a function of the
  bank and the distances; the control is a print-and-stop. Pinned by a test that builds the
  pair table before and after running the control and asserts the frames are identical.
- **Nothing was let through.** The result behind the gate is a refutation. A gate relaxation
  that admits a negative result is not a rescue of anything.

The cost is real and is why both bands are printed: a gate at 99.778% per cell has less
power against a single genuinely broken operator than one at 98%.

**The cos theta guard: I asked the wrong question, and T2-A5's table cannot answer the one I
asked.** I said the guard's tolerance should be set from the solver's convergence. It cannot
be. `|cos theta| <= 1` is Cauchy-Schwarz, which holds exactly for *any* two vectors whether
or not `P_N` is a true projector -- so if numerator and denominator are formed from the same
vectors in the same solve, no residual and no idempotency defect can produce a violation,
and 1,744 of them above 1e-3 are impossible. Loosening the tolerance to 12% to match the
additivity defect would legitimise the violations by adopting a threshold that makes them
vanish, when the cause may be one line. Two cases, and they need different answers:

1. The ratio is formed consistently -- then the violations are a formula or caching defect
   (an analytic or cached norm in the denominator, or a different iteration count between
   numerator and denominator), and the tolerance is not the problem.
2. `cos theta` is computed through a polarization-type identity from norms of `P_N(a +/- b)`
   -- then the additivity defect (2-12%) enters directly, violations of that order are
   expected, and the tolerance should be set from the defect.

**The diagnostic that separates them is the magnitude distribution of `|cos theta| - 1` per
operator, regressed on that operator's additivity defect -- not the count above a fixed
threshold.** Track 2's observation that the count does not track the defect (sv 30 has the
smallest defect and the second most violations) does not settle it: a count above a fixed
cut is a visible tail, not a population, which is D87's error in a new place. Until this is
settled, `cos theta = +0.882` should not be described as good to a few percent; under case 2
its band is roughly 0.78 to 0.99, and whether that matters depends on what it is being
compared against. **Guard unchanged at 1e-3, as their default proposed, but for a different
reason: not pending a tolerance, pending the formula.**

**Status:** 758 tests passing, 45 of them T1-B's and 28 the runner's; `gap.py` gains
`rank_correlation` and `gap_sign_from_thresholds`, `prereg.py` gains
`GAP_PREDICTION_DIRECTION`, `GAP_BARRED_PREDICTORS` and `GAP_FACE_BANK_PREDICTION`.
Fingerprint `fe192e610104` -> `ce668b3e9b98`, amendments now six. On branch `T1-B`. T1-B
still has no fitted result: it needs the per-pair gap table. But its headline no longer
depends on one, because the direction is settled by the identity.

### D90 - T1-A's failure reconciled pair by pair, and "sv 120 is out of order" localised to one transposition [T1-A; builds on D89; `scripts/t1a_rule_anatomy.py`]

Track 2 holds the CT bank and the per-pair coverage table; Track 1 holds the operator grid.
The rule's predictions depend on the grid alone, so the whole of T1-A's confusion matrix can
be reconciled here without waiting for their table. Run: `scripts/t1a_rule_anatomy.py`.

**Their numbers are sound, checked independently.** The per-cell rates have to resolve to
whole pair counts against predicted-safe denominators computed from the local distance
matrix, and all of them do: sv->sv 0.167 x 30 = 5 unsafe and 0.333 x 15 = 5 false-safe,
la->sv 0.889 x 18 = 16 and 1.000 x 6 = 6, la->la 0.333 x 6 = 2 and 0.667 x 3 = 2, sv->la 0
of 18. Pooled, that is **23/72 = 0.3194** trivial against their reported 0.319 and
**13/36 = 0.3611** false-safe against their reported 0.361. Two pooled figures and six
per-cell rates reconstructed to the digit from operator data plus their rates. The table is
arithmetically consistent and the denominators are what they should be.

**The rule is not uninformative. Inside sparse view it is exactly inverted.** The rule calls
36 of 72 pairs safe; D89's identity guarantees 36 cannot be unsafe; the two sets **overlap
on 12**. Chance would give 18 (hypergeometric sd 2.14), so the rule's safe set is
anti-correlated with the truth at z = -2.81 -- which is what a Youden J of -0.10 is made of.
On the 30 sv->sv pairs the disagreement is **total**: every pair the rule calls safe is a
pair the identity says may be unsafe, and vice versa, because within a family the distance
and the threshold both rise with ill-posedness, so `distance <= 0` is precisely `tau_b <=
tau_a` is precisely `gap >= 0`. The inversion is a property of the operator grid and needs no
coverage number to state.

**The six false-safe la->sv pairs, by name.** The rule calls exactly six la->sv pairs safe,
and Track 2 report false-safe 1.000 there, so all six are wrong: la 90, la 120 and la 150
each into **sv 120 and sv 180** -- the only two sparse-view operators whose null-space
dimension is at or below the limited-angle value of 64000, which is what puts them on the
negative side of the signed distance. The worst cell in the entire table, la 90 -> sv 120 at
0.377, is one of these six. The rule's single worst behaviour is concentrated in a set the
grid identifies in advance.

**"sv 120 is out of order" is one adjacent transposition, and it is testable from one
column.** (Corrected in D93: the prediction below holds -- `tau(sv 120) < tau(sv 180)`, 0.893
against 0.972, 13 of 13 splits -- but the localisation in this section does not. Track 2's
full ordering puts sv 90 below sv 180 as well, so it is not one transposition, and the
parsimonious reading is that **sv 180** sits two positions too high rather than that sv 120
is anomalous. `rho(views, tau) = -0.83` within sparse view, not -1.00. Left in place rather
than edited away.) The threshold order was declared from the acquisition knob before any coverage
number was consulted: fewer views and narrower arcs mean larger reconstruction error, larger
score, higher threshold. Against the five coverage values Track 2 quoted, that order
generates seven constraints and **six hold**. The one violation is:

    from sv 30:  sv 120 = 0.676  vs  sv 180 = 0.731   implies  tau(sv 120) < tau(sv 180)

and the knob order says the reverse, because sv 120 has fewer views than sv 180. This is the
**stronger** of the two kinds of constraint the script forms: both cells share the
calibration operator sv 30, so `tau_a` cancels exactly and no matched value is needed --
verified by a test that widens the matched range to [0, 1] and shows the within-row verdicts
do not move while every against-matched verdict goes undecided. It is also exactly what
"hardest to cover from almost every row" requires, since a deployment operator is hard to
cover from most rows precisely when its own threshold is low.

So the anomaly is **one operator's score scale**, not a defect in the coverage machinery and
not a problem with 72 cells: sv 120's positive scores sit lower than sv 180's despite sv 120
having 120 views against 180. **Checkable from one column of Track 2's existing
`ordered_pairs.csv`: is `tau(sv 120) < tau(sv 180)`?** A suspicious coincidence sits beside
it -- sv 120's `null_space_dim` is 64000, exactly the value shared by all three limited-angle
operators, despite sv 120 being a full-arc acquisition. Worth checking how sv 120's operator
was built before concluding anything about its physics.

**What is fitted here and what is derived.** One bit: the two families' knobs do not compare,
so the relative placement of the limited-angle and sparse-view blocks cannot come from the
knob. It is taken from the 18 sv->la pairs, the cell where every pair covers, and it says the
limited-angle block sits above the whole sparse-view block. Everything stated about the other
54 pairs is then a consequence. One bit fitted on 18 pairs, reconciling 54, is a real
compression -- but it is one bit fitted, and the script says so in its own output rather than
in a footnote.

**The prediction to check when the per-pair table arrives.** The identity fixes the gap's
sign and says nothing about magnitude, so it makes no claim about how far any cell falls. It
does make one claim with no fitted parameter: **a pair with `tau_a < tau_b` has `gap <= 0`,
covers at least as well as the deployment operator's own matched cell, and therefore cannot
be unsafe.** There are 36 such pairs. Eighteen of them are the sv->la cell, where Track 2
already report 0 unsafe, so that part is confirmed; the other 18 are per-pair checks. A
single unsafe pair among the 36 falsifies D89's account, and that is the outcome to look for
rather than a confirmation to collect.

**Status:** 13 new tests, all passing. (The "774" first written here was a subset run,
not the suite; the suite is 1,078. Corrected in D91.) `scripts/t1a_rule_anatomy.py` plus
`runs/t1a_anatomy/{t1a_pair_anatomy,t1a_reconciliation,t1a_order_constraints}.csv`. No new
frozen choice, so the fingerprint is unchanged at `ce668b3e9b98`. On branch `T1-B`.

---

### D91 - T1-B's direction was the wrong *kind* of frozen choice; and the 72-of-72 that supports D86 supports nothing [T1-B; rules on Track 2's T2.11 Sec. 1i, four items]

Track 2 ran `t1b_gap_prediction.py` unmodified against fingerprint `fe192e610104` and
reported T1-B **reportable on 0 of 4 protocol x scope cells, the same verdict in all four
arm x definition combinations.** That stands. Nothing below changes it, and it goes in the
paper as a negative result on this bank.

Four items came with it, each with a default. All four defaults are taken. Two of them are
taken for different reasons than the ones offered, and one of the numbers offered in
support has to come back out.

#### 1. Direction: their default stands, and the frozen choice was wrong in a way a flip would not fix

The question asked was "do you want a mirrored (non-increasing) fit added as a new
pre-registered analysis?", default no. **Default taken for this bank.** Fitting
`-distance` over all 72 pairs, looked at after the result and in-sample, is a description;
adopting it would be choosing a sign after seeing the sign it has to match, which is free
whatever the justification. Track 2 flagged this themselves, correctly, before I did.

But the diagnosis "a non-decreasing fit cannot follow a falling gap" stops one step short.
By D89 `sign(gap) == sign(tau_a - tau_b)` exactly, so **the direction of the gap-distance
relation is a property of how the calibration threshold orders the operators** - not of the
mechanism, and not of the hypothesis. The two banks already disagree about it: rho -0.70 to
-1.00 within the CT family cells against +0.976 on the face-prior bank. So no single frozen
constant could have been right for a project that uses two banks. The defect is not that
`increasing` was the wrong value. It is that a constant was the wrong *kind* of choice.

**The ruling: `GAP_DIRECTION_SOURCE = "threshold_ordering"`**, amendment seven. The
direction is computed by `gap.direction_from_thresholds`, which takes the recorded
thresholds and the signed distance and nothing else - no coverage, no gap, no fitted value.
`prereg.GAP_DIRECTION_RULE` writes it out so it can be run by hand off one CSV.

**This is the first of the seven amendments that *widens* what may be claimed, and it is
fenced accordingly.** Four fences, all in the module:

1. `GAP_DIRECTION_BANKS_BARRED = ("lidc_ct",)`. On any bank whose per-pair gaps have
   already been seen, the rule may be *shown* and may not be *reported*.
   `DirectionCall.reportable` is False there regardless of every other condition. The CT
   bank is named, so T1-B on LIDC stays 0 of 4 and cannot be revived by this amendment.
2. The rule costs **one bit**, the same bit for all 72 pairs of a bank, where
   `GAP_BARRED_PREDICTORS` forbids the 72 per-pair values of `tau_a - tau_b`. That bit is
   still a bit taken from a barred quantity, which is exactly why fence 1 exists. Stating
   the size of the concession is the point: one global sign, checkable as one number,
   against 72 pairwise values that would determine the answer outright.
3. `GAP_DIRECTION_DETERMINACY_MIN_RATIO = 3.0`. Near-flat tau leaves the ordering
   undetermined and the correlation still returns a sign, from noise. The across-operator
   tau range must exceed the median per-operator tau sampling spread - T2-A4's measured
   quantity - by this factor, or the direction is recorded undetermined and nothing is
   reported under this source. `tau_noise` has **no default**: `determinate` is `None`
   until a measured value is supplied, and `reportable` is False while it is `None`. The
   alternative to measuring it is assuming the tau spread is real, and T2-A4 exists
   because a conformal threshold is a single order statistic whose spread may be entirely
   sampling noise.
4. `GAP_FACE_BANK_PREDICTION`, frozen at `ce668b3e9b98` *before* this amendment, must pass
   first. It says tau must *fall* with the distance on the face bank. If tau rises there
   too, D89's account is wrong, this entire path closes, and the two banks' sign difference
   is reported as unexplained.

So the only new run this authorises is on the face bank, after the tau check, and it is a
genuine out-of-sample test of a magnitude claim rather than a sign claim. If someone later
wants to argue the fence away, fence 1 names a bank rather than stating a principle, which
is the only kind of fence that cannot be reinterpreted.

**Why the LOO numbers are the strongest evidence against the frozen direction, stronger
than any rho.** Fit-one-family-predict-the-other returned `distance = constant = views/arc
= null-dim`, MAE 0.0707, **delta exactly 0** - four different predictors agreeing to the
last digit is not four equally good predictors, it is PAVA collapsing every one of them to
the training mean. The right report is "the estimator was inert", not "the distance is
indistinguishable from the baselines"; a reader given the second sentence concludes
something about the distance. Leave-one-operator-out within family then gives distance
0.0884 against constant 0.0792, **delta -0.018 [-0.026, -0.006]** - out of sample,
operator-clustered, and excluding zero. Under partial collapse a surviving step points the
wrong way and actively costs accuracy. That interval is the result.

**The prereg earned its keep on the other baseline.** Distance beat null-space dimension by
+0.005 **[+0.0001, +0.009]** - a lower bound four decimal places from zero, on one of two
baselines. `GAP_PREDICTION_BASELINES` requires beating **both**, constant included, so the
verdict is 0 of 4 and that +0.0001 never reaches a sentence. A conjunction requirement
written in September caught a number that would have been reported in November.

#### 2. The 72-of-72 gap-sign agreement is not evidence for D86, and comes out of its support

D88 predicted, before the fit, that the registry and across-split arms would agree more
closely on the gap than on the level. Track 2 reports median -0.061 -> -0.001 and mean
absolute 0.066 -> 0.024. **The prediction held and the magnitude side is stronger than
claimed.** Then: "Gap sign agrees at 72 of 72 pairs. So the common displacement cancels to
first order." That inference is invalid.

By D89 the gap's sign is `sign(tau_a - tau_b)` under *either* arm. So 72-of-72 says exactly
one thing: **the two arms rank the nine thresholds the same way.** Demonstrated both
directions, in tests:

- Two arms sharing a threshold vector exactly and differing only in score scale: sign
  agreement **100%** with mean `|gap difference|` of 0.09 on the fixture, and 0.22 in the
  nine-operator simulation. Arbitrarily large non-cancellation, perfect agreement.
- Two arms with *identical* score distributions - perfect cancellation, every gap
  difference zero to machine precision - and one adjacent threshold pair transposed:
  agreement falls below 100%.

The statistic is high when the claim is false and low when the claim is true. It is the
**eighth** instance of this project's recurring failure mode, and it is partly mine: D88
framed the prediction on magnitudes, correctly, but said "agree far more closely" without
naming the statistic, and an unnamed statistic gets chosen after the fact.
`prereg.GAP_ARM_SIGN_AGREEMENT_BARRED` now bars it and names its replacement.

**The two difference tables are not independent, and that is what makes the magnitude test
sharp.** With `gap(a,b) = level(b,b) - level(a,b)`, subtracting the arms gives, as algebra:

    gap_diff(a,b) == level_diff(b,b) - level_diff(a,b)

exactly - residual 0.0, not 1e-12, in `gap.arm_difference_identity`. The gap-difference
table *is* the level-difference table minus its own diagonal. So "the gap agrees better
than the level" means precisely: **the displacement at the matched cell is strongly
correlated with the displacement at the transfer cell**, which is D86's claim stated in a
measurable form. At zero correlation the gap difference is no smaller than the level
difference, so the statistic can fail.

**How much correlation 0.024/0.066 forces, without an equal-variance assumption.** Writing
`s` for the ratio of the two displacements' spreads and `r` for their correlation,
`sd(gap)/sd(level) = sqrt(s*s - 2*r*s + 1)`, which over all `s` bottoms out at
`sqrt(1 - r*r)` at `s = r`. A ratio of 0.364 therefore forces **`r >= 0.93` whatever the
spreads turn out to be** - verified by brute force over `s` and by 400,000 random
`(s, r)` draws below the bound, zero violations.

I first obtained 0.934 by inverting the ratio under equal variance and was about to report
it as an estimate. On this module's own fixture that shortcut overstates `r` by a factor of
1.3, because the matched displacement carries **6%** of the transfer displacement's spread
- unequal by a factor of 17. The bound is assumption-light where the inversion is not; both
are in the code, and the test
`test_inverting_the_ratio_directly_would_overstate_the_correlation` pins the gap between
them so the shortcut cannot quietly come back. One assumption does remain: the bound is in
standard deviations and Track 2 reported mean absolute deviations, which agree only for
similar distributional shape. **So the number to report is not the bound. It is
`corr(level_diff(a,b), level_diff(b,b))` measured directly** - one line on tables that
already exist, and the quantity D86 actually rests on.

#### 3. The gate, the cos-theta guard, and T2-E

**Gate (their item 2): already done, in D89, before this message arrived.** Their default
was "you update the gate", and the correction is a family-wise one -
`CONTROL_FAMILY_LEVEL` with a per-cell Bonferroni level, because 0.98 applied to nine cells
leaves a 16.6% family-wise spurious-failure rate, not 2%. `--ignore-control` should not be
needed on a re-run. **Predicted, so it can be wrong:** la 90 degrees at 98%
[0.795, 0.896] becomes roughly [0.779, 0.912] at the corrected 99.778% per-cell level and
should pass - a normal-approximation rescaling of their reported band, to be replaced by
their bootstrap.

**And the gate passing must not be read as la 90 degrees being calibrated.** Its point
estimate sits 5.4 points below nominal 0.90. Nine cells at 99.778% cannot resolve a
5-point shortfall, so the corrected gate is *weak* by construction - which is why the
runner prints `status_uncorrected` and the uncorrected band beside it every time. I
corrected a stopping condition in the direction that makes it easier to pass, after it
fired, for the fourth time in this project; the fence is that the gate cannot reach any
number the rule reports, pinned by `test_the_gate_does_not_enter_any_number_the_rule_reports`.

**cos-theta guard (their item 3): default taken, reason replaced.** Guard unchanged at
1e-3, but not "until you set the tolerance from the T2-A5 table" - T2-A5 *cannot* set it.
`|cos theta| <= 1` is Cauchy-Schwarz, which holds for any two vectors whether or not `P_N`
projects, so no CG residual and no idempotency or additivity defect can breach it when
numerator and denominator come from the same solve. The 2-12% additivity defect is real and
irrelevant to this guard. T2-A6 is the diagnostic that can speak to it: the *magnitude*
distribution of `|cos theta| - 1` per operator against that operator's additivity defect.
A count above a fixed threshold is a visible tail, not a population - which is also why
"sv 30 has the smallest additivity defect and the second most violations" is not the
paradox it looks like.

**T2-E (their item 4): default taken, and fig 9 should not be improved.** A flat fitted
line laid over a visibly falling cloud of 72 gaps is the single clearest statement of the
refutation in the paper, and the instinct to redraw it with a fit that follows the cloud is
the thing to resist. Keep it as drawn. A reversed panel may be added beside it only if the
face-bank tau check passes, and must be labelled a diagnostic.

#### One discrepancy to resolve, and it is small

The same four family-cell correlations came back as -0.72, -0.70, -0.70, -1.00 with pooled
-0.09 in the T1-A run and -0.72, **-0.75**, -0.70, -1.00 with pooled **-0.11** here. One
cell and the pooled figure moved. Most likely the two runs used different arms or
definitions, in which case the pair is a sensitivity and should be reported as one. If
both came from the same arm and definition on the same bank, one of them is wrong. Worth
one line to say which, given how often this project's numbers have moved for reasons that
turned out to matter.

**Status:** 1,071 tests passing, 7 skipped, 18 of them new here. `prereg.py` amendments
seven and eight; `gap.direction_from_thresholds`, `gap.DirectionCall` and
`gap.arm_difference_identity`. Fingerprint `ce668b3e9b98` -> **`4a7426a10a50`**, so every
table produced under the old one is visibly from an earlier freeze - including Track 2's
T1-B run at `fe192e610104`, which is three amendments and three fingerprints further back and whose 0-of-4 verdict
is unaffected. On branch `T1-B`, uncommitted: Track 2 cannot see D89, D90 or this entry
until it is pushed.

---

### D92 - The Beta law transfers; the matched band does not. T1-D gets a positive result that needs no bank [T1-D, local, derived then verified]

Everything handed to Track 2 on 21 Nov needs their artefacts. One Track 1 row needed
nothing: T1-D's calibration-conditional side is a statement about split conformal, and
D89's identity extends it from the matched cell to every transfer cell.

**The law.** With `F_a`, `F_b` the positive-score CDFs at the calibration and deployment
operators, `tau_a` the `k`-th smallest of `n` calibration positives, and coverage the
fraction clearing the threshold:

    U := F_a(tau_a) ~ Beta(k, n + 1 - k)                  [probability integral transform]
    coverage(a -> b) = 1 - F_b(tau_a) = 1 - G_ab(U),      G_ab := F_b o F_a^{-1}

**A transfer cell's coverage is the same Beta variable pushed through the operator pair's
probability-probability map, and the matched cell is the case `G = identity`.** Verified
draw by draw at `2e-16` before any of what follows was written, so this is algebra, not a
model. `reliability.transfer_beta_reference` builds it by sampling the exact Beta and
evaluating the *empirical* `G`, so no density is estimated and no asymptotics are used.

#### 1. The matched Beta band is wrong at a transfer cell, and it can be too narrow

`G'(u) = f_b/f_a` at the threshold, so the spread rescales by that density ratio. Measured
against the first-order prediction across a **7.1x** range of spreads:

| deployment scale | transfer mean | transfer sd | sd ratio | `G'` | ratio / `G'` |
|---|---|---|---|---|---|
| 0.7 (easier) | 0.8251 | 0.04343 | **1.605** | 1.603 | 1.0015 |
| 1.0 (matched) | 0.9008 | 0.02677 | 0.989 | 0.990 | 0.9996 |
| 1.6 | 0.9565 | 0.01309 | 0.484 | 0.490 | 0.9869 |
| 2.4 (harder) | 0.9793 | 0.00611 | **0.226** | 0.221 | 1.0220 |

Agreement to ~2% over a factor of seven. **The direction is the finding:** the ratio
crosses 1, so a certificate quoting the matched Beta band at a transfer cell is not
conservative - at the easier deployment operator the true spread is **1.6x** the band it
would quote. `TransferBetaReference.matched_band_is_too_narrow` is a property rather than a
note, because that is the case a deployment certificate must not get wrong. This is what
T1-D's row asked for - "naive transfer should be shifted off the Beta law" - except that it
is *rescaled* as well as shifted, and the rescaling is the part that bites.

#### 2. The sign of the gap, in the law rather than in the bank

The table above re-derives D89 without a bank. The **more** ill-posed deployment operator
*over*-covers (mean 0.9793, gap -0.0785) because its high scores clear an inherited low
threshold easily; the operator that **loses** coverage is the **easier** one (0.8251, gap
+0.0757). So any operator distance that tracks ill-posedness must get the gap's sign
backwards. T1-A's refutation and T1-B's 0-of-4 are now both consequences of a two-line
identity, and the test is named for the counterintuitive direction rather than the
expected one, because the first version of it was named backwards and asserted the truth.

#### 3. D86's mechanism is a theorem within a row, and 72-of-72 is about the cohort

At a fixed calibration draw every deployment operator in one row is a monotone function of
the same scalar `U`, so a row's cells are **exactly comonotone** - rank correlation `+1.0000`
across 40,000 draws, not approximately. That is D86's "one calibration draw displaces every
cell together", derived rather than observed.

**But only within a row.** Nine rows have nine draws. Nine *independent* rows land on the
same side with probability `2 * (1/2)^9 = 0.0039`; measured, **0.00392**. With one cohort
imaged through all nine operators, **1.00000**. So the registry arm sitting below the
across-split mean at **72 of 72** pairs is not evidence for the within-row mechanism - it
is evidence that the nine operators share a calibration cohort. D86's remedy is right and
unaffected; the mechanism it states is one layer short of what makes the number come out
at 72 rather than at 64 or 8.

**This is the second time a 72-of-72 has meant something other than what it was read as**
(D91 was the first, on gap signs between label arms). The shape of the error is the same
both times: a count over 72 pairs that is really a statement about 9 operators. The pairs
are not the sample size - which `GAP_CLUSTER_UNIT` already says for intervals, and which
evidently needs saying for counts too.

**One question for Track 2, one line:** within a single split, do all nine operators
calibrate on the same patient ids? If yes, point 3 is closed and D86 is proved. If no,
72-of-72 at p = 0.004 is itself the anomaly and wants explaining.

#### What this does not claim

The law is conditional on exchangeability *within* each operator's own positives, which is
the assumption split conformal already needs and which the region-clustering caveat at the
end of this module qualifies. It says nothing about whether `G_ab` is estimable from
operator geometry - that was T1-B's question and the answer there was no. It predicts the
spread *ratio* between cells, not the absolute spread, for which the finite test set and
the split overlap corrections already in `SpreadCheck` still apply.

**Status:** `reliability.transfer_beta_reference`, `reliability.probability_map`,
`reliability.TransferBetaReference`; 10 new tests, `tests/test_reliability.py` 40 -> 50;
full suite **1,081 passed, 7 skipped**.

**`TRANSFER_SPREAD_PREDICTION` frozen before the number exists**, on the same footing as
`GAP_FACE_BANK_PREDICTION`: the across-split spread of cell `(a, b)` must be the spread of
cell `(a, a)` times `f_b/f_a` at the threshold -- 72 predictions from artefacts that
already exist, no fitted parameter, falsifiable on Track 2's 13-split table. Amendment
nine; fingerprint `4a7426a10a50` -> **`f3b3f1b6aed8`**. The entry first claimed the
deployment operator's matched cell "would not be `G'` at all"; that is false, since the
spread ratio divides out the Beta's scale, so at the bank's ~10,760 positives per operator
either reference gives the same number. Corrected in the constant and pinned by
`test_either_matched_cell_gives_the_same_spread_ratio_at_equal_counts`. Branch `T1-B`.

---

## Open Questions

### Q1 — "Operator distance" — **RESOLVED by D19** (retained for the reasoning)
The coverage-gap-vs-operator-distance plot is the paper's central figure, and the x-axis of that figure is currently undefined. This choice determines the weighting scheme (D4) *and* the form of the bound (D5, now Track 3's), and Track 3 is blocked on it. Blueprint v3.0 §12.1 widens the candidate list to five: null-space projection distance; subspace and principal-angle distances (geodesic, chordal, projection); operator and pseudo-inverse norms; sampling-support difference; spectral divergence in singular-value decay or conditioning.

Three requirements, all binding:
1. **Computable without ground truth** — it is a function of the two operators only.
2. **Monotone in something a radiographer would recognise** as "a bigger change to the scan."
3. **Defined identically for both modalities** (D15) — it must accept a k-space mask *and* a projection-angle set. This is new in v3.0 and it prunes the list: sampling-support difference is the natural MRI-only candidate and does not obviously generalise; the projector and principal-angle distances do, since `P_null` exists for any linear A.

The T1.1 projector utilities (WI-2) are the substrate for the subspace-distance candidates, and following D18 they are unblocked, so this question is answerable now rather than after Track 2 delivers.

### Q2 — Region definition for the flags — **RESOLVED by D20** (retained for the reasoning)
"Per-region" is used throughout the blueprint without fixing what a region is — pixel, patch, superpixel, or annotated anatomical/pathological structure. It determines the multiplicity correction, the flag-area metric, and how fastMRI+ annotations are matched to flags. Note the tension: fastMRI+ labels are lesion-level bounding boxes, so evaluation naturally lives at that scale even if calibration runs on patches.

### Q3 — How many solver draws per reconstruction — **statistical half RESOLVED (8 draws); cost half still open**
The score combines null-space structure with cross-sample disagreement, so it needs several solver draws per measurement set. Draws x calibration-set size x number of operator regimes is the dominant compute cost of the whole project. Following D18 the question splits: the *statistical* half is measurable now on a mock sampler; the *cost* half (seconds per draw) waits on T2.2. Multiply at integration.

**Statistical half, measured** (`scripts/draw_budget.py`, 128^2, R=4, injected hallucination, 5 trials, reference 256 draws). The metric is **rank** correlation against the reference, not RMS error: conformal calibration thresholds the score, so a uniformly mis-scaled score that orders regions identically produces identical flags. A reordering is what costs.

| draws | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| mean rank corr | 0.948 | 0.949 | 0.959 | 0.958 | 0.967 | 0.970 |
| worst trial | 0.935 | 0.934 | **0.951** | 0.944 | 0.951 | 0.958 |

**Ruling: 8 draws.** It is the smallest budget whose *worst* trial clears 0.95, and 64 draws buys only +0.011 mean over it. Worst-trial rather than mean, because the budget has to hold for every item in the bank.

**Two caveats, both real.** (a) This is measured against a mock with a known 10x spread contrast between hallucinating and clean regions; a real prior's contrast is unknown, so **re-measure once T2.2 lands**. (b) The first attempt at this measurement used a *spatially uniform* mock spread and produced a meaningless plateau at ~0.9 — with homogeneous noise `s_disagree` estimates the same constant in every region, carries no ranking signal, and the correlation then measures the mock rather than the score. The mock's spread must be heteroscedastic or the experiment is void. Recorded because the same trap applies to any future re-measurement.

### Q4 — What experiment shows weighted recalibration beating the Mondrian backbone? [OPEN, owned by T1.4]
D3 makes Mondrian the guaranteed backbone and D4 makes operator-distance weighted recalibration the headline contribution. But Mondrian is *exact* in the known-finite-operator regime and weighted CP is only approximate there (D6), so a reviewer will reasonably ask why the approximate method is the contribution. The background research reached the same conclusion independently, calling group-conditional CP "a serious competitor to the whole merger, and arguably the honest primary method."

The answer exists and is defensible — weighting buys what Mondrian structurally cannot: (a) **thin groups**, where per-regime calibration data is too scarce for a stable threshold, exactly the high-acceleration regimes D10 warns about; and (b) **continuous operator change**, where there is no finite group to calibrate within. But neither case is currently an explicit line in the blueprint's experimental protocol (§16), which means the paper's headline method has no experiment that demonstrates its advantage over its own baseline. T1.4 must name that experiment: sweep calibration-set size per regime down to where Mondrian's thresholds become noisy and weighting wins, and include off-grid deployment operators that no group covers. Report both, not just the coverage curve.

### Q5 — Scoop recheck — **done August 2026 (D30); standing, repeat at write-up**
The two "verified empty" cells (operator-shift-robust conformal coverage; flow-matching × conformal) were verified as of **mid-2026**, and the research documents explicitly flag that absence of evidence is not proof, naming VOILA (Fridovich-Keil), Schniter/Ahmad and Sulam as the groups to monitor monthly across cs.CV, eess.IV and stat.ME. D11 already mandates an early preprint for priority. The sweep has not been repeated since the research was written. Repeat it before further design work, and again at write-up.

### Q6 — Does `s_free` fire on detailed anatomy rather than on hallucination? [**ANSWERED on a development bank — D43**; standing until a reportable bank exists; owned by T1.1/T1.4]

**Answer: no.** On the first real-prior bank, `s_free`'s correlation with the ground truth's own null-space content is **−0.035**, and its separation *strengthens* when anatomical content is held fixed (detail-matched AUC 0.933, +0.715 over the anatomy control). The component responds to invention, not to detail. Full result and its two cautions — 3 items, in-sample, face prior — in **D43**. Re-run at the first reportable bank; the command is one line and the verdict rule is pre-registered, so the re-run cannot be steered by knowing this outcome.

**Status, September 2026.** Half of Q6 turned out to be answerable from the definitions. **D38** derives the exact identity linking `s_free`, `free_content` and `err_free`, and it settles the mechanism: under the relative criterion the label is monotone in `u = s_free/free_content`, so `s_free` alone is mismatched to its own label by however much anatomical content varies between regions. The risk Q6 names is therefore real and structural, not merely possible. **D40** adds that `cos θ` — how much of the reconstruction's hidden content agrees with the truth's — is recoverable per region, which is the quantity Q6 has been circling.

The empirical half still needs a real-prior bank, and is one command once A2(a) lands:

```
PYTHONPATH=src python3 scripts/q6_attribution.py runs/w02_solver_smoke-56bc92fa6a76906d --tau 1.75
```

**The verdict rule is pre-registered**, in `caliber/scoring/attribution.py`'s constants, written before the bank was available so it cannot be tuned to what the data says — the same discipline D37 applied to τ and D28 to the weights. `s_free` is declared *invention-driven* only if its detail-matched AUC clears 0.60 over at least two surviving strata, exceeds the anatomy control's own detail-matched AUC by 0.05, and keeps a partial rank correlation of 0.20 against invention with content held fixed. It is declared *anatomy-confounded* if it beats the anatomy control by less than 0.02 with no partial correlation, or if stratifying destroys its separation.

**One finding from building it, which changes how the result must be read.** Stratifying on a *continuous* confound only attenuates it. A component constructed to be an exact multiple of `free_content` — a pure anatomy meter, no invention signal whatever — still scores 0.87 detail-matched with four quantile bins, which an absolute threshold would call a strong detector. Scoring `free_content` itself through the identical path gives 0.87 too, and the difference of zero is the honest answer. **The paired comparison against the anatomy control is the load-bearing instrument; the stratified AUC alone is not sufficient and would have answered Q6 wrongly.** Strong confounding also starves the strata (one of four survived in that control), so the number of surviving strata is reported beside every verdict.

Validated against four banks with known answers: an anatomy-confounded control, an invention-driven control, a pure-noise control, and an identity-consistent bank where the oracle ratio must beat the raw component. `tests/test_q6_attribution.py`, 23 gates.

**Known limitation of the rule, stated rather than patched.** A candidate that is *defined* as a ratio with the control (the oracle) has a partial correlation that no longer means what the rule assumes, and lands as `ambiguous` despite the best AUC and the smallest flagged area in the table. The rule is left alone — it is pre-registered — and ratio-type candidates are read off AUC and flagged area instead.

---

**Original entry, retained:**
**Sharpened by D35:** with `s_residual` structurally dead for data-consistent solvers, the score is effectively `s_free` + `s_disagree`. If `s_free` also turns out to track anatomy rather than invention, the whole thing reduces to cross-sample disagreement — a posterior-spread detector, which is a *known baseline*, not this project's contribution. That makes Q6 considerably more load-bearing than when it was written.
`s_free` measures ‖P_N x̂‖ — the magnitude of the reconstruction's hidden content. But real anatomy *has* large null-space content whether or not the prior invented any of it, which is structurally the same trap D21 identified in the label criterion. The injection test shows `s_free` alone does detect an injection **on top of** anatomy (it is the *excess* that stands out), so the component is not useless. The open risk is region-level FDR on real data: if `s_free` is uniformly high across all textured tissue, it will flag detail rather than invention, and the weighting (D28) will have to suppress it.

This cannot be settled on synthetic images — it needs the real prior plus the D21 criterion. **T1.4's ablation must report per-component FDR/power separately**, not just the combined score, or a `s_free` that is doing nothing useful will hide inside an acceptable-looking total. Note the possible outcome, flagged early: if `s_disagree` alone matches the combination's detection quality, that is an honest and publishable negative result about the operator decomposition (D2), and better found at reference-implementation scale than in Phase 3.

---

## Track 1 Task Log

Per-task records live in `T1.<n>_*.md`. Decisions and problems arising during a task are logged below as `T1.<n>-D<k>`. Task names follow blueprint **v3.0 §18** (see D14 for the change from v2).

### T1.1 — Operator-aware score design
See `T1.1_score_design.md`. Branch `T1.1`.

WI-1 **closed by D20** (patch grid is the risk unit). WI-2 **superseded** — Track 2 shipped range/null-space projectors for both modalities with exact adjoints and health checks, so build on `operators/base.py` rather than reimplementing. The Week 2 criterion question is **answered by D21**.

**WI-3 delivered (D25–D28).** Components ratified unchanged; combination replaced with a standardised non-negative affine rule in `scoring/score.py`. `SCORE_VERSION` → `v1-components-ratified` (component arithmetic **unchanged**, so existing banks are numerically identical and need no regeneration — only the label differs); `COMBINATION_VERSION` tracked separately because the combination is applied from cache at analysis time and will be revised more often.

**WI-4 delivered.** Reference implementation exercised end to end by `tests/test_score_combination.py` — 8 gates, all passing, including the constructed-hallucination injection test, the frozen-scales regression test protecting the headline result, the affineness check T3.3 depends on, and a structural guard that the score's signature cannot see ground truth. Building it surfaced **D29** (the `.real` cast breaking data consistency), which is referred to Track 2 as D-024.

**WI-5 delivered — 8 draws** (`scripts/draw_budget.py`); see Q3 for the table and the two caveats.

_Remaining: none of the WIs. T1.1 is complete pending the weight selection in D28, which needs the design split and therefore real data (C2)._

### T1.2 — Shift-robust calibration
See `T1.2_shift_robust_calibration.md`. **Implemented and verified on synthetic banks** — all three tools, thin-group refusal, ESS tracking (D32–D34). 15 gates passing, including the coverage collapse (0.76 miss at nominal 0.1) and its restoration by Mondrian. Awaiting real banks (C2) for the reported result; fallbacks [7][8][9] deferred per the risk register. Grouping and split semantics are settled ahead of it by **D22, D23, D24**. `calibration/__init__.py` holds Track 2's reference split-conformal and Mondrian implementations, explicitly marked as superseded by T1.2 and not for reported results.

### T1.3 — Operator-distance formulation
See `T1.3_operator_distance_formulation.md`. **Delivered — D19.** `null_space_projection` (Frobenius) with `ambient` normalisation, locked for Week 3. Unblocks Track 3's T3.3, which derived contingently against "the T3.2-recommended aggregate subspace distance"; that is now the ratified choice.

### T1.4 — Experimental design
_Not started. Owns Q4._ Carries four inherited instructions: operator grids spread evenly in **distance** not acceleration (D19); region size is a declared ablation axis (D20); noise becomes a group axis if the protocol varies it (D22); seed-to-seed variation is the noise floor a real shift must clear (D24).

### T1.5 — Cross-track integration
See `T1.5_cross_track_integration.md`. _Week 1 sign-off delivered_ — D19–D24 close `Documentation/week1_kickoff.md` §G. Standing work: the shared interfaces (score bank, operator registry) and adjudication continue through the term.

### T1.6 — Analysis and course correction
See `T1.6_analysis_and_course_correction.md`. **Phase 1 gate delivered** — `T1.6_analysis_and_course_correction.md`. Verdict: Phase 1 complete but for reproduced baselines (T2.6), blocked on solver integration (T2.2); feasibility demonstrated; no scope reduction warranted. Scoop recheck done (D30); novelty framing tightened (D31). Standing from here: owns Q5 and each subsequent phase gate.

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
| **S1** | The operator-distance definition (Q1 / T1.3) — the quantity the bound must be stated in | T3.2, T3.3 | **Delivered — D19** |
| **S2** | The region definition (Q2 / T1.1 WI-1) — fixes the multiplicity correction, the flag-area metric, and how annotations are matched | T2.1 (C4), T2.4 | **Delivered — D20** |
| **S3** | Score-bank format and the non-conformity score specification | T2.3, T1.5 | Partial — format signed off (D19–D24); the score *combination rule* is still `v0-placeholder` |
| **S4** | Experimental protocol, operator grids and ablation matrix | T2.5, T2.6, T3.6 | Open — now carries four instructions from D19–D24 (distance-even grids, region-size sweep, noise-as-group-axis, seed-variation noise floor) |

S1 and S2 are delivered, which unblocks Track 3's T3.3 and Track 2's Week 2. **The remaining Track 1 critical-path item is the score combination rule** (T1.1 WI-3): `scoring/score.py` still ships `SCORE_VERSION = "v0-placeholder"`, and nothing downstream can be reported until it is real. Note T3.3 §5's constraint on it — the null-space-energy and disagreement terms may need to be *designed* Lipschitz in `d_op`, not checked afterwards, or L1's bound will not close.

### One ask back to Track 2

| # | Track 1 asks | For | Cost |
|---|---|---|---|
| **A1** | One nullable bank column `free_content` = ‖P_N x_true‖ per region / `ref_intensity` | D21's criterion | Additive nullable column — backwards-compatible, no `SCHEMA_VERSION` bump (spec §6) |

### D93 - One cohort closes D86; and four of my own claims break, one of them a floor the data walks under [T1-B, T1-D; rules on Track 2's T2.11 Sec. 1j; amendments 10-12]

Track 2's run (f1d0012, 1,097 tests) answers both outstanding questions, confirms two frozen
predictions, and falsifies three things I wrote. The falsifications are the useful part of
this entry, so they come first.

#### 1. The correlation floor I reported was wrong, and the data goes under it

D91 reported `r >= 0.93` for the common-mode correlation between the two label arms'
difference tables, from the assumption-free bound `r >= sqrt(1 - ratio^2)` with a ratio of
0.364. Track 2 measured `common_mode_correlation = +0.788`.

**The bound is right; I fed it the wrong quantity.** The algebra is in sd's --
`sd(gap)/sd(level) = sqrt(s^2 - 2*r*s + 1)`, minimised at `s = r` with value
`sqrt(1 - r^2)`. The 0.364 I substituted was a ratio of **mean absolute** differences
(0.024 / 0.066). In sd's the ratio is 0.0287 / 0.0425 = 0.675, which gives a floor of
**0.738**, and the measured 0.788 clears it.

So the corrected statement is `r >= 0.738`, measured `+0.788`. D86's common-mode claim
survives and is weaker than I reported: the common displacement is most of the level
difference, not nearly all of it.

This is worse than an overstatement. A floor of 0.93 with a true value of 0.788 is a
**false floor** -- a claim of the form "at least", published, that the data violates. I
caught the equal-variance version of this error in D91 and replaced it with the
assumption-free bound; I then fed the assumption-free bound a mean-absolute input. The
lesson that generalises is not about this bound: **a dimensionless ratio does not carry a
label saying which norm produced it**, and sd-algebra silently accepts an L1 ratio. The
`common_mode_correlation_lower_bound` field is computed from `sd_difference_ratio` in code
and always was; only the number I wrote into the prose came from the wrong column.

Ninth instance of the project's recurring failure -- a statistic returning the hoped-for
number -- and the sixth of them mine.

#### 2. "At equal positive counts" was a false premise, and the retraction reverses a correction

`TRANSFER_SPREAD_PREDICTION` (amendment 9) carried the clause: *"at equal positive counts
the deployment operator's matched cell gives the same ratio, so this choice is not
load-bearing on a bank with ~10,760 per operator."*

The bank's positive counts run **3,157 (sv 180) to 21,762 (la 90)**. 10,760 is their mean.
The premise was false and the choice is load-bearing, as Track 2 measured:

| Reference cell | rho | log-log slope |
|---|---|---|
| Calibration operator's matched cell | **+0.977** | **1.08** |
| Deployment operator's matched cell | +0.88 | 0.66 |

**The sequence matters more than the error.** I first wrote that the reference is
load-bearing. I then "corrected" that to not-load-bearing on the basis of my own fixture --
which had equal counts *by construction*, so it could not have returned anything else. The
original instinct was right and the correction was the mistake. A fixture that confirms a
claim because the claim is built into the fixture is the same failure mode wearing a test's
clothing. **Tenth instance**, and the seventh of them mine.

The clause is retracted in place with the retraction left visible, because the clause
licensed the wrong reference and a silently rewritten freeze is not a freeze.

#### 3. The Beta unit: a real local defect, inside a finding the repo already had

Track 2's incidental line -- *"tau sat between the 0.03 and 0.18 levels of F_a over the
splits"* -- does not fit the Beta law at any positive count on the bank. Working backwards
from the span of 13 draws, only a count in the tens reproduces it: 21,762 regions imply a
span of 0.007, 3,157 imply 0.018, and the observed span is **0.15**, which needs an
effective count near **45**.

**I then wrote this up as a discovery, and it is not one.** The comment block above
`ClusteredCoverage` in `reliability.py` already says it, in these words: *"Regions are not
independent: on T2-A's CT bank, 16,400 calibration regions behave like roughly 35 units,
because the unit is the subject."* It goes further than I did: it records that the region
count gives a band of +/-0.011 that **rejects a correctly calibrated bank** while the
subject count gives +/-0.10 that **accepts almost anything**, concludes that *"neither is a
measurement, so neither should be the gate"*, and names the cluster bootstrap as the gate.
The module also already **measures** the thing per cell -- `ClusteredCoverage.design_effect`,
`.effective_n`, and an `.independent_sd` carrying both variance terms -- and `as_row()`
emits all three.

I was editing that file and did not read it. The order of operations was: find a striking
number, verify the number, write it up, then discover the module had it. This is **not**
another instance of the project's statistic-shaped failure mode and should not be added to
that tally -- the number was real and my arithmetic was right. It is a different and more
ordinary failure: not reading the module being edited. Worth recording separately because
the remedy is different. A frozen constant cannot prevent it; reading can. The correction is
not subtle: **nothing in this section is a new finding about clustering.**

**And one of my two estimates was wrong by the error that file documents.** I estimated a
second effective count of 174 from the width of Track 2's la 90 interval, comparing it
against `sqrt(p(1-p)/n_cal)` -- a *one-term* reference. `independent_sd`'s docstring exists
precisely to say that the reference must be
`sqrt(Var_Beta(n_cal) + p(1-p)/n_test)`, because comparing against the Beta term alone
"makes genuinely independent regions look clustered by a factor of about sqrt(2)", which is
how an earlier `design_effect` came out at 2.6 on a fixture with no clustering. **The 174 is
withdrawn.** Redone with both terms it is not stable enough to quote, and it should not be
quoted at all: the number is computed per cell from the bootstrap and should be read from
`t1a`'s output, not estimated from a published interval width. No bracket is frozen.

**What survives, and it is narrow.** Two things.

First, a cheap cross-check that is new as a *method* rather than a finding: **tau's own
motion across splits recovers the effective count without a bootstrap**, giving ~45 against
the module's ~35 from resampling. A threshold level and a coverage interval are unrelated
quantities and they agree within 30%, which is worth having because the tau version costs
one column and no replicates.

Second, the actual defect, which is mine from yesterday and local:
`TransferBetaReference.interval` and `.contains` shipped with **no label**, while every
other use of the Beta in this codebase carries one -- `make_figures` draws it as the band
"split conformal follows if every *region* were independent" and calls the distance to the
clustered band "the clustering"; `t1a_safety_rule` emits it as "the independence-assumed
idealisation, with the design effect beside it". Mine read as a certificate. Amendment 11
makes them refuse unless a unit is named, documents them as the rescaled idealisation, and
states that **a band at the subject count is not a gate either** -- the property is
`rescalable`, not `certifiable`, because nothing here certifies anything. The dimensionless
results, `slope` and `sd_ratio`, stay available, since they do not depend on the unit.

#### 4. Why rho +0.977 certified nothing, and the form of the failure mode that is new

The prediction in amendment 9 was frozen before the number existed, is a true consequence of
D92's law, and passed at rho +0.977 with 100% of pairs within 2x. It was also **blind to a
unit error of order 100 standing immediately beside it**, because the predicted quantity is
a *ratio of two spreads* and the reference's `n` appears in numerator and denominator and
divides out. Verified in code: changing `n_independent` from 10,762 to 41 leaves `sd_ratio`
and `slope` bit-identical while the quoted band widens 16x. This holds whatever the right
effective count turns out to be, which is why it is the part of Section 3 that is actually a
finding: the blindness is a property of the *statistic*, not of the number it was fed.

Every previous instance of this project's recurring failure was a statistic that came out
well because it had been chosen, framed, or inverted after the fact. This one is a different
and worse shape: **the statistic was correct, frozen in advance, and still uninformative
about the error next to it.** A dimensionless agreement certifies a law's *shape* and says
nothing about its *scale*. Amendment 12 freezes `DIMENSIONLESS_PASS_WARNING`, which requires
one dimensioned check beside any such agreement -- here, the matched cell's own across-split
spread against its Beta sd at the independent count, which does not divide out.

**Eleventh instance, and a new form.** The running tally: seven up to the
bootstrap-interval defect, D91's eighth, and three in this entry -- Section 1's false floor,
Section 2's fixture-confirmed correction, and this one. Four of the first seven were mine,
the eighth partly so, and all three here are mine: **eight of the eleven**. Section 3's
overclaim is a fourth error in this entry and is deliberately **not** counted, because it is
a different failure and folding it in would blur the pattern the tally exists to show.

#### 5. What is NOT explained, stated as such

Track 2 reports the extreme pair la 90 -> sv 180 at **3.83 measured against 2.38
predicted**, a factor of 1.61, while the median over 72 pairs is 0.97. My first hypothesis
was curvature: tau wanders over a 0.15-wide band of `F_a` while `slope` is a central
difference at half-width 0.02, so the linearisation is being asked to summarise a band it
was not built for.

**I tested it and it is not the explanation.** On gamma populations the exact pushforward
(`sd_ratio`, the sd of `1 - G(U)` over the full Beta) and the linearisation (`slope`) agree
to 0.6-0.8%, and they still agree to ~1% when the Beta is given n = 41 instead of n = 21,762.
Curvature accounts for about one part in a hundred of a 61% discrepancy.

So the compression at the extremes is **open**. The leading remaining candidate is that the
measured spread carries test-set noise whose effective unit is also the subject, not the
region, which would inflate the measured side worst where the deployment count is smallest --
and sv 180 is the smallest count on the bank, which is the pair in question. That is a
hypothesis with a mechanism and no number behind it, so it is recorded as one and not
frozen as a prediction. No amendment is taken for it.

#### 6. Both questions answered, and D86's mechanism closes

**One cohort: yes.** Side is `split_for(subject id, salt)`; the operator does not enter. The
same 41 calibration subjects at all nine operators, in all 13 splits, checked on the bank.

This closes the gap D92 left. D92 proved that within one calibration row every cell is a
monotone function of the same scalar `U`, so a row's cells are exactly comonotone -- but
nine rows are nine draws, and nine *independent* rows would land on the same side with
probability 0.0039. One shared cohort gives 1.00000. The cohort is shared, so the 72-of-72
one-directional displacement is **the mechanism, measured**, and not a coincidence at
p = 0.004. **D86's argument is now a theorem plus a verified premise rather than an
argument**, and the retraction in D91 stands unchanged: the *gap-sign* count was never the
evidence; the common-mode correlation is, at +0.788 against a floor of 0.738.

**The two rho sets: same arm, different definition.** T1-A used nominal-minus-transfer,
T1-B matched-minus-transfer. A sensitivity, neither wrong, and all four arm x definition
columns are tabulated. The discrepancy I flagged in D91 is resolved and no one was wrong.

#### 7. The direction rule's determinacy test was not sufficient -- Track 2 found the hole

On the CT bank the rule's pooled `rho(d, tau_b - tau_a)` is **+0.05** while every family
cell is **+0.55 to +1.00**, and the determinacy test passed anyway, because it checked
tau's *range* against tau's noise (12.5x, well clear of the frozen floor of 3.0) and never
asked whether the correlation it actually reads was resolved. A rule one bootstrap resample
away from returning the opposite direction was being called determinate.

**The cause is Simpson's.** The two families' tau levels are offset -- every limited-angle
threshold sits above every sparse-view one -- so pooling the 72 pairs mixes a between-family
level difference into a within-family trend and can cancel it exactly. On a fixture built to
reproduce the bank's pattern (pooled -0.01, cells +0.79 to +0.94):

| Reading | sign flips under bootstrap resampling |
|---|---|
| Pooled rho | **52.4%** |
| Within-cell unanimity | **0.1%** |

A coin flip against a stable verdict. Amendment 10 adds
`GAP_DIRECTION_DETERMINACY_SCOPE = "within_family_cell"` and
`GAP_DIRECTION_POOLED_RHO_BARRED`: the rule reads the four family cells, their signs must
agree, and the pooled value may be shown but may not set the direction. `meta` is now
required for a reportable call, and `DirectionCall.reportable` needs all three of
determinacy, unanimity, and an unbarred bank.

**This narrows what may be claimed.** It adds a condition, and both banks pass it -- CT at
+0.55 to +1.00 and face at -0.89 to -0.99, each unanimous. Amendment 7 remains the only
entry in the tuple that widens.

I am not taking Track 2's offered default ("nothing changes, the bank is barred anyway").
The bar on CT is about gaps already seen; it does nothing about a rule that would hand the
next bank a direction drawn from noise. A defect found in a rule is fixed in the rule.

#### 8. Two frozen predictions passed, and one of them is the direction rule working

**`GAP_FACE_BANK_PREDICTION` holds.** Frozen at fingerprint `ce668b3e9b98`, long before the
face bank's pairs were completed. `rho(distance, tau_b - tau_a) = -0.94` over all 72 pairs,
-0.89 to -0.99 within cells, with Track 2 computing the 19 missing pairs from the bank's own
256^2 distances. tau follows the knob exactly there (+1.00 in both families); on CT it is
-0.83 and -1.00.

So the direction rule, which takes thresholds and distances and no outcome, returns
**non-decreasing** on the face bank and **non-increasing** on CT -- the correct direction on
each, and the face bank was genuinely out of sample. That is the amendment-7 widening
earning its fence.

**The T1-A gate passes 9 of 9 without `--ignore-control`.** D91 predicted la 90 would move
to roughly `[0.779, 0.912]`; measured `[0.779, 0.906]`. The lower end is exact and the upper
is 0.006 wide.

**The warning D91 attached to that prediction stands and matters more than the pass.** la
90's point estimate is **0.847**, which is **5.3 points under nominal**. The cell passes
because the interval is wide, not because the cell is calibrated. Passing a gate is not
evidence of calibration and must not be reported as any.

#### 9. The baseline set is smaller than it looks: null-dim is the view count

Track 2 resolved sv 120's `null_space_dim` of 64,000: the column is ambient minus
measurements, `102400 - 320 * views`. Every 120-view operator gets 64,000 and the operator
is built like the other five.

**So `null_space_dimension` is an affine function of `views` within the sparse-view family**
-- `102400 - 320v` -- and therefore rank-identical to it. Any rank-based method sees one
predictor, not two. This is why `views/arc` and `null_space_dim` returned near-identical MAE
in T1-B and why both ordered baselines collapsed the same way: they were the same baseline.

T1-B's four predictors are three: the distance, a constant, and the acquisition knob in two
sets of units. `BASELINE_SCOPE` already confines `acquisition_parameter` to within-family,
which is why this never produced a wrong number -- but the four-baseline framing overstated
how much the distance was being asked to beat, and anything written as "beats three
baselines" must be written as "beats a constant and the knob".

#### 10. D90's "one adjacent transposition" is wrong; the anomalous operator is sv 180

D90's testable prediction **holds**: `tau(sv 120) < tau(sv 180)`, 0.893 against 0.972, in
13 of 13 splits. Its localisation does not. Track 2's full ordering is

    sv 120 < sv 90 < sv 180 < sv 60 < sv 45 < sv 30 < la 150 < la 120 < la 90

and sv 90 (0.933) is below sv 180 as well, so this is not one transposition. The knob order
is sv 180 < sv 120 < sv 90 < sv 60 < sv 45 < sv 30; the rest of the family is perfectly
ordered, and the single displacement that explains the data is **sv 180 sitting two
positions too high**. D90 named sv 120 as the anomaly; the parsimonious reading is that sv
180's threshold is too high, not that sv 120's and sv 90's are both too low.
`rho(views, tau) = -0.83` within sparse view, not -1.00.

The heading of D90 is left in place with this correction recorded against it, on the
standing principle that a corrected record has to show what it corrected.

#### 11. The dichotomy: a deployment screen that needs no deployment data

Track 2's `gap_sign_from_thresholds` check: 72/72 on the registry, 934/934 decidable over 13
splits, 72/72 on the face bank. That is D89's identity, so it had to pass; it is a check on
the bank, not a finding.

What is worth taking is the split they report beside it. Of D90's 36 pairs with
`tau_a < tau_b`, **none is below nominal** (lowest 0.904); the other 36 are **all** below.
A perfect dichotomy, and by the identity it must be: `tau_a < tau_b` forces
`F_b(tau_a) <= F_b(tau_b)`, hence `coverage(a->b) >= coverage(b->b)`.

The consequence is operational. **Which direction a transfer will fail is readable from the
two calibration thresholds alone, before any deployment data exists.** Not the magnitude --
the magnitude needs the density ratio and that is what Section 5 leaves open -- but the
sign, exactly, with no scores, no labels, and no deployment set. For a coverage claim that
is the part a user needs first, and it is the strongest thing T1-B has that does not depend
on a fit.

#### 12. Rulings on Track 2's three items

**(1) The cos-theta guard. I am not taking the default, and I am not setting a tolerance
either.** T2-A6 shows the excess above 1 does not track the additivity defect
(rho -0.44 to +0.07, n = 9), and that a region must already be near `cos = 1` to cross. So
the excess is not a proxy for solver quality and may not be cited as one -- which also means
no tolerance read off the T2-A6 table would mean anything.

The structural point: `P_N` is linear, so `P_N(x_hat - x) = P_N(x_hat) - P_N(x)` *exactly*,
and with `e` formed that way the law of cosines gives `|cos| <= 1` by Cauchy-Schwarz
identically. The excess the bank shows is therefore entirely the three solves failing to be
linear with each other. Hence:

- The **reported** `cos` is formed from two solves, where it obeys Cauchy-Schwarz by
  construction. The 1e-3 guard applies there, where it is a real numerical check.
- The three-solve excess becomes a **logged diagnostic with no gate**, per operator. It is
  the only thing that measures the solver's linearity defect, so it is kept -- but
  `err_free` must not *replace* it, because that would make the guard vacuous and hide the
  defect rather than fix it.

Adopt on the next bank, with both channels emitted. This is consistent with D91's point that
Cauchy-Schwarz is projector-independent and so T2-A5 cannot set the tolerance.

**(2) The determinacy test.** Not the default. Fixed in the rule -- Section 7, amendment 10.

**(3) T2-E.** Default taken for figures 6-9, and fig 9 stays untouched. Figure 10 should
gain a second series only after Section 5's open question is settled, not before: a figure
that shows predicted against measured with a 1.61x miss at the extreme and no account of it
is a figure that invites the account to be invented. One figure is worth adding and costs
nothing: coverage against `tau_a - tau_b`, 72 points, with the nominal line and zero line
making four quadrants of which two are empty by theorem. It is Section 11 drawn, and because
it cannot fail it is a figure about the mechanism, not evidence for it.

#### 13. Where T1-B stands, and the face bank's reportability

T1-B on the face bank is **reportable as a coverage-transfer result and not as a
hallucination-detection result.** The face bank's label defect bars T1-A there -- a
false-safe rate is a detection claim and inherits the label's defects whole. The coverage
gap is a coverage quantity, and the project's standing scope is that coverage claims survive
the label's defects while detection claims do not. `GAP_FACE_BANK_PREDICTION` was frozen
specifically so that bank would be the out-of-sample test, and it is.

Under the frozen criterion the result is **1 of 4**, not a win: the distance beats the
constant in all four cells and is never beaten, but does not separate from views/arc
(+0.006 `[-0.001, +0.016]` and +0.011 `[-0.000, +0.024]`). `GAP_PREDICTION_BASELINES`
requires beating both, so three cells do not report, and that criterion is not relitigated
here.

Section 9 explains why the ceiling is where it is: on nine operators in two families the
distance and the knob are nearly collinear, and the knob was two of the four baselines. The
honest headline for T1-B across both banks is **the signed distance carries information
about the coverage gap beyond a constant, and none beyond the acquisition knob** -- 1 of 4
on the face bank, 0 of 4 on CT, with the direction rule accounting for the difference
between the two.

| Claim | Status |
|---|---|
| Transfer coverage law `1 - G_ab(U)` (D92) | Holds, exact, unaffected |
| Spread ratio `= f_b/f_a` (amendment 9) | Holds, rho +0.977 -- and certifies shape only |
| Matched Beta band at a transfer cell | Wrong twice: by `G'`, and by the unit |
| Matched Beta band as a certificate | **Withdrawn**; not a gate at any count |
| "Regions are not the unit" as a new finding | **Withdrawn** -- already in `reliability.py` |
| Effective count = 174 from the bootstrap width | **Withdrawn** -- one-term reference |
| Common-mode correlation >= 0.93 | **Retracted**; `>= 0.738`, measured +0.788 |
| D86's mechanism | **Closed** -- shared cohort verified |
| Direction rule on both banks | Holds, face bank out of sample |
| Transfer *sign* from thresholds alone | Holds as a theorem; a screen needing no deployment data |
| Extreme-pair compression (1.61x) | **Open**, and not curvature |

Fingerprint `f3b3f1b6aed8` -> `289b0f72f828`. Suite 1,096 passed, 7 skipped.

---

### D94 - The 1.61x triaged: two mechanisms die by sign, and the dimensioned check was already in Track 2's message [T1-D, local, no new bank data; amendment 13]

D93 Sec. 5 left one thing open -- Track 2's extreme pair la 90 -> sv 180 at **3.83 measured
against 2.38 predicted** -- and recorded one hypothesis for it. This entry closes two of the
three candidates, both by the same argument, and finds that the number needed to settle the
other open item had been sitting in Track 2's message all along.

#### 1. The dimensioned check amendment 12 demanded: run, passed, and already reported

Amendment 12 required one dimensioned check beside any dimensionless agreement, and named
it: the matched cell's across-split spread against its Beta sd at the independent count.
That check needs no new run. Track 2 had already reported the number, as an aside:

> tau sat between the 0.03 and 0.18 levels of F_a over the splits, wider than your +/-0.02.

`U := F_a(tau_a)` is `Beta(k, n + 1 - k)` **exactly**, for `n` independent calibration
scores. So the across-split range of `U` carries `n`, which is precisely what the predicted
spread *ratio* divides out. Thirteen draws, by simulation:

| Reference count | `Beta(k, n+1-k)` | median min | median max | against observed [0.03, 0.18] |
|---|---|---|---|---|
| 41 subjects | Beta(4, 38) | **0.034** | **0.177** | matches |
| 3,157 positives (sv 180) | Beta(315, 2843) | 0.090 | 0.110 | excluded |
| 21,762 positives (la 90) | Beta(2176, 19587) | 0.097 | 0.103 | excluded |

The check **passes at the subject count and excludes the region count by a factor of order
300**. The dimensioned companion to a dimensionless pass is therefore no longer pending on
Track 2's disk; it was in the sentence that flagged the problem.

**And D93 Sec. 3's own reading of it was over-precise.** That section inverted the same span
to "an effective count near **45**". The statistic does not carry a point estimate.
`count_from_level_range` compares the observed range against the simulated range of the same
number of Beta draws and keeps the counts whose central 90% contains it:

```
observed width 0.15 over 13 splits  ->  bracket (10, 70), no holes
  consistent: 10 15 20 25 30 35 41 50 60 70
  excluded:   85 and above, through 50,000
```

So the honest statement is **a bracket of 10 to 70**. It cannot separate 35 from 50 -- the
module's figure from the bootstrap and the cohort size are both inside it, and nothing here
chooses between them -- and it rules out the region count overwhelmingly. The method is
frozen as `prereg.COUNT_FROM_LEVEL_RANGE`, which fixes that it yields a bracket and never a
count, before it is applied to the face bank. Two properties of the statistic are recorded
with it: it is calibration-side only, so it carries no test-set term, which is what makes it
clean and also what stops it being a coverage certificate; and `k = floor(alpha*(n+1))`
steps in integers, so the likelihood is not monotone in `n` and the consistent set can have
holes, which `LevelRangeCount.has_holes` reports rather than hides.

#### 2. Curvature: the conclusion survives, the reasoning did not

D93 Sec. 5 ruled curvature out on gamma populations, at "0.6-0.8%, and still ~1% at n = 41".
Track 2's remark was a direct criticism of the premise -- `slope` is a central difference at
half-width 0.02 and tau roams a 0.15-wide band -- so the check was re-run over `U`'s **true**
Beta spread at 41 units, and across seven score-distribution families rather than one.

| Pair | `slope` | exact at n=41 | exact at n=21,762 | inflation at 41 |
|---|---|---|---|---|
| gamma k=2, scale 1.0 -> 0.35 | 3.149 | 3.014 | 3.099 | **0.957** |
| normal -> bimodal mixture | 2.684 | 2.478 | 2.563 | **0.923** |
| lognormal s=0.6 -> 1.4 | 0.877 | 0.883 | 0.814 | 1.007 |
| uniform -> beta(0.5,0.5), crossing | 1.100 | 1.089 | 1.069 | 0.990 |
| t(3) -> normal, heavy tail | 1.014 | 1.001 | 1.030 | 0.986 |
| gamma k=2 -> k=6 | 0.246 | 0.375 | 0.275 | 1.525 |
| lognormal s=1.4 -> 0.6 | 0.053 | 0.176 | 0.054 | **3.315** |

**Curvature has the wrong sign where it would be needed.** In the large-slope regime -- the
regime the 2.38 pair is in -- it *deflates*, 0.92 to 0.96 across families. It inflates only
where `|G'|` is far below 1, and such a cell has a spread well under the matched cell's and
is harmless. A power-law reading says the same thing without a family at all: in the lower
tails `G(u) = c*u^p`, `c` cancels from every ratio, and the inflation is a function of `p`
alone -- 0.99 to 1.01 over the whole admissible range `p < 1`, which is the range a pair
with coverage under nominal must be in.

So the D93 conclusion stands and the number behind it did not generalise: 1% was one family,
not a bound. This is the third time in two days that a claim of mine was right for a weaker
reason than I gave, and it is the kind of correction worth making loudly, because a
right-for-the-wrong-reason claim survives exactly until someone checks it in a different
family.

#### 3. Amendment 13(a): the spread ratio is not exactly dimensionless

The table above also narrows **amendment 12 itself**, which asserted that "the reference's
`n` divides out of both sides" without qualification. It does not, quite. `sd_ratio` is
`sd[1 - G(U)] / sd[U]`; `n` divides out only while `G` is linear across `U`'s range, and
`U`'s range is itself `sd[U]`, which is set by `n`. Measured: agreement to about 3% where
`|G'| >= 1`, disagreement up to **3.3x** where `|G'| << 1`.

The remedy amendment 12 froze is unaffected, and the regime it holds in is the regime that
matters -- the cells where the ratio exceeds 1 are the cells `matched_band_is_too_narrow`
exists for. `DIMENSIONLESS_PASS_REGIME` records the condition and bars quoting a ratio far
below 1 without its unit.

**Also checked, because it would have been a real defect:** whether the reference count can
flip `matched_band_is_too_narrow`, the flag documented as "the case a deployment certificate
must not get wrong". Over a 49-pair lognormal grid the only flips are the seven diagonal
cells, where `G` is the identity, the true ratio is exactly 1, and the flag is a tie rather
than an answer. **No off-diagonal pair flips.** The flag is safe at either count.

#### 4. Test-side noise: refuted by sign, and it makes the gap worse

D93 Sec. 5's "leading remaining candidate" was that the measured spread carries test-set
noise whose unit is the subject, inflating the measured side worst where the deployment count
is smallest -- and sv 180 is the smallest count on the bank. **It is wrong, and wrong in the
direction that matters.** Using the two-term variance `ClusteredCoverage.independent_sd`
already uses, with `V_U` at 41 subjects and `G' = 2.38`:

| `n_test` | transfer sd | matched sd | predicted ratio | measured / predicted |
|---|---|---|---|---|
| 35 | 0.1357 | 0.0676 | **1.995** | 1.92 |
| 100 | 0.1176 | 0.0539 | **2.182** | 1.76 |
| 500 | 0.1088 | 0.0467 | 2.329 | 1.64 |
| infinite | 0.1065 | 0.0448 | 2.380 | 1.61 |

Test noise moves the predicted ratio **down** from 2.38 at every finite test size, so it
widens the discrepancy from 1.61x to as much as **1.92x** rather than closing it. The reason
I got the direction wrong is legible: I reasoned about the numerator and forgot the
denominator gets a test term too, and relative to its own much smaller signal -- `V_U`
against `G'^2 * V_U` -- the matched cell's term is the **larger** fractional addition. Any
noise common to both cells pulls a ratio toward 1, and this ratio is above 1.

#### 5. What survives, with its own shape problem stated

One mechanism has the right sign, and it is structural rather than fitted.
`coverage(a->a) = 1 - F_a(tau_a)` is a **rank statistic of a's own scores**, so it is exactly
Beta however a refit moves that operator's score distribution. `coverage(a->b) = 1 - F_b(tau_a)`
is not: it depends on where a's quantile lands in b's distribution, and a per-split refit
moves the two independently. **That variance enters transfer cells only** -- the matched
reference is immune by construction, so the ratio's denominator cannot absorb it.

Simulated, with each operator's score scale given an independent per-split wobble:

| refit sigma | measured / predicted |
|---|---|
| 0.00 | 0.92 |
| 0.05 | 0.95 |
| 0.10 | 1.04 |
| 0.20 | 1.34 |
| 0.35 | **1.92** |

It reaches the required size. **But it has the wrong shape.** Track 2's discrepancy is
concentrated at the extremes -- median 0.97 over 72 pairs, 1.61 at the most distant one --
and in simulation this mechanism is nearly distance-independent (1.14 at matched scales,
1.02 to 1.05 across increasing dissimilarity). So it is recorded as **necessary but not
sufficient**: it is the only candidate left with the right sign, it cannot by itself produce
a discrepancy that grows with distance, and **no amendment is taken for it**.

The test is cheap and runs on output Track 2 already has: re-estimate `G_ab` from each
split's own scores and read the across-split spread of `G_ab(u0)`. If that spread grows with
pair distance, this closes; if it is flat, something fourth is at work.

#### 6. Status

| Claim | Status |
|---|---|
| Dimensioned check beside the dimensionless pass | **Run and passed** -- at the subject count, from Track 2's own aside |
| Effective count "near 45" (D93 Sec. 3) | **Replaced** by a bracket of 10 to 70 |
| Region count as the Beta's unit | **Excluded** by a factor of order 300 |
| Curvature as the 1.61x | **Ruled out**, now by sign and across seven families |
| "1% curvature error" as a general bound | **Withdrawn** -- one family; range is 0.92 to 3.3 |
| Amendment 12's unqualified "n divides out" | **Narrowed** to `|G'| >= 1` |
| `matched_band_is_too_narrow` across counts | Safe -- only identity ties flip |
| Test-side noise as the 1.61x | **Refuted by sign**; it widens the gap to 1.92x |
| Per-split refit wobble in `G_ab` | Right sign, right size, **wrong shape** -- open |
| Extreme-pair compression | **Still open**, with one candidate and a stated test |

No new trap is counted here. The tally stands at eleven, eight of them mine: Sec. 2 and
Sec. 3 are overstatements narrowed by measurement a day later, not statistics chosen to come
out well, and folding them in would blur the pattern the tally exists to show. The stale
"five of eight" in `prereg.py`'s own docstring is corrected to eight of eleven in this round,
along with the claim that every instance was a choice made after seeing a number -- the
eleventh was not, which is the entire reason amendment 12 exists.

Fingerprint `289b0f72f828` -> `a2ab9a1fdfe4`. Suite 1,106 passed, 7 skipped.

---

### D95 - T1-C's baselines, and a frozen name that did not fix its own direction [T1-C, local, no bank result yet; amendment 14]

T1-C is the last item on the work plan's pre-27-November list that had not started, and the
last place a CALIBER-beats-a-baseline comparison can still come out positive: T1-A is
refuted on CT, T1-B is 0 of 4 there and 1 of 4 on the face bank. The three baselines are
built and tested; **no T1-C number exists yet**, because the comparison needs a bank pass
and both banks are Track 2's.

#### 1. The frozen name did not determine the arm, and the gap had to be closed on a property

`prereg.TOOL2_BASELINES` has named `max_quantile` since 18 November. Flags are
`score >= tau` (`apply_flags`), so a **lower** threshold flags more and `-inf` flags
everything -- which means "max-quantile" could be read as the largest per-group threshold
or the smallest, and those are opposite choices. The name does not say which, and picking
the one that flattered Tool 2 would be this project's recurring failure mode at its purest.

So it was settled on the property the name appeals to -- *trivially safe* -- with both
readings run on a fixture where they disagree. Four calibration operators with positive
scores centred at 1.0, 2.0, 3.0, 4.0 and the deployment operator at 2.5, which is the
interior fold the harness exists for:

| per-group threshold | tau | coverage at the unseen operator | valid at 0.90 |
|---|---|---|---|
| smallest (`g_a`) | 0.3665 | **1.0000** | yes |
| `g_b` | 1.3319 | 0.9850 | yes |
| `g_c` | 2.3296 | 0.6200 | no |
| largest (`g_d`) | 3.3572 | **0.0400** | no |

The largest per-group threshold is not trivially safe; it is the least safe choice on the
table. So `max_quantile` is the **minimum** certifiable per-group threshold, and
`test_the_conservative_reading_of_max_quantile_is_the_valid_one` runs both candidate
readings so the decision cannot be quietly reversed later.

**One sub-decision, recorded because it changes the arm.** A group too thin to certify
returns `-inf` from `split_threshold`. Letting that into the minimum would send this arm to
"flag everything" whenever any group anywhere was thin -- a baseline that compares against
nothing. Uncertifiable groups are excluded from the minimum and recorded in `results`
instead, so `reportable` still goes False and the caller can see which groups dropped out.

#### 2. Amendment 14: the bandwidth grid, frozen before the run

`TOOL2_SELECTION_SCOPE` already required the kernel bandwidth to be chosen inside each fold
on calibration operators only, but it named **no candidate set**. A grid assembled after
seeing which bandwidth won would make that nested selection cosmetic -- the same defect as
selecting on the held-out operator, one level up. `TOOL2_BANDWIDTH_GRID = (0.25, 0.5, 1.0,
2.0, 4.0)`, as multiples of D33's median non-zero distance, recomputed per fold over that
fold's retained groups. D33's own default (1.0) is on the grid, so the grid does not move
the method's existing choice. Every point is reported under `TOOL2_REPORT_ALL_WEIGHTINGS`,
so it cannot be mined for a best case, and the verdict prints the span rather than the best
point.

#### 3. `TOOL2_REQUIRE_BOTH_VALID` is a per-fold pairing, and the difference is 53 points

The rule could be read as "average each arm over the folds it is valid on, then compare".
That compares two different fold sets, and it is exactly what let `nearest` post the
smallest area in D-070 Sec. 3 while under-covering in three folds of seven. Measured on a
four-fold fixture where Tool 2 is valid everywhere and the safe rule only in two:

| reading | saving against the trivially-safe rule |
|---|---|
| arm averages over all four folds | **+32.8%** |
| paired, restricted to the two folds where both are valid | **-20.8%** |

A 53-point swing, and the two readings disagree about the sign. The verdict pairs per fold
and prints `folds_both_valid` beside every row, because a headline resting on two folds is
not the same claim as one resting on seven.

**The pathology itself is now a test.** On the same fixture the two invalid arms post the
*smallest* areas of any arm -- 0.3100 and 0.0200 against the valid arms' 0.5100 and 0.8113 --
so ranking on area alone selects the broken ones. That is the whole content of
`TOOL2_REQUIRE_BOTH_VALID`, and it was a comment until now.

#### 4. Two implementation choices worth a line each

**Extended the existing harness rather than writing a second one.** `--t1c` swaps the arm
set inside `scripts/tool2_holdout.py` and leaves the Q4 path untouched. The reason is not
elegance: I cannot run any of this against a real bank here, so Track 2 runs it cold, and
the loading, fold, interiority and validity code in that script is the code they have
already run successfully. A fresh script would have put all of it back in play.

**The module is `calibration/tool2_baselines.py`, not `baselines.py`.** There is already an
`evaluation/baselines.py` -- non-conformal reference *detectors*, a different thing from
calibrators -- and two modules of that name degrade the file-parametrised test IDs to
`baselines.py0` and `baselines.py1`, which is unreadable in a failure report.

#### 5. What is needed, and what it cannot be

One bank pass, which is request one to Track 2:

```
PYTHONPATH=src python scripts/tool2_holdout.py runs/<bank> --t1c \
    --distances-csv runs/<...>/ct_distances.csv --out runs/t1c/lidc.csv
PYTHONPATH=src python scripts/tool2_holdout.py runs/<bank> --t1c --exclude-nearest 1 \
    --out runs/t1c/lidc_sparse.csv
```

The second form is D75's sparse-grid case and is not optional: on a dense grid the
weighting has little to win, and a result from the full grid alone would describe the grid.

**What this cannot become.** If Tool 2 is not valid at every unseen operator there is no
efficiency claim to make, and the verdict refuses to print one -- an area number beside a
validity failure is the `nearest` error again. If it is valid and saves nothing, that is the
reportable result and the headline says so.

Fingerprint `a2ab9a1fdfe4` -> `0ea2d2fa9b3a`. Suite 1,135 passed, 7 skipped.

### D96 - The dimensioned check could not fail, so it proved nothing; and two gates that disagreed with each other [T1-D, both banks, Track 2's D94 pass; amendment 15]

Track 2 ran D93 and D94 on both banks and returned level vectors, brackets, design
effects, the survivor's test, the quadrant panel and the face bank's spread ratio. Four of
those land on Track 1's side. One of them retracts a frozen claim made the previous day.

#### 1. The level I specified was k/n by construction, and the only computable version confounds the two counts

Amendment 12 required a **dimensioned** check beside any pass that came from a
dimensionless ratio. Amendment 13(b) declared that check run and passed, off an aside in
Track 2's D93 message: tau sat between the 0.03 and 0.18 levels of `F_a` over 13 splits,
and thirteen draws of `Beta(4, 38)` have median min 0.034 and median max 0.177 while the
region count gives [0.097, 0.103]. I wrote that it "passes at the subject count and
excludes the region count by a factor of order 300".

Track 2 implemented the wording I froze -- *the fraction of that split's calibration
positives at or below its threshold* -- and found it empty. `tau` **is** the k-th smallest
of those positives, so that fraction is `k/n` whatever the scores do. Measured over all 234
cells it runs 0.0990 to 0.1001. It is a constant and it carries no `n`.

The only version that carries anything is **test-side**: the split's test positives at or
below tau, which is `1 - matched coverage`. Its across-split variance is then a sum,

```
var[level] = var_Beta(n_calibration) + var_test(n_test)
```

and variances add, so the count read off the total lies **below both** counts. The estimand
is their harmonic combination, confirmed over 60 seeds a row:

| n_calibration | n_test | harmonic | median bracket | holds n_calibration | holds harmonic |
|---|---|---|---|---|---|
| 40 | 50 | 22 | (10, 50) | 67% | 78% |
| 5000 | 50 | 50 | (10, 100) | **0%** | 85% |
| 5000 | 3300 | 1988 | (1000, 2000) | 23% | 62% |
| 40 | 3300 | 40 | (10, 70) | 92% | 92% |
| 100 | 10000 | 99 | (15, 200) | 92% | 92% |

The decisive row is the second. Build the world the check exists to reject -- calibration
side **at** the region count, test side at subject scale, which is this bank -- and the
bracket comes back (10, 100) and **excludes the region count in 200 of 200 seeds, with the
region count true.** The check returns the same verdict whether the hypothesis is true or
false. Track 2's "region count excluded at 9 of 9 on each bank" is therefore a property of
the statistic and not a finding about the bank, and 18 of 18 across two banks adds nothing
to it.

The method recovers the calibration count only where the test side is much the larger (the
last two rows). On these banks both sides are subject-scale, which is the one regime where
the bracket looks right and establishes nothing.

#### 2. What this is an instance of

**Statistics trap twelve, and Track 1's ninth.** The eleven before it were, with one
exception, a choice made after seeing a number. This one was frozen in advance, named
before it was applied to the second bank, and carried its own refusal to gate. None of that
helped, because the defect was not in *when* it was chosen but in what it could return:

> Freezing a statistic in advance makes it honest about when it was chosen. It says
> nothing about whether it can come out the other way. Only a calculation under the
> hypothesis the statistic is supposed to reject can say that.

That sentence is now in `prereg.py`'s module docstring, and the test that should have been
written on 22 November exists as
`test_the_level_range_cannot_exclude_a_region_scale_calibration_side`. It builds the
rejected world and asserts that the check wrongly rejects it -- a test that passes by
demonstrating the method's blindness, so the retraction cannot be quietly undone.

Worth stating plainly: the check was biased **towards the answer I wanted.** Test-side
noise inflates the width, a wider width reads as a smaller count, and a smaller count is
what excludes the region count. Track 2 saw the shape of this and said so -- *"the widest
ranges are at the operators with fewest positives, so the test side's own sampling is in
the range and the brackets lean low"* -- in a caveat under item 2. The caveat was the
finding.

#### 3. The conclusion survives, on evidence that can fail, and it corrects the conclusion too

The claim 13(b) was taken to support -- that the Beta's reference unit is the calibration
subject and not the region -- stands on the **clustered bootstrap**, which resamples
subjects and so measures the design effect directly, and which `BETA_REGION_COUNT_USE` has
always named as the only certificate path. Track 2's item 4, matched cells, per operator:

| bank | effective n | design effect | transfer-cell design effect |
|---|---|---|---|
| CT | 13 to 154, median 44 | 47 to 186, median 125 | median 54 |
| face | 19 to 46, median 28 | 61 to 211, median 110 | median 29 |

It also corrects the conclusion. The effective count is **not one number**: it tracks each
operator's positive count across an order of magnitude. So `BETA_REFERENCE_UNIT` names the
**unit** -- what one independent draw is -- and may not be read as naming a count. Freezing
the count would repeat amendment 7's defect exactly: a constant where the right object is a
function of the bank, which is a different error from a wrong value. The 10-to-70 bracket
is struck from that constant's docstring, and D93's "near 45" and 174 were already gone.

Also fixed: `LevelRangeCount.excludes` returned False for a count that was never a
candidate. The obvious call is `excludes(bank_region_count)`, and a region count is never a
round number -- 5021, 3089, 8983 on this bank -- so the honest answer "never tested" was
being delivered as "not excluded". It raises now.

#### 4. The direction rule's note quoted the wrong correlation, on both banks

`GAP_DIRECTION_SOURCE`'s note said the two banks *order tau oppositely (rho -0.70..-1.00
against +0.976)*. Those are the **gap** correlations. The rule's input is
`rho(d, tau_b - tau_a)`, and D89's identity -- three lines above, in the same note --
forces the two to have opposite signs. Reproduced from both banks' `ordered_pairs.csv`:

| bank | rho(d, tau_b - tau_a) | rho(d, gap) | rule's call |
|---|---|---|---|
| CT (lidc_ct) | +0.546 to +1.000 | -0.657 to -1.000 | non-increasing |
| face-prior | -0.886 to -0.980 | +0.886 to +0.980 | non-decreasing |

Pooled on CT: **+0.049**, against the amendment-10 record of +0.05. Track 2's numbers
reproduce to three decimals on both banks. The sentence's conclusion is untouched -- the
banks do order tau oppositely -- but the figures attached to it were the wrong quantity,
and `GAP_DIRECTION_RULE` explicitly invites running the rule by hand from one CSV, so
anyone doing that would have found the opposite sign on both banks and concluded they had
erred. Corrected, with both columns now tabulated beside each other.

#### 5. The face bank's fit cannot discriminate, and the CT bank's disagreement is not optional to disclose

`GAP_FACE_BANK_PREDICTION` passed: face tau falls with the distance, unanimously, -0.89 to
-0.98 across four family cells. Run the rule on that and it calls a **non-decreasing** fit
-- the same shape as `GAP_PREDICTION_DIRECTION`, the Week-2 constant **refuted** on the CT
bank. On the face bank the rule's answer and the refuted constant's answer coincide.

So a face-bank isotonic fit agrees with the rule and with the refuted hypothesis at once
and tells them apart not at all, while the bank where they disagree is the bank barred from
reporting a direction. `FACE_BANK_DIRECTION_COINCIDENCE` is frozen: the coincidence must be
stated wherever the face fit appears, and the CT bank's opposite sign must be disclosed
beside it. The bar on the CT bank was a bar on letting it **set** the direction; it was
never a licence to omit that the two banks answer oppositely.

#### 6. `reportable=True` on all 72 rows of a bank stamped DEVELOPMENT ONLY

Two orthogonal senses of the word meet in `ordered_pairs.csv`. The per-cell `reportable`
column is **statistical** -- is this estimate determined well enough to quote. The
`claim_stamp` is **provenance** -- may any number from this bank be quoted at all. On the
face bank every one of the 72 rows carries `reportable = True` under
`claim_stamp = DEVELOPMENT ONLY`, so `df[df.reportable]` selects 72 development rows for
the write-up. `spread_ratio.log` says it outright: *NOT REPORTABLE - development only, no
number below may appear in the write-up.*

`t1a_safety_rule.py` now conjoins the column and keeps the statistical bit as
`reportable_statistical`, so an existing filter becomes conservative rather than wrong: on
a development bank it selects nothing. `CLAIM_STAMP_BINDS` records which gate wins, and
that a frozen prediction passing on a development-only bank has passed **on a
development-only bank**, stamp beside it.

**This is load-bearing for the paper, not a tidy-up.** CT is reportable for coverage and
barred from setting the direction. Face gives the direction and is development-only. As
things stand the direction rule has **no bank on which its output is both unbarred and
reportable** -- which is a different sentence from the one D93 recorded, that the rule "is
correct on both banks".

`scripts/restamp_bank.py` recomputes a bank's claims from its manifest alone, reads no row
and needs no pass. The face bank is `w07_ct_onboarding`, old enough that its manifest may
predate `reportable_claims` entirely, in which case D-018's boolean is giving all-or-nothing
and a re-stamp fixes it for free. If instead the prior is inadmissible for the modality,
the second bank can never carry coverage, and that is a fact the publication plan needs in
November rather than in December. Asked, not assumed.

#### 7. The survivor's test is flat, and its consolation does not replicate

D94 left one candidate for the extreme-pair compression: the matched cell is a rank
statistic of its own scores and so immune to a per-split refit, the transfer cell is not,
so refit wobble enters transfer cells only. The test was pre-specified -- re-estimate
`G_ab` per split from that split's own scores, read the across-split spread against pair
distance. Track 2 ran it. Reproduced here:

| bank | median sd of G(u0) | rho(abs distance, sd) | by distance quartile |
|---|---|---|---|
| CT | 0.0513 | **+0.087** | 0.045, 0.063, 0.055, 0.048 |
| face | 0.0365 | **+0.005** | 0.036, 0.038, 0.038, 0.040 |

Flat. **The candidate is refuted by its own pre-specified test**, which makes three
mechanisms dead by three different arguments: curvature by sign, test-side noise by sign,
refit wobble by flatness.

Track 2 offered one consolation -- on CT, `rho(sd of G, log measured/predicted) = +0.41`,
so the mechanism "is not unrelated". On the face bank that correlation is **+0.049**. It
does not replicate, so it is a CT coincidence and not a partial mechanism, and it is not
carried forward.

#### 8. What the second bank actually says about the discrepancy, which is not what the first said

The 1.61x was never a median. CT: median measured/predicted **0.97**, rho +0.977, log-log
slope 1.08, within 2x at 100% -- 1.61 was the single most distant pair. Face: median
**1.28**, rho +0.851, slope 0.80, within 2x at 89%. And the shape differs:

| bank | log(measured/predicted) by distance quartile |
|---|---|
| CT | -0.052, -0.029, -0.038, -0.018 |
| face | -0.012, **+0.161**, **+0.357**, **+0.529** |

On CT the discrepancy is flat in distance and the median sits slightly *below* 1. On face
it **grows monotonically with distance**, reaching 1.70x in the far quartile. So the second
bank turns what looked like one outlying pair into a trend -- and no mechanism triaged so
far has the right shape for it, since all three died and the two that die by sign die on
arguments that do not depend on the bank.

One decomposition is available without a new run, and it splits the face result in two.
`spread_ratio.log` already reports a count-matched subset -- pairs whose positive counts
are within 1.5x -- as a standard secondary line, so the cut is Track 2's and not one I
chose after seeing the trend. Applied to the face excess:

| bank | all 72 pairs | count-matched | rho(abs d, excess), matched | far quartile, matched |
|---|---|---|---|---|
| CT | 0.97 | 0.94 (n=18) | **-0.112** | -0.099 |
| face | 1.28 | **1.08** (n=28) | **+0.418** | **+0.524** (1.69x) |

So the two halves of the face result have different causes. Its **level** is largely
positive-count mismatch: matching the counts moves the median from 1.28 to 1.08, nearly
closing it. Its **distance trend survives the control** at rho +0.418, with the far
quartile still at 1.69x. On face, distance and count mismatch are themselves correlated
(rho **+0.550**, against +0.140 on CT), which is why the two had to be separated before
either could be read.

The honest status: the Beta law predicts the **ordering** of transfer spreads well on both
banks (rho +0.977 and +0.851), predicts the **magnitude** to within 8% once positive counts
are matched, and leaves **one open residual** -- a distance-growing inflation present on
the face bank at matched counts and absent on CT. That is a sharper open item than D94's
was, and narrow enough to be stated as a limitation rather than a mystery. No mechanism is
proposed: three have died, and inventing a fourth to fit a trend on one bank is where this
project's recurring failure lives.

#### 9. Provenance note

`lidc/spread_ratio.log` is stamped `prereg f3b3f1b6aed8` while everything else in the
archive is at `a2ab9a1fdfe4`, so that one file is the D92-era output and was not re-run.
The spread computation does not read any constant amendments 10 to 13 touched, so the
numbers stand; the stale stamp is noted rather than acted on.

#### 10. Answers to Track 2's two questions

**The frozen level estimate: test-side, for a smaller purpose than it was requested for.**
`level_calib` is `k/n` and is dropped. `level_pooled` mixes labels across splits -- face
runs 0.006 to 0.348 from label re-valuation -- and is a different quantity, not read.
`level_test` is the right column, and what it measures is the **matched coverage estimate's
own combined effective count**, which is worth having. It is not the dimensioned check and
may not be cited against the region count.

**The two-solve cos channel: accepted as proposed**, built into the bank writer when the
next bank is scheduled, three-solve form logged ungated, LIDC untouched. One fence: LIDC's
reconstructions are not kept, so the channel can never be back-filled there, and with face
development-only it would exist on no bank whose numbers may be published. Build it;
nothing may become load-bearing on it until a bank that carries coverage has it.

#### 11. Status

| Item | State |
|---|---|
| Dimensioned check via the level range (amendment 13b) | **Retracted**; the check cannot fail |
| Beta reference unit = calibration subject | **Stands**, on the clustered bootstrap |
| Effective n as one number | **Withdrawn**; 13 to 154 per operator, tracks positive count |
| Curvature as the discrepancy | Ruled out, by sign, D94 |
| Test-side noise as the discrepancy | Refuted by sign, D94 |
| Per-split refit wobble as the discrepancy | **Refuted by flatness**, both banks |
| Face excess LEVEL (1.28) | **Mostly positive-count mismatch**; 1.08 at matched counts |
| Face excess DISTANCE TREND | **Open, no candidate**; survives count-matching at rho +0.418 |
| Direction rule's note | Corrected; wrong correlation quoted on both banks |
| Direction rule reportable anywhere | **No**, pending the face bank's re-stamp |
| `reportable` vs `claim_stamp` | Conjoined in the writer |

Amendment 15; fingerprint `0ea2d2fa9b3a` -> `9c0221cae495`. Suite 1,138 passed, 7 skipped.

### D97 - My pre-specification passed and the claim is refused; and the grid cannot tell the distance from the knob [T1-B, T1-D, local, off Track 3's row 5; amendment 16]

Track 3 merged T3-A through T3-D (c4941df, 3,210 insertions, suite now 1,183). Two of
their results reach into Track 1's open items, and following one of them up produced a
worse finding about Track 1's own headline comparison.

#### 1. Track 3 hands over a candidate for the residual D96 left open

T3.6 row 5, measured on the LIDC bank and reported as a **negative** result for their own
purposes: within family cells the operator distance does not order the score-law shift
(rho -0.16 to +0.18 on 66 of 72 pairs), and **the law shift follows the view ratio** (up to
+0.92 within sparse view). Their phrasing: *"the distance counts absolute change."*

D96 left exactly one open residual -- the face bank's spread excess grows with distance at
matched positive counts, rho +0.418 -- and recorded that no fourth mechanism would be
proposed, because three had died and inventing one is this project's failure mode. Row 5 is
not me inventing one. It is another track's independent measurement, and it makes a
specific prediction: my predicted spread ratio is `f_b/f_a`, a **ratio**, so if the law
shift is driven by the view ratio while the distance measures absolute change, a ratio
prediction should fail exactly where the two diverge.

So the test was legitimate to run. It was pre-specified first, in writing, before any
number: a pass required (a) the view ratio to order the face excess at rho >= +0.40, at
least as well as the distance, **and** (b) the partial correlation of distance given the
view ratio to fall below +0.20. The pre-specification also barred me from trying any other
predictor on this residual in this round.

**Both conditions passed.** The view ratio orders the excess at **+0.429** against the
distance's +0.418, and the partial correlation of distance given the view ratio collapses
to **+0.068**.

#### 2. The pass is refused, because the pre-specification was defective

It asked the question in one direction only. Run symmetrically:

| face bank, count-matched (n=28) | raw rho | partial, given the other |
|---|---|---|
| abs distance vs excess | +0.418 | **+0.068** |
| abs log view ratio vs excess | +0.429 | **+0.128** |

Both collapse. That is what collinearity does: the two predictors correlate at **+0.916**,
and of the 28 count-matched pairs exactly **1** has distance and view-ratio ranks differing
by more than a quarter of the sample (3 of 72 unmatched). Neither variable survives
controlling for the other, and that is not evidence for either.

A partial correlation near zero means the two cannot be separated. It does not mean the
variable controlled for is the cause. My criterion (b) would have licensed the reading that
Track 3's row 5 explains my residual -- a tidy close, one day after D96 -- and the only
reason it did not is that I ran the direction I had not specified.

**This is the same failure shape as D96, one day later, moved up a level.** D96's defect
was a statistic that could not come out the other way. This one is a *test design* that
could not come out the other way, written by someone who had just spent a day on the first
defect. Freezing the test in advance did not help again, and for the same underlying
reason: what has to be checked is not when a criterion was chosen but whether it can fail.
The symmetric pair is now barred from being omitted anywhere in the project
(`SPREAD_RESIDUAL_UNATTRIBUTABLE`).

The residual therefore stays exactly where D96 left it: open, sized, consistent with Track
3's finding, and **not** explained by it.

#### 3. The same measurement is a harder limit on T1-B, which matters more

`GAP_PREDICTION_BASELINES` requires the distance to beat `acquisition_parameter`, and
`BASELINE_SCOPE` already restricts that baseline to within-family pairs because no scalar
knob orders a limited-angle operator against a sparse-view one. The freeze also already
called this "the objection that makes rho = +0.976 weak within a family". That was
qualitative. Measured, in the only scope where the baseline exists:

| family cell | pairs | rho(abs distance, abs log knob ratio) |
|---|---|---|
| sparse_view | 30 | **+0.923** (CT), **+0.957** (face) |
| limited_angle | 6 | **+0.970** (CT), **+0.970** (face) |

The baseline is a near-duplicate of the predictor it is meant to control, on both banks.
So **T1-B's within-family comparison is not identified**: whichever of the two wins, wins
by a margin that 30 and 6 pairs cannot resolve, and the result cannot be reported as the
measure carrying information beyond the knob.

The cause is the grid's shape, not the distance measure. The grid is two one-dimensional
arms -- sparse view turns the view count at a full arc, limited angle turns the arc at 120
views -- so within an arm the distance is nearly a monotone function of the one knob that
moved. It follows that:

- T1-B's claim has to live on the **36 cross-family pairs**, which is where no scalar knob
  is defined and so where the measure is not duplicated. That is half the grid, and it is
  the half `BASELINE_SCOPE` already identified as the interesting one.
- The remedy is a **grid change, not an analysis change**. Interior operators that move
  both knobs at once -- 90 views at 120 degrees, 60 views at 150 -- break the duplication.
- **It must be settled before the next bank is built.** No reanalysis of the existing 72
  pairs can recover a contrast the grid never varied independently, so this is a request
  with a deadline rather than a caveat.

#### 4. Track 3's other result, noted and not consumed

H-T3C offers a mechanism for the thing D89 recorded as unexplained and `GAP_FACE_BANK_PREDICTION`
still calls a bank property: *why* the two banks order tau oppositely. LIDC's
marginal positives are aligned (+0.57 to +0.65) with their score coupled to the hidden
truth and tau rises; the face bank's are orthogonal, edge-bound and decoupled and tau
falls. The pre-named falsifier -- LIDC orthogonal and truth-driven while tau rises -- did
not fire, and both banks fell in the regime assigned to them in advance.

Track 3 is careful about what that buys: *"H-T3C remains a description of two regimes,
consistent with six banks, not a law"*, and one sub-prediction failed as worded. Nothing in
Track 1's freeze changes on it. Recorded here because D96 Sec. 5 says the two banks' opposite
signs must be disclosed wherever the face fit appears, and H-T3C is now the thing to
disclose beside them -- as a hypothesis with a surviving falsifier, which is more than
"unexplained" and less than a mechanism.

#### 5. Status

| Item | State |
|---|---|
| Face residual explained by the view ratio | **Not shown**; both partials collapse at +0.916 collinearity |
| Face residual | **Open**, sized at rho +0.418, unchanged from D96 |
| T1-B within-family margin | **Not identified**; knob and distance +0.92 to +0.97 collinear |
| T1-B claim scope | The **36 cross-family pairs** |
| Grid remedy (interior operators) | **Needed before the next bank**, not re-analysable after |
| One-sided partial correlations | **Barred** project-wide for collinear predictors |
| Why the banks order tau oppositely | Track 3's H-T3C, falsifier did not fire; hypothesis |
| Merge of Track 3's T3-A..T3-D | Clean; suite **1,183 passed, 7 skipped** |

Amendment 16; fingerprint `9c0221cae495` -> `cc6bb1e631a7`. Suite 1,183 passed, 7 skipped.

### D98 - T1-D's last blocked item was already run, and its certificate number was in the wrong units [T1-D, both banks, off Track 2's D94 logs; amendment 17]

T1-D's remaining piece was worst-operator coverage -- *the minimum over cells, what a
deployment certificate needs* -- recorded as blocked on Track 2's per-cell output. It was
not blocked. `t1a_safety_rule.py` has called `worst_operator_coverage` since it was
written, and the numbers were sitting in the `t1a_d94.log` files in the archive, under a
heading that says `== T1-D worst cell ==`. Nobody read them, mine included; the work plan
went on listing the row as T2-A dependent.

#### 1. The number a certificate would have been written from was computed on independent regions

`worst_operator_coverage` fed Clopper-Pearson the raw `n_hallucinated` count, which treats
every hallucinated region as an independent Bernoulli trial. The design effects are in the
same dataframe, in a column the function did not read: **47 to 186** on the CT bank.

`BETA_REGION_COUNT_USE` has barred this since 22 November -- *no Beta band may set a gate;
the clustered bootstrap is the only certificate path*. The function violated a bar that was
already in force, for six days, because it called its band a **Clopper-Pearson interval**.
An exact binomial bound *is* a Beta quantile; the implementation calls `_beta_ppf` on the
next line. The bar and the violation were the same object under two names.

Recomputed in effective units, both banks, matched and transfer:

| bank | cell set | worst cell | observed min | shipped (regions) | corrected (effective units) | too high by |
|---|---|---|---|---|---|---|
| CT | matched, 9 | sv 120 | 0.7948 | 0.7725 | **0.3968** | 0.376 |
| CT | transfer, 72 | la 90 -> sv 120 | 0.2690 | 0.2398 | **0.0365** | 0.203 |
| face | matched, 9 | la 90 | 0.8551 | 0.8423 | **0.5185** | 0.324 |
| face | transfer, 72 | sv 180 -> sv 30 | 0.6660 | 0.6423 | **0.4286** | 0.214 |

**No verdict flips.** `certifiable_at_0.90` was already False in all four cases, so nothing
published is wrong today. But the numbers would have been quoted, and the failure mode is
live rather than hypothetical: a fixture that certifies at 0.90 on independent regions
**stops certifying** once the measured clustering is carried. That is a false certificate,
and it is now
`test_a_grid_that_certifies_on_regions_can_fail_on_effective_units`.

`design_effect_col` now takes no default and the function refuses rather than assumes --
the same refusal `TransferBetaReference.interval` makes, for the same reason, after the same
error. The independent-regions value is still computed and carried as
`simultaneous_lower_regions`, labelled as the idealisation, so the gap between the two is
visible instead of being a choice someone has to remember.

**T1-C is unaffected, and this was checked rather than assumed.** Its per-fold validity runs
through `cluster_bootstrap_coverage`, not through this function, so the thing built two days
ago does not share the defect.

#### 2. The third naming defect in three days

- **D95:** `max_quantile`, one name with two opposite readings -- with flags at `score >= tau`
  the largest and smallest per-group thresholds are different arms.
- **D96:** `reportable`, one word with two orthogonal senses -- statistical and provenance --
  which disagreed on all 72 face rows.
- **D98:** Beta band and Clopper-Pearson, two names over one object, where a bar on the first
  did not reach code that used the second.

The lesson is not to choose better names. It is that **a rule stated in one vocabulary does
not enforce itself over code written in another**, so a bar on a quantity has to name that
quantity's aliases. `EXACT_BINOMIAL_IS_A_BETA_BAND` now does: Clopper-Pearson by name, and
Wilson, Jeffreys and Agresti-Coull as the same error by whatever interval, since what is
barred is treating regions as independent trials rather than any particular formula.

#### 3. What T1-D now says, which is the project's thesis stated as a number

On the **reportable** bank, with clustering carried:

- At the **worst matched** operator, the simultaneous lower bound on coverage is **0.397**
  against a nominal 0.90. Conformal calibration on its own operator cannot be certified at
  0.90 here -- 41 subjects and a design effect of 186 at that cell leave the bound that
  wide. This is a statement about the bank's power, not about conformal prediction.
- At the **worst transfer** cell it is **0.0365**, with an observed minimum of **0.269**.

The 0.269 is the number that matters and it is not a width artefact: a threshold calibrated
on limited-angle 90 degrees and deployed on sparse-view 120 catches **27%** of
hallucinated regions where it promises 90%. That is a 63-point shortfall, measured, on the
bank whose coverage claims are reportable.

**That is the project's motivating claim, stated as the thing it was always supposed to be:
not that transfer degrades gracefully, but that a naive transferred certificate is wrong by
enough to be dangerous.** It is also exactly the gap Tool 2 (T1-C) has to close, which makes
T1-C's blocked bank pass the most valuable run outstanding -- there is now a measured
63-point hole for it to be measured against.

#### 4. Status

| Item | State |
|---|---|
| T1-D worst-operator coverage | **DONE on both banks**; was never blocked, only unread |
| Certificate bound's units | **Fixed**; design effect required, no default, refuses to guess |
| Worst transfer cell, CT, reportable | **0.269 observed**, 0.0365 simultaneous, against 0.90 |
| Worst matched cell, CT | 0.7948 observed, 0.397 simultaneous -- bank power, not a method failure |
| Any verdict changed by the fix | **No**; all four were already not certifiable |
| False-certificate path | **Closed**, and pinned by a test that fails if it reopens |
| T1-C | Unaffected, checked; its validity is clustered |

Amendment 17; fingerprint `cc6bb1e631a7` -> `f387c2b4f3ab`. Suite 1,191 passed, 7 skipped.

### D99 - T1-B's cross-family fit was inert for the wrong recorded reason, and there is a claim behind the face bank's stamp [T1-B, local reconstruction, face bank DEVELOPMENT ONLY]

D97 narrowed T1-B's claim onto the 36 cross-family pairs. The obvious objection is that
this is where D91 recorded the estimator as **inert** -- *"distance = constant = views/arc =
null-dim at MAE 0.0707 with delta exactly 0"* -- so the scope D97 points at would be the
scope where nothing can be measured. That objection is wrong, for a reason the work plan
had recorded incorrectly.

#### 1. The inertness is the direction, not the range

The work plan's caveat blames range: *"fitting on the 6 within-limited-angle pairs leaves
78-93% of the held-out points outside the training range, where a monotone fit is clamped
and is therefore a constant."* True for that one direction, and measured here on both
banks:

| train -> predict | CT, outside training span | face |
|---|---|---|
| sv->sv -> la->la | **0 of 6 (0%)** | **0 of 6 (0%)** |
| la->la -> sv->sv | 28 of 30 (93%) | 26 of 30 (87%) |
| sv->la -> la->sv | 5 of 18 (28%) | 4 of 18 (22%) |
| la->sv -> sv->la | 5 of 18 (28%) | 4 of 18 (22%) |

So range explains one of four directions and not the headline one: training on the 30
sparse-view pairs leaves **nothing** out of range.

The actual cause is the frozen direction. Measured on synthetic data with the CT bank's
shape -- the gap *falls* with the signed distance (D89) -- PAVA's non-decreasing fit
returns the training mean to machine precision:

| direction | fitted span | equals training mean |
|---|---|---|
| increasing (frozen) | **0.000000** | **yes, to 1e-12** |
| decreasing (the rule's call on CT) | 0.120984 of a 0.122199 response span | no |

`IsotonicFit`'s own docstring already said this would happen. Nobody connected it to the
"inert" verdict, and the work plan wrote down the wrong remedy as a result. **The remedies
differ:** a range problem needs a denser grid, and a direction problem is what amendment 7's
threshold-ordering rule already fixed. Pinned by
`test_the_frozen_direction_returns_the_training_mean_on_reversed_data`.

#### 2. So T1-B is blocked four ways, each for a different structural reason

| scope / bank | state | why |
|---|---|---|
| CT, frozen direction | inert | direction refuted; PAVA returns the mean exactly |
| CT, rule direction | barred | `GAP_DIRECTION_BANKS_BARRED` names `lidc_ct` |
| face, rule direction | live | but the bank is `DEVELOPMENT ONLY` for coverage (D96) |
| any bank, within-family | not identified | knob and distance +0.92 to +0.97 collinear (D97) |

The binding constraint is therefore **the face bank's claim stamp** -- not the grid, not the
estimator, not the distance measure. That makes the `restamp_bank.py` request from D96 the
gate on T1-B's only live magnitude claim, which is a stronger reason to prioritise it than
the one I gave Track 2.

#### 3. Is there a claim behind that gate? Indicatively yes, and it is fragile

Run on the face bank -- **DEVELOPMENT ONLY, no number here may be written up** -- to decide
whether the restamp is worth prioritising, which is a legitimate use of a development bank
and not a claim. Driven through `gap.py`'s own `folds`, `held_out` and `compare_to_baseline`
rather than a reimplementation, with `meta` rebuilt from the operator ids and D93's resolved
`null_space_dim = 102400 - 320*views`. The scope split comes out at 45 within-family and 36
cross-family, matching `BASELINE_SCOPE` exactly, and `acquisition_parameter` is NaN on
exactly those 36.

Cross-family scope, `delta = mae_baseline - mae_method`, so positive favours the distance:

| protocol | baseline | delta | interval | verdict |
|---|---|---|---|---|
| cross_family | constant | +0.0290 | [+0.0041, +0.0497] | distance_better |
| cross_family | null_space_dimension | +0.0228 | [+0.0118, +0.0342] | distance_better |
| cross_family | acquisition_parameter | -- | -- | undefined here (all 72 dropped) |
| leave_one_operator_out | constant | +0.0295 | [+0.0023, +0.0541] | distance_better |
| leave_one_operator_out | null_space_dimension | +0.0111 | **[-0.0005, +0.0246]** | indistinguishable |

**One protocol passes and the other fails, by a lower bound of -0.0005.** That is a hair,
and a hair is not a result. `GAP_PREDICTION_PROTOCOLS` freezes both protocols, so the
honest summary is: the cross-family claim is **live, borderline and protocol-dependent**,
and it would be decided by the real pipeline run rather than by this reconstruction.

Two reasons not to lean on these numbers harder than that. **The reconstruction is not the
run:** I set the matched diagonal's signed distance to 0.0 and rebuilt `meta`, both of which
touch the training sets. Its `acquisition_parameter` deltas come out at +0.0042 and +0.0101
against D93's recorded +0.006 [-0.001, +0.016] and +0.011 [-0.000, +0.024] -- same sign,
same magnitude, same verdict, which validates the shape and not the digits. **And the
within-family failure is the one D97 disqualified:** `acquisition_parameter` is the only
baseline the distance does not separate from, at [-0.0002, +0.0080] and [-0.0015, +0.0190],
and that comparison is exactly the unidentified one.

The scoped reading this rests on was **not** chosen now. Amendment 2, 20 November: *"
GAP_PREDICTION_BASELINES scoped by family and extended with null_space_dimension: the
acquisition parameter is undefined on cross-family pairs. Found from the completed distance
matrix, no coverage number in existence."* Verified in the trail rather than from the
constant's docstring, and pinned by a test that asserts the date and the
no-coverage-number stamp.

#### 4. What this does not license

D96's `FACE_BANK_DIRECTION_COINCIDENCE` still holds: on the face bank the rule's direction
and the refuted `GAP_PREDICTION_DIRECTION` are the same shape, so none of the above is
evidence for the frozen direction. It is a **magnitude** claim -- does the distance predict
the gap better than the baselines defined on this scope -- which is a different question
from the direction claim, and the two must not be run together in the write-up.

No amendment: nothing frozen changed. Two tests added.

#### 5. Status

| Item | State |
|---|---|
| Why the cross-family fit was inert | **Direction, not range**; work plan's reason corrected |
| Range on the headline direction | **0% out of span** on both banks |
| T1-B binding constraint | ~~**The face bank's claim stamp**, nothing else~~ **WITHDRAWN by D100 -- the block is bank disagreement; the stamp was never the only one** |
| Cross-family claim behind that gate | **Live, borderline**: 1 of 2 protocols, by 0.0005. *D100: on the face-prior bank this is a development value only, not an estimate of the LIDC effect* |
| Within-family claim | Not identified (D97), and it is the only baseline that fails |
| Evidence for the frozen direction | **None**, and face cannot provide it (D96) |

Suite 1,193 passed, 7 skipped. Fingerprint unchanged at `f387c2b4f3ab`.

---

### D100 -- The face-prior bank's scope stays as it is, T1-C's ceiling fails, and D99's "single binding constraint" is withdrawn (24 Nov)

Track 2's §1l answered the three questions from the 24 Nov reply and put one rule question
back with a default. Their work is at `ce6aa56`, pre-registration `cc6bb1e631a7`; this entry
is written against `f387c2b4f3ab` and leaves `4fdb8e4f658b`.

#### 1. The rule question, and the ruling

`restamp_bank.py runs/w07_ct_onboarding-0389934dc7d7aaed`, show mode:
`totalsegmentator + natural_image  claims [] -> none (unchanged)`. The important part is
Track 2's explanation, not the output: this is **not** a stale manifest. The re-stamp
recomputes the claims from the substrate, the sampler and the checkpoint, and
`prior_claims` returns nothing because a natural-image prior is not a medical prior for CT
(D80, P-18). D80's coverage-only case is a medical prior of *another anatomy* -- that is the
LIDC bank. A face prior is not that case.

So the question is a rule question: should a non-medical prior be admitted for coverage
claims? Track 2's default was "no change".

**Ruled: no change.** `PRIOR_ADMISSIBILITY_UNCHANGED`. The coverage mathematics is not in
doubt -- split conformal is distribution-free and does not care what the prior was trained
on. But a coverage claim is a claim *about a score distribution*, and what a face prior on
CT changes is what that distribution is. Admitting it would license quoting coverage for a
distribution that arises only from a prior-modality mismatch.

The reason to record the ruling rather than just make it: the request reached me as what
looked like the only route to a positive T1-B. A rule relaxed after learning which bank
holds the result is this project's recurring failure **one level up** -- not a statistic
chosen after seeing a number, but an admissibility criterion chosen after seeing which bank
gives the number. The ruling was therefore made at *apparent* cost. Section 3 shows the cost
was illusory, which is luck and not judgement, and is not a reason to trust the next such
ruling more.

#### 2. T1-C: the ceiling fails, so this bank cannot answer the question

Both runs D75 asks for, `tool2_holdout.py --t1c` unmodified, Track 2's 320^2 distances:

| run | interior folds | weighted | pooled | oracle | max_quantile |
|---|---|---|---|---|---|
| full grid | 7 | 3 | 3 | **4** | 7 |
| `--exclude-nearest 1` | 6 | 3 | 3 | **4** | 6 |

The script refused the efficiency claim, correctly. Where the arms are valid the weighted
arm flags 0.476 to 0.518 against pooled's 0.473 (0.501 to 0.515 against 0.500 sparse), so
the weighting flags **no less** area than pooling at any frozen bandwidth.

**The decisive cell is the oracle's, not the weighted arm's.**
`group_conditional_oracle` calibrates on the held-out operator's own calibration rows. It
is not deployable; it exists to show the headroom. It is invalid at 3 of 7 folds. That
means at least 3 of the 7 folds are shut to *every* operator-aware method, including a
perfect one, so "weighted reaches 3 where pooled reaches 3" is being read against a ceiling
of 4 and not of 7.

Three things were checked rather than assumed before reading that as real, because the
obvious suspicion was that it was my own script's defect:

- `split_threshold` applies the finite-sample correction as `k = floor(alpha * (n + 1))`
  and the docstring argues the flooring explicitly. It errs **conservative**, so the
  deficit is not the quantile convention.
- `valid` is `bool(metrics["hi"] >= nominal)` where `metrics` comes from
  `cluster_bootstrap_coverage` on `sample_id`. This is a **lenient** criterion on a
  **subject-clustered** interval: a thin fold has a wider interval and so passes more
  easily. A fold cannot fail by thinness alone, and failing means the whole clustered
  interval sits below nominal. This also confirms amendment 17's note that T1-C does not
  carry the region-count unit error -- it was recorded then and is re-verified here.
- The harness already refuses a fold whose clusters appear on both sides (line 467).

What is left is a **within-operator** deficit, where there is no operator shift to blame.
At the design effects this bank carries -- 47 to 186, median 125 -- a region-level
conformal threshold calibrated on some subjects does not deliver its coverage on others.
T1-C presumes within-operator calibration transfers across subjects; on this bank it does
not. `TOOL2_CEILING_FAILS`.

**This is a scope result, not a verdict on distance weighting.** The honest headline is
that the only arm valid at every unseen operator is `max_quantile`, which is valid by
construction and buys nothing, and that the comparison T1-C was built to make needs a bank
whose within-operator calibration holds.

#### 3. Why the stamp was never the only block, which withdraws a D99 finding

D99 §5 recorded "T1-B binding constraint: **the face bank's claim stamp**, nothing else".
**That is withdrawn.** `BANK_DISAGREEMENT_BLOCKS_T1B`. Track 2's count-match cut table is
what shows it: the two CT banks do not merely differ in power, they disagree in sign and in
kind.

| quantity | LIDC (coverage-stamped) | face-prior (unstamped) |
|---|---|---|
| rho(distance, tau_b - tau_a) | +0.546 to +1.000 | -0.886 to -0.980 |
| direction the rule then calls | non-increasing | **non-decreasing** |
| rho(distance, log spread ratio) by cut | 0.00, 0.00, -0.11, -0.09, +0.12 | +0.60, +0.52, +0.42, +0.28, +0.40 |

The bank that may be quoted and the bank that carries the effect return **opposite**
direction verdicts, and the spread trend present at every cut on one is absent at every cut
on the other. Lifting the stamp would have licensed quoting a magnitude measured where the
sign of the underlying relation is reversed. So the stamp was never the only block, and
repairing it would not have unblocked T1-B.

Track 3's re-runs corroborate this independently, and the reading is theirs to make: the
marginal positives' composition is nearly disjoint between the banks -- on LIDC score-free
0.83, under-recovery 0.12, edge 0.05; on the face-prior bank score-free 0.04,
under-recovery 0.00, edge 0.96. The positives are not the same kind of object on the two
banks.

One thing the cut table settles in the other direction: the face-prior **level** is count
mismatch (median measured/predicted falls 1.28 to 1.05 as counts are matched) while the
**trend** survives every cut. Only the trend was ever a candidate finding, and it still is
-- as a development value that motivates a measurement on an admissible bank, never as an
estimate of the LIDC bank's effect.

#### 4. The header boolean, fixed

`tool2_holdout.py` printed `reportable False` for the LIDC bank. Track 2 identified it
exactly: that is D-018's old boolean, which is true only for a bank carrying *every* claim,
so it reads False beside a bank that is coverage-reportable -- which is the certificate this
script exists to write. The **gate** was always right (line 375 tests
`carries(provenance, COVERAGE)`); only the header lied. It now prints
`describe(provenance)`.

This is the fourth naming collision in the family (`max_quantile`'s two readings,
`reportable`'s statistical and provenance senses, Clopper-Pearson as a Beta band, and now
the bank boolean against the claim set). Amendment 17's general form covers it: a bar
written in one vocabulary does not enforce itself over code written in another.

#### 5. Not acted on

The `err_free_two_solve` channel is built and checked -- three-solve reaches 1.0005, the
two-solve form's largest value is 0.999997 -- and rests on nothing yet, correctly: the LIDC
reconstructions are not kept so it cannot be back-filled, and the face-prior bank is
development-only. It becomes usable when a bank carrying coverage is written with it.

#### 6. Status

| Item | State |
|---|---|
| Face-prior bank's claims | **None, and the rule stands** (D100) |
| T1-C validity claim | **None**; 3 of 7, against a ceiling of 4 of 7 |
| T1-C efficiency claim | **None**; weighting flags no less than pooling at any bandwidth |
| Why T1-C fails | **Within-operator**, at design effects 47-186 -- a scope result |
| T1-B binding constraint | **Bank disagreement**, not the stamp (D99 §5 withdrawn) |
| Face-prior magnitude | Development value motivating a measurement; not an estimate |
| Face-prior level vs trend | Level is count mismatch (1.28 -> 1.05); trend survives every cut |

Amendment 18; fingerprint `f387c2b4f3ab` -> `4fdb8e4f658b`. Eight tests added.

---

### D101 -- Every operator-aware refinement is positive only on the bank that carries no claims; the core result is not (24 Nov)

D100 ruled on the face-prior bank and withdrew the claim that its stamp was T1-B's only
block. Following that through the older documents turned up something larger, and it is not
a new measurement -- every number below was already recorded. Nobody had put them in one
table.

#### 1. What §3c of `PROJECT_STATUS.md` is actually measured on

Section 3c is titled "Tool 2 -- the primary contribution -- works where it was argued to".
Its numbers come from `T2.9_track1_week10_runs.md` §3, and that document's command block
names the bank: `CT=runs/w07_ct_onboarding-<hash>`.

That is the **face-prior bank** -- `totalsegmentator + natural_image`, the bank Track 2
re-stamped this morning as `claims [] -> none`. The LIDC bank did not exist when §3c was
written: `PROJECT_STATUS.md` is dated 17 November and T2-A was built on the 19th. Both banks
have nine operators and therefore seven interior folds, which is why the two tables look
comparable at a glance and are not.

#### 2. The two banks, on every quantity where both have a value

| quantity | face-prior (`claims: none`) | LIDC (`claims: coverage`) |
|---|---|---|
| Tool 2 weighted arm valid at | **6 of 7** folds | **3 of 7** |
| the ceiling (own-operator calibration) | **6 of 7** | **4 of 7** |
| coverage-loss direction | loses as views are removed, 0.890 -> 0.693 | **over-covers**, 0.964 at 30 views |
| rho(distance, tau_b - tau_a) | -0.886 to -0.980 | **+0.546 to +1.000** |
| direction the rule then calls | non-decreasing | **non-increasing** |
| rho(distance, spread ratio), by cut | +0.60, +0.52, +0.42, +0.28, +0.40 | 0.00, 0.00, -0.11, -0.09, +0.12 |
| marginal positives: score-free / edge | 0.04 / **0.96** | **0.83** / 0.05 |
| coverage spread across 13 splits | 0.034 | 0.051 |

**Where both banks have a value they disagree in sign or in kind, without exception.** Two
of these were already written down as open puzzles -- T2.11 asks "why the direction reversed
with the prior" and says of the spread "why it is wider here is not established" -- and D100
added two more. The pattern is the finding: these two banks do not behave alike on anything
measured, so a result from one is not evidence about the other in either direction.

#### 3. The part that matters for what Track 1 can claim

Sorting Track 1's results by which bank carries them:

| result | bank | state |
|---|---|---|
| coverage fails under operator shift, graded | both | **holds** |
| group-conditional calibration restores nominal | LIDC | **holds**, 9 of 9 operators, 0.892 to 0.905 |
| signed distance orders the gap, rho = +0.976 | face-prior | 8 pairs, 13-split means |
| T1-A safety rule | LIDC | **REFUTED** (D89): pooled false-safe 0.361 [0.09, 0.67] against trivial 0.319 |
| T1-B gap prediction | LIDC | reportable on **0 of 4** cells (D91); blocked entirely (D100) |
| T1-C distance-weighted recalibration | LIDC | **no validity claim, no efficiency claim** (D100) |
| T1-D worst transfer cell | LIDC | catches **26.9%** where it promises 90% |

So the honest statement is narrower than "the project has positives and negatives", and it
cuts in a specific place:

- **The core result survives on a bank that may be quoted.** Coverage fails under operator
  shift and group-conditional calibration restores it at all nine LIDC operators. That is
  the central claim, it is on the coverage-stamped bank, and nothing in D89 to D101 touches
  it.
- **Every operator-aware *refinement* beyond that is positive only on the face-prior bank
  and negative or absent on the LIDC bank.** The distance ordering, the safety rule, the gap
  prediction and the distance-weighted recalibration are four separate elaborations of the
  same idea -- that the *amount* of the shift is usable -- and the admissible bank declines
  all four.

This is a sharper reading of the project's standing scope limit, not a new defect. It says
the surviving contribution is the demonstration and the fix, and that the operator-aware
layer is, on present evidence, unsupported where it would count.

#### 4. What §3c's table could not show even on its own bank

Two defects in the comparison itself, both recorded after §3c was written and neither
carried back to it:

- **The baseline is `nearest`, retired by D-070 §3.** It "posted the smallest flagged area
  of any arm while averaging coverage 0.547", so beating it on coverage while flagging more
  area is not an efficiency result. The replacement baselines (`pooled`, `max_quantile`,
  `group_conditional_oracle`) did not exist yet.
- **One `weighted` arm is reported, not a bandwidth grid.** `TOOL2_REPORT_ALL_WEIGHTINGS`
  and D95 exist because reporting a single bandwidth after seeing which one won is
  cosmetic selection. §3c's 6 of 7 is one bandwidth.

So §3c is not merely measured on the wrong bank for a claim; its comparison would not be
reportable now on any bank. It is left in place and marked, not deleted -- the run happened
and the numbers are real development output.

#### 5. Status

| Item | State |
|---|---|
| §3c's bank | **The face-prior bank** (`w07_ct_onboarding`), claims none |
| Banks agreeing on any measured quantity | **None found**, 8 checked |
| Core claim (failure + group-conditional fix) | **Holds on the coverage-stamped bank** |
| Operator-aware refinements (T1-A, B, C, distance ordering) | **0 of 4 supported on that bank** |
| §3c's own comparison | Retired baseline + single bandwidth; unreportable now on any bank |
| New measurements taken | **None.** Every number here was already recorded |

**Amendment 19, against my own first judgement.** I closed this entry "no amendment:
nothing frozen changed, and no statistic was computed". The second half is true and the
conclusion did not follow from it. This narrows what may be claimed -- it bars citing either
CT bank as evidence about the other -- and the risk it guards is concrete: T1-F is a paper
draft, and §3c's 6-of-7 is the most attractive Tool 2 number in the project. A banner in a
markdown file does not stop a draft reaching for it; `BANK_RESULTS_ARE_NOT_INTERCHANGEABLE`
does, and the bar is written to cut both ways, so T1-C's within-operator failure may not be
cited as refuting the method either.

Fingerprint `4fdb8e4f658b` -> `ba2ab7d6e338`. Four tests added, one of which asserts the
§3c banner is still in `PROJECT_STATUS.md` -- a reader reaching for the Tool 2 number opens
that file, not this one. `PROJECT_STATUS.md` patched at §1 (two rows), §3c, §5 (section
claim and bank table) and §6.

### D102 -- The falsification I shipped with the ceiling reading fired, and following it took down D97's shelter for T1-B as well (24 Nov)

Track 2 delivered `runs/t1c` with the CSVs and logs (T2.11 §1m, `3a53922`, 1,210 pass,
merged at `f608eac`), plus a `w13` dry-run grid check I had asked for. Everything below is
measured off those files. **The face-prior bank was deliberately not consulted**, because
`BANK_RESULTS_ARE_NOT_INTERCHANGEABLE` binds me while I am the one using it.

#### 1. The ceiling reading is withdrawn, on the test it was issued with

D100 read T1-C's headline as *"the undeployable ceiling reaches only 4 of 7, so at least 3
folds are shut to every operator-aware method."* I sent it to Track 2 with a falsification
attached: *if the weighted arm is valid anywhere the oracle is not, the oracle is not a
per-fold bound and I withdraw the ceiling reading.*

`weighted_h0.25` is valid at **la120** and **la90**, where the oracle is not. So it goes.

Track 2's mechanism is right as far as it goes — the arms pass for opposite reasons, the
oracle covering 0.859 and 0.848 at those folds while weighted/pooled over-cover at 0.98 to
1.00 — but it does not say *why* the oracle fails exactly there, and that turns out to be
the whole story.

#### 2. What replaces it is not the opposite headline either

Track 2's own summary — *"some operator-aware arm is valid at 6 of 7; only sv45 has none"* —
is read off the same criterion and fails the same way. Four measurements on the delivered
CSVs:

| what was measured | value |
| --- | --- |
| own-operator arm, point coverage across all 7 folds | **0.8520 ± 0.0144**, below nominal at **7 of 7**, one direction |
| its limited-angle range vs its sparse-view range | 0.8478–0.8666 vs 0.8252–0.8664 — **overlapping** |
| its valid/invalid flips sorted by interval half-width | the **3 narrowest intervals are exactly the 3 invalid folds** (2 of 2 in the sparse run) |
| interior cells landing within 0.02 of nominal | **1 of 56**, and from below |
| valid stamps whose *point* coverage is under nominal | **8 of 29** |
| sparse-view folds where any of the 8 arms reaches nominal | **0 of 4** (best 0.8959) |

The own-operator arm sits 4.8 points low at every fold, with a spread (sd 0.0144) thirteen
times tighter than pooled's (0.1857), and with the two operator families overlapping
completely — so the operator explains none of it.

> **Corrected the same day, before any commit.** I first wrote that this was *"a
> near-constant offset that the treatment variable does not move, so a calibration defect
> rather than a shift effect"*, and that no arm's deficit was attributable to operator shift
> until a cheap same-operator run resolved it. **That run already exists, and Track 2 named
> the section in the same message that carried the table: T2.11 §1e.** Every T1-C number is
> on the **registry split**, which ranks 1 to 4 of 13 draws at every operator. The same arm
> over 13 patient splits averages **0.897 — nominal at all nine to within 0.008.** Split
> conformal promises coverage over the draw of the calibration set, and the spread comes
> from the 41 calibration subjects a test-side interval never resamples. My own standing
> rule covers this exactly: read the partner document that already has the data.

So **nominal 0.90 is the wrong reference for any T1-C cell.** Against the matched control
measured on the *same* split, the own-operator arm meets or beats it at **7 of 7** (mean
**+0.0195**, from +0.0005 at la90 to +0.0578 at sv90). The arm is sound; it was being held
to a level this split does not sit at.

**The verdict survives, which is why this is a correction and not a retraction.** Against
the same-split matched control, `pooled` and `weighted_h0.25` reach it at **3 of 7** — all
three limited-angle folds and **0 of 4** sparse-view — while `max_quantile` reaches it at
7 of 7. No efficiency claim is rescued, and the shape of the T1-C result is unchanged.

What the correction does sharpen is §2's other finding: with a common offset across folds,
interval width is the only thing left that can vary, which is exactly why the valid/invalid
flips sort by it.

And the validity criterion is not measuring methods. Its flips sort *perfectly* by interval
width in both runs. `valid` is `hi >= nominal`, so at fixed point coverage a fold passes or
fails on how wide its bootstrap interval is — i.e. on n. "Valid at k of 7" is a sample-size
statistic.

One further thing, recorded so it is not mistaken for a result later: within a fold there is
**one** deployment operator, so every arm is a single threshold there and all eight lie on
that fold's own area–coverage curve by construction. Checked anyway — 0 of 42 interior cells
clear their own interval half-width above the chord of their area-neighbours, the largest gap
being +0.0135 against a half-width of 0.0614. That is the design, not a finding, and it is
why **an arm ranking is not available from this experiment at all.**

#### 3. A route to T1-B appeared on existing data, and it is the seventh recurrence

The bank of record has four operators at **120 views** (`sv v120.a180` and the three
`limited_angle v120.a{90,120,150}`). Across their six pairs the view count and the
null-space dimension are *both* constant, so a predictor built from "how many views" is
constant by construction. That is the decorrelated subset T1-B has been missing, and it
already exists. Direction pre-stated before running: greater distance → larger `|dtau|`.

ρ(distance, `|dtau|`) = **+0.9429**, exact p = 0.0167.

It is not a result, for two independent reasons, and both had to be checked for:

1. **The rival predictor ties it exactly.** ρ(distance, `|log arc ratio|`) = **+1.0000** on
   those six pairs — the geometric distance and the arc ratio rank them *identically* — and
   the arc ratio scores the same **+0.9429** against `|dtau|`. Holding views fixed leaves
   arc as the only thing still moving, so the subset swaps the views confound for the arc
   confound.
2. **The number is the statistic's own ceiling.** Six pairs built from four operators are
   not six independent observations; the null must permute the four operator *labels*
   (4! = 24), not the six pair values (6! = 720, which is what produced the 0.0167).
   Under the correct null, +0.9429 is reached by **1 of 24** relabellings and never exceeded.
   The estimate *is* the attainable maximum and p = 1/24 is the test's floor.

Another recurrence of the statistics trap, and the first caught by a rival-predictor check
rather than by a sign — worth noting, because a sign check would have passed this one.

#### 4. Following that down took two more of my own entries with it

`BASELINE_COLLINEARITY` said: *"the cross-family cells are where the measure is not
duplicated by a knob, because no scalar knob is defined there -- those 36 pairs are where
T1-B's claim has to live."* D97 built on it. It is wrong.

A scalar **is** defined there. Every operator in the grid has both a view count and an
angular range; it is only the *implemented* `_acquisition_knob` that is undefined across
families, because it switches meaning by family and so ends up comparing 120 views against
90 degrees. Read the arc uniformly — sparse-view operators span 180° — and the baseline
exists on all 36 pairs. Measured there:

| predictor on the 36 cross-family pairs | ρ with `\|dtau\|` |
| --- | --- |
| `\|log arc ratio\|` | **+0.8524** |
| operator distance | **−0.5046** |
| `\|log views ratio\|` | −0.4592 |

A three-level scalar read off the operator name predicts the threshold gap better than the
geometric measure, in the direction the hypothesis wants, while **the distance runs
backwards**. The arc ratio's ties are real and stated — 3 values, 12 pairs each — and its
group means separate cleanly (0.2395, 0.3682, 0.7250). Transferred coverage is predicted by
none of the three (−0.05 to +0.09), consistent with the over-covering already recorded.

So **T1-B on this bank is not merely unidentified, it is refuted on its own response
variable in its own designated cell.** That is stronger than the blockage D97 recorded, and
it replaces it.

#### 5. The grid: ruled not to be built, and what to build instead

Track 2 fixed my requested operators in a config, dry-ran them, and — the part that matters —
re-ran the distance statistics before building anything. That discipline is what produced
this ruling, and it should be the default for every future grid request.

- **w13 is not to be built for this purpose.** Adding `v90/a120` and `v60/a150` moves
  ρ(distance, null-space dimension difference) from **+0.950** over 72 ordered pairs to
  **+0.952** over 110. Thirty-eight new pairs and about **25 GPU hours** for 0.002 on the
  statistic that does the blocking. What they *do* fix — arc duplication inside
  `limited_angle`, +0.970 → −0.307 — was never the binding confound. My request named the
  wrong target.
- They also land next to the sparse-view operators in distance (0.061 from `sv60`, 0.099
  from `sv90`), so they add weight to the existing views axis rather than opening a new one.
- **What T1-B needs is a family at the same view count *and* the same angular range,**
  differing only in which views are taken — uniform, non-uniform, disjoint blocks, half-step
  offset, all at e.g. 120 views over 180°. There both trivial scalars are constant by
  construction, so the distance is not a rank-duplicate of anything available for free.
  Check it on distances first, as w13 was checked.
- **The single acquisition knob is retired.** `_acquisition_knob` returns arc for
  `limited_angle` and assumes 120 views, giving knob difference 0 for `v90/a120` against
  `v120/a120` at 4 of 20 pairs. A knob that collapses distinct operators to zero makes the
  trivial baseline artificially *weak* and so flatters the distance. Frozen as the tuple
  **(views, arc, null-space dimension)**. The fix moves against T1-B, which is the only
  direction a baseline should ever be fixed in.
- `null_space_projection`, which Track 2 measured without being asked, is the one candidate
  that is not a restatement of a knob: +0.201 against views and +0.369 against null-space
  dimension over 110 pairs, against growth's +0.920 and +0.952, and it ranks pairs
  differently (ρ = +0.317 between the two measures). But its dynamic range is a third of
  growth's — cv 0.127 against 0.378, max/min 1.94 against 9.36 — so part of that
  decorrelation is compression. **A candidate to test against the gap, not a fix to adopt.**

#### 6. Status

| item | after D102 |
| --- | --- |
| T1-C ceiling reading (D100) | **Withdrawn** on its own pre-issued falsification |
| T1-C as an operator-shift experiment | **Readable, against the right reference.** Its cells sit on the registry split (low draw, §1e), so they must be compared to the same-split matched control, not to 0.90 |
| "valid at k of 7", either direction | **Not a method statistic** — the flips sort by interval width |
| T1-B on the LIDC bank | **Refuted**, not blocked: the arc ratio beats the distance at +0.8524 vs −0.5046 |
| D97's cross-family shelter | **Withdrawn** — the knob is undefined only as implemented |
| w13 build | **Ruled out for this purpose**; +0.950 → +0.952 for ~25 GPU h |
| Acquisition knob | Retired as a scalar; frozen as (views, arc, null-space dimension) |
| New bank passes requested | **None.** Everything here is off files already delivered |

**Amendment 20.** `GRID_CANNOT_DECORRELATE_DISTANCE` added, `TOOL2_CEILING_FAILS` rewritten,
`BASELINE_COLLINEARITY` corrected in two places. Fingerprint `ba2ab7d6e338` →
`979a8dab4176`. Nineteen tests added, including ones that recompute the +1.0000 arc
duplication and the 1-in-24 ceiling from the stored numbers rather than asserting the prose.

**No run is requested.** The diagnostic this entry originally asked for had already been
done — T2.11 §1e's 13-split table — which is recorded above as the correction. Nothing in
D102 waits on Track 2, and nothing waits on a bank pass.
