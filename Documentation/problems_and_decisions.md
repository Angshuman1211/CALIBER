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

### Q6 — Does `s_free` fire on detailed anatomy rather than on hallucination? [OPEN, owned by T1.1/T1.4]
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
