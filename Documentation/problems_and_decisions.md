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
