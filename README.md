# CALIBER

**Operator-shift-robust conformal hallucination flagging**
Distribution-free reliability certificates for generative-prior image
reconstruction that remain valid when the acquisition changes between
calibration and deployment.

> Existing conformal reliability certificates for image reconstruction assume
> the acquisition operator is fixed. This project shows the guarantee breaks
> when the operator shifts between calibration and deployment, restores
> validity through group-conditional calibration and operator-distance-aware
> recalibration, demonstrates the result across two imaging modalities with
> structurally different operators, and derives a coverage-gap bound expressed
> in quantities computable from the operators themselves.

This repository is **Track 2 — Systems, Data Engineering and Cross-Domain
Validation** (Blueprint Sec. 18; Track B in the weekly cards). It owns the
reconstruction and evaluation infrastructure that Tracks 1 and 3 build on.

Track 1 (methodology, calibration, theoretical guarantees) lives alongside it
in [Documentation/](Documentation/) - see
[Documentation/problems_and_decisions.md](Documentation/problems_and_decisions.md)
for the Track 1 decision log and the cross-track contracts it depends on.

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

## Layout

The package mirrors the five-stage method pipeline of Blueprint Sec. 10.

```
src/caliber/
  operators/    the known acquisition operator A, its range/null-space split,
                the shared operator registry, distance measures     Stage 2
  solvers/      pretrained diffusion and flow-matching back-ends behind one
                black-box posterior-sampler interface               Stage 1
  scoring/      region definition, operator-aware scoring, and the
                ground-truth hallucination criterion                Stage 3
  scorebank/    the cached, shared data format all three tracks read
  calibration/  group-conditional and weighted recalibration        Stage 4
                (Track 1 owns this module)
  evaluation/   coverage, efficiency, detection, gap-vs-distance    Stage 5
  data/         dataset access and preprocessing

configs/        version-controlled experiment definitions
docs/           interface specs, data access, cross-track decision log,
                Track 2 problems log
scripts/        CPU-only analysis entry points
tests/          correctness gates
```

## The two ideas the codebase is built around

**1. Compute once, analyse forever.** Reconstructions and scores are produced
once on a GPU and cached as a *score bank*. Every subsequent calibration
experiment, coverage curve, ablation, and numerical check reads the cache and
needs no GPU at all (Sec. 17). This is what makes the scope affordable on a
6 GB laptop GPU, and what lets all three tracks work in parallel rather than in
series. See [docs/score_bank_spec.md](docs/score_bank_spec.md).

**2. One definition of every operator.** Operators and the distance between
them are defined in a single registry used identically in MRI, CT, and the
theoretical derivations (Sec. 19). Week 9 audits this explicitly. See
[docs/operator_registry_spec.md](docs/operator_registry_spec.md).

## Invariants the code enforces

These are structural, not conventions — each is enforced in code and covered by
a test.

- **The solver is held fixed within a coverage curve** (Sec. 13). A config
  listing multiple solvers is rejected, and any score bank whose `run_id` spans
  more than one solver fails validation. The Week 10 solver ablation runs one
  experiment per solver and merges afterwards.
- **Operators pass a health check before any production run.** A wrong adjoint
  or a non-idempotent projector corrupts every score in the bank while leaving
  the numbers looking plausible, so both are gated on every run.
- **Ground truth never reaches the score.** Score components are functions of
  the measurements alone; the true image enters only through the `err_*`
  evaluation columns. The method must work without it — that is the premise
  (Sec. 4.1).
- **Approximations are recorded as data, not comments.** `exact_pinv: false`
  and `null_space_dim_method: dimension_count` travel with every CT operator,
  so the caveats reach the manifest and the write-up.

## Track 2 tasks

| Task | Deliverable | Status |
|---|---|---|
| T2.1 Data & operator pipeline | Dataset access, preprocessing, reproducible operator generation | operators done; fastMRI downloading (D-017) |
| T2.2 Solver integration | Pretrained back-ends behind the common interface | interface fixed; back-ends Week 2 |
| T2.3 Score-bank production | Cached reconstructions and scores across the grid | schema v0 done |
| T2.4 Evaluation harness | Coverage, efficiency, region-level detection | metrics done |
| T2.5 Cross-domain experiments | Protocol in CT plus appendix domain | CT operators done; data Week 7 |
| T2.6 Baseline reproduction | Published comparators under the shared protocol | Weeks 3-6 |
| T2.7 Figures & packaging | Visualisations and reproducibility package | Weeks 10-11 |

## Where things stand

**Done.** Operator layer for both modalities with exact adjoints and health
checks; all five candidate distance measures; score-bank schema, writer, and
validator; region grid and operator-aware scoring; evaluation harness; config
system with the controlled-comparison guard; runner; synthetic and open-data
development substrates; 66 tests. The CT path runs end to end on real LoDoPaB
data (`configs/experiments/w07_ct_smoke.yaml`).

**Data.** All CT data is local and structured. fastMRI+ annotations and an
open-licensed diffusion prior are fetched. fastMRI raw k-space is downloading
off-machine (D-017), with one licensing check still open there.
`python scripts/fetch_data.py --list` shows the full inventory,
`--verify` checks what is already on disk for truncation.

**Development substrate.** M4Raw (open, CC-BY, same `.h5` layout) carries MRI
pipeline work while fastMRI lands — wired as the `m4raw` dataset, with the one
gap that it has no pathology annotations (D-015). It cannot carry a reported
result, and the code says so rather than trusting anyone to remember: substrates
and samplers that are development-only are marked in code, and every manifest
records whether its bank is reportable (D-018).

**Three findings from Week 1 that need a decision, all in
[docs/decision_log.md](docs/decision_log.md):**

- **D-020 — the v0 label threshold is calibrated to phantoms.** On real MRI
  k-space, `err_free > 0.10` labels 100% of regions hallucinated, so region FDR
  is identically zero and the flag says nothing. MRI's own decomposition is
  sound (ratio 0.97, against CT's 1.33), so this is the threshold, not the
  projector. The Week 2 criterion cannot be an absolute constant.

- **D-012 — the CT null projector amplifies.** FBP is not the pseudo-inverse,
  so `‖P_N x‖/‖x‖` reaches 1.74 at 30 views when an orthogonal projection can
  never exceed 1. Worse, it degrades monotonically *in the same direction as
  operator distance*, so a CT coverage curve could look monotone because our
  decomposition degrades, not because reliability does. Needs a
  conjugate-gradient pseudo-inverse before any CT coverage curve is reported.
- **D-011 — randomised spectral estimation barely separates CT operators.**
  The decay exponent moves 1.688 → 1.676 between 180 and 20 views. L2 leans on
  spectral decay; Track 3 should know this before Week 6.

**Next.** Wire DPS and FlowDPS behind the existing interface (Week 2, T2.2);
run the Week 1 kickoff from [docs/week1_kickoff.md](docs/week1_kickoff.md),
which collects every question needing a cross-track answer — the eight spec
sign-off items, D-013's blocking choice of CT substrate, and the referral of
D-011 and D-012 to Tracks 3 and 1.

## References

Blueprint v3.0 (July 2026); Computer Vision Weekly Cards MASTER.
Datasets: fastMRI [28], fastMRI+ [29], LoDoPaB-CT [30], TotalSegmentator [31].
Solvers: DPS [23], DDNM [25], FlowDPS [26].
Conformal: [1-10], with non-exchangeable conformal [3] underpinning the
theoretical programme.
