# CALIBER

**Do distribution-free reliability certificates for image reconstruction survive a change of
acquisition operator?** They do not, and this repository measures how badly, what fixes it,
and where the obvious extensions of that fix stop working.

---

## The problem

A generative prior reconstructing an image from incomplete measurements has to invent the part
the measurements do not determine. Sometimes it invents structure that is not there — a lesion
that does not exist, an edge that is not an edge — and because the invention lives in the
forward operator's null space, **no data-consistency check can see it.** The reconstruction
agrees with the measurements perfectly and is still wrong.

Conformal prediction offers a way to flag such regions with a guaranteed miss rate and no
distributional assumptions, and a growing literature applies it to reconstruction. That
guarantee rests on exchangeability between the data used to calibrate the flag and the data it
is deployed on. **Changing the acquisition — a different undersampling mask, fewer projection
angles, a narrower angular arc — breaks that assumption, and the guarantee fails silently:
the flag still reports 90% reliability and no longer delivers it.**

This is not a hypothetical. Retrospective undersampling means the operator is a design choice,
and a calibration set collected under one protocol is routinely used under another.

## What this work establishes

**1. The failure is real, large, and graded.** On sparse-view and limited-angle CT with a
chest-CT diffusion prior, a conformal threshold transferred between acquisition operators falls
below its nominal 0.90 coverage at **36 of 72 ordered operator pairs**, at 22 of them below
0.80. Read as a certificate rather than as an average: in the worst pair, a flag promising to
catch 90% of hallucinated regions **catches 26.9%** — a 63-point shortfall. On accelerated
brain MRI with a brain-MRI prior, transfer from a 1-D mask to 2-D variable-density masks covers
**0.000 to 0.159**.

**2. Conditioning on the operator restores it.** Group-conditional (Mondrian) calibration —
one threshold per operator, exact under exchangeability within each group — reaches nominal
coverage at **all nine CT operators**, 0.892 to 0.905 as the mean over 13 patient splits, and
0.80 to 0.91 on MRI. The cost is real and reported: flagged area and region-level false
discovery rate both rise. This is the textbook estimator applied to a new partition; the
contribution is the demonstration and the measurement, not the estimator.

**3. Conditioning on *how far* the operator moved does not work.** This is the part that was
expected to be the main contribution, and it is a measured negative. Four refinements were
specified in a frozen configuration *before* the evaluation data finished being produced, each
with the baselines it had to beat named in advance:

| refinement | result |
|---|---|
| Pre-deployment safety rule from the operators alone | **refuted** — false-safe rate 0.361 against a trivial rule's 0.319 |
| Monotone prediction of the coverage gap from operator distance | **worse than a constant** out of sample, −0.018 [−0.026, −0.006] |
| Distance-weighted recalibration | **no validity claim and no efficiency claim** |
| The distance ordering itself | the **angular range read off the operator's name** outpredicts the geometric distance, +0.85 against −0.50, the distance running backwards |

An independent analysis of the score distributions reaches the same verdict from the other
side. **Operator identity is what conditioning can use; operator distance, on this grid and
this evaluation set, is not.**

**4. A coverage-gap bound that holds.** The gap is bounded by the Kolmogorov distance between
the label-conditional score laws, with a reduction to operator distance under stated
assumptions. The matched form holds at **72 of 72 ordered pairs** and is close to attained
where under-coverage is real.

**5. A caution about how this field validates uncertainty maps.** The standard protocol is to
correlate posterior standard deviation with absolute error and draw an ROC for high-error
regions. Run beside a **scale-only null** that reads no posterior information at all — just
the magnitude of what the prior invented — the protocol is **outscored by that null at 9 of 9
operators on two banks whose prior does not match the anatomy.** Where the prior does match,
the posterior wins, but it contributes 0.04 to 0.12 AUROC over the null rather than the 0.92
to 0.94 the protocol reports.

The mechanism is an exact identity. With `s` the reconstruction's hidden-component magnitude,
`c` the truth's, and `θ` their alignment, `err² = s² + c² − 2sc·cos θ`. At `cos θ ≈ 0` the
error *is* essentially a scale, so anything monotone in that scale scores well. And `cos θ`
turns out to be set by whether the prior matches the anatomy — median **+0.07** under a
natural-image prior, **+0.60 to +0.88** under matched medical priors, in both modalities.

**An evaluation needs three extra columns to tell those regimes apart:** a scale-only null,
the margin over it rather than the level, and an alignment measure. This result does not depend
on our own score being any good, which is the main reason to read it.

## Scope, stated precisely

Precise scope-limiting is part of the contribution, so these are stated up front rather than
buried.

- **Known linear/affine forward operators** at both calibration and deployment — true for
  retrospective MRI undersampling and for CT projection geometries, where the mask or view set
  is chosen. Not unknown or estimated operators.
- **The guarantee is exact only in the discrete known-operator regime.** For continuous
  operator change the target is an explicit coverage-*gap bound*, not exact coverage.
- **Operator shift is not covariate shift.** It changes the measurement process itself, so the
  textbook weighted-conformal correction does not transfer, and no new exact finite-sample
  guarantee is claimed for arbitrary shift.
- **Coverage is a mean over calibration draws, and a single draw can be much worse.**
  Per-draw standard deviation 0.051, individual draws from 0.795 to 0.991. A deployment that
  draws one calibration set can land near 0.80.
- **Two generative priors, and results are not interchangeable between them.** The
  natural-image-prior banks and the matched medical-prior banks disagree in sign or in kind on
  every quantity where both have a value, including the direction of the gap-distance
  relationship. Every number in this repository is attached to a named bank, and a bank's
  manifest records which claims it carries.
- **No external validation of the hallucination criterion.** An independently published
  patch-wise measure covers the same modalities and regimes and has not been run against ours.
- **One solver family carries the results**, one operator grid, single-coil MRI, and the CT
  forward operator is ours rather than a scanner's.

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

The smoke run reproduces the central phenomenon in miniature. A threshold calibrated at
acceleration R=2 and transferred to more aggressive accelerations loses coverage, and
group-conditional calibration restores it:

| method | R=2 | R=4 | R=8 |
|---|---|---|---|
| naive transferred | 0.90 | 0.59 | 0.41 |
| matched per operator | 0.90 | 0.90 | 0.90 |
| group-conditional | 0.90 | 0.90 | 0.90 |

*Nominal 0.90, on synthetic phantoms with a reference sampler. This is an interface smoke
test, not a result.* Note the price of restoration: flagged area rises from 0.44 to 0.83, which
is why flag area is a reported metric everywhere alongside coverage — a valid but
over-conservative flag is of little use.

## Design

The framework is a **solver-agnostic audit layer over pretrained priors.** It audits
reconstruction models rather than training them, which is what makes the study feasible on a
single GPU.

Reconstructions and non-conformity scores are computed once across the whole operator grid and
cached as a **score bank**. Every calibration experiment, coverage curve and numerical check
then runs on the cache with no GPU, which decouples the statistical and analytical work from
reconstruction time entirely.

```
src/caliber/
  operators/    the known acquisition operator A, its range/null-space split,
                the operator registry, and distance measures
  solvers/      pretrained diffusion and flow-matching back-ends behind one
                black-box posterior-sampler interface
  scoring/      region definition, operator-aware scoring, and the ground-truth
                hallucination criterion
  scorebank/    the cached score-bank format, self-describing, with a manifest
                recording which claims the bank carries
  calibration/  group-conditional and distance-weighted recalibration
  evaluation/   coverage, efficiency, detection, gap-versus-distance, paired
                estimands, transfer penalties, and the frozen pre-registration
  theory/       the coverage-gap bound and its comparison against measurement
  data/         dataset access and preprocessing

configs/        version-controlled experiment definitions
scripts/        CPU-only analysis entry points
tests/          1,244 correctness gates
Documentation/  the written record — see below
```

**Reproducibility.** Config-driven runs, fixed seeds, checkpointed and resumable bank builds,
self-describing banks, and a CPU-only analysis path. Analysis choices that could be made after
seeing a number — the response variable, the error metric, the baselines, the resampling unit,
the matching tolerances — are frozen in a single module with a content fingerprint stamped onto
every result table, so a table produced under a different freeze is visible without reading
git history. Amendments are dated and recorded rather than edited in.

## Data

Nothing is committed; everything is fetched.

- **MRI** — accelerated reconstruction from raw k-space: fastMRI, with fastMRI+ pathology
  annotations for region-level evaluation, and M4Raw as an openly available brain substrate.
- **CT** — ground-truth volumes from **TotalSegmentator**, with 117 aligned anatomical masks
  per subject supplying the region overlay, re-projected through this project's own
  parallel-beam operator. The operator is ours for the same reason the MRI masks are: angle
  subsampling and angular-range restriction have to be applied, and varied, retrospectively.
- **Priors** — diffusion score models for brain MRI and lung CT, ported and verified against
  the original implementations to ~1e-5, plus a natural-image model retained only as a
  deliberately mismatched control.

See [`Documentation/reproducing.md`](Documentation/reproducing.md) for what to fetch, under
what terms, and the command behind every reported number. Dataset provenance and licensing are
in [T2.1 Appendix A](Documentation/T2.1_data_and_operator_pipeline.md#appendix-a--data-access-and-provenance).
The repository carries no model weights: a converter and the download commands ship instead.

## Documentation

The written record is extensive because the negative results are the contribution, and a
negative result is only readable if the reasoning behind it is inspectable. Four entry points:

| start here | for |
|---|---|
| [`PROJECT_STATUS.md`](Documentation/PROJECT_STATUS.md) | the whole project in one document — what is established, what is false, what does not exist |
| [`T1.7_preprint_draft.md`](Documentation/T1.7_preprint_draft.md) | the paper: the failure, the fix, the measured boundary, and the evaluation caution |
| [`reproducing.md`](Documentation/reproducing.md) | from a clean clone to every reported number |
| [`problems_and_decisions.md`](Documentation/problems_and_decisions.md) | the decision log — every methodological choice, why it was made, and the ones that were later withdrawn on measurement |

The remaining files are per-task records, named for the task that produced them, and are best
reached from the four above rather than read in order.

**A note on the withdrawals.** Several results in the decision log are marked withdrawn or
refuted, including some that had been reported as findings. Where that happened the original
wording is kept beside the correction rather than deleted, because the point of a log is to
show what was believed when. If a number in an older document disagrees with the decision log,
the log is right.

## Status

Four phases across one academic term; the first three are complete and gated. The empirical
core and the group-conditional fix are measured on banks that carry coverage claims, the
coverage-gap bound holds against measurement, and the operator-aware refinements are each
measured and each unsupported. What remains is submission: a method paper on the failure, the
fix, the boundary and the evaluation caution, and a companion analytical paper on the bound.

Development is organised as three concurrent workstreams — methodology and calibration,
systems and cross-domain validation, and analytical foundations — integrating through the
score bank and operator registry, with a shared decision log for adjudication. Work lands on
`main` by pull request from per-task branches.
