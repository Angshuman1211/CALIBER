# Score-Bank Specification v0

**Status:** draft for Week 1 sign-off by all three tracks
**Schema version:** `0.1.0` (`caliber.scorebank.schema.SCHEMA_VERSION`)
**Owner:** Track 2 (T2.3) · **Consumers:** Track 1 (T1.2), Track 3 (T3.6)

Blueprint references: Sec. 17 (compute strategy), Sec. 19 (shared interfaces),
Week 1 and Week 3 cards.

---

## 1. Why this exists

Reconstructions are computed **once**, on a GPU, and cached. Every subsequent
calibration experiment, coverage curve, ablation, and numerical check reads the
cache and needs no GPU at all. That single decision is what makes the project's
scope affordable on the available hardware, and it is what lets Track 1 and
Track 3 work in parallel with Track 2 rather than behind it.

The bank is the only data interface between the three tracks. If a track needs
something that is not in the bank, the answer is to add a column — never to
re-run reconstruction.

## 2. Three design rules

Each rule exists to prevent re-running GPU work, and each one has already paid
for itself once in a project of this shape.

**Rule 1 — store components, not conclusions.**
The three score components (`s_free`, `s_disagree`, `s_residual`) are stored
separately, never pre-combined. Track 1 finalises the score in Week 3 (T1.1)
and will re-weight it several times; every re-weighting must cost zero GPU
seconds.

**Rule 2 — store continuous quantities, not labels.**
`err_total`, `err_free`, and `ann_coverage` are stored raw. The ground-truth
hallucination criterion (O2) is a Week 2 decision that will be revisited; a
bank full of booleans would have to be regenerated when it changes.

**Rule 3 — store provenance on every row.**
`run_id`, `config_hash`, `operator_id`, `solver_id`, `solver_checkpoint`, and
`seed` travel with each row. Any figure must be traceable to the run that
produced it, without consulting a lab notebook.

## 3. On-disk layout

```
runs/<run_id>/
  manifest.json              provenance, environment, embedded operator registry
  operator_registry.json     the registry snapshot, standalone
  scores.parquet             the long-format table below
  recons/                    OPTIONAL float16 posterior samples, small subset only

  scores.partial.parquet     ONLY while running or after a crash (D-038)
  manifest.partial.json      ditto; carries "complete": false
```

**Complete and partial banks (D-038).** A finished bank's manifest carries
`"complete": true`. A run with `run.checkpoint_every: N` also writes the two
`*.partial.*` files every N items and removes them once the finished bank is on disk.
A directory holding **only** partial files is an interrupted run: `read_bank` refuses
it unless called with `allow_partial=True`, because a partial bank is a valid table
whose rows give no sign the run stopped early. Restarting the same config resumes
from the checkpoint; a checkpoint from a different config is refused (D-040). Banks
written before D-038 have no `complete` key and are read as complete. The table schema
below is unchanged.

A bank is **self-describing**: the manifest embeds the operator registry, so
Track 3 can compute operator distances and Track 1 can calibrate from a copied
directory with no access to the machine that produced it.

`recons/` is off by default (`run.cache_reconstructions: 0`). A full grid of
float32 posterior samples is orders of magnitude larger than the score table
and would defeat the point of caching. Enable it for a fixed handful of items
used in figures.

## 4. Table schema

One row per **(item, operator, solver, region)**.

### Provenance
| Column | Type | Notes |
|---|---|---|
| `schema_version` | string | Must equal `0.1.0` throughout a bank |
| `run_id` | string | `<experiment name>-<config hash>` |
| `config_hash` | string | 16 hex chars over the resolved config |

### Item identity
| Column | Type | Notes |
|---|---|---|
| `dataset` | string | `fastmri`, `lodopab`, `synthetic_phantom` |
| `split` | string | `calib`, `test` |
| `sample_id` | string | Volume or file stem |
| `slice_idx` | int32 | 2-D single slice per the Sec. 17 compute profile |

### Acquisition operator — joins to the operator registry
| Column | Type | Notes |
|---|---|---|
| `operator_id` | string | Primary key into `operator_registry.json` |
| `modality` | string | `mri`, `ct` |
| `operator_family` | string | `random1d`, `sparse_view`, `limited_angle`, … |
| `calib_group` | string | The group for group-conditional calibration (Tool 1) |

### Solver
| Column | Type | Notes |
|---|---|---|
| `solver_id` | string | Family, name, checkpoint, step count |
| `solver_family` | string | `diffusion`, `flow_matching`, `reference` |
| `solver_checkpoint` | string | Exact weights used |
| `n_posterior` | int16 | Posterior draws per item |
| `seed` | int32 | Per-item seed; runs are reproducible from it |

### Region
| Column | Type | Notes |
|---|---|---|
| `region_id` | int32 | Index into the region grid |
| `region_row`, `region_col` | int16 | Grid position |
| `region_area` | int32 | Pixels; constant under the v0 patch grid |

### Score components — Rule 1
| Column | Type | Notes |
|---|---|---|
| `score_version` | string | Bump when the component definitions change |
| `s_free` | float32 | RMS energy of the null-space component |
| `s_disagree` | float32 | Cross-sample spread, free component only |
| `s_residual` | float32 | Backprojected measurement residual |

### Ground truth — Rule 2
| Column | Type | Notes |
|---|---|---|
| `criterion_version` | string | Bump when the O2 criterion changes |
| `err_total` | float32 | Region RMSE, reconstruction vs truth |
| `err_free` | float32 | Region RMSE restricted to the free component |
| `ann_coverage` | float32 | Fraction of the region under an annotation |
| `ref_intensity` | float32 | Scale used to normalise the above |

**Why `err_free` is separate from `err_total`:** a region can be wrong because
the prior invented structure, or because it smoothed structure away. Only the
first is hallucination in the Blueprint's sense (Sec. 4). Restricting to the
free component is what distinguishes them, and the distinction has to survive
into the bank or the criterion cannot be revised later.

## 5. Invariants

Enforced by `caliber.scorebank.schema.validate`, tested in
`tests/test_scorebank.py`:

1. No duplicate rows for the key `(dataset, split, sample_id, slice_idx,
   operator_id, solver_id, seed, region_id)`.
2. One schema version per bank.
3. No nulls in score or error columns.
4. `n_posterior >= 1`.
5. **One solver per `run_id`.** More than one is the signature of a violated
   controlled comparison (Sec. 13) and is reported as a validation problem.
   The solver ablation runs one experiment per solver and merges afterwards.

## 6. Changing this spec

Bumping `SCHEMA_VERSION` is a cross-track decision. Record it in
`decision_log.md` with the reason and the migration path for existing banks.
Adding a nullable column is backwards-compatible and does not require a bump.

## 7. Open questions for Week 1 sign-off

- [ ] **Region definition.** v0 is a 16px patch grid. Does the O2 risk
      definition (Week 2) need anatomically derived regions instead? If so,
      `region_area` stops being constant and region-level rates become weighted.
- [ ] **`calib_group` granularity.** Currently modality/family/severity. Too
      fine starves groups of calibration data (the declared Tool 1 trade-off);
      too coarse mixes regimes that genuinely differ.
- [ ] **Calibration/test split.** Currently a column. Should split assignment
      be fixed per `sample_id` in the registry so it is identical across every
      run and modality?
- [ ] **Multi-coil MRI.** v0 reduces coils by root-sum-of-squares, so the
      studied operator is single-coil Cartesian undersampling. Confirm this is
      acceptable and state it in the Experimental Setup.
