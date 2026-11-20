# Reproducing Track 2's Numbers

**Owner:** Track 2 · **Started:** Week 12 (T2.11) · **Status:** first cut. It covers the two medical-prior banks the paper rests on (D-070) and the protocol replication. The CT rows are commands only: the LIDC bank's control failed as written and the rest waits on a ruling (T2.11 §1e).

This is the path from a clean clone to the numbers Track 2 reports. Every command here has been run on the project machine and is recorded, with its output, in the weekly record named beside it. Nothing below ships data or weights: the repository carries code, configs and tests, and each of the rest is fetched from its owner (T2.11 §5).

---

## 1. What you need that the repository does not give you

| | Where it comes from | Terms | Lands in |
|---|---|---|---|
| M4Raw brain k-space, validation split (240 volumes, 30 subjects) | Zenodo 8056074 | CC BY 4.0 | `data/raw/m4raw/multicoil_val/` |
| TotalSegmentator CT v2.0.1 (100 of 1,228 subjects used) | Zenodo 10047292 | CC BY 4.0 | `data/raw/totalsegmentator/<subject>/` |
| fastMRI knee `singlecoil_val` (199 volumes) | NYU, by application | no redistribution | `data/raw/fastmri/singlecoil_val/` |
| Song et al.'s BraTS and LIDC score models | the authors' Google Drive, linked from `yang-song/score_inverse_problems` | none stated | `checkpoints/song_inverse_problems/` |
| `ddpm-celebahq-256` (development banks only) | Hugging Face | Apache-2.0 | `checkpoints/ddpm-celebahq-256/` |

`python scripts/fetch_data.py --list` prints the full table with sizes and what is manual. T2.1 Appendix A has the provenance and the verification of each archive.

The medical-prior results need M4Raw, TotalSegmentator and the two score models. fastMRI and the face prior are needed only for the development banks and for the fastMRI row of the protocol replication.

---

## 2. Environment

```bash
pip install -e ".[dev,solvers,convert,figures]"
python -m pytest -q          # 968 passed, 2 skipped at T2.11
```

The banks were built on one laptop GPU (RTX 3050, 6 GB) with `torch 2.13.0+cu126`. Every analysis runs on the CPU from a finished bank (score-bank rule 1). Two things from the problem log save time on Windows:

- Set `OPENBLAS_NUM_THREADS=1` for the analysis scripts (P-38).
- A long run needs free *system* commit memory, not GPU memory (P-38, P-41). Do not read a bank's checkpoint files while its runner is writing them (P-39).

---

## 3. The priors

```bash
# download checkpoint_26 (BraTS) and checkpoint_101 (LIDC) from the authors' Drive into
#   checkpoints/song_inverse_problems/brats_ncsnpp_ve/  and  .../ct2d_320_ncsnpp_ve/
python scripts/convert_song_checkpoint.py \
    checkpoints/song_inverse_problems/brats_ncsnpp_ve/checkpoint_26 \
    --out checkpoints/song_inverse_problems/brats_ncsnpp_ve/ncsnpp_brats240.pt
python scripts/convert_song_checkpoint.py \
    checkpoints/song_inverse_problems/ct2d_320_ncsnpp_ve/checkpoint_101 \
    --config lidc320 --out checkpoints/song_inverse_problems/ct2d_320_ncsnpp_ve/ncsnpp_lidc320.pt
python -m pytest -q tests/test_ncsnpp.py
```

The converter refuses a checkpoint with a layer it cannot place. `tests/test_ncsnpp.py` checks the port against values recorded from Song et al.'s own Flax model (agreement ≈ 1×10⁻⁵ of the score's magnitude, T2.10 §2 and §8).

---

## 4. The banks

A bank is a function of its config (D-041): the same config gives the same draws in any process, and the run directory is named by the config's hash. If your hash matches, you built the same bank.

| Bank | Config | Run id | Pairs | GPU time |
|---|---|---|---|---|
| MRI, M4Raw + BraTS prior | `w11_m4raw_brats_gpu.yaml` | `w11_m4raw_brats_gpu-2643c9bfa71f4793` | 2,160 | 2.3 h |
| CT, TotalSegmentator + LIDC prior | `w12_ct_lidc_gpu.yaml` | `w12_ct_lidc_gpu-e2d9d249aab63e6b` | 900 | ≈ 21 h |

```bash
python -u -m caliber.runner configs/experiments/w11_m4raw_brats_gpu.yaml
python -u -m caliber.runner configs/experiments/w12_ct_lidc_gpu.yaml
```

Both checkpoint every 10 pairs and resume from the checkpoint when the same command is run again (D-042). `--dry-run` builds the operators, runs their health check and prints the run id without touching the GPU.

Each manifest records what the bank may be quoted for. Both of these carry **coverage claims and not detection claims** (D80, D82, D-072):

```bash
python scripts/restamp_bank.py runs/<run_id>     # prints the claims; --write applies them
```

---

## 5. The numbers

`M` is the MRI bank with patient ids, `C` the CT bank. One subject is one patient on CT, so it clusters on `sample_id`.

```bash
export OPENBLAS_NUM_THREADS=1
python scripts/add_patient_ids.py runs/w11_m4raw_brats_gpu-2643c9bfa71f4793 --data-root data/raw/m4raw/multicoil_val
M=runs/w11_m4raw_brats_gpu-2643c9bfa71f4793_by_patient
C=runs/w12_ct_lidc_gpu-e2d9d249aab63e6b
MLOW=mri.random1d.R2.c0.08.s0;  MHIGH=mri.random1d.R8.c0.08.s0
CLOW=ct.sparse_view.v180.a180.dfull;  CHIGH=ct.sparse_view.v30.a180.dfull
```

### MRI, medical prior (recorded in T2.10 §6)

| Number | Value | Command |
|---|---|---|
| Naive coverage, 1-D mask → `variable_density2d` R2 / R4 / R8 | 0.000 / 0.016 / 0.118 | `coverage_by_patient.py $M --low $MLOW --high $MHIGH --transfer` |
| Group-conditional, same cells | 0.857 / 0.890 / 0.901 | same |
| Below nominal on every one of 13 patient splits | 13 of 13 | `split_noise_floor.py $M --low $MLOW --high $MHIGH` |
| cos θ, median | +0.596 | `label_comparison.py $M --calibration-operator $MLOW` |

Thirty patients resampled 13 ways is a robustness check against the choice of split. It is not 13 replications (D82).

### CT, medical prior (T2.11; values pending the bank)

Run in this order. The first command is the pre-registered control (T2.11 §1c): if its group-conditional arm does not sit at nominal, stop. On the bank as built the control misses on the registry split (0.831) and holds over 13 patient splits (0.897); Track 1's D86 rules the bank read on that table (T2.11 §1e to §1g). Every number from it is reported on the registry split with the both-sides interval and beside the across-split mean: `pair_coverage_by_split.py $C --cluster sample_id --out-dir <dir>`.

| Step | Command |
|---|---|
| 1. Matched-operator control, alone (add `--transfer` only once it has passed) | `coverage_by_patient.py $C --cluster sample_id --low $CLOW --high $CHIGH` |
| 2. cos θ, before any score result is read (T2-C) | `label_comparison.py $C --calibration-operator $CLOW` |
| 3. Patient-split noise floor | `split_noise_floor.py $C --cluster sample_id --low $CLOW --high $CHIGH` |
| 4. Operator distances, all pairs | `ct_distances.py $C --all-pairs --measures null_space_growth null_space_projection --out <csv>` |

### The protocol replication (T2.11 §3)

| Bank | Scale-only control against the posterior spread, AUROC | Command |
|---|---|---|
| fastMRI, face prior | 0.978 against 0.932; above at 9 of 9 operators | `protocol_replication.py <fastmri bank>_by_patient --dispersion <dispersion.parquet>` |
| TotalSegmentator, face prior | 0.997 against 0.994; above at 9 of 9 | `protocol_replication.py <ct bank> --cluster sample_id` |
| M4Raw, BraTS prior | 0.895 against 0.937; matches at 7 of 9 | `protocol_replication.py $M` |

---

## 6. What this does not reproduce

- **Anything that needs fastMRI** without your own NYU approval.
- **Bit-identical banks on other hardware.** D-041 makes the draws a function of the config. It does not make floating-point reductions identical across GPUs, so expect agreement to the tolerance `dispersion_baseline.py` checks (1×10⁻² relative on `s_disagree`), not equality.
- **The development banks' history.** The face-prior banks (T2.3 to T2.9) are not reportable and are not listed here; each weekly record has its own appendix.
- **Track 1's and Track 3's analyses.** They read these banks. Their commands live in their own records.
