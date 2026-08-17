# `data/` — layout

Nothing here is committed except this file. See
[../docs/data_access.md](../docs/data_access.md) for provenance and licensing,
and [../docs/decision_log.md](../docs/decision_log.md) D-005 and D-013 for why
some of the downloaded archives are not needed.

## Layout

Fetch everything scriptable with `python scripts/fetch_data.py --all`.

```
data/
  archives/                    downloaded zips, kept for re-extraction
    lodopab/*.zip
    totalsegmentator/*.zip
    m4raw/*.zip
  raw/
    lodopab/                   ✅ validation + test extracted (56 shards)
      ground_truth_validation_*.hdf5    28 shards, 3522 slices
      ground_truth_test_*.hdf5          28 shards, 3553 slices
      patient_ids_rand_*.csv            per-slice patient IDs — see below
    totalsegmentator/          ✅ first 100 of 1228 subjects extracted
      s0000/ct.nii.gz                   CT volume, 1.5 mm isotropic
      s0000/segmentations/*.nii.gz      117 aligned organ masks
    m4raw/                     ⏳ open MRI k-space stopgap (D-015)
      multicoil_val/*.h5              240 volumes, ISMRMRD, fastMRI-compatible
    fastmri/                   ⏳ downloading off-machine (D-017)
      singlecoil_val/*.h5             keep splits in separate directories
  annotations/
    fastmri_plus/              ✅ knee.csv, brain.csv + file lists
  interim/                     preprocessed caches, regenerable

checkpoints/                   (outside data/, gitignored)
  ddpm-celebahq-256/           ✅ 434 MB, 113.7 M params, verified loadable
```

## MRI status

`fastmri_plus` annotations are here — 969 annotated knee files, 16,154 boxes,
top labels Meniscus Tear (5,658) and Cartilage partial-thickness loss (2,985).
They stay useless until the **images** arrive; that download is now running
off-machine (D-017), with one licensing check still open there.

M4Raw fills the gap for development only: open raw k-space in the same `.h5`
layout, but 0.3 T brain instead of 3 T knee and **no pathology annotations**.
It is wired as the `m4raw` dataset and marked unreportable in code, so a bank
built from it announces itself as such (D-018). See D-015 for exactly what it
can and cannot carry.

```bash
python scripts/fetch_data.py m4raw_val --extract
caliber configs/experiments/w02_m4raw_smoke.yaml   # real k-space, no checkpoint
```

## Verify before trusting a ✅

A truncated download keeps a valid zip header and loses only the central
directory at the end, so it looks present and correct until something tries to
extract it. The M4Raw archive sat marked as fetched at 0.8 GB of 2.8 GB.

```bash
python scripts/fetch_data.py --verify    # checks every archive on disk
```

## Archive inventory

| Archive | Zip | Extracted | Status |
|---|---|---|---|
| `ground_truth_validation.zip` | 1.6 GB | 1.8 GB | ✅ extracted |
| `ground_truth_test.zip` | 1.6 GB | 1.8 GB | ✅ extracted |
| `Totalsegmentator_dataset_v201.zip` | 23.6 GB | 30.1 GB | ✅ 100/1228 subjects (2.6 GB) |
| `ground_truth_train.zip` | 15.9 GB | 18.1 GB | ⛔ not needed — D-013 |
| `observation_train.zip` | 29.9 GB | — | ⛔ not needed — D-005 |
| `observation_test.zip` | 3.0 GB | — | ⛔ not needed — D-005 |
| `observation_challenge.zip` | 3.1 GB | — | ⛔ not needed — D-005 |

**~49 GB of archives are recoverable** (`ground_truth_train` +
`observation_*`). Disk is the binding constraint: 27 GB free, and a full
TotalSegmentator extraction needs 30.1 GB. Deleting the three `observation_*`
zips alone frees 36 GB and unblocks it.

Nothing has been deleted — those were slow downloads and the call is yours.

## Why `observation_*` is not needed (D-005)

They hold LoDoPaB's low-dose sinograms at LoDoPaB's own fixed geometry. This
project re-projects ground-truth images with its own registered
`CTProjectionOperator`, because the acquisition operator is the variable under
study and a fixed supplied geometry makes the operator grid impossible.

## Why `ground_truth_train` is not needed (D-013)

The project audits pretrained priors and trains nothing (Sec. 17). The train
split exists to train reconstruction models. Validation + test give ~7,000
slices, far more than the ~200-item runs the protocol calls for.

## Two things to know about this data

**Patient IDs matter for split integrity.** `patient_ids_rand_*.csv` gives one
patient ID per slice, in slice order. LoDoPaB slices come from patients with
many slices each, so a random slice-level calibration/test split puts the same
patient on both sides. That inflates apparent coverage and quietly violates the
exchangeability the conformal guarantee rests on. Split by patient, not by
slice.

**TotalSegmentator masks belong to TotalSegmentator's own CT volumes**, not to
LoDoPaB's — they are different cohorts (LoDoPaB derives from LIDC-IDRI). They
cannot be used to annotate LoDoPaB slices. See D-013 for what follows.

## Extraction commands

```bash
# remaining TotalSegmentator subjects (needs ~28 GB free)
python -c "import zipfile; zipfile.ZipFile(
  'data/archives/totalsegmentator/Totalsegmentator_dataset_v201.zip'
  ).extractall('data/raw/totalsegmentator/')"

# LoDoPaB train, only if more calibration data is ever needed
python -m zipfile -e data/archives/lodopab/ground_truth_train.zip data/raw/lodopab/
```
