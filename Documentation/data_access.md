# Data Access — Track 2 (T2.1)

**The only Week 1 item with an external dependency outside our control.**
fastMRI access is credentialed and has approval lead time; the Week 3
deliverable depends on it. Start the application before anything else.

Nothing in `data/` is committed. This file is the provenance record.

---

## Status board

Everything scriptable is fetched by `python scripts/fetch_data.py --all`
(`--list` shows the inventory). Downloads resume on interruption.
`--verify` checks every archive already on disk for truncation.

| Resource | Ref | Needed by | Access | Status |
|---|---|---|---|---|
| fastMRI raw k-space (`singlecoil_val`) | [28] | **Week 3** | Application + approval | ☑ downloaded ☑ extracted ☑ verified — 199 volumes / 7,135 slices / 19.4 GB · **reportable**, D-017 resolved |
| fastMRI raw k-space (`singlecoil_test`) | [28] | — | Application + approval | ☑ downloaded · **unusable**, P-13 — pre-undersampled at R=8, no ground truth |
| fastMRI raw k-space (`singlecoil_train`) | [28] | — | Application + approval | ☐ **not downloaded, not needed** — 72.7 GB; no config references it and the project trains nothing |
| fastMRI+ pathology annotations | [29] | Week 3 | Public GitHub | ☑ downloaded (969 files, 16,154 boxes) · 155 of the 199 val volumes annotated |
| M4Raw open raw k-space | — | Weeks 2-3 | Open (Zenodo, CC-BY) | ☑ downloaded ☑ extracted (240 volumes) ☑ pipeline verified · D-015 |
| DDPM natural-image prior | — | Week 2, 10 | Open (HF, Apache-2.0) | ☑ downloaded, verified |
| Medical score priors | [27] | Week 2 | Google Drive, manual | ☐ — see D-016 |
| LoDoPaB-CT ground truth (val + test) | [30] | Week 7 | Open (Zenodo) | ☑ downloaded ☑ extracted |
| LoDoPaB-CT ground truth (train) | [30] | — | Open (Zenodo) | ☑ downloaded · **not needed**, D-013 |
| LoDoPaB-CT observations | [30] | — | Open (Zenodo) | ☑ downloaded · **not needed**, D-005 |
| TotalSegmentator | [31] | Week 7 | Open | ☑ downloaded ☑ 100/1228 subjects extracted |
| AbdomenCT-1K | [32] | Week 7 (alt) | Registration | not needed if TotalSegmentator suffices |
| Natural-image benchmarks | — | Week 10 | Open | ☐ deferred |

**All CT data is already on local disk and structured.** fastMRI is moving
(D-017): links are in hand and the transfer runs off-machine, because the local
disk cannot hold it alongside a full TotalSegmentator extraction. One licensing
check remains open there and it is load-bearing — see D-017.

> **Verify archives before trusting the board.** The M4Raw zip sat recorded as
> fetched at 0.8 GB of 2.8 GB: a truncated download keeps a valid zip header
> and only loses the central directory at the *end*, so nothing notices until
> an extraction is attempted. `python scripts/fetch_data.py --verify` now
> checks every archive on disk, and `fetch` refuses to report success on an
> archive it cannot open.

Two decisions came out of inspecting what was downloaded, both open at the
Week 1 kickoff: **D-013** (TotalSegmentator masks do not align with LoDoPaB
images, so one of them has to change role) and **D-014** (splits must be by
patient, not by slice). See [decision_log.md](decision_log.md), and
[week1_kickoff.md](week1_kickoff.md) for the sign-off agenda.

`data/` carries no README of its own — this file is the single data document.
Archive inventory, extraction commands, and what can be deleted are in §6 below.
Layout is in §6 too, so a fresh clone with an empty `data/` has one place to look.

---

## 0. What is blocked, and what is not

**fastMRI is the only true blocker.** It requires a signed Data Sharing
Agreement plus an approved application naming your institution and intended
use — a legal agreement between you and NYU that cannot be scripted, and whose
terms forbid the third-party mirrors that would otherwise route around it.

Consequences, stated plainly:

- Reported MRI results need fastMRI. There is no substitute for the
  annotation-linked evaluation, because no open raw-k-space dataset ships
  pathology annotations.
- Everything *else* can proceed now. M4Raw (D-015) supplies open, fully-sampled
  raw k-space in fastMRI's own `.h5` layout, which is all retrospective
  undersampling needs. Solver wiring, score-bank production, calibration, and
  the coverage-vs-distance machinery can all be built and tested against it.

Apply at <https://fastmri.med.nyu.edu/> before anything else this week.

## 1. fastMRI — primary empirical substrate

**Why raw measurements are non-negotiable.** The acquisition operator is the
variable under study. It must be *chosen*, not inherited. Fully-sampled k-space
lets us synthesise the whole operator grid retrospectively, which is also what
makes the operator known at both calibration and deployment — the assumption
the exact guarantee rests on (Sec. 13). A dataset of finished images cannot
support this project at all.

**Access.** Apply at <https://fastmri.med.nyu.edu/>. Approval is not instant;
treat it as the long pole. You receive time-limited download links by email.

**What to fetch first.** `knee_singlecoil_val` (~7 GB). Single-coil avoids coil
sensitivity estimation entirely, and the validation split is small enough to
develop against while the training split downloads. Knee is the split fastMRI+
annotates most densely.

```
data/raw/fastmri/singlecoil_val/*.h5
```

Each file holds `kspace` of shape `(slices, H, W)` complex, plus a
`reconstruction_esc` reference.

**Landing the files.** The loader searches recursively, so extra nesting from
staging the archive elsewhere is harmless — `singlecoil_val/singlecoil_val/`
reads the same as `singlecoil_val/`. Two things are not harmless:

- **Keep the splits in separate directories.** Configs point at
  `data/raw/fastmri/singlecoil_val`, not at `data/raw/fastmri`, precisely so a
  recursive search cannot sweep train and val into one index and put the same
  volume on both sides of a calibration/test split. That is D-014 in its
  fastMRI form.
- **A redistribution shipping only `reconstruction_*` images is useless here.**
  The loader raises rather than falling back, because the operator has to be
  applied to fully-sampled k-space; it cannot be inferred from a finished
  image.

**Preprocessing** (`caliber.data.fastmri`): `ifft2` → centre crop to the working
resolution → magnitude → 99th-percentile normalisation. Multi-coil volumes are
reduced by root-sum-of-squares.

> The studied operator is therefore **single-coil Cartesian undersampling**.
> That is a real narrowing of scope and belongs in the Experimental Setup
> section, not left implicit.

### 1a. M4Raw — the open stopgap (D-015)

Open raw k-space in fastMRI's own `.h5` layout, so the fastMRI reader runs
unchanged. It fills the development gap while fastMRI access is pending, and
nothing more: **0.3 T brain rather than 3 T knee, and no pathology
annotations**, so it cannot carry any annotation-linked evaluation. It is wired
as the `m4raw` dataset and marked unreportable in code, so a bank built from it
announces itself as such (D-018).

Each volume ships `kspace` of shape `(18, 4, 256, 256)` complex64 plus
`reconstruction_rss`. Volume attributes also carry `patient_id`, which is what
the split-by-patient rule (D-014, D-022) needs to be enforceable here.

```bash
python scripts/fetch_data.py m4raw_val --extract
caliber configs/experiments/w02_m4raw_smoke.yaml   # real k-space, no checkpoint
```

## 2. fastMRI+ — pathology annotations

Public: <https://github.com/microsoft/fastmri-plus>. Bounding boxes per slice
with columns `file, slice, study_level, x, y, width, height, label`.

```
data/annotations/fastmri_plus/knee.csv     # 16,154 boxes across 969 files
data/annotations/fastmri_plus/brain.csv
```

These provide the region-level labels that let a flag be judged on whether it
fires **where it matters** (Sec. 14) rather than merely somewhere.

**The boxes are drawn on fastMRI's 320x320 `reconstruction_esc` grid, not on
k-space.** Knee k-space inverts to 640x368, so rasterising there misplaces
every box — silently, because a misplaced mask is still a valid mask. The
loader crops the image to the annotation grid first and then applies one
identical crop to both. See **D-019**, which has the measurements.

One consequence worth knowing before setting the working resolution: at
256x256 every box survives the centre crop; at 128x128 only 76.7% do.

## 3. LoDoPaB-CT — cross-domain substrate

Open on Zenodo: <https://zenodo.org/records/3384092>. No approval needed, so it
can be fetched any time before Week 7.

```
data/raw/lodopab/ground_truth_*.hdf5
```

**We use the ground-truth images and re-project them with our own registered
operator, not LoDoPaB's supplied sinograms.** Using the shipped sinograms would
fix the acquisition at LoDoPaB's geometry and make the operator grid
impossible. The cost is that the CT forward model is ours, not the scanner's —
which belongs in Assumptions and Honest Scope.

**Patient IDs are shipped separately and matter for split integrity.**
`patient_ids_rand_*.csv` gives one patient ID per slice, in slice order.
LoDoPaB slices come from patients contributing many slices each, so a random
slice-level calibration/test split puts the same patient on both sides. That
inflates apparent coverage and quietly violates the exchangeability the
conformal guarantee rests on. Split by patient, never by slice — D-014, and
D-022 for the operator-independence requirement on top of it.

## 4. CT anatomical annotations

TotalSegmentator [31] (open, ~100 structures) or AbdomenCT-1K [32]
(registration). Either supplies the region-level ground truth that makes CT
flag evaluation mirror MRI.

Labels are supplied per volume and must be resampled onto the same
centre-cropped grid as the image. An off-by-one here mislabels every CT region
without any visible symptom, so this needs its own verification step when
wired (Week 7).

**TotalSegmentator's masks segment TotalSegmentator's own CT volumes, not
LoDoPaB's.** They are different cohorts — LoDoPaB derives from LIDC-IDRI — so
LoDoPaB slices cannot be annotated with them, and the two cannot simply be
paired. This is what D-013 resolves; each TotalSegmentator subject ships
`ct.nii.gz` plus 117 aligned organ masks, which is the same image-plus-region
pairing MRI gets from fastMRI+.

## 5. Storage and compute budget

The available GPU is a 6 GB RTX 3050. Sec. 17's compute profile — 2-D single
slice, reduced resolution, no prior training — is therefore a hard constraint
rather than a preference, and risk R7 is live from day one.

Rough budget:

| Item | Size |
|---|---|
| fastMRI singlecoil_val | ~7 GB |
| LoDoPaB ground truth (val + test) | ~4 GB |
| TotalSegmentator, full extraction | 30.1 GB |
| Score bank, full MRI grid | ~100 MB (region rows are tiny) |
| Cached reconstructions | Only enabled for a fixed handful of items |

The score table being negligible next to the raw data is the whole point of
Sec. 17: the expensive artefact is produced once, and the artefact everyone
actually reads is small enough to copy around freely.

## 6. Layout, archive inventory, and what can be deleted

### Directory layout

`data/` is gitignored in full and carries no README of its own (D-025), so this
is the map for a fresh clone.

```
data/
  archives/                    downloaded zips, kept for re-extraction
    lodopab/*.zip
    totalsegmentator/*.zip
    m4raw/*.zip
  raw/
    lodopab/
      ground_truth_validation_*.hdf5    28 shards
      ground_truth_test_*.hdf5          28 shards
      patient_ids_rand_*.csv            one patient ID per slice, in slice order (§3)
    totalsegmentator/
      s0000/ct.nii.gz                   CT volume, 1.5 mm isotropic
      s0000/segmentations/*.nii.gz      117 aligned organ masks
    m4raw/
      multicoil_val/*.h5                kspace (18, 4, 256, 256) + reconstruction_rss
    fastmri/
      singlecoil_val/*.h5               keep each split in its own directory (§1)
  annotations/
    fastmri_plus/                       knee.csv, brain.csv + file lists
  interim/                              preprocessed caches, regenerable

checkpoints/                            (outside data/, also gitignored)
```

Two things the layout itself encodes, both explained above: splits live in
separate directories so a recursive search cannot put one volume on both sides
of a calibration/test split (§1), and `patient_ids_rand_*.csv` is required
rather than incidental, because a slice-level split leaks a patient across
calibration and test (§3).

### Archive inventory

Sizes are measured, not estimated — `python scripts/fetch_data.py --verify`
reports them. Download state is deliberately **not** recorded here: it changes
under you, and a status marker committed to version control is a claim that
goes stale the moment it is true. Run `--list` for current state.

| Archive | Zip | Extracted | Needed? |
|---|---|---|---|
| `ground_truth_validation.zip` | 1.57 GB | 1.8 GB | Yes — CT substrate |
| `ground_truth_test.zip` | 1.58 GB | 1.8 GB | Yes — CT substrate |
| `Totalsegmentator_dataset_v201.zip` | 23.58 GB | 30.1 GB | Yes — CT annotations (D-013) |
| `M4RawV1.1_multicoil_val.zip` | 2.81 GB | 2.7 GB | Yes — MRI stopgap (D-015) |
| `ground_truth_train.zip` | 15.94 GB | 18.1 GB | **No** — D-013 |
| `observation_train.zip` | 29.94 GB | — | **No** — D-005 |
| `observation_challenge.zip` | 3.13 GB | — | **No** — D-005 |
| `observation_test.zip` | 3.00 GB | — | **No** — D-005 |
| `observation_validation.zip` | 2.94 GB | — | **No** — D-005 |

**~55 GB is recoverable** — `ground_truth_train` (15.94 GB) plus the four
`observation_*` zips (39.01 GB). Disk is the binding constraint on this project:
a full TotalSegmentator extraction alone needs 30.1 GB and fastMRI adds ~7 GB on
arrival, so the `observation_*` zips are usually what has to go first. Nothing
is deleted automatically; the call belongs to whoever owns the machine.

**Why `observation_*` is not needed (D-005).** They hold LoDoPaB's low-dose
sinograms at LoDoPaB's own fixed geometry. This project re-projects ground-truth
images with its own registered `CTProjectionOperator`, because the acquisition
operator is the variable under study and a fixed supplied geometry makes the
operator grid impossible. See §3.

**Why `ground_truth_train` is not needed (D-013).** The project audits pretrained
priors and trains nothing (Sec. 17). The train split exists to train
reconstruction models. Validation plus test give ~7,000 slices, far more than
the ~200-item runs the protocol calls for.

### Extraction

```bash
# remaining TotalSegmentator subjects (needs ~28 GB free)
python -c "import zipfile; zipfile.ZipFile(
  'data/archives/totalsegmentator/Totalsegmentator_dataset_v201.zip'
  ).extractall('data/raw/totalsegmentator/')"

# LoDoPaB train, only if more calibration data is ever needed
python -m zipfile -e data/archives/lodopab/ground_truth_train.zip data/raw/lodopab/
```

## 7. Ethics and licensing

fastMRI and LoDoPaB are de-identified, publicly released research datasets used
under their stated terms. fastMRI's terms prohibit re-distribution, so the
open-source release (O7) ships **code, configs, and score banks** — never
images. Confirm before Week 12 that releasing derived score banks is within
terms; if it is not, release the generating scripts and a synthetic
demonstration bank instead.
