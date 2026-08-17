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
| fastMRI raw k-space | [28] | **Week 3** | Application + approval | ☑ links obtained · ⏳ downloading off-machine, D-017 |
| fastMRI+ pathology annotations | [29] | Week 3 | Public GitHub | ☑ downloaded (969 files, 16,154 boxes) |
| M4Raw open raw k-space | — | Weeks 2-3 | Open (Zenodo, CC-BY) | ⏳ resuming — was truncated, see below · stopgap, D-015 |
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

See [../data/README.md](../data/README.md) for the archive inventory,
extraction commands, and what can be deleted.

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

## 4. CT anatomical annotations

TotalSegmentator [31] (open, ~100 structures) or AbdomenCT-1K [32]
(registration). Either supplies the region-level ground truth that makes CT
flag evaluation mirror MRI.

Labels are supplied per volume and must be resampled onto the same
centre-cropped grid as the image. An off-by-one here mislabels every CT region
without any visible symptom, so this needs its own verification step when
wired (Week 7).

## 5. Storage and compute budget

The available GPU is a 6 GB RTX 3050. Sec. 17's compute profile — 2-D single
slice, reduced resolution, no prior training — is therefore a hard constraint
rather than a preference, and risk R7 is live from day one.

Rough budget:

| Item | Size |
|---|---|
| fastMRI singlecoil_val | ~7 GB |
| LoDoPaB ground truth | ~4 GB |
| Score bank, full MRI grid | ~100 MB (region rows are tiny) |
| Cached reconstructions | Only enabled for a fixed handful of items |

The score table being negligible next to the raw data is the whole point of
Sec. 17: the expensive artefact is produced once, and the artefact everyone
actually reads is small enough to copy around freely.

## 6. Ethics and licensing

fastMRI and LoDoPaB are de-identified, publicly released research datasets used
under their stated terms. fastMRI's terms prohibit re-distribution, so the
open-source release (O7) ships **code, configs, and score banks** — never
images. Confirm before Week 12 that releasing derived score banks is within
terms; if it is not, release the generating scripts and a synthetic
demonstration bank instead.
