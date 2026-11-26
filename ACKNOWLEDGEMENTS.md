# Acknowledgements

This project is an audit layer. It trains nothing and collects nothing: every image, every
generative prior and every reconstruction algorithm it measures was produced by someone else
and released for others to build on. The work below is what made the study possible on a
single GPU, and the honest summary is that the contribution here is the measurement, not the
machinery.

`NOTICE` is the legal attribution record — the licences, the required citations, and the one
adapted source file. This page credits the same work in prose and adds the methods and
software that carry no licence obligation but without which there would be no project. Where
the two differ, `NOTICE` governs.

---

## Data

**No data is distributed with this repository.** Everything below is fetched from its
original source under its own terms; see [`Documentation/reproducing.md`](Documentation/reproducing.md)
for what to fetch and how.

**fastMRI** — Knoll, Zbontar et al., *fastMRI: A Publicly Available Raw k-Space and DICOM
Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning*,
Radiology: Artificial Intelligence, 2020. Obtained from NYU Langone Health under its data
sharing agreement. Raw k-space is the reason the MRI side of this study is a retrospective
undersampling study rather than a simulation.

**fastMRI+** — Zhao et al., *fastMRI+: Clinical Pathology Annotations for Knee and Brain Fully
Sampled Multi-Coil MRI Data*, Scientific Data, 2022. The pathology bounding boxes make
region-level evaluation possible at all.

**M4Raw** — Lyu et al., *M4Raw: A multi-contrast, multi-repetition, multi-channel MRI k-space
dataset for low-field MRI research*, Scientific Data, 2023. CC BY 4.0. An openly available
brain substrate with real k-space, which is what let the MRI arm proceed on an anatomy the
medical prior actually matches.

**TotalSegmentator** — Wasserthal et al., *TotalSegmentator: Robust Segmentation of 104
Anatomic Structures in CT Images*, Radiology: Artificial Intelligence, 2023. CC BY 4.0. Its
aligned anatomical masks are what supply the region overlay on the CT side; without them
there is no region-level CT evaluation.

**LIDC-IDRI** — Armato III et al., *Data From LIDC-IDRI*, The Cancer Imaging Archive, 2015.
CC BY 3.0. Used here indirectly: the chest-CT score model was trained on it by its authors.
*The authors acknowledge the National Cancer Institute and the Foundation for the National
Institutes of Health, and their critical role in the creation of the free publicly available
LIDC/IDRI Database used in this study.*

**BraTS** — the Brain Tumor Segmentation challenge data, used indirectly as the training
corpus of the brain-MRI score model.

## Generative priors

**Song, Shen, Xing and Ermon**, *Solving Inverse Problems in Medical Imaging with
Score-Based Generative Models*, ICLR 2022, and the accompanying
[`score_inverse_problems`](https://github.com/yang-song/score_inverse_problems) release. The
BraTS and LIDC score models are the reason this project has a prior that matches the anatomy
at all, and therefore the reason any number here is quotable. They carry no stated licence, so
**this repository distributes no weights**: a converter and the download commands ship
instead. `src/caliber/solvers/ncsnpp.py` is adapted from the related
[`score_sde_pytorch`](https://github.com/yang-song/score_sde_pytorch) (Apache-2.0) and keeps
its original header.

**SegGuidedDiff** — the unconditional breast-MRI and CT-organ diffusion models, wired as a
second, independent prior family. Two families are what turn "does the result depend on which
prior?" from an assumption into an ablation.

**`ddpm-celebahq-256`** — a natural-image diffusion model, retained deliberately as a
*mismatched* control. It carries no claims in this work, and the contrast between it and the
medical priors is what produced the evaluation caution in the paper.

## Reconstruction algorithms

The solvers are treated as black-box posterior samplers behind one interface, and none is
modified:

- **DPS** — Chung et al., *Diffusion Posterior Sampling for General Noisy Inverse Problems*,
  ICLR 2023.
- **DDNM** — Wang, Yu and Zhang, *Zero-Shot Image Restoration Using Denoising Diffusion Null-
  Space Model*, ICLR 2023. Its structural null-space decomposition is also what the project's
  score design argues against as a confound, which is a debt of a different kind.
- **FlowDPS** — flow-matching posterior sampling, for the flow-prior arm.

## Methods

The statistical backbone is **split conformal prediction** (Vovk, Gammerman and Shafer;
Papadopoulos et al.; Lei et al.) and its **Mondrian / group-conditional** form, which is the
estimator this project's one positive result rests on. The weighted-exchangeability machinery
of **Tibshirani, Barber, Candès and Ramdas** is the standard route for covariate shift, and
the reason operator shift needs a different argument is that it changes the measurement
process rather than the covariate distribution.

Two independently published results shaped what this work claims:

- **Doula**, *Conformal Calibration Transfer* (arXiv:2609.10737), whose impossibility result
  for label-free, uniformly valid correction is cited in our limitations as support for a
  negative result of ours rather than as competing work.
- **Kc, Zeng, Soni and Badano**, *sFRC for assessing hallucinations in medical image
  restoration* (arXiv:2603.04673), which is the external criterion we name as available and
  have not yet run — an open item, stated rather than quietly omitted.

## Software

PyTorch, NumPy, SciPy, pandas, PyArrow, Hugging Face `diffusers`, scikit-image, pytest, and
Matplotlib. The score-bank design — compute reconstructions once, cache the scores, and run
every calibration and analysis on CPU afterwards — is what made a three-front study affordable
on one laptop GPU, and it is built entirely on the above.

## People

Developed by **Angshuman Chakravertty**, **Kuchuru Sai Krishna Reddy** and **Md Rayyan**
across three concurrent workstreams — methodology and calibration, systems and cross-domain
validation, and analytical foundations — integrating through a shared score-bank format, a
shared operator registry, and a common decision log.

A note worth recording: several results in this repository were withdrawn or refuted after
they had been written up, in a few cases by the person who had written them. The decision log
keeps the original wording beside each correction. That only works when the person who found
the error is not the only one who could have.

---

*If something here is miscredited or missing, it is an error rather than an omission on
purpose — please open an issue.*
