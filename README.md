# Automatic Segmentation and 3D Geometric Characterization of Molar Root Canals Using Deep Learning

**MSc Thesis (TFM)** — IIT Scholarship, ICAI, Universidad Pontificia Comillas.

Deep-learning pipeline to support endodontic planning: from cone-beam CT (CBCT) volumes to 3D models of molar root canals and their geometric parameters.

---

## Table of contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Repository structure](#repository-structure)
- [Projects](#projects)
- [Data](#data)
- [Getting started](#getting-started)
- [Current status](#current-status)
- [References and acknowledgements](#references-and-acknowledgements)

---

## Overview

Root canal anatomy is one of the main factors determining the success of endodontic treatment. Accurately identifying the internal morphology of the tooth, and estimating geometric parameters such as **diameter, curvature, length or cross-section**, gives the specialist valuable information to select instruments and plan the treatment.

This thesis develops an artificial-intelligence methodology for the automatic segmentation of molar root canals from CBCT images, followed by the computation of geometric metrics on the resulting 3D models.

## Pipeline

| Stage | Goal |
| :---: | --- |
| 1 | Automatic localization and extraction of the tooth |
| 2 | Three-dimensional segmentation of the root canals |
| 3 | Detailed 3D reconstruction of the internal anatomy |
| 4 | Computation of geometric metrics on the generated 3D models |

## Repository structure

```
.
├── 1_Data_Validation/    Project 1: dataset validation and manual vs. ground-truth analysis
├── 2_UNet/               Project 2: training framework (UNet, Attention UNet, ...)
├── .gitignore
└── README.md
```

> Only the code is versioned. Datasets, model weights, experiment results, virtual environments and personal documents (papers, meeting notes, reports) are kept locally and excluded through `.gitignore`.

## Projects

### [`1_Data_Validation/`](1_Data_Validation/) — Data validation

Study of the **Pulpy3D** dataset (dental CBCT):

- Ground-truth quality checks: empty masks, consistency between `gt_pulp_mandible` and `gt_instance`, molars 36/46.
- Manual segmentation (3D Slicer) vs. ground truth: DICE and HD95 for the whole tooth, the root canal region and the inferior alveolar nerve (IAN).
- Dataset used: 393 valid patients (30 excluded for empty ground truth).

Details: [`README.md`](1_Data_Validation/README.md) and [`ORGANIZACION.md`](1_Data_Validation/ORGANIZACION.md) (script-by-script map, in Spanish).

### [`2_UNet/`](2_UNet/) — Segmentation training

Training of 3D UNet, Attention UNet and variants (UNETR, VNet, multi-head) to segment the pulp and the IAN in CBCT.

| Component | Contents |
| --- | --- |
| `experiments/` | Pulp, IAN, semantic, instance and multi-head segmentation |
| `models/` | UNet3D, PosPadUNet3D, Attention UNet, UNETR, VNet3D, MultiHead |
| `losses/` | Jaccard, Dice + Cross-Entropy |
| `configs/` | One `.yml` per experiment, plus augmentation and preprocessing |

Details: [`README.md`](2_UNet/README.md) (in Spanish).

## Data

Data is **not** included in the repository.

| Dataset | Description | Location (local) |
| --- | --- | --- |
| Pulpy3D | 393 CBCT patients with ground-truth masks. [Download](https://drive.google.com/drive/folders/1M5iU1urLOp1rSxKOm7WCzodAKcZrqT5O?usp=sharing) | `1_Data_Validation/datasets/Pulpy3D/` |
| Excluded patients | 30 patients removed for empty ground truth | `1_Data_Validation/datasets/Pulpy3D_excluidos/` |
| 3D Slicer segmentations | 33 patients with manual segmentation (`.seg.nrrd`) | `2_UNet/datasets/Segmentaciones 3DSlicer/` |

## Getting started

```bash
git clone https://github.com/ClaudiaAgromayor/tfm_dl_root_canal.git
cd tfm_dl_root_canal
```

- **Data analysis** (`1_Data_Validation`): create the environment from `analysis/requirements.txt` and run the scripts from the project root, e.g. `python analysis/canales/canales_por_capa.py`.
- **Training** (`2_UNet`): `pip install -r requirements.txt`, then `python main.py -c configs/pulp_segmentation.yml --verbose`. An NVIDIA GPU is required (the code uses CUDA).

See each project's README for the full setup.

## Current status

- [x] Dataset validation and cleaning (Pulpy3D)
- [x] Manual vs. ground-truth analysis (DICE, HD95)
- [x] Training framework ported and bug-fixed (`2_UNet`)
- [ ] Adapt the dataloader to the 3D Slicer segmentations (`.nrrd`) and define the training labels
- [ ] Train and evaluate the segmentation networks
- [ ] 3D reconstruction and geometric metrics (diameter, curvature, length, cross-section)

## References and acknowledgements

The training code builds on:

- [Pulpy3D-Seed](https://github.com/mahmoudgamal0/Pulpy3D-Seed) — Gamal et al., Pulpy3D dataset and baseline.
- [alveolar_canal](https://github.com/AImageLab-zip/alveolar_canal) — AImageLab, ToothFairy Challenge (MICCAI 2023).

Key literature:

- Gamal et al. — *Automatic Mandibular Semantic Segmentation of Teeth Pulp Cavity and Root Canals, and Inferior Alveolar Nerve on Pulpy3D Dataset*.
- Cipriano et al. (2022) — *Deep Segmentation of the Mandibular Canal: A New 3D Annotated Dataset of CBCT Volumes*.
- *Cross-Frequency Collaborative Training Network and Dataset for Semi-supervised First Molar Root Canal Segmentation* (2025).
- Milletari et al. (2016) — *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*.
- Lin et al. (2017) — *Feature Pyramid Networks for Object Detection*.
- Gao et al., BMC Oral Health 26:691 (2026) — root canal segmentation in CBCT.
