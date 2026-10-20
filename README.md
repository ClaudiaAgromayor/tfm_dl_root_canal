# Automatic Segmentation and 3D Geometric Characterization of Molar Root Canals Using Deep Learning

**MSc Thesis (TFM)** — IIT Scholarship, ICAI, Universidad Pontificia Comillas.

Deep-learning pipeline to support endodontic planning: from a clinical cone-beam CT (CBCT) to a 3D model of the molar root canals and their geometric parameters. High-resolution micro-CT (µCT) of extracted teeth is used to obtain accurate labels.

---

## Table of contents

- [Overview](#overview)
- [Approach](#approach)
- [Repository structure](#repository-structure)
- [Data](#data)
- [Getting started](#getting-started)
- [Current status](#current-status)
- [References and acknowledgements](#references-and-acknowledgements)

---

## Overview

Root canal anatomy is one of the main factors determining the success of endodontic treatment. Estimating geometric parameters such as **length, curvature, volume or diameter** gives the specialist valuable information to select instruments and plan the treatment.

The goal is to segment the root canals of the lower first molars (36/46) automatically from CBCT images and to compute those measurements on the resulting 3D model.

**The main difficulty is the ground truth.** A clinical CBCT (voxel ≈ 75 µm in endodontic protocols) does not show the apical part of the canal well enough to be labelled by hand, whereas a µCT of an extracted tooth (voxel 16 µm) does, and its canal can be obtained by thresholding. But µCT is only possible on extracted teeth, and its grey values are not comparable to a CBCT.

## Approach

Public data sets are not used. Labels come from µCT and the network learns on CBCT-like images:

1. **Labels from µCT.** Each µCT volume is intensity-normalised and its canal is obtained by thresholding.
2. **Simulated CBCT.** Each µCT is degraded to look like a 75 µm CBCT (resolution, blur, noise, contrast). Since the simulated image is derived from the µCT, **the label is already aligned and no registration is needed**.
3. **Segmenter.** A 3D network is trained on simulated CBCT with µCT labels, and then applied to real 75 µm CBCT volumes.
4. **Evaluation on real CBCT**, held out from training, against manual segmentations.
5. **Measurements** (centreline, length, curvature, volume, diameter) computed on the predicted canal.

| Stage | Goal |
| :---: | --- |
| 1 | Tooth localisation and cropping in the CBCT |
| 2 | 3D segmentation of the root canals |
| 3 | 3D reconstruction of the internal anatomy |
| 4 | Geometric measurements on the 3D model |

## Repository structure

```
.
├── 1_Data_Validation/
│   └── analysis/microct/     µCT preprocessing (normalisation + canal thresholding)
│   └── analysis/...          earlier analysis of the Pulpy3D data set (legacy, see below)
├── 2_UNet/                   training framework (UNet, Attention UNet, ...)
├── .gitignore
└── README.md
```

> Only the code is versioned. Data, model weights, results, virtual environments and personal documents (papers, meeting notes, reports) are kept locally and excluded through `.gitignore`.

**Legacy.** The scripts under `1_Data_Validation/analysis/` (except `microct/`) and the configs in `2_UNet/` were written for the Pulpy3D CBCT data set (0.3 mm voxel). The project **no longer uses Pulpy3D data**: that code is kept for reference, and the training framework will be adapted to the µCT-based data.

### [`1_Data_Validation/analysis/microct/`](1_Data_Validation/analysis/microct/)

`preprocess_microct.py` reads a Nikon XT volume (`.raw`, uint16, described by its `.xtekVolume`), normalises it and extracts the canal. For each tooth it writes the normalised volume, the canal mask, an `info.json` and quality-control images.

- **Normalisation:** two robust anchors per volume (air and median tooth grey level) so that one threshold works across scans.
- **Canal:** the canal is air inside the tooth, so it is found as the cavity enclosed by tooth tissue in the axial slices; short cavities (pores, crown caries) are discarded.
- **Resolution:** `--downsample` trades speed for detail. Thin canals are lost at 64 µm; use a lower value for final labels.

### [`2_UNet/`](2_UNet/) — Segmentation training

3D UNet, Attention UNet and variants (UNETR, VNet, multi-head). Details: [`README.md`](2_UNet/README.md) (in Spanish).

## Data

Data is **not** included in the repository. Expected local layout:

| Folder | Content |
| --- | --- |
| `1_Data_Validation/datasets/MicroCT/Diente_XX/` | µCT of an extracted tooth (Nikon XT, 16 µm isotropic): `.raw`, `.xtekVolume`, `.xtekct` |
| `1_Data_Validation/datasets/CBCT_75um/` | real CBCT, 75 µm isotropic, with a manual canal segmentation (NRRD), used for calibration and testing |

CBCT volumes come from patients and must not be shared or committed.

## Getting started

```bash
git clone https://github.com/ClaudiaAgromayor/tfm_dl_root_canal.git
cd tfm_dl_root_canal
```

Environment (tested with Python 3.12 and an NVIDIA RTX 5080; the CUDA build of PyTorch must match the driver):

```bash
conda create -n tfm python=3.12 -y
conda activate tfm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install numpy scipy scikit-image matplotlib nibabel SimpleITK
pip install -r 2_UNet/requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Run the µCT preprocessing from the repository root:

```bash
python 1_Data_Validation/analysis/microct/preprocess_microct.py 1_Data_Validation/datasets/MicroCT/Diente_35 --downsample 4 --out 1_Data_Validation/results/microct
```

## Current status

- [x] Environment on the GPU machine (CUDA PyTorch working)
- [x] µCT normalisation and canal thresholding (3 teeth)
- [ ] Review the masks against an expert segmentation (none available yet)
- [ ] Canal labels at the full µCT resolution (16 µm)
- [ ] CBCT simulator at 75 µm, calibrated on the real CBCT
- [ ] Adapt the training framework to the simulated data and train
- [ ] Evaluate on real CBCT
- [ ] Geometric measurements (length, curvature, volume, diameter)

## References and acknowledgements

The training code builds on:

- [Pulpy3D-Seed](https://github.com/mahmoudgamal0/Pulpy3D-Seed) — Gamal et al., code and baseline.
- [alveolar_canal](https://github.com/AImageLab-zip/alveolar_canal) — AImageLab, ToothFairy Challenge (MICCAI 2023).

Key literature:

- Gao et al., BMC Oral Health 26:691 (2026) — root canal segmentation in CBCT guided by µCT.
- Lin et al., J Endod 47:1933–1941 (2021) — µCT-guided AI for pulp cavity and tooth segmentation on CBCT.
- Pauwels et al., Dentomaxillofac Radiol (2015) — CBCT grey values and Hounsfield units.
- Billot et al., Med Image Anal (2023) — SynthSeg: segmentation trained on synthetic images with randomised contrast and resolution.
- Milletari et al. (2016) — *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*.
