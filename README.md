# MSc Thesis: Automatic Segmentation and Three-Dimensional Geometric Characterization of Molar Root Canals Using Deep Learning to Support Endodontic Planning

Master's thesis carried out under the IIT Scholarship (ICAI, Universidad Pontificia Comillas).

## Description

Root canal anatomy is one of the main factors determining the success of endodontic treatments. Accurately identifying the internal morphology of the tooth and estimating geometric parameters such as diameter, curvature, length or cross-section can provide highly valuable information for the specialist when selecting instruments and treatment strategies.

This MSc thesis proposes the development of an artificial-intelligence-based methodology for the automatic segmentation of molar root canals from cone-beam computed tomography (CBCT) images. Using deep learning, a workflow consisting of three stages will be addressed:

1. Automatic localization and extraction of the tooth.
2. Three-dimensional segmentation of the root canals.
3. Obtaining a detailed 3D reconstruction of the internal anatomy.

Subsequently, relevant geometric metrics will be computed for the generated 3D models.

## Structure

> Only `1_Data_Validation/` and `2_UNet/` (the code) are on GitHub. Data, papers, meeting notes and reports are kept locally (see `.gitignore`).

| Folder | Contents |
| --- | --- |
| `00_Papers/` | Reference articles (see index below) |
| `01_Reuniones_y_notas/` | Meeting notes `p0`…`p8` in order, and hand-drawn sketches |
| `02_Informes_y_presentaciones/` | BRILLA (V2, V3), thesis presentations and microCT pipeline presentation |
| `1_Data_Validation/` | Project 1: validation of the Pulpy3D dataset (393 good patients + 30 excluded) and manual segmentation vs. GT. See its `ORGANIZACION.md` |
| `2_UNet/` | Project 2: training of networks (UNet, Attention UNet...) + 3D Slicer segmentations |

## Paper index (`00_Papers/`)

| File | Title | Original name |
| --- | --- | --- |
| `Gamal_Pulpy3D_segmentacion_pulpa_canales_IAN.pdf` | Automatic Mandibular Semantic Segmentation of Teeth Pulp Cavity and Root Canals, and Inferior Alveolar Nerve on Pulpy3D Dataset | `1419_paper.pdf` |
| `Cipriano_2022_Mandibular_Canal_dataset_CBCT.pdf` | Deep Segmentation of the Mandibular Canal: A New 3D Annotated Dataset of CBCT Volumes | (same) |
| `2025_Cross-Frequency_segmentacion_canales_primer_molar.pdf` | Cross-Frequency Collaborative Training Network and Dataset for Semi-supervised First Molar Root Canal Segmentation | `2504.11856v1.pdf` |
| `Milletari_2016_V-Net.pdf` | V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation | `1606.04797v1.pdf` |
| `Lin_2017_Feature_Pyramid_Networks.pdf` | Feature Pyramid Networks for Object Detection | `1612.03144v2.pdf` |
| `2022_Open-Full-Jaw_dataset_FEM.pdf` | Open-Full-Jaw: An open-access dataset and pipeline for finite element models of human jaw | `2209.07576v1.pdf` |
| `Elgarba_2025_IA_implantes_virtuales.pdf` | Clinical Feasibility of AI-Driven Automated Virtual Dental Implant Placement | `CID-27-0.pdf` |
| `Meto_2025_Review_CBCT_IA_RA_RV_dental.pdf` | The Integration of CBCT, AI, AR and VR in Dental Diagnostics, Surgical Planning, and Education: A Narrative Review | `applsci-15-06308.pdf` |
| `Gao_2026_BMC_segmentacion_canales_radiculares_CBCT.pdf` | Gao et al., BMC Oral Health 26:691 (2026): root canal segmentation in CBCT | `s12903-026-07918-2.pdf` |
| `Niane_Informe_practicas_Bigue.pdf` | Internship report: automating the detection of first molar root canal dimensions | `rapport de stage.pdf` |
