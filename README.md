# TFM: Segmentación automática y caracterización geométrica tridimensional de conductos radiculares en molares mediante Deep Learning para el apoyo a la planificación endodóntica

Trabajo Fin de Máster realizado en la Beca IIT (ICAI, Universidad Pontificia Comillas).

## Descripción

La anatomía de los conductos radiculares constituye uno de los principales factores que condicionan el éxito de los tratamientos endodónticos. La identificación precisa de la morfología interna del diente y la estimación de parámetros geométricos como el diámetro, la curvatura, la longitud o la sección transversal pueden proporcionar información de gran valor para la selección de instrumental y estrategias de tratamiento por parte del especialista.

Este Trabajo Fin de Máster propone el desarrollo de una metodología basada en inteligencia artificial para la segmentación automática de conductos radiculares en molares a partir de imágenes de tomografía computarizada de haz cónico (CBCT). Utilizando aprendizaje profundo, se abordará un flujo de trabajo compuesto por tres etapas:

1. Localización y extracción automática del diente.
2. Segmentación tridimensional de los conductos radiculares.
3. Obtención de una reconstrucción 3D detallada de la anatomía interna.

Posteriormente, se calcularán métricas geométricas relevantes para los modelos 3D generados.

## Estructura

> En GitHub solo están `1_Data_Validation/` y `2_UNet/` (el código). Los datos, papers, notas de reuniones e informes se quedan en local (ver `.gitignore`).

| Carpeta | Qué hay |
| --- | --- |
| `00_Papers/` | Artículos de referencia (ver índice abajo) |
| `01_Reuniones_y_notas/` | Notas de reuniones `p0`…`p8` en orden y bocetos a mano |
| `02_Informes_y_presentaciones/` | BRILLA (V2, V3), presentaciones del TFM y del pipeline microCT |
| `1_Data_Validation/` | Proyecto 1: validación del dataset Pulpy3D (393 pacientes buenos + 30 excluidos) y segmentación manual vs GT. Ver su `ORGANIZACION.md` |
| `2_UNet/` | Proyecto 2: entrenamiento de redes (UNet, Attention UNet...) + segmentaciones de 3D Slicer |

## Índice de papers (`00_Papers/`)

| Fichero | Título | Nombre original |
| --- | --- | --- |
| `Gamal_Pulpy3D_segmentacion_pulpa_canales_IAN.pdf` | Automatic Mandibular Semantic Segmentation of Teeth Pulp Cavity and Root Canals, and Inferior Alveolar Nerve on Pulpy3D Dataset | `1419_paper.pdf` |
| `Cipriano_2022_Mandibular_Canal_dataset_CBCT.pdf` | Deep Segmentation of the Mandibular Canal: A New 3D Annotated Dataset of CBCT Volumes | (igual) |
| `2025_Cross-Frequency_segmentacion_canales_primer_molar.pdf` | Cross-Frequency Collaborative Training Network and Dataset for Semi-supervised First Molar Root Canal Segmentation | `2504.11856v1.pdf` |
| `Milletari_2016_V-Net.pdf` | V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation | `1606.04797v1.pdf` |
| `Lin_2017_Feature_Pyramid_Networks.pdf` | Feature Pyramid Networks for Object Detection | `1612.03144v2.pdf` |
| `2022_Open-Full-Jaw_dataset_FEM.pdf` | Open-Full-Jaw: An open-access dataset and pipeline for finite element models of human jaw | `2209.07576v1.pdf` |
| `Elgarba_2025_IA_implantes_virtuales.pdf` | Clinical Feasibility of AI-Driven Automated Virtual Dental Implant Placement | `CID-27-0.pdf` |
| `Meto_2025_Review_CBCT_IA_RA_RV_dental.pdf` | The Integration of CBCT, AI, AR and VR in Dental Diagnostics, Surgical Planning, and Education: A Narrative Review | `applsci-15-06308.pdf` |
| `Gao_2026_BMC_segmentacion_canales_radiculares_CBCT.pdf` | Gao et al., BMC Oral Health 26:691 (2026): segmentación de canales radiculares en CBCT | `s12903-026-07918-2.pdf` |
| `Niane_Informe_practicas_Bigue.pdf` | Internship report: automatizar la detección de las dimensiones del canal radicular del primer molar | `rapport de stage.pdf` |
