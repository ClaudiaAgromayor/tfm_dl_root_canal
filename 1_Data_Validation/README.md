# 1_Data_Validation

Estudio y validación de datos del dataset **Pulpy3D** (CBCT dental):

- **Calidad del ground truth** de Pulpy3D: máscaras vacías, coherencia entre `gt_pulp_mandible` y `gt_instance`, molares 36/46.
- **Segmentación manual (3D Slicer) vs ground truth**: DICE y HD95 del diente completo, de la zona de canales radiculares y del nervio alveolar inferior (IAN).

El mapa completo (qué hace cada script, dónde guarda sus resultados y qué problemas se han corregido) está en **[ORGANIZACION.md](ORGANIZACION.md)**.

## Uso rápido

Desde esta carpeta, con el entorno `.venv` (ya tiene todo instalado):

```
.venv\Scripts\python.exe analysis/canales/canales_por_capa.py
.venv\Scripts\python.exe analysis/ian/plot_ian_hd95.py P48
```

Los resultados salen en `results/`, con la misma estructura de carpetas que `analysis/`.

## Datos

- `datasets/Pulpy3D/`: 393 pacientes de Pulpy3D (`data.nii.gz` + `gt_*.nii.gz`). Descarga: [enlace](https://drive.google.com/drive/folders/1M5iU1urLOp1rSxKOm7WCzodAKcZrqT5O?usp=sharing).
- `datasets/Pulpy3D_excluidos/`: los 30 pacientes quitados por tener el GT vacío.
- `datasets/P459_3DSlicer_prueba/`: primera segmentación manual de prueba (P459).
- Segmentaciones manuales de 3D Slicer (33 pacientes): en `../2_UNet/datasets/Segmentaciones 3DSlicer/`.

## Proyectos relacionados

- `../2_UNet`: entrenamiento de redes de segmentación (UNet, Attention UNet...). Contiene el framework de entrenamiento que antes estaba aquí, basado en el ToothFairy Challenge (MICCAI 2023) y en [alveolar_canal](https://github.com/AImageLab-zip/alveolar_canal) de AImageLab.
