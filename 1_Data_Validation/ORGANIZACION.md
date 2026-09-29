# 1_Data_Validation — organización

Proyecto 1 de la beca: **estudio y validación de datos**. Compara las segmentaciones manuales hechas en 3D Slicer con el ground truth (GT) de Pulpy3D y revisa la calidad del propio dataset.

Qué hay en cada carpeta, qué hace cada fichero y por qué está donde está. Al final: **ficheros rotos, repetidos o sin uso**.

## Dónde está cada proyecto

```
Beca IIT/
├── 00_Papers/                ← artículos de referencia (renombrados: Autor_Año_Tema.pdf)
├── 01_Reuniones_y_notas/     ← p0…p8 (notas de reuniones) + bocetos
├── 02_Informes_y_presentaciones/  ← BRILLA, presentaciones TFM / microCT
├── 1_Data_Validation/        ← ESTE proyecto (antes Pulpy3D_Dataset/Pulpy3D_GitHub)
├── 2_UNet/                   ← proyecto 2: entrenamiento de redes (UNet, Attention UNet...)
│   └── datasets/Segmentaciones 3DSlicer/   ← las segmentaciones manuales viven aquí ahora
└── (Pulpy3D_Dataset/ ya no existe: lo útil se movió aquí)
```

- El **código de entrenamiento** (`main.py`, `eval.py`, `models/`, `experiments/`, `losses/`...) se ha movido a `../2_UNet`, con todos los errores ya corregidos. Aquí solo queda el análisis de datos.
- Las **segmentaciones de 3D Slicer** están en `../2_UNet/datasets/Segmentaciones 3DSlicer/`, y los scripts de aquí ya apuntan a esa ruta.
- **Pulpy3D:** la descarga original tenía 423 pacientes. `datasets/Pulpy3D/` tiene los 393 buenos (los que usan todos los scripts) y `datasets/Pulpy3D_excluidos/` los 30 quitados.
- **Los 30 pacientes excluidos** son exactamente los de `results/dataset_checks/pacientes_canal_sin_info.csv` (16, `gt_pulp_mandible` vacío o casi) y `pacientes_instance_vacios.csv` (14, `gt_instance` vacío o casi). Comprobado sobre la descarga original: ninguno pasa de 589 vóxeles en esa máscara.

La organización separa tres cosas:

| Qué | Dónde | Regla |
| --- | --- | --- |
| **Datos** (entrada, no se tocan) | `datasets/` | Solo datos originales. Nada generado por scripts. |
| **Código** | `analysis/` | Solo `.py`. Nada de imágenes ni Excel. |
| **Resultados** (salidas) | `results/` | Todo lo que generan los scripts, con **la misma estructura de carpetas que `analysis/`**. |

> **Cómo ejecutar:** siempre desde la raíz de `1_Data_Validation`, p. ej. `.venv\Scripts\python.exe analysis/canales/canales_por_capa.py`.
> Muchos scripts guardan en `Path("results") / ...` (relativo a desde dónde ejecutas), y los scripts que se importan entre sí están en la misma carpeta.
> Dependencias: `analysis/requirements.txt`, ya instaladas en `.venv`.

---

## Vista general

```
1_Data_Validation/
├── README.md, ORGANIZACION.md
├── datasets/
│   ├── Pulpy3D/                              ← 393 pacientes + dataset.json + splits.json
│   ├── Pulpy3D_excluidos/                    ← los 30 pacientes quitados por GT vacío
│   └── P459_3DSlicer_prueba/                 ← primera segmentación manual de prueba (P459)
├── analysis/                                 ← MIS SCRIPTS
│   ├── dataset_checks/   calidad del GT de Pulpy3D
│   ├── manual_vs_gt/     manual vs GT, diente completo
│   ├── canales/          manual vs GT, solo la zona de canales
│   ├── ian/              nervio alveolar inferior (IAN)
│   ├── por_paciente/pXX/ scripts ajustados a mano para cada paciente (IAN + vista 3D)
│   ├── exploracion/      pruebas y visores de los primeros días
│   ├── utils/
│   └── requirements.txt
├── results/                                  ← MIS RESULTADOS (ignorado por git)
│   ├── dataset_checks/  manual_vs_gt/  canales/  ian/  por_paciente/pXX/
│   ├── exploracion/     imágenes sueltas de pruebas
│   └── obsoleto/        versiones antiguas que se han sustituido
├── docs/Tabla_Exclusion_Inclusion_ToothFairy.pdf
└── .venv/                                    ← entorno con todo lo que usan los análisis
```

---

## 1. Framework de entrenamiento → movido a `../2_UNet`

Todo el código para entrenar (`main.py`, `eval.py`, `requirements.txt`, `configs/`, `dataloader/`, `experiments/`, `losses/`, `models/`, `optimizers/`, `schedulers/`) está ahora en `../2_UNet`. Su descripción y los pasos para entrenar están en `../2_UNet/README.md`. Sigue en el historial de git de este repositorio (commits anteriores).

---

## 2. `datasets/` — datos

### `datasets/Pulpy3D/` (393 pacientes, 20 GB)
Cada `P<N>/` contiene `data.nii.gz` (CBCT), `gt_pulp`, `gt_pulp_mandible`, `gt_instance`, `gt_ian`, `gt_instance_ian` (`.nii.gz`), y en 281 pacientes también `gt_pulp_ian.nii.gz`. 258 tienen `meta.json` (`{"circular": true}`).
- `dataset.json`: mapa de etiquetas.
- `splits.json`: reparto train/val/test para entrenar sobre Pulpy3D (315 / 39 / 39, semilla 33).

### `datasets/Pulpy3D_excluidos/` (30 pacientes)
Los pacientes quitados del dataset por tener `gt_instance` o `gt_pulp_mandible` vacío o casi (ver arriba). Mismo formato que `Pulpy3D/`. No los usa ningún script; se guardan como prueba de por qué se excluyeron.

### `datasets/P459_3DSlicer_prueba/`
`1 data.nrrd` + `Segmentation_1.seg.nrrd`: la primera segmentación manual de prueba (P459, marzo). La usan los scripts de `analysis/exploracion/test_P459/`. No está en `../2_UNet/datasets/Segmentaciones 3DSlicer/` porque no es una escena completa (no tiene `.mrml`).

### Segmentaciones de 3D Slicer (33 pacientes, 2 GB) → en `../2_UNet/datasets/Segmentaciones 3DSlicer/`
Cada `P<N>/` es una **escena de 3D Slicer**: `<fecha>-Scene.mrml` (la escena), `<fecha>-Scene.png` (miniatura), `1 data.nrrd` (el CBCT) y `Segmentation.seg.nrrd` (la segmentación manual; en P5 se llama `Canal P5.seg.nrrd`). No hay que borrar nada de ahí: la escena `.mrml` hace referencia a esos ficheros por nombre.

Casos especiales (todos referenciados por su `.mrml`, por eso se quedan):
- **P1**: además tiene los GT exportados (`1 gt_*.nrrd`) y `1 NIfTI to DICOM Conversion.*`.
- **P195**: `Segmentation-label.nrrd` + `Segmentation_ColorTable.csv` (la usan `por_paciente/p195` e `ian/ian_cortes_verticales.py`) y `1 data_1.nrrd`.
- **P133**: `Displacement to color.txt` (tabla de colores de Slicer).

---

## 3. `analysis/` — mis scripts

### `analysis/dataset_checks/` → `results/dataset_checks/`
Revisan el GT oficial de Pulpy3D (no las segmentaciones manuales, salvo el último).

| Archivo | Antes se llamaba | Qué hace |
| --- | --- | --- |
| `check_instance_vacia.py` | `pacientes_instance.py` | Pacientes con `gt_instance` vacío → `pacientes_instance_vacios.csv` |
| `check_pulp_mandible_vacia.py` | `pacientes_canal.py` | Pacientes con `gt_pulp_mandible` vacío → `pacientes_canal_sin_info_2.csv` |
| `dice_pulp_vs_instance_todos.py` | `verificar_pulp_instance.py` | DICE `gt_pulp_mandible` vs `gt_instance` en los 393 pacientes → `dice_pulp_vs_instance.csv` |
| `dice_pulp_vs_instance_P459.py` | `compare_verificar.py` | Lo mismo para un solo paciente (P459), corte a corte, y muestra el peor corte |
| `solape_pulp_con_molares_36_46.py` | `pacientes_malos_1molar.py` | En los pacientes "malos", ¿la pulpa toca el 36, el 46, ambos o ninguno? → `pacientes_malos_1molar_resumen.csv` |
| `capas_anotadas_slicer.py` | `pacientes_capas_anotadas.py` | Capas Z y rango anotado en cada segmentación manual → `pacientes_cortes_profundidad.txt` |

En `results/dataset_checks/` también hay CSV de versiones anteriores de estos scripts (`pacientes_canal_sin_info.csv`, `pacientes_dice_bueno.csv`, `pacientes_dice_malo.csv`, `pacientes_malos_tipo.txt`, `num_pixeles_segmentaciones.csv`, `spacing_headers_summary.csv`, `pacientes_cortes_profundidad.xlsx`).

### `analysis/manual_vs_gt/` → `results/manual_vs_gt/`
Segmentación manual (3D Slicer) vs `gt_instance` (36 = derecha, 46 = izquierda), diente completo.

| Archivo | Antes | Qué hace | Salida |
| --- | --- | --- | --- |
| `dice_manual_vs_gt_reglas.py` | `compare_copy.py` | DICE por paciente con reglas `SOLO_36/SOLO_46/AMBOS`, quitando el IAN; también DICE del IAN | `dice_manual_vs_gt_reglas_final.xlsx`, `figuras_dice/` |
| `dice_hd95_diente_paso_a_paso.py` | `dice_paso_a_paso.py` | DICE + HD95 3D + análisis corte a corte para la lista `CASES` | `dice_paso_a_paso/` |
| `figuras_manual_vs_gt.py` | `fotos.py` | Figura GT / manual / superposición de cada paciente | `figuras/` |

### `analysis/canales/` → `results/canales/`
Igual que lo anterior, pero solo en las capas donde el diente ya se ha dividido en canales (≥ 3 manchas por corte). Los scripts se importan entre sí:

```
canales_finales.py   (módulo base: TARGETS, carga de datos, máscaras, DICE, figuras)
 ├─ canales_por_capa.py  ─┬─ hd95_canales_finales.py
 │                        └─ hd95_diente_completo.py
 ├─ previews_antes_inicio_canales.py
 └─ previews_P5_z60_67.py
```

| Archivo | Antes | Qué hace | Salida |
| --- | --- | --- | --- |
| `canales_finales.py` | `compare_canales_finales.py` | **Módulo base** + 1.ª versión del análisis de canales | `canales_finales/` |
| `canales_por_capa.py` | `canals_analysis.py` | **Versión buena** del análisis de canales (separa lados en P217/P343), DICE por capa | `canales_por_capa/` + `resumen_canales_pacientes.xlsx` |
| `hd95_canales_finales.py` | (igual) | HD95 3D solo en la zona de canales | `hd95_canales_finales/` |
| `hd95_diente_completo.py` | `hd95_paso_a_paso.py` | DICE + HD95 3D del diente entero | `hd95_diente_completo/` |
| `excel_canales_desde_P111.py` | `build_canals_detail_workbook.py` | Excel con una hoja por paciente (desde P111): tabla + 3 imágenes. Requiere haber ejecutado `canales_por_capa.py` | `canales_por_capa/resumen_canales_pacientes_desde_P111.xlsx` |
| `previews_antes_inicio_canales.py` | `compare_preview_prestart.py` | Imágenes de las 5 capas antes del inicio de canales (P5, P35) | `canales_finales/<P>/prestart_slices/` |
| `previews_P5_z60_67.py` | `generate_specific_slices_P5.py` | Capas z = 60–67 de P5 | `canales_finales/P5/prestart_more/` |

### `analysis/ian/` → `results/ian/`

| Archivo | Antes | Qué hace |
| --- | --- | --- |
| `ian_targets.py` | (igual) | Configuración única del IAN: rutas, etiquetas y lado del IAN de cada paciente |
| `ian_metrics.py` | (igual) | Funciones: cargar IAN GT/manual, elegir lado, DICE, HD95 |
| `plot_ian_hd95.py` | (igual) | Figura que explica el HD95 de un paciente: `python analysis/ian/plot_ian_hd95.py P48` → `hd95_explicado/` |
| `ian_cortes_verticales.py` | `compare_ian_vertical_slices.py` | IAN manual vs GT en cortes verticales (eje Y) → `cortes_verticales/`, `hd95_cortes_verticales/` |

También en `results/ian/`: `ian_hd95_summary.csv/.xlsx` (de un script que ya no existe), `test_ian_metrics_P35.txt` (prueba de `ian_metrics`: DICE 0.313, HD95 247 vox) y `P61_gt_ian_por_lado/` (el IAN de P61 separado en derecho/izquierdo; antes estaba metido dentro del dataset).

### `analysis/por_paciente/pXX/` → `results/por_paciente/pXX/`
31 pacientes. **Son copias del mismo script ajustadas a mano** (lado, recorte en Y, etiqueta del IAN). Antes estaban en `ian_hd95/` en la raíz, mezclando código con imágenes; ahora los `.py` están aquí y los PNG/HTML en `results/por_paciente/pXX/`.

| Archivo (nombre nuevo) | Antes | Qué hace |
| --- | --- | --- |
| `ian_dice_hd95.py` | `ian_hd95_dice_hd_<N>.py` | DICE 3D + peor HD95 2D (eje Y); guarda `p<N>_worst_hd95_y<Y>.png` y abre la vista 3D |
| `ian_vista3d.py` | `ian_hd95_plot3d.py` | Vista 3D interactiva GT vs manual → `ian_3d_visualization.html` |
| `canales_vista3d.py` (solo p137, p208, p217, p343, p394, p402, p466) | `canales_plot3d.py` | Igual pero del molar (36/46) → `canales_3d_visualization.html` |
| `ian_visor2d.py` (solo p35) | `ian_hd95_plot2d.py` | Visor 2D con slider |

Los HTML se abren con doble clic en cualquier navegador.

### `analysis/exploracion/` → `results/exploracion/`
Scripts de los primeros pasos. Útiles para mirar datos; **no producen resultados finales**.

| Archivo | Antes | Qué hace |
| --- | --- | --- |
| `visor_nii_slider.py` | `read_nii.py` | Enseña los `.nii.gz` de P35 y un visor con slider del diente 36 |
| `prueba_transformaciones_P35.py` | `test_transformations.py` | Prueba flips del IAN manual de P35 para alinearlo con el GT |
| `overlay_ian_P35.py` | `visualize_overlay_comparison.py` | Superposición IAN manual (rotado 180°) vs GT en P35 |
| `primera_version_dice_46.py` | `compare.py` | 1.ª versión del DICE manual vs GT (solo diente 46), sustituida por `manual_vs_gt/dice_manual_vs_gt_reglas.py` |
| `test_P459/comparacion_dice.py` | `Pulpy3D_Dataset/Test/` | DICE por capa de la segmentación de prueba de P459 vs `gt_pulp` → `results/exploracion/test_P459/dice_por_capa.csv` + `salida_comparacion_dice/` |
| `test_P459/comparar_segmentacion_pulpa.py` | (igual) | Figuras segmentación de prueba vs GT de P459 → `comparacion_side.png`, `superposicion.png` |
| `test_P459/comparar_tamanos.py` | (igual) | Compara tamaños/forma de los volúmenes `.nrrd` y `.nii.gz` de P459 |

### `analysis/utils/`
| Archivo | Qué hace |
| --- | --- |
| `convert_csv_sep_decimal.py` | Pasa `results/canales/hd95_diente_completo/hd95_global_summary.csv` a formato Excel español (`;` y coma decimal) |
| `nii_to_dicom.py` | Convierte cada `.nii.gz` de Pulpy3D a una serie DICOM (lee `datasets/Pulpy3D`, escribe en `results/dicom/Pulpy3D_DICOM`) |
| `combine_nii_to_dicom.py` | Igual, pero combina CBCT + máscaras en una sola serie DICOM (`results/dicom/Pulpy3D_DICOM_Combined`) |

### `results/excels/`
Excels de resultados hechos a mano (`Resultados_Datos_1*.xlsx/.xlsm`: general, canales, IAN). Antes estaban sueltos en `Beca IIT/`. La versión `_canales_OLD` está en `results/obsoleto/`.

---

## 4. Ficheros ROTOS, REPETIDOS o SIN USO

### Rotos en los análisis (arreglados)
| Fichero | Problema | Arreglo |
| --- | --- | --- |
| `por_paciente/p59/ian_dice_hd95.py` | Copia de p61 sin adaptar: **leía los datos de P61 y sobrescribía la imagen de P61** | Lee P59 y guarda en p59. Imagen regenerada |
| `por_paciente/p81/ian_dice_hd95.py` | Copia de p100 sin adaptar: **leía P100 y sobrescribía la imagen de P100** | Lee P81. Imagen regenerada |
| `por_paciente/p208/ian_dice_hd95.py` | Imagen de fondo recortada con `95:265` y máscaras con `97:241`: superposición desalineada | Ambos `97:241`. Imagen regenerada |
| `por_paciente/*/ian_dice_hd95.py` (28 de 31) | El Y del PNG se calculaba sumando 80 o 122 fijos, en vez del inicio del recorte | Ahora `Y_CROP_START` = inicio del recorte, así que el Y es el índice del corte en la segmentación de 3D Slicer. **Las 31 imágenes están regeneradas**; las antiguas están en `results/obsoleto/por_paciente_Y_mal_calculado/`. El valor HD95 no cambia |
| `ian/plot_ian_hd95.py` | Leía la etiqueta del IAN de un Excel **que nunca existió** → siempre usaba la etiqueta 2 (mal en P217, P343 y P447, donde el IAN es la 3) | Lee `results/canales/canales_por_capa/resumen_canales_pacientes.xlsx` |
| `exploracion/primera_version_dice_46.py` | Buscaba `pacientes_dice_bueno.csv` en la raíz → no hacía nada | Ruta corregida |
| `exploracion/overlay_ian_P35.py`, `prueba_transformaciones_P35.py` | Guardaban PNG **dentro del dataset** | Guardan en `results/exploracion/` |
| `../2_UNet/datasets/Segmentaciones 3DSlicer/P195/*.mrml` | La escena tenía 2 segmentaciones vacías y una miniatura que apuntaban a ficheros que no existen | Quitados esos nodos; la segmentación buena (`Segmentation.seg.nrrd`) no se toca. Copia de la escena original en `results/obsoleto/` |
| `.venv/` | Le faltaban scipy, matplotlib, plotly y scikit-image | Instaladas. **Usa `.venv` para los análisis**: es el único entorno que lo tiene todo (el Python del sistema no tiene SimpleITK) |
| `README.md` | Decía `configs/segmentation.yml`, que no existe | Corregido |

Nota: todos los scripts de `por_paciente` recortan en Y una ventana que deja fuera los extremos del IAN. Es a propósito ("tu recorte manual"): se compara solo la zona central del canal. Lo he comprobado paciente a paciente y cada recorte está ajustado a su propio paciente, también en P59 y P81.

### Rotos en el framework (arreglados; el código está ahora en `../2_UNet`; sin probar porque aquí no hay GPU NVIDIA)
| Fichero | Problema | Arreglo |
| --- | --- | --- |
| `datasets/Pulpy3D/splits.json` | **Vacío** (0 pacientes en train/val/test): no se podía entrenar. La versión del último commit tenía 30 pacientes que no existen y val/test vacíos | Reparto nuevo y reproducible (semilla 33) con los 393 pacientes: **315 train / 39 val / 39 test** |
| `experiments/experiment.py` → `predict` | Estaba **fuera de la clase** (mal indentado): `do_predict: True` fallaba. Además desempaquetaba mal los datos y guardaba las predicciones sin la geometría del paciente (`affine=np.eye(4)`) | Dentro de la clase; usa la affine del CBCT; soporta MultiHead (1 = pulpa, 2 = IAN) |
| `experiments/experiment.py` → `load` | Fallaba al cargar en CPU o con torch ≥ 2.6 | `map_location` y `weights_only=False` |
| `main.py` | Recogía `train()` como (iou, dice, hd95), pero devuelve (loss, iou, hd95): **el log llamaba "IoU" a la pérdida** | Corregido |
| `main.py` | Con Adam + Plateau pasaba el número de época como si fuera la métrica | Plateau siempre recibe el DICE de validación |
| `main.py` | `do_test`/`do_predict` tras entrenar cargaban `./checkpoints/last.pth` (no existe), no el modelo recién entrenado | Usan el `best.pth` del entrenamiento |
| `main.py` | Comprobaba si existe el checkpoint **después** de intentar cargarlo | Se comprueba antes |
| `main.py` | wandb con proyecto vacío y la clave borrada: fallaba sin `--debug` | Por defecto guarda en local (offline); `--debug` lo desactiva |
| `main.py` | Mensaje de aviso con una variable inexistente (`config.augmentations`) | Corregido |
| `main.py` | Con val/test vacíos dividía entre 0 | Avisa y para con un mensaje claro |
| `eval.py` | **El HD95 multiclase no era un HD95** (media del 5 % mayor de una norma de la diferencia) | HD95 real por clase (MedPy), sin contar el fondo |
| `eval.py` | DICE/IoU multiclase calculados sobre probabilidades | Sobre la predicción final (argmax), como en binario |
| `eval.py` | El DICE por clase se acumulaba desde la época 1 y nunca se reiniciaba | Se reinicia cada época |
| `losses/DiceCrossEntropy.py` | **Softmax aplicado dos veces** (el modelo ya termina en softmax y `CrossEntropyLoss` lo vuelve a aplicar) | NLL sobre `log(pred)` |
| `experiments/multihead.py` | Unión pulpa + IAN con **suma**: vale 2 donde se solapan y rompía IoU/DICE (`1 & 2 = 0`) | Máximo en vez de suma |
| `experiments/ian_segmentation.py`, `instance.py` | Cargaban **dos veces** los datasets (primero los de pulpa y luego los suyos) | Solo cambian la etiqueta o la transformación |
| `schedulers/SchedulerFactory.py` | `gamma`/`patience` a `None` rompían; `verbose` ya no existe en torch reciente | Valores por defecto; sin `verbose` |
| `dataloader/Pulpy.py` | El sampler por etiqueta buscaba `gt`, que no existe en MultiHead | Usa la etiqueta del experimento |
| `configs/augmentation.yml` | `degrees: [15, 15]` = girar **siempre exactamente 15°** | `[-15, 15]` = aleatorio entre -15° y 15° |
| `configs/*.yml` | Guardaban los entrenamientos en `./results`, mezclados con los análisis | `./results/entrenamientos` |

### Lo único que no se puede arreglar desde aquí
| Qué | Por qué | Cómo arreglarlo |
| --- | --- | --- |
| `../2_UNet/datasets/Segmentaciones 3DSlicer/P59/*.mrml` | El preset de renderizado 3D apunta a `C:/Users/user/Documents/CT-Chest-Contrast-Enhanced.vp.json` (otro ordenador) y su contenido no está guardado en la escena. **Solo afecta al color del render 3D**, no a la segmentación | En 3D Slicer: abrir la escena → Volume Rendering → elegir el preset *CT-Chest-Contrast-Enhanced* → File → Save |

### Repetidos
| Qué | Estado |
| --- | --- |
| 8 copias descomprimidas `*.nii/` dentro de `datasets/Pulpy3D`, 1,3 GB | **Borradas** (idénticas byte a byte a su `.nii.gz`) |
| `ian_hd95.zip`, `_test_log.txt`, `__pycache__/` | **Borrados** (copia antigua, duplicado y caché) |
| `por_paciente/*` | 31 × 2 scripts casi idénticos (cambian lado, recorte y etiqueta). Se podrían unificar en un solo script + una tabla |
| `dice_score`, `hd95_3d`, `align_shape_with_permutations`, `extract_label_map` | Copiadas en 6–8 scripts |
| `canales_finales.py` vs `canales_por_capa.py` | Dos algoritmos para el inicio de los canales; el bueno es `canales_por_capa.py`. El otro se mantiene porque se importan sus funciones |
| `models/*/networks_other.py` | Idénticos en dos carpetas; vienen así del repo original |
| `results/obsoleto/` | Versiones antiguas ya sustituidas. Se puede borrar entero |

### Sin uso
| Qué | Comentario |
| --- | --- |
| `gt_pulp_ian.nii.gz`, `meta.json` del dataset | No los lee ningún script; son parte del dataset oficial |
| `models/PosPadUNet3D.py`, `UNet3D.py`, `UNETR.py`, `InstanceUNETR.py`, `VNet3D.py` | Ningún config los usa, pero se pueden elegir con `model.name` |
| `ian_targets.py` → `OUTPUT_XLSX`, `OUTPUT_CSV` | Nadie las usa (el script que escribía `ian_hd95_summary` ya no está) |
| `results/exploracion/*` | Imágenes de pruebas antiguas |

---

## 5. Cómo entrenar

El entrenamiento está en el proyecto `../2_UNet`: los pasos están en `../2_UNet/README.md`.
