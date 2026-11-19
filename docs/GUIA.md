# Guía de uso del proyecto

**Para qué sirve este documento.** Es el **manual**: qué es el proyecto en cinco líneas, por dónde pasan los datos, **dónde está cada carpeta y cada fichero, qué hace cada script y cómo se ejecuta, cómo repetirlo todo con un diente nuevo, cómo abrir las salidas, las preguntas que te pueden hacer y un glosario**. Es lo primero que hay que leer.

**A quién va dirigido.** A Claudia: estudiante de informática sin conocimientos de odontología. Sirve para aprender a manejar el proyecto y para explicárselo a Luis (tutor) y a los dentistas.

**Qué NO está aquí:**
- Por qué se decidió cada cosa, qué números salieron, qué se descubrió y qué falta → [PROYECTO.md](PROYECTO.md).
- Cómo se miden longitud, curvatura, ángulos y diámetros → [MEDICION.md](MEDICION.md).
- Qué dicen los papers → [LITERATURA.md](LITERATURA.md).

**Cómo leerla.** Las frases son cortas y cada término técnico se explica la primera vez (al final, apartado 8, hay un glosario). La etiqueta **[no verificado]** marca lo que no se ha comprobado ejecutándolo o con un experto. Los números «medidos» salen de ejecuciones del 2026-10-28 o anteriores (cada uno dice cuál). El comando `python` es `C:\Users\tfg\.conda\envs\tfm\python.exe` (entorno conda `tfm`), o simplemente `python` con `(tfm)` activo. Si tienes poco tiempo: apartados **1 y 2** (el problema y el camino de los datos, 10 minutos), luego **7** (preguntas frecuentes) y después el resto según lo que necesites ejecutar.

---

## 1. El problema en 5 líneas

1. Un endodoncista (dentista que trata el interior del diente) necesita conocer los **conductos radiculares**: los tubitos de la raíz por donde pasan el nervio y los vasos. Quiere su longitud, curvatura, volumen y diámetro.
2. En la clínica solo hay un **CBCT** (un escáner 3D de rayos X, con vóxeles de 75 µm). Es borroso y ruidoso: los conductos finos se distinguen mal.
3. Segmentar a mano cada conducto en un CBCT cuesta mucho y no hay datos etiquetados por expertos. Por eso queremos que una red neuronal lo haga sola.
4. Truco: un **micro-TC (µCT)** de un diente extraído se ve nítido (vóxel de 16 µm) y el conducto sale casi solo por umbral. Con ese µCT **fabricamos CBCT falsos** (más borrosos, con ruido). Como el falso viene del µCT, **la etiqueta ya está alineada con la imagen: no hay que registrar nada**.
5. Entrenamos la red con los falsos y la probamos en el CBCT real de Isa. Después se mide sobre la segmentación (línea central y distancia a la pared; ya hecho sobre los µCT).
6. **Uso final:** el dentista meterá un CBCT entero (siempre de 75 µm) y pedirá el 36 o el 46; el sistema segmentará y medirá solo los conductos de ese diente. Hoy la red **no** distingue un diente de sus vecinos: ver PROYECTO.md, apartado 8.

---

## 2. El camino de los datos, paso a paso

```
(0) datasets/MicroCT/Diente_XX/            .raw (uint16, 2,4-2,9 GB) + .xtekVolume (tamaño y vóxel)
        |
        |  preprocess_microct.py     [python analysis/microct/preprocess_microct.py datasets/MicroCT/Diente_XX]
        v
(1) results/microct/Diente_XX/             norm.npy (imagen normalizada), canal.npy (máscara completa),
        |                                  info.json, qc.png, qc_sagital_coronal.png
        |  split_canal.py            [python analysis/microct/split_canal.py results/microct/Diente_XX]
        v
(2) misma carpeta + canal_solo_conducto.npy (LA ETIQUETA), camara_y_cavidades.npy, canal_completo.npy,
        |           info.json ampliado (orientación, orificio, ápice), control_split.png, control_apice.png
        |
        +--> export_slicer.py        -> results/para_slicer/*.nrrd   (para REVISAR en 3D Slicer)
        |
        |  calibrate_cbct.py         (aparte, con el CBCT real de Isa) -> results/cbct_calibration/Isa_caso00/calibration.json
        |        |                    mide grises, ruido y borde del CBCT real; sirve para fijar los rangos del simulador
        v        v
(3) simulate_cbct.py  (simulador, lee los dientes de (2) y simulate_cbct_config.yaml)
        |   parches de 128x128x128 vóxeles de 75 µm: imagen simulada + etiqueta blanda + mapa de pesos
        |   (en el entrenamiento se generan al vuelo; con la línea de comandos se guardan en results/sim_cbct/*.npz)
        v
(4) 2_UNet/train_sim.py  -> 2_UNet/results/entrenamientos_sim/<nombre>/last.pt (pesos), log.txt, metrics.json
        |
        v
(5) medidas del conducto sobre la etiqueta del µCT: medir_conductos.py -> results/medidas/   (HECHO; ver MEDICION.md)
(6) [PENDIENTE] aplicar la red a un CBCT real y medir sobre su predicción. El uso final (recortar al 36/46) está en PROYECTO.md, apartado 8.
```

| Paso | Qué entra | Qué sale | Script | Comando (desde `D:\tfm_claudia\1_Data_Validation`) |
|---|---|---|---|---|
| 1 | `.raw` + `.xtekVolume` de un diente | `norm.npy`, `canal.npy`, `info.json`, 2 PNG de control | `analysis/microct/preprocess_microct.py` | `python analysis/microct/preprocess_microct.py datasets/MicroCT/Diente_35` |
| 2 | la carpeta del paso 1 | etiqueta `canal_solo_conducto.npy` y más | `analysis/microct/split_canal.py` | `python analysis/microct/split_canal.py results/microct/Diente_35` |
| 2b | carpeta del paso 2 | NRRD para Slicer (32 µm) | `analysis/microct/export_slicer.py` | `python analysis/microct/export_slicer.py results/microct/Diente_35 --out results/para_slicer` |
| 3a | CBCT real + segmentación de Isa | `calibration.json` + figura | `analysis/microct/calibrate_cbct.py` | ver sección 4.4 |
| 3b | carpetas del paso 2 + yaml + calibración | parches `.npz` + figuras | `analysis/microct/simulate_cbct.py` | ver sección 4.5 |
| 4 | carpetas del paso 2 + yaml del simulador | `last.pt` (solo prueba de montaje) | `2_UNet/train_sim.py` | `cd ..\2_UNet` y `python train_sim.py -c configs/sim_canal.yml --holdout Diente_52` |
| 5 | carpeta del paso 2 | línea central, longitud, curvatura, ángulos, diámetros, figuras | `analysis/microct/medidas/medir_conductos.py` | `python analysis/microct/medidas/medir_conductos.py results/microct/Diente_35` |

Qué significa cada paso en cristiano:
- **Paso 1.** El µCT trae el diente en gris. Se reescalan los grises para que "aire = 0" y "dentina = 1" valgan en todos los dientes. El conducto es un hueco con aire dentro del diente, así que se encuentra buscando "huecos encerrados por el diente".
- **Paso 2.** La máscara del paso 1 incluye también la cámara pulpar (la sala grande de la corona donde se juntan los conductos) y agujeros de caries. Claudia decidió que la etiqueta sea **solo el conducto**: se corta en un plano llamado **orificio** (donde los conductos se funden en la cámara).
- **Paso 3.** El simulador degrada el µCT hasta que "parece" un CBCT: lo emborrona, lo baja a 75 µm y le añade ruido.
- **Paso 4.** La red (una U-Net 3D) aprende a devolver la etiqueta a partir de la imagen simulada.

---

## 3. Mapa de carpetas y ficheros que tengo que conocer

Raíz del repositorio: `D:\tfm_claudia`. Hay tres `.gitignore` (raíz, `1_Data_Validation/`, `2_UNet/`).

| Carpeta | Qué hay | ¿Va a git? |
|---|---|---|
| `CLAUDE.md` | Contexto del proyecto y estado al cerrar cada día. Lo lee Claude Code al empezar. | **Sí** |
| `README.md` | Presentación del repositorio y comandos básicos. | **Sí** |
| `docs/` | Los cuatro documentos: `GUIA.md` (esta), `PROYECTO.md`, `MEDICION.md` y `LITERATURA.md`. | **Sí** |
| `1_Data_Validation/datasets/MicroCT/` | Los µCT originales (Diente_35, 51, 52): `.raw` de 2,4–2,9 GB. | **No** (datos) |
| `1_Data_Validation/datasets/CBCT_75um/` | `Isa_caso00` (el CBCT que se usó para calibrar) e `Isa_caso01…10` (entrega de noviembre): cada carpeta con `CBCT.nrrd`, `Segmentation.seg.nrrd` (borrador), `DICOM_original.zip` y `LEEME.txt`; los casos 5, 9 y 10 solo tienen DICOM. **Datos de paciente: no compartir; los zips llevan nombres de pacientes.** | **No** |
| `1_Data_Validation/datasets/Pulpy3D/` | Solo dos `.json` pequeños del proyecto antiguo (Pulpy3D, descartado). | Sí (2 ficheros viejos) |
| `1_Data_Validation/results/microct/Diente_XX/` | **Las etiquetas finales a 16 µm** (ver sección 5). Unos 23 GB en total (medido). | **No** |
| `1_Data_Validation/results/microct_32um/` | Las mismas a 32 µm, solo para comparar. | **No** |
| `1_Data_Validation/results/para_slicer/` | NRRD para revisar en 3D Slicer, con `LEEME.txt`. | **No** |
| `1_Data_Validation/results/cbct_calibration/Isa_caso00/` | `calibration.json` y `calibration.png`: lo medido en el CBCT real. | **No** |
| `1_Data_Validation/results/sim_cbct/` | Parches simulados de ejemplo (`.npz`, `.json`, `.png`). | **No** |
| `1_Data_Validation/results/prueba_cbct_isa/` | Resultado de la prueba de la red con el CBCT de Isa (datos de paciente). | **No** |
| `1_Data_Validation/results/medidas/` | Salida de las medidas (`Diente_XX/`), la validación con tubos artificiales y las figuras de explicación (`explicacion/`). | **No** |
| `1_Data_Validation/results/para_isa/` y `para_isa_v2/` | Imágenes para enseñar a la endodoncista Isa (v1 y v2). | **No** |
| `1_Data_Validation/analysis/microct/` | **El código nuevo del TFM** (ver sección 4). | **Sí** |
| `1_Data_Validation/analysis/microct/diagnostico/` | Scripts que reproducen las cifras del análisis y las pruebas. Con su `README.md`. | **Sí** |
| `1_Data_Validation/analysis/microct/medidas/` | Medición del conducto: línea central, longitud, curvatura, ángulos, diámetros (ver 4.11). | **Sí** |
| `1_Data_Validation/analysis/cbct_real/` | `ordenar_cbct_75um.py`: ordena y renombra los CBCT reales (`Isa_caso00…10`). | **Sí** |
| `1_Data_Validation/analysis/` (resto) | Análisis antiguo de Pulpy3D. Solo de referencia. | Sí |
| `2_UNet/` | Entrenamiento: `train_sim.py`, `models/`, `losses/`, `dataloader/`, `configs/`, `metrics_canal.py`. Parte viene de Pulpy3D y no se usa. | **Sí** (código) |
| `2_UNet/results/entrenamientos_sim/` | Cada entrenamiento: `last.pt` (pesos), `log.txt`, `metrics.json`, `config.yml`. | **No** |
| `.vscode/`, `test_gpu.py` | Ajustes del editor y una prueba suelta de la GPU. | **No** (no se han añadido) |

Regla simple: **lo que va a git es código y documentos; lo que no va es todo lo que sea dato, resultado, pesos o carpeta de paciente.** Los `.gitignore` ya lo aplican (`datasets/MicroCT`, `datasets/CBCT_75um`, `results/`, `checkpoints/`). Los documentos personales de la beca (`00_Papers/`, `01_Reuniones_y_notas/`, `02_Informes_y_presentaciones/`) están ignorados en el `.gitignore` de la raíz.

---

## 4. Los scripts uno por uno

Todos se ejecutan con el entorno `tfm`. Los de `analysis/microct/` aceptan rutas relativas desde cualquier directorio (las rutas por defecto cuelgan de `1_Data_Validation/`).

### 4.1 `preprocess_microct.py` — del `.raw` a la máscara completa
- **Qué hace:** normaliza el µCT (aire = 0, dentina = 1) y saca la máscara del conducto buscando huecos encerrados por el diente en cada corte axial (el eje z es el eje largo del diente).
- **Lee:** `.raw` + `.xtekVolume` (o `.xtekct`) de la carpeta del diente.
- **Escribe** en `results/microct/Diente_XX/`: `norm.npy`, `canal.npy`, `info.json`, `qc.png`, `qc_sagital_coronal.png`.
- **Parámetros importantes:**
  - `--downsample` (1 = 16 µm, que es lo definitivo; 2 = 32 µm; 4 = 64 µm). Con otro valor escribe en `results/microct_32um`, etc., para no pisar las finales.
  - `--thr 0.5` (umbral del canal), `--min-mm3 0.05` (volumen mínimo de un hueco), `--min-len-mm 1.5` (largo mínimo).
  - `--gap-mm 0.13` (unir trozos separados por menos de 0,13 mm en z), `--min-piece-mm3 0.005` (quitar motas), `--min-thick-mm 0.05` (quitar rayas finas de caries).
  - `--out`, `--all` (varios dientes de una carpeta), `--force` (permite pisar una carpeta con etiquetas ya separadas).
- **Funciones principales:** `open_volume` (abre el `.raw` sin cargarlo entero en RAM), `normalization_anchors` y `normalize` (las dos anclas: aire y tejido), `canal_mask` (la detección del conducto), `remove_specks` (quita motas), `guard_overwrite` (se niega a pisar etiquetas finales).
- **Ejemplo:** `python analysis/microct/preprocess_microct.py datasets/MicroCT/Diente_35`
- **Tiempo medido:** 129,3 s a 16 µm y 24,5 s a 32 µm, para el diente 52 **incluyendo comparar con lo guardado**; el script solo tarda algo menos (no medido por separado).
- **Aviso:** el comando antiguo de los README (`--downsample 4 --out results/microct`) ya no existe: aborta si va a pisar etiquetas finales.

### 4.2 `split_canal.py` — etiqueta solo conducto y análisis del ápice
- **Qué hace:** separa el **conducto** (del orificio al ápice) de la **cámara y cavidades**, decide hacia dónde está la corona y mide dónde acaba cada conducto respecto a la punta de la raíz.
- **Lee:** `norm.npy`, `canal.npy`, `info.json`.
- **Escribe:** `canal_completo.npy` (copia de `canal.npy`), `canal_solo_conducto.npy` (la etiqueta), `camara_y_cavidades.npy`, `info.json` ampliado (`orientacion`, `separacion`, `apice`), `control_split.png`, `control_apice.png`.
- **Parámetros:** `--area-camara-mm2 1.5` y `--sostenido-mm 0.1` (criterio del orificio), `--root-min-mm2 0.3`, `--lumen-radius-mm 0.2`.
- **Funciones:** `orientation` (corona = extremo con más esmalte), `section_stats` (sección 2D mayor del conducto en cada corte), `find_floor` (busca el orificio), `roots_and_apex` (raíces, conductos y distancia a la punta), `figures`, `process`.
- **Ejemplo:** `python analysis/microct/split_canal.py results/microct/Diente_35`
- **Tiempo medido:** 81,7 s para el diente 52 **incluyendo calcular hashes de 8 ficheros grandes**; el script solo tarda menos (no medido por separado).

### 4.3 `export_slicer.py` — para mirar las etiquetas en 3D Slicer
- **Qué hace:** recorta cada diente a su caja y lo baja a 32 µm; guarda imagen y tres máscaras en NRRD comprimido.
- **Escribe** en `results/para_slicer/`: `Diente_XX_imagen_uint8.nrrd`, `..._mascara_solo_conducto.nrrd`, `..._mascara_camara_y_cavidades.nrrd`, `..._mascara_completa.nrrd`, `LEEME.txt`, `resumen_exportacion.json`.
- **Funciones:** `block_mean2` (promedia 2×2×2), `block_major2` (mayoría para máscaras), `write_nrrd`, `export`.
- **Ejemplo:** `python analysis/microct/export_slicer.py results/microct/Diente_35 results/microct/Diente_51 results/microct/Diente_52 --out results/para_slicer`
- **Tiempo:** [no medido].

### 4.4 `calibrate_cbct.py` — medir el CBCT real
- **Qué hace:** mide en el CBCT de Isa los grises de esmalte y dentina, el ruido en zonas uniformes y cuánto emborrona el escáner (anchura del borde). Esos números fijan los rangos del simulador.
- **Lee:** el NRRD del CBCT y la segmentación de Isa (solo para saber dónde está el conducto).
- **Escribe:** `calibration.json` y `calibration.png`.
- **Funciones:** `load_crop`, `grey_levels`, `block_noise`/`noise_stats`, `edge_profiles`, `measure` (la misma `measure` se usa luego para medir el simulador).
- **Ejemplo:**
  `python analysis/microct/calibrate_cbct.py "datasets/CBCT_75um/Isa_caso00/CBCT.nrrd" --seg datasets/CBCT_75um/Isa_caso00/Segmentation.seg.nrrd --out results/cbct_calibration/Isa_caso00`
- **Tiempo:** [no medido].

### 4.5 `simulate_cbct.py` y `simulate_cbct_config.yaml` — el simulador
- **Qué hace:** de un bloque de µCT (aleatorio, centrado en el conducto) fabrica un parche de 128³ vóxeles de 75 µm (9,6 mm de lado): imagen, etiqueta blanda (fracción de conducto por vóxel), etiqueta binaria, máscara del diente y mapa de pesos.
- **Pasos (todos con valores aleatorios dentro de rangos del yaml):** elegir el recorte → pasar a grises de CBCT (aire = −1000) y rellenar conducto/cámara con gris de pulpa → emborronar (PSF gaussiana) → bajar a 75 µm → ruido correlacionado + variación suave de brillo → etiqueta con la misma geometría.
- **Parámetros de la línea de comandos:** `--n` (nº de parches), `--seed`, `--out`, `--figs` (nº de figuras), `--real-cbct` / `--real-seg` (para dibujar el real al lado), `--check` + `--calibration` (comprueba el simulador con las mismas medidas que el CBCT real), `--center` (parámetros en el centro de cada rango).
- **Claves del yaml:** `crop.third_weights: [1,1,1]` (parches a partes iguales entre tercios apical, medio y coronal), `crop.reduccion: exacta`, `intensity.*`, `psf.*`, `noise.*`, `labels.*` (qué pasa con la cámara).
- **Clases y funciones:** `Tooth` (un diente preprocesado), `Simulator` (con `sample_params`, `pick_start`, `patch`, `crop_size`), `blur`, `reduce_to`, `box_weights`, `dilate_box`, `check`.
- **Ejemplos:**
  `python analysis/microct/simulate_cbct.py results/microct/Diente_35 results/microct/Diente_51 --n 8 --seed 0 --out results/sim_cbct --figs 4`
  `python analysis/microct/simulate_cbct.py results/microct/Diente_35 --check --center --calibration results/cbct_calibration/Isa_caso00/calibration.json`
- **Tiempo medido:** `patch()` 0,41–0,45 s en GPU y 87 s en CPU; el bucle de la línea de comandos (comprime y guarda) 1,0–1,6 s por parche.
- **Importante:** cada diente lee su tamaño de vóxel de su `info.json`. Con 16 µm el resultado es idéntico al de antes; con otra resolución solo se probó con una copia a 32 µm [no verificado con un µCT real de otra resolución].

### 4.6 `2_UNet/train_sim.py` — entrenar la red
- **Qué hace:** entrena una U-Net 3D con parches generados al vuelo por el simulador. Dejando un diente fuera (`--holdout`) se valida con parches fijos de ese diente.
- **Lee:** `configs/sim_canal.yml` (que apunta a `results/microct` y al yaml del simulador).
- **Escribe** en `2_UNet/results/entrenamientos_sim/<title>_<tag>/`: `last.pt`, `log.txt`, `metrics.json`, `config.yml`.
- **Parámetros:** `-c` (config), `--holdout Diente_52` (o varios; sin valores = ninguno fuera, todos entrenan), `--iterations`, `--tag`, `--seed`, `--third-weights {config,equilibrado,proporcional}`, `--val-patches`.
- **Piezas:** `models/CanalUNet3D.py` (U-Net 3D, 5,65 M parámetros), `losses/SoftDiceBCE.py` (pérdida Dice + BCE con etiqueta blanda y pesos), `dataloader/SimCBCT.py` (parches al vuelo; `normalize_patch` normaliza el gris de cada parche con los percentiles 1 y 99,5), `metrics_canal.py` (`dice`, `hd95`, `third_map`, `metrics_by_thirds`).
- **Ejemplo:** `cd 2_UNet` y `python train_sim.py -c configs/sim_canal.yml --holdout Diente_52 --iterations 1000 --third-weights equilibrado --seed 33 --tag mi_prueba`
- **Tiempo medido:** unos 1,65 s por iteración, pico de 10,7–11,5 GB de GPU (de 16); 200 iteraciones ≈ 340 s.
- **Aviso:** no hay validación cruzada completa, ni planificador de la tasa de aprendizaje, ni test aparte. Es un montaje, no un entrenamiento final.

### 4.7 `imagen_para_isa.py` y `imagen_para_isa_v2.py` — imágenes para la endodoncista
- **Qué hacen:** cortes longitudinales de un diente (corona arriba) con la máscara en azul y una flecha amarilla donde acaba, para preguntar a Isa dónde termina el conducto.
- **v1** (`results/para_isa/`): un plano por diente, otro perpendicular y un ampliado. **v2** (`results/para_isa_v2/`): un plano por cada conducto (A = el más largo, B = el otro), buscando el que más conducto contiene sin romperlo en trozos; más un corte a ~90° y un ampliado de los últimos 3 mm. El plano oblicuo con los dos conductos no sale bien en ningún diente, así que no se genera.
- **Parámetros v2:** `--paso-angulo 2`, `--desplazamiento-mm 0.1`, `--min-frac-90 0.30`, `--min-frac-oblicuo 0.6`, `--min-cobertura-rel 0.9`.
- **Ejemplo:** `python analysis/microct/diagnostico/imagen_para_isa_v2.py` (genera `results/para_isa_v2/*.png` y `resumen_v2.json`).
- **Tiempo:** v1 ~25 s [aprox.]; v2 [no cronometrado por separado; los tres dientes salieron en unos pocos minutos].
- **Ojo:** Isa solo vio el diente 52 (tres imágenes, v1). Lo que dijo del final del conducto vale solo para el 52.

### 4.8 Los scripts de `analysis/microct/diagnostico/`
Todos aceptan `--help`, no escriben en `results/microct*` y reproducen cifras del análisis. Tabla corta (la completa, con resultados esperados, está en `diagnostico/README.md`):

| Script | Para qué sirve (una frase) |
|---|---|
| `comparar_resultados.py` | Comprueba que lo regenerado es idéntico a lo guardado (hashes, json). |
| `barrido_gap.py` | Prueba distintos valores de la holgura `--gap-mm` para unir trozos. |
| `grosor_piezas.py` | Mide el grosor de las piezas de máscara (para distinguir conducto de rayas de caries). |
| `secciones_y_orificio.py` | Sección mayor por corte y cuánto se mueve el orificio al cambiar el umbral de área. |
| `dice_32_vs_16.py` | Compara la máscara a 16 µm con la de 32 µm (Dice). |
| `ver_apice.py` | Cortes ampliados del final de un conducto hasta la punta. |
| `imagen_para_isa.py` / `imagen_para_isa_v2.py` | Imágenes para Isa (sección 4.7). |
| `regresion_simulador.py` | Guarda/compara parches con recortes fijos (para comprobar que un cambio no altera nada). |
| `muestreo_por_tercios.py` | Cuántos centros de parche caen en cada tercio del conducto. |
| `calibracion_reduccion.py` | Efecto de la reducción exacta sobre la calibración con Isa. |
| `prueba_vox_distinto.py` | Comprueba que el simulador usa el vóxel de cada diente (copia a 32 µm). |
| `prueba_alineacion_y_pesos.py` | Comprueba que imagen y etiqueta están alineadas y los pesos del orificio. |
| `ruido_medida_sesgo.py` | Explica por qué el ruido medido sube con rangos aleatorios (sesgo de la medida). |
| `tiempos_simulador.py` | Tiempo por etapas y CPU frente a GPU. |
| `comprobaciones.py` | Varias comprobaciones (segmentación de Isa, ventanas de reducción, ruido, tercios...). |

Ejemplo: `python analysis/microct/diagnostico/dice_32_vs_16.py`.

### 4.9 `2_UNet/prueba_cbct_isa.py` — red entrenada frente al CBCT real
- **Qué hace:** pasa el CBCT de Isa por una red entrenada (ventanas de 128³ con 50 % de solape) y compara con la segmentación de Isa.
- **Lee:** el NRRD del CBCT, la segmentación de Isa y un `last.pt`. **Escribe** en `1_Data_Validation/results/prueba_cbct_isa/`: probabilidad y máscara (`.npy` y `.nrrd`), `resultados.json`, 3 axiales y 1 longitudinal en PNG.
- **Parámetros:** `--checkpoint` (obligatorio), `--margen-mm 3`, `--contexto-vox 32`, `--ventana 128`, `--paso 64`, `--umbral 0.5`, `--z-camara 511` (a partir de qué z la segmentación de Isa cuenta como cámara), `--grises-sim N` (compara el gris que ve la red en N parches simulados frente al real).
- **Ejemplo:** `cd 2_UNet` y `python prueba_cbct_isa.py --checkpoint results/entrenamientos_sim/sim_canal_montaje_modelo_de_prueba_3dientes_1000it/last.pt --grises-sim 60`
- **Tiempo medido:** unos 76 s en total (6 s de inferencia, el resto lectura del NRRD, HD95 e imágenes).

### 4.10 Código antiguo que NO se usa
`2_UNet/main.py`, `eval.py`, `models/UNet3D.py`, `PosPadUNet3D.py`, `UNETR.py`, `VNet3D.py`, `AttentionSegmentation/`, `MultiHead/`, `dataloader/Pulpy.py` y casi todo `1_Data_Validation/analysis/` (salvo `microct/`) vienen de Pulpy3D/ToothFairy. No los he ejecutado en esta etapa [no verificado que sigan funcionando].

### 4.11 `analysis/microct/medidas/` — medir el conducto
- **Qué hace:** a partir de `canal_solo_conducto.npy`, traza la línea central de cada conducto y mide longitud (arco y cuerda), curvatura, ángulos y diámetros, con figuras de control.
- **Ficheros:** `canal_medidas.py` (las funciones), `medir_conductos.py` (el programa), `validar_conducto_artificial.py` (prueba con tubos de medidas exactas) y `figuras_explicacion.py` (dibuja las figuras de MEDICION.md).
- **Ejemplo:** `python analysis/microct/medidas/medir_conductos.py results/microct/Diente_35 results/microct/Diente_51 results/microct/Diente_52` → `results/medidas/Diente_XX/` (~1–1,5 min por diente [medido]).
- **Todo el método, el código, el error y los resultados:** [MEDICION.md](MEDICION.md).

---

## 5. Los ficheros de salida

Para cada diente, en `results/microct/Diente_XX/`. El orden de los ejes es **(z, y, x)**; z es el eje largo del diente (la corona está hacia z alto en los tres dientes actuales). El vóxel es **0,016 mm** (16 µm) isotrópico.

| Fichero | Qué es | Forma / tipo / unidad | Cómo abrirlo |
|---|---|---|---|
| `norm.npy` | µCT normalizado: aire = 0, dentina ≈ 1, esmalte > 1 | `(z,y,x)` float16, sin unidad (2,4–2,9 GB) | Python: `np.load(f, mmap_mode="r")` |
| `canal.npy` | máscara completa del conducto + cámara + cavidades | `(z,y,x)` uint8 0/1 | Python (igual) |
| `canal_completo.npy` | copia de `canal.npy` hecha por `split_canal.py` | igual | Python |
| **`canal_solo_conducto.npy`** | **LA ETIQUETA**: del orificio al ápice | `(z,y,x)` uint8 0/1 | Python; en Slicer vía `para_slicer` |
| `camara_y_cavidades.npy` | lo quitado de la etiqueta | `(z,y,x)` uint8 0/1 (solo + esto = completa, sin solape) | Python |
| `coords_canal_solo_conducto_s6.npy` | caché del simulador: coordenadas de vóxeles del conducto cada 6 | `(N,3)` int32 | Python (se regenera sola) |
| `info.json` | anclas de normalización, umbral, volúmenes, orientación, orificio, ápice por raíz | texto | cualquier editor |
| `qc.png`, `qc_sagital_coronal.png` | cortes con la máscara, para ojo | imagen | visor de imágenes |
| `control_split.png`, `control_apice.png`, `control_antes_despues.png` | control del orificio, del ápice y de los filtros | imagen | visor |

Claves útiles de `info.json`: `voxel_mm`, `forma_zyx`, `anclas`, `volumen_canal_mm3`, `orientacion.corona_en_z`, `separacion.z_orificio_vox`, `separacion.canal_solo_z_vox`, `separacion.volumen_mm3` (completo / solo conducto), `separacion.avisos`, `apice.raices[].conductos[]` (`z_fin_vox`, `yx_fin_vox`, `mm_a_la_punta_local`...).

Otros ficheros de salida:

| Fichero | Qué es | Cómo abrirlo |
|---|---|---|
| `results/microct/resumen_etapa1.json` | resumen numérico de los tres dientes | editor |
| `results/para_slicer/*.nrrd` | imagen (uint8, gris 0–255) y 3 máscaras (0/1), a 32 µm, recortadas a la caja del diente | **Slicer** (abajo) o `nrrd.read(...)` |
| `results/cbct_calibration/.../calibration.json` | lo medido en el CBCT real (grises, ruido, bordes) | editor |
| `results/sim_cbct/.../*.npz` | un parche: `image` (128³ float32, grises tipo Hounsfield), `label_soft` (fracción de conducto), `label` (0/1), `tooth`, `weight` (0 = ignorar en la pérdida), `chamber` | `np.load(f)["image"]` |
| `results/sim_cbct/.../*.json` | todos los parámetros aleatorios sorteados para ese parche | editor |
| `2_UNet/results/entrenamientos_sim/*/last.pt` | pesos de la red + config | `torch.load(f)` |
| `.../metrics.json`, `log.txt` | pérdida por iteración y métricas de validación | editor |

### Abrir en Python
```python
import numpy as np, json
f = r"D:\tfm_claudia\1_Data_Validation\results\microct\Diente_52"
norm = np.load(f + r"\norm.npy", mmap_mode="r")            # no carga 2,4 GB: lee por trozos
lab  = np.load(f + r"\canal_solo_conducto.npy", mmap_mode="r")
info = json.load(open(f + r"\info.json"))
v = info["voxel_mm"]                                        # 0,016 mm
print(norm.shape, lab.sum() * v**3, "mm3")                  # ~11,7 mm3 en el diente 52
corte = np.asarray(norm[600]); mascara = np.asarray(lab[600])   # un corte axial (z = 600)
```

### Abrir en 3D Slicer
1. `File > Add Data > Choose File(s)`: elige `Diente_XX_imagen_uint8.nrrd` y las máscaras de `results/para_slicer/`.
2. En las máscaras marca **LabelMap** (si no, se abren como una imagen en gris).
3. Mira las tres vistas con la máscara encima. Para 3D: `Segment Editor > Import/Export` o `Volume Rendering`.
4. **Los NRRD se comprobaron releyéndolos con SimpleITK (tamaño, espaciado, sumas); no se han abierto en Slicer** [no verificado]. Es lo primero que te toca hacer.

Para el CBCT real (`.nrrd` de Isa): se abre igual. Los ejes del NRRD son (x, y, z); si lo lees con `nrrd.read`, para trabajar con el orden (z,y,x) del resto usa `.transpose(2,1,0)`.

---

## 6. Cómo repetirlo todo desde cero con un diente nuevo

Supón que el diente nuevo se llama `Diente_60` y su carpeta con el `.raw` y el `.xtekVolume` ya está en `1_Data_Validation/datasets/MicroCT/Diente_60`. Ejecuta desde `D:\tfm_claudia\1_Data_Validation` con `(tfm)` activo.

| # | Comando | Tiempo |
|---|---|---|
| 1 | `python analysis/microct/preprocess_microct.py datasets/MicroCT/Diente_60` | ~2 min [medido en el 52: 129 s con la comparación incluida] |
| 2 | Mira `results/microct/Diente_60/qc.png` y `qc_sagital_coronal.png`: ¿el conducto azul tiene sentido? | a ojo |
| 3 | `python analysis/microct/split_canal.py results/microct/Diente_60` | ~1 min [medido en el 52: 82 s con hashes] |
| 4 | Mira `info.json["separacion"]["avisos"]`, `control_split.png` y `control_apice.png` | a ojo |
| 5 | `python analysis/microct/export_slicer.py results/microct/Diente_60 --out results/para_slicer` | [no medido] |
| 6 | Abre en 3D Slicer lo exportado y comprueba el orificio y el final del conducto (o pídeselo a Isa) | a mano |
| 7 | Prueba el simulador: `python analysis/microct/simulate_cbct.py results/microct/Diente_60 --n 4 --seed 0 --out results/sim_cbct --figs 4` | ~1–2 s por parche [medido] |
| 8 | (opcional) `python analysis/microct/diagnostico/dice_32_vs_16.py` si has generado también la versión a 32 µm (`--downsample 2`) | ~1 min [aprox.] |
| 9 | Entrenar: `cd ..\2_UNet` y `python train_sim.py -c configs/sim_canal.yml --holdout Diente_60 --iterations 1000 --tag nuevo` | ~1,65 s por iteración [medido]; 1000 iteraciones unos 25–30 min [estimado, ver nota] |
| 10 | Medir el conducto: `python analysis/microct/medidas/medir_conductos.py results/microct/Diente_60` y mirar `control_perfil.png`, `control_secciones.png` y los `avisos` de `medidas.json` | ~1–1,5 min [medido en 35, 51, 52] |

Notas:
- El simulador y el entrenamiento **descubren los dientes por carpeta**: basta con que `results/microct/Diente_60/` tenga `info.json`, `norm.npy` y `canal_solo_conducto.npy`. No hay lista que editar.
- Con la corona hacia z bajo, `split_canal.py` voltea z para trabajar y deshace el volteo al guardar [leído en el código; los tres dientes actuales tienen la corona hacia z alto, así que ese camino **no está probado con un diente real** — no verificado].
- Pide el **orificio** a un experto: el plano está ajustado con solo 3 dientes. Con ~40 hay que revisar los avisos de cada uno.
- Memoria: a 16 µm el preprocesado trabaja con volúmenes de varios GB; la máquina del laboratorio tiene 127,7 GB de RAM [medido]. El pico de RAM [no medido]. El portátil de Claudia no tiene GPU NVIDIA: la simulación y el entrenamiento van en la máquina del laboratorio; los pasos 1–5 en el portátil [no probado ahí].

---

---

## 7. Preguntas que te pueden hacer y respuestas honestas

| Pregunta | Respuesta honesta |
|---|---|
| ¿Cómo sabéis que la máscara del µCT es correcta? | No lo sabemos con un experto. Se obtiene por una regla (umbral 0,5 + huecos encerrados) y es estable entre 16 y 32 µm (Dice ~0,97–0,99), pero eso mide estabilidad, no exactitud. Hay que revisarla en 3D Slicer y pedir a Ana una máscara de experto. |
| ¿Dónde acaba el conducto? | Hasta el final de la máscara. Isa dijo que, en el diente 52, ese final coincide con el foramen (el agujero de salida) que ella marcaría. Del 35 y el 51 no hay opinión de experto. |
| ¿Por qué solo el conducto y no la cámara? | Fue una decisión de Claudia; el corte entre ambos es una convención de ±0,1 mm. La cámara está guardada aparte (`camara_y_cavidades.npy`) por si se quiere usar. |
| ¿Por qué simular un CBCT? | Porque no tenemos etiquetas de experto sobre CBCT ni pares µCT–CBCT del mismo diente [no verificado]. Simulando, la etiqueta ya está alineada. El riesgo es que el simulador no se parezca lo bastante a un CBCT real. |
| ¿El simulador se parece a un CBCT real? | En las cosas que se miden (grises de dentina y esmalte, ruido, desenfoque) sí, con un solo CBCT. Eso no prueba que engañe a la red; se verá al aplicarla a uno real. |
| ¿Con 3 dientes se puede entrenar? | No para un resultado fiable. Sirve para comprobar que el montaje funciona. Se espera tener ~40 µCT. |
| ¿Y en un CBCT real? | En una prueba de montaje con el CBCT de Isa (3 dientes de entrenamiento, 1000 iteraciones), Dice 0,33 en la zona de conducto frente a su borrador, con muchos falsos positivos. Es solo una prueba de montaje. |
| ¿Cuánto vale el Dice? | En una prueba de montaje con 2 dientes de entrenamiento: apical ~0,58, total ~0,70–0,73 en el diente dejado fuera (1000 iteraciones, una semilla). Es solo para ver que funciona, no un resultado. |
| ¿Qué es el Dice y el HD95? | Dice mide cuánto se solapan la predicción y la referencia (1 = idéntica, 0 = nada). HD95 mide la distancia (en mm) a la que están los peores bordes (percentil 95). |
| ¿Qué precisión tienen las medidas? | Sobre la etiqueta del µCT y con tubos artificiales de medidas exactas: longitud −0,5…−0,3 % a 16 µm y −1,8…−0,5 % a 75 µm; ángulos < 1°; diámetro equivalente ±1 % (16 µm) y −5…+0,5 % (75 µm). Sobre la predicción de una red en un CBCT real **todavía no se ha medido**. Detalle y límites: MEDICION.md. A 75 µm el diámetro apical es impreciso. |
| ¿Por qué 16 µm si el CBCT es de 75 µm? | Para tener etiquetas finas y bajarlas a 75 µm con fracción de conducto por vóxel; a 64 µm se pierden conductos. |
| ¿Funciona con otro escáner? | [no verificado]. Los rangos del simulador son anchos a propósito, pero solo hay un CBCT para calibrar. |
| ¿La orientación del diente importa? | Sí: la red se entrena con el eje largo del diente en z y la corona hacia z alto, y solo con volteos y giros de 90° en el plano xy. Un diente inclinado en el CBCT puede fallar [no verificado]. |
| ¿Qué datos se suben a GitHub? | Solo código y documentos. Datos de µCT, CBCT del paciente, resultados y pesos están ignorados. |
| ¿Por qué no usasteis Pulpy3D? | Se descartó por decisión previa (ni Pulpy3D ni los datos de Fer); no sé el motivo detallado [no verificado]. Usar sus datos solo para localizar el diente (36/46) está por decidir: PROYECTO.md, apartado 8. |
| ¿Qué falta para tener la tesis? | Revisar etiquetas con expertos, tener ~40 dientes, la segmentación de solo el diente pedido (36/46) con un simulador con vecinos, un volumen completo simulado, entrenar con validación cruzada y evaluar en CBCT reales con medidas sobre la predicción. Lista ordenada: PROYECTO.md, apartado 9. |
| ¿Y si el dentista mete el CBCT entero? | Hoy la red no distingue un diente de sus vecinos (se entrenó con un diente aislado). El diseño propuesto es localizar el diente (un clic del dentista o un modelo de numeración), recortar y segmentar con el diente marcado: PROYECTO.md, apartado 8. |
| ¿Qué resolución tendrán los CBCT? | Todos de 75 µm (campo pequeño), como los de Isa (decidido el 2026-11-19). |
| ¿Dónde están las respuestas a lo que no está comprobado? | PROYECTO.md, apartado 10 (dudas y preguntas abiertas) y los apartados 5 y 6. |

---

## 8. Glosario (30 términos)

1. **Vóxel:** el "píxel 3D"; el cubito mínimo de un volumen. Aquí mide 16 µm (µCT) o 75 µm (CBCT). 1 µm (micra) = una milésima de mm.
2. **µCT (micro-TC):** escáner 3D de muy alta resolución para piezas pequeñas, como un diente extraído.
3. **CBCT:** escáner 3D de rayos X de las clínicas dentales; más borroso y con más ruido.
4. **Conducto radicular:** tubito dentro de la raíz por donde pasan el nervio y los vasos.
5. **Cámara pulpar:** cavidad grande de la corona donde se juntan los conductos.
6. **Orificio:** entrada del conducto en el suelo de la cámara. En el código, un plano horizontal donde acaba la etiqueta hacia la corona.
7. **Ápice:** la punta de la raíz.
8. **Foramen apical:** agujero por donde el conducto sale al hueso, cerca del ápice. El dentista mide el conducto hasta ahí.
9. **Corona / raíz:** parte visible del diente / parte que va dentro del hueso. Un molar (muela) tiene varias raíces.
10. **Esmalte / dentina:** capa dura exterior de la corona (más densa, más brillante en la imagen) / material del interior del diente.
11. **Umbral:** valor de gris a partir del cual se decide "esto es objeto" o "esto es fondo".
12. **Efecto de volumen parcial:** un vóxel que contiene dos materiales tiene un gris intermedio; el borde real cae a mitad de camino.
13. **Máscara / etiqueta:** volumen de 0 y 1 que marca qué vóxeles son conducto. "Blanda" = fracción de conducto por vóxel (entre 0 y 1).
14. **PSF (función de dispersión del punto):** cuánto emborrona un escáner un punto; aquí, una gaussiana.
15. **Parche:** recorte 3D pequeño (128³ vóxeles) que se da a la red.
16. **U-Net:** tipo de red neuronal que "comprime" la imagen y la "descomprime" para devolver una máscara.
17. **Dice:** medida de solapamiento entre dos máscaras (0 = nada, 1 = idénticas).
18. **HD95:** distancia de Hausdorff al percentil 95: cuánto se separan los bordes de dos máscaras (en mm), ignorando el 5 % de los peores puntos.
19. **3D Slicer / NRRD:** programa gratuito para ver y editar volúmenes médicos 3D / formato de fichero de volumen que Slicer abre.
20. **Holdout / checkpoint:** dejar un diente entero fuera del entrenamiento para probar con él / fichero (`last.pt`) con los pesos guardados de la red.
21. **Línea central:** la curva que recorre el centro del conducto del orificio al ápice. Aquí se obtiene recentrando cada punto en el centro de gravedad de la sección perpendicular (MEDICION.md).
22. **Centroide (centro de gravedad):** el punto donde se equilibraría una forma plana recortada en cartulina; media de las posiciones de sus píxeles.
23. **Sección perpendicular:** la imagen 2D que se obtiene cortando el volumen con un plano perpendicular a la línea central en un punto.
24. **EDT (transformada de distancia):** para cada vóxel de una máscara, la distancia a la pared más cercana; en el centro de un tubo vale su radio.
25. **Curvatura κ y radio de curvatura R:** κ (mm⁻¹) = cuánto gira la dirección del conducto por mm recorrido; R = 1/κ es el radio del círculo que mejor imita la curva (κ = 0 es recto).
26. **Ángulo de Schneider:** el ángulo clásico de curvatura de un conducto, medido sobre una radiografía 2D entre el tramo recto y la recta que une el foramen con el punto donde empieza a curvarse. Aquí hay una versión en 3D.
27. **Diámetro equivalente:** el diámetro del círculo con la misma área que la sección del conducto.
28. **Hounsfield (HU):** escala de densidad en TC (aire ≈ −1000, agua ≈ 0). Que los grises del CBCT de Isa sean HU reales no está verificado.
29. **Logit:** salida de la red antes de la sigmoide; `sigmoide(logit)` es la probabilidad de «conducto».
30. **Longitud de trabajo:** distancia desde un punto de referencia de la corona hasta donde se prepara el conducto (no es lo mismo que la longitud del conducto).
