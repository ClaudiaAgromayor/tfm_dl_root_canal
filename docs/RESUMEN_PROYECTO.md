# Resumen del proyecto: segmentar conductos radiculares en CBCT con µCT sintético

> Documento de referencia para alguien que programa pero no sabe odontología ni imagen médica.
> Escrito el 2026-10-28 a partir de la lectura del código real, no de CLAUDE.md ni de memoria.
> Estado del repositorio: **sin commit** de nada de lo que se describe aquí.

## 0. Cómo leer este documento

**Citas.** Cada afirmación técnica lleva entre corchetes `[alias:línea]` o `[alias:desde-hasta]`. Los alias son:

| Alias | Fichero |
|---|---|
| `P` | `1_Data_Validation/analysis/microct/preprocess_microct.py` |
| `S` | `1_Data_Validation/analysis/microct/split_canal.py` |
| `C` | `1_Data_Validation/analysis/microct/calibrate_cbct.py` |
| `SIM` | `1_Data_Validation/analysis/microct/simulate_cbct.py` |
| `Y` | `1_Data_Validation/analysis/microct/simulate_cbct_config.yaml` |
| `X` | `1_Data_Validation/analysis/microct/export_slicer.py` |
| `DS` | `2_UNet/dataloader/SimCBCT.py` |
| `UN` | `2_UNet/models/UNet3D.py` (heredado de Pulpy3D) |
| `NET` | `2_UNet/models/CanalUNet3D.py` |
| `LOSS` | `2_UNet/losses/SoftDiceBCE.py` |
| `MET` | `2_UNet/metrics_canal.py` |
| `TR` | `2_UNet/train_sim.py` |
| `CFG` | `2_UNet/configs/sim_canal.yml` |
| `MF` | `2_UNet/models/ModelFactory.py` (heredado) |
| `PL` | `2_UNet/dataloader/Pulpy.py` (heredado) |
| `PC` | `2_UNet/configs/pulp_segmentation.yml` (heredado) |
| `CL` | `CLAUDE.md` · `RD` = `README.md` (raíz) · `GPU` = `test_gpu.py` |

Las rutas de datos y resultados son relativas a `1_Data_Validation/` (ahí cuelgan `datasets/` y `results/`). `datasets/` y `results/` **no están en git** (y contienen datos de paciente en el caso del CBCT).

**Etiquetas de verificación.** Cada número lleva una de estas:

- **[reejecutado hoy]**: lo volví a calcular el 2026-10-28 con el comando indicado y coincide con lo guardado (los pasos se llaman V1…V9 y E-a…E-e, y están descritos en el Apéndice A).
- **[leído de fichero]**: está en un fichero de resultados (`info.json`, `metrics.json`…) que no he vuelto a generar hoy.
- **[no reproducible desde el repo]**: salió de un script temporal de la sesión que **no se ha copiado** al repositorio (ver 9). Los scripts que sí se copiaron están en `1_Data_Validation/analysis/microct/diagnostico/` (con su `README.md`) y sus resultados se marcan **[reejecutado hoy]**.
- **[no verificado]**: lo afirma alguien (un comentario del código, CLAUDE.md…) y no lo he comprobado.

**Comandos.** Todos se lanzan desde la raíz del repositorio (`D:\tfm_claudia`) con el entorno conda `tfm`:
`C:\Users\tfg\.conda\envs\tfm\python.exe` (Python 3.12, PyTorch 2.14.1+cu130 **[reejecutado hoy]**, versiones en el Apéndice A). Abreviamos ese intérprete como `python`.

---

## 0b. Cambios hechos después de la primera versión de este documento (2026-10-28)

Esta versión recoge cambios de código hechos tras la primera; las citas de línea ya están actualizadas. **Sin commit.**

1. **El preprocesado ya no puede sobrescribir las etiquetas finales.** `preprocess_microct.py` escribe por defecto en `results/microct` solo a 16 µm; con otra resolución va a `results/microct_32um`, `microct_64um`… [P:195-201]; se niega a escribir en una carpeta con etiquetas separadas o hecha con otra resolución salvo con `--force` [P:204-217, P:225]; el `--downsample` por defecto pasa de 4 a 1 [P:257]; la salida ya no depende del directorio de trabajo. Probado: el comando antiguo de los README aborta en 0,8 s y la prueba a 64 µm va a su propia carpeta (que borré después). Los dos README están corregidos.
2. **Vóxel por diente.** El simulador lee el vóxel de cada `info.json` y calcula el tamaño del bloque (`crop_size`) en vez de suponer 16 µm [SIM:213-218]. Con los interruptores en modo antiguo, el resultado es **idéntico** al de antes (diferencia máxima 0,0 en imagen, etiqueta blanda y pesos; 12 parches) y con una copia a 32 µm da el mismo parche físico (3.6).
3. **Reducción a 75 µm exacta.** Sustituye las ventanas solapadas de 106 µm por cajas exactas de 75 µm [SIM:101-126]. Efecto sobre la calibración contra Isa: despreciable (3.6).
4. **Muestreo por tercios.** Los parches se centran a partes iguales en los tercios apical, medio y coronal del conducto [SIM:238-269]. Apical: 13,9 % → 31,9 % de los centros. Efecto sobre el entrenamiento: **inconcluso** con 200 iteraciones (mitad de los entrenamientos sin despegar) y, con 1000 iteraciones en una vuelta y una semilla, **mismo Dice apical final** (0,58) con convergencia más rápida del equilibrado (3.12).
5. **Contradicciones de 5.3:** corregidas las de código y documentación; ver 5.3.
6. **Scripts de diagnóstico copiados** a `analysis/microct/diagnostico/`, parametrizados, con `README.md` y probados.
7. **Comentario de Isa** sobre las imágenes del final del conducto (informal, relatado por Claudia): la máscara termina donde ella pondría el foramen, **pero solo vio el diente 52** (3.4(e)).

## 1. Resumen de una página

**Problema.** Un endodoncista trata los conductos de la raíz de un diente (los "tubos" por donde van el nervio y los vasos). Para planificar necesita saber longitud, curvatura, volumen y diámetro de cada conducto. En la clínica solo hay un **CBCT** (tomografía de rayos X con vóxeles de unos 75 µm = 0,075 mm), que es demasiado borroso en la punta de la raíz (el **ápice**) para que nadie dibuje el conducto a mano con fiabilidad; sin dibujo manual no hay datos para entrenar una red neuronal. Objetivo: segmentar automáticamente los conductos de los primeros molares inferiores (36/46) en un CBCT y medirlos [CL:5-6].

**Idea.** Un **µCT** (micro-tomografía, vóxel de 16 µm) de un diente extraído sí muestra el conducto con nitidez, y su conducto es "aire dentro del diente", así que se puede sacar casi sin ayuda humana. Entonces: (1) sacar el conducto del µCT por reglas geométricas; (2) **fabricar un CBCT falso** degradando ese mismo µCT (borrosidad, ruido, 75 µm, tejidos alrededor); (3) entrenar una red 3D con los falsos, donde la etiqueta ya coincide con la imagen porque sale del mismo volumen (no hay que alinear imágenes); (4) aplicarla a CBCT reales y medir sobre su salida.

**Estado actual (2026-10-28).**
- Hecho y reproducido hoy: sacar el conducto del µCT a 16 µm en 3 dientes (V2, V3); separar "solo conducto" de "cámara pulpar y cavidades" (V5); calibrar contra un CBCT real (V4); simulador de CBCT que reproduce las medidas de ese CBCT (V6); montaje de entrenamiento extremo a extremo (V8).
- Los 3 µCT dan volúmenes de "solo conducto" de 11,2 / 8,0 / 11,7 mm³ (dientes 35 / 51 / 52) **[leído de fichero]** (`results/microct/Diente_XX/info.json`, clave `separacion.volumen_mm3`).
- Se entrenó 200 iteraciones por vuelta como **prueba de montaje**, no como entrenamiento; **no se saca ninguna conclusión de calidad**.

**Lo que falta (lo importante).**
1. Ninguna máscara hecha por un experto sobre el µCT: todo lo que se llama "etiqueta" es la salida de un algoritmo sin validar [CL:19]. Claudia revisará en 3D Slicer.
2. Solo 3 µCT (de ~40 que están en traspaso) y no está comprobado que 2 de ellos sean molares.
3. Un único CBCT real, con una segmentación "en curso" (borrador) de Isa, para calibrar el simulador.
4. Faltan: entrenar de verdad, evaluar en el CBCT real, el volumen completo simulado, la distancia al ápice, y la medición geométrica (línea central + distancia a la pared) del diámetro, longitud y curvatura.

---

## 2. Esquema del pipeline, de µCT a medidas

```
 .raw (µCT Nikon, uint16, 16 µm)          CBCT real de Isa (NRRD, 75 µm)
        │                                         │
 [3.1] normalizar (aire=0, tejido=1)              │
        │                                         │
 [3.2] umbral 0,5 + "hueco encerrado"             │
        │                                         │
 [3.3] limpieza a 16 µm (motas, láminas)          │
        │  → canal_completo.npy (= canal.npy)     │
        ├──[3.4/3.5] orientación, ápice, separar  │
        │        conducto ≠ cámara                │
        │  → canal_solo_conducto.npy  (ETIQUETA)  │
        │  → camara_y_cavidades.npy               │
        │                                         ▼
        │                              [3.7] calibrar: grises, ruido, borde
        │                                         │ (rangos del azar, Y)
        ▼                                         ▼
 [3.6] SIMULADOR de CBCT a 75 µm  ◄────────────── rangos
        │   imagen + etiqueta blanda + pesos
        ▼
 [3.8] dataset por carpetas  →  [3.9] red 3D  ←  [3.10] pérdida  →  [3.12] entrenamiento
                                      │                                      │
                                      ▼                                      ▼
                              [3.11] métricas por tercios        (todavía NO: aplicar a CBCT real
                                                                   y medir longitud/curvatura/diámetro)
```

Dos "carriles" que conviene no confundir: el **carril del µCT** (pasos 3.1–3.5) produce las etiquetas y vive en 16 µm; el **carril del CBCT** (3.6–3.12) trabaja en 75 µm y es donde entrena la red.

Datos de entrada reales (en `datasets/`, fuera de git): tres µCT `MicroCT/Diente_35`, `Diente_51`, `Diente_52`: `.raw` uint16 con orden (z, y, x), dimensiones 790×1230×1473, 763×1230×1455 y 735×1108×1451 vóxeles, vóxel 0,015999 mm **[leído de fichero]** (`results/microct/Diente_XX/info.json`, `forma_zyx` y `voxel_mm`). Un CBCT real `CBCT_75um/Isa_caso01` de 555×667×688 vóxeles de 0,075 mm con una segmentación manual de Isa **[leído de fichero]** (cabecera del NRRD; ver Apéndice A, E-a). Los "números" de los dientes (35, 51, 52) son **numeración de la colección, no códigos dentales** [CL:17].

---

## 3. Los pasos, uno a uno

### 3.1 Normalización de los grises del µCT

**(a) Qué hace.** El µCT de Nikon da enteros de 16 bits (`uint16`) en unidades arbitrarias. Se reescalan para que el aire valga 0 y el "tejido del diente" valga 1 [P:19-22, P:80-84].

**(b) Por qué así.**
- *Problema que resuelve:* el valor "bruto" cambia de un escaneo a otro. En los 3 dientes, el gris del aire es 7481 / 10341 / 8276 y el del tejido 26424 / 35786 / 26520 **[leído de fichero]** (`info.json`, clave `anclas`), un 35 % de diferencia en el tejido entre dientes. Un umbral fijo en bruto no serviría para todos.
- *Cómo:* dos "anclas" robustas por volumen: aire = mediana de lo que queda por debajo del umbral de Otsu; tejido = mediana de lo que queda por encima [P:71-77]. Con `I' = (I − aire)/(tejido − aire)` el valor 0,5 es el punto medio entre aire y tejido, que es donde cae el borde real de una estructura aunque la imagen esté desenfocada (**efecto de volumen parcial**: un vóxel que cubre mitad aire y mitad diente tiene gris intermedio) [P:19-22].
- *Alternativas descartadas o no usadas:*
  - *Umbral fijo en bruto:* descartado por la diferencia entre escaneos (arriba).
  - *Z-score (media 0, desviación 1):* no fija qué valor es "el borde"; depende de cuánto aire haya en la caja del escaneo. **[no probado]**, es un razonamiento.
  - *Calibrar a unidades Hounsfield:* no hay fantoma (objeto de calibración) en los datos; **[no verificado]** si el escáner lo permitiría.
- *Matiz que se descubrió después:* el ancla "tejido" es la mediana de **todo** lo que está sobre Otsu, que mezcla dentina y esmalte. Por eso la dentina acaba en 0,975 y no en 1,0: pico medido en los 3 dientes [Y:21], y el esmalte en ≈1,6 [X:6, Y:23]. No pasa nada, pero obliga a recordar que "1" no es "dentina".

**(c) Cómo funciona el código.**
- `open_volume` abre el `.raw` con `np.memmap` (un array NumPy que vive en disco y se lee por trozos cuando se indexa, sin cargarlo en RAM) con forma `(z, y, x)` [P:56]. Comprueba que el tamaño del fichero es exactamente `nx·ny·nz·2` bytes y, si no, avisa de fichero incompleto [P:52-55]. Las dimensiones salen del `.xtekVolume` (texto `clave=valor`) con una expresión regular [P:39-42, P:48-51].
- `downsample_mean` (bajar la resolución un factor `f`) promedia bloques `f×f×f` leyendo **por losas** de `f` cortes, para no cargar los 2,9 GB enteros [P:59-67]. Coste: una pasada, O(N) en vóxeles, memoria de una losa + el resultado.
- `normalization_anchors` calcula Otsu (un método que elige el umbral que mejor separa dos "montones" de valores en el histograma) sobre una submuestra `vol[::3, ::3, ::3]`, es decir, 1/27 de los vóxeles [P:73-74], para ir rápido.
- `normalize` resta y divide **en el sitio** (`-=`, `/=`) para no duplicar los ~5,7 GB que ocupa un volumen `float32` a 16 µm [P:80-84].
- *Parámetros:* `--downsample` (1 = 16 µm, 2 = 32 µm, 4 = 64 µm; por defecto **1**, las etiquetas finales) [P:257]. La carpeta de salida depende de la resolución (`results/microct` a 16 µm; `microct_32um`, `microct_64um`…) [P:199-201] y el script se niega a sobrescribir una carpeta con etiquetas separadas (`canal_solo_conducto.npy`…) o calculada con otra resolución salvo con `--force` [P:204-217, P:225, P:272]. A 64 µm se pierden conductos finos (visto en el diente 52) [CL:24, no re-probado hoy].

**(d) Resultado.** Con `--downsample 1` el diente 52 sale idéntico byte a byte al guardado (anclas, `norm.npy`, `canal.npy`) **[reejecutado hoy]** (V2; el paso completo, que incluye comparar con lo guardado, tardó 129,3 s). Comando: `python 1_Data_Validation/analysis/microct/preprocess_microct.py 1_Data_Validation/datasets/MicroCT/Diente_52 --downsample 1 --out <carpeta>`.

**(e) Límites.** Las anclas dependen de que haya aire y tejido en la caja del escaneo; **[no probado]** con un escaneo que no los tenga. La normalización guarda `norm.npy` en `float16` [P:236]: 3 cifras decimales de precisión, suficiente aquí pero no para restar volúmenes cercanos. Dientes con restauraciones metálicas (grises muy altos) no se han tratado; el diente 35 tiene un pico de gris por encima de 2 en unidades normalizadas que podría ser una restauración **[no verificado]**.

---

### 3.2 Umbral del conducto: "hueco encerrado por el diente"

**(a) Qué hace.** Define el conducto como lo que es aire (gris ≈ 0) pero está **rodeado de diente en cada corte axial** [P:24-26, P:127-129].

**(b) Por qué así.**
- *El problema:* en un diente extraído, el conducto está vacío y vale lo mismo que el aire de fuera. Un umbral de gris no los distingue.
- *La solución:* el eje z es el eje largo del diente, así que en cada corte perpendicular (axial) el conducto es un agujero dentro de una sección de diente. `binary_fill_holes` rellena esos agujeros y la diferencia "relleno − diente" es el conducto [P:127-129].
- *Alternativas descartadas o no usadas:*
  - *Rellenar huecos en 3D:* falla porque el conducto se abre al exterior por el ápice y por la cámara; en 3D "no está encerrado" y se perdería entero. Por cortes axiales sí está encerrado casi siempre. **[razonado, no probado como variante]**.
  - *Región creciente desde un punto semilla:* necesita que una persona marque el orificio de cada diente; no sirve para ~40 dientes sin intervención. **[no probado]**.
  - *Segmentación manual por un experto:* sería la referencia, pero no existe para estos µCT [CL:19, CL:44].
  - *Red neuronal ya entrenada (Pulpy3D, datos de Fer):* descartada por decisión del proyecto [CL:9].
- *Umbral 0,5:* el punto medio entre aire y tejido (ver 3.1) [P:21-22, P:260].

**(c) Cómo funciona el código.**
- `largest_component` etiqueta las componentes conexas (`ndi.label`: recorre el volumen asignando el mismo número a vóxeles vecinos conectados) y se queda con la mayor [P:88-93]. Sirve para eliminar restos sueltos (por ejemplo, trozos del soporte) y quedarse con "el diente". Coste: O(N) con memoria de un volumen de enteros de 32 bits (≈5,7 GB a 16 µm) **[estimado, no medido]**.
- `canal_mask` hace `tooth = largest_component(norm > thr)`, luego `np.stack([ndi.binary_fill_holes(s) for s in tooth])`: un bucle en Python sobre los ~1450 cortes, cada uno con relleno 2D [P:127-128]. `cavity = filled & ~tooth` [P:129].
- Los datos viven en matrices booleanas de `(z, y, x)`; el eje 0 es z. Cualquier función que "mire cortes" indexa `arr[z]`.

**(d) Resultado.** Es la base de todo lo que sigue; sus números finales están en 3.3.

**(e) Límites.**
- Un conducto **no encerrado** en un corte no se detecta en ese corte. Pasa en el ápice (se abre al exterior) y en la cámara abierta a una caries. Se verificó a mano para el ápice (ver 3.4) [S:132-146].
- Asume que el eje largo del diente es z. El código no lo comprueba; en los 3 dientes lo es [CL:17] **[no verificado para los ~40 restantes]**.
- Las cavidades de la corona (caries, poros) también son "huecos encerrados" y entran en la máscara; por eso existen 3.3 y 3.5.

---

### 3.3 Limpieza a 16 µm: motas, trozos partidos y láminas de caries

**(a) Qué hace.** Convierte "todos los huecos encerrados" en "los conductos", descartando tres tipos de basura:
1. **Motas de ruido** (huecos de unos pocos vóxeles) [P:96-107].
2. **Huecos cortos o pequeños** (poros, bolsas de la corona): menos de 0,05 mm³ o menos de 1,5 mm de largo en z [P:141, P:261-263].
3. **Láminas finas** (el borde de una caries): grupos cuyo radio inscrito en el percentil 90 es menor de 0,05 mm [P:143-147].
Y **une** antes los trozos del mismo conducto separados por menos de 0,13 mm en z [P:130-133].

**(b) Por qué así.** Cada regla responde a un fallo observado, en este orden:
- *Fallo 1: a 16 µm el diente 51 daba 16,8 mm³ en 7 trozos, y a 32 y 64 µm 23 mm³ en 1.* La hipótesis inicial ("el ruido abre huecos en la pared") **era falsa**: el conducto estaba encerrado casi siempre, pero en cortes sueltos (z = 361 y 440) dejaba de estar encerrado, el 3D se partía en trozos de ~1,25 mm y el filtro de largo mínimo (1,5 mm) los tiraba. Arreglo: medir el largo **después de unir** trozos separados por menos de `gap_mm` [P:116-118, P:264-265]. **[reejecutado hoy]** con `diagnostico/barrido_gap.py --dientes 51 --downsample 1 --gaps 0 0.13 0.3 --min-piece-mm3 0` (sin quitar motas, como el barrido original): hueco 0 → 16,85 mm³ en 7 componentes; 0,13 mm → 23,18 mm³ en 1; 0,3 mm → 23,24 mm³ en 2. El resultado final queda en `info.json` del 51: 23,139 mm³ en 1 componente.
- *Fallo 2: la unión arrastraba motas.* Con la unión puesta, el diente 35 tenía 60.837 piezas sueltas de 1–10 vóxeles (1,4 mm³) en la máscara, porque las motas que quedaban encima o debajo del conducto se pegaban a él. Arreglo: quitar las motas **antes** de unir (`remove_specks`) [P:96-107, P:129]. **[no reproducible desde el repo]** como cifra exacta (60.837 piezas salieron de una máscara intermedia que ya no existe); `diagnostico/grosor_piezas.py` cuenta las motas de cualquier máscara, y la intermedia se puede regenerar con `preprocess_microct.py --min-piece-mm3 0 --out <otra carpeta>`.
- *Fallo 3: rayas de caries (diente 35, 3,9 mm³).* Alrededor del agujero de una caries salían rayas de 1–2 vóxeles de grosor que pasaban los filtros anteriores. Se descartan los grupos unidos cuyo **radio inscrito p90** (distancia a la pared, percentil 90) sea < 0,05 mm [P:143-147]. Se mide sobre el **grupo unido**, no sobre trozos sueltos, para no perder los trozos finos que prolongan el conducto hacia el ápice [P:119-124].
- *Alternativas descartadas:*
  - *Cierre morfológico del diente (cerrar huecos de la pared):* era la propuesta inicial; descartada porque la causa no era una pared rota. **[probado solo en diagnóstico]**.
  - *Suavizar con gaussiana antes de umbralizar:* propuesta inicial; **[no probado]** porque la causa real era otra y suavizar mueve el borde del canal (sesga el volumen).
  - *Filtrar solo por tamaño o longitud:* las rayas de caries medían 6 mm de largo, pasaban.

**(c) Cómo funciona el código.**
- `remove_specks`: etiqueta componentes 3D, cuenta vóxeles con `np.bincount`, y mapea un vector booleano `ok[pl]` sobre el volumen de etiquetas (indexación "fancy": `ok` es una tabla de consulta indexada por etiqueta) [P:102-107]. Coste O(N).
- La unión: `ndi.binary_dilation` con un elemento estructurante `ones((2g+1, 1, 1))` dilata **solo a lo largo de z** `g = round(gap_mm/voxel)` cortes (8 a 16 µm) [P:130-132]. La dilatación se usa únicamente para decidir qué trozos forman un grupo; la máscara final no se modifica [P:116-118, P:152].
- Por grupo: volumen `ndi.sum(cavity, lab, ...)`, largo = alto de la caja menos `2g` (se resta la holgura añadida) [P:136-138]. Para el grosor, `ndi.distance_transform_edt` sobre la caja del grupo (rellena de 1 vóxel) da la distancia de cada vóxel de conducto a la pared; el p90 es `np.percentile(..., 90)` [P:143-144]. Coste del EDT: O(N de la caja); para el grupo principal es una caja de cientos de millones de vóxeles en `float64` **[estimado, no medido]**.
- `process` guarda `norm.npy` (float16), `canal.npy` (uint8), `info.json` con parámetros y los grupos descartados [P:236-247].
- *Parámetros y qué pasa si cambian:*
  - `--gap-mm` (0,13) [P:264]: barrido hecho 0 → 0,5 mm: con 0 el 51 se rompe en 7 trozos; con ≥ 0,3 el 51 empieza a conservar un trozo extra de 0,05 mm³ **[reejecutado hoy]** con `diagnostico/barrido_gap.py` (ver Fallo 1).
  - `--min-thick-mm` (0,05) [P:268]: los conductos finos del 51 miden 0,068–0,105 mm de radio p90 y los trozos de caries ≤ 0,032 mm **[no reproducible desde el repo]** con la máscara actual (ya filtrada); `diagnostico/grosor_piezas.py` sobre una máscara intermedia (`--min-thick-mm 0`) lo reproduce; hay margen pero **solo con 3 dientes**.
  - `--min-piece-mm3` (0,005) [P:266] y `--min-len-mm` (1,5) [P:262].
  - `--downsample` por defecto **1** [P:257] (antes 4): el filtro de grosor no está pensado para 64 µm (una raya de 1 vóxel ya mide 0,064 mm, por encima de 0,05) [P:123-124].

**(d) Resultado** (con `--downsample 1` y `--downsample 2`, parámetros por defecto):

| Diente | Volumen canal 32 µm | Volumen canal 16 µm | Componentes (32 / 16) |
|---|---|---|---|
| 35 | 21,0699 mm³ | 21,1987 mm³ | 2 / 2 |
| 51 | 23,0378 mm³ | 23,1388 mm³ | 1 / 1 |
| 52 | 37,2499 mm³ | 36,8845 mm³ | 1 / 1 |

**[leído de fichero]** (`results/microct_32um/Diente_XX/info.json` y `results/microct/Diente_XX/info.json`, claves `volumen_canal_mm3` y `componentes_canal`). Reejecutado hoy: 52 a 16 µm (V2) y 51 a 32 µm (V3), idénticos al guardado. Dice entre 32 y 16 µm (16 µm reducido por mayoría de bloques): 0,974 / 0,977 / 0,989 **[reejecutado hoy]** con `diagnostico/dice_32_vs_16.py` (coincide con `results/microct/resumen_etapa1.json`: rejilla de 32 µm 0,9739 / 0,9773 / 0,9888). Es **estabilidad entre resoluciones del mismo método**, no exactitud frente a un experto.
Comandos: `python 1_Data_Validation/analysis/microct/preprocess_microct.py 1_Data_Validation/datasets/MicroCT/Diente_35 --downsample 1 --out 1_Data_Validation/results/microct` (y `--downsample 2 --out .../microct_32um`).
Tiempo: los pasos V2 y V3 completos (preprocesado **más** la comparación con lo guardado, que lee varios GB) tardaron 129,3 s a 16 µm y 24,5 s a 32 µm **[reejecutado hoy]**; el script solo tarda algo menos (no medido por separado). Memoria de pico: **[no medida]**.

**(e) Límites.**
- Los tres parámetros nuevos (0,13 mm, 0,05 mm, 0,005 mm³) están ajustados mirando solo 3 dientes [CL:86].
- El 51 conserva una cavidad grande de la corona (~6 mm³) dentro de su "grupo principal" en este paso; se resuelve en 3.5 por estar por encima del orificio.
- Un conducto muy fino (< ~30 µm de ancho) no se ve a 16 µm; no hay forma de saber que existe **[no verificado]**.

---

### 3.4 El ápice: ¿hasta dónde llega la máscara?

**(a) Qué hace.** Para cada conducto mide a cuántos mm de la punta de su raíz acaba la máscara, y mira si más abajo el hueco se abre al exterior (el conducto "sale" por el ápice) [S:76-156].

**(b) Por qué así.**
- *Pregunta clínica:* la longitud de trabajo de un conducto llega hasta el foramen apical (el agujero de salida en la punta de la raíz). Si la máscara se queda corta, la longitud medida también.
- *Por conducto y no por diente:* un molar tiene 2 raíces y 3–4 conductos con puntas a distinta altura. "El último corte con diente" de todo el diente solo vale para la raíz más larga; en raíces-lámina anchas (dos conductos por raíz) y en el 51 (las dos láminas salen unidas como una "raíz" de 288 mm³) tampoco sirve para el segundo conducto [S:124-127]. Por eso se define la **punta local**: último corte con diente a menos de 0,6 mm del eje del conducto [S:124-127].
- *Alternativas descartadas o no usadas:* (1) usar solo el extremo del diente (explicado arriba); (2) esqueletizar el conducto y buscar puntas del esqueleto (más complejo, no probado); (3) medirlo a mano en cada diente (no escala a ~40).

**(c) Cómo funciona el código.**
- *Raíces:* se cuenta por corte cuántas secciones de diente grandes (≥ 0,3 mm²) hay; el último corte con ≥ 2 es la **furca** (donde las raíces se separan) y las raíces son las componentes 3D por debajo de ella [S:83-91].
- *Conductos de una raíz:* componentes del conducto dentro de la raíz, unidas con la misma holgura de 0,13 mm [S:108-113]; un trozo cuyo extremo está a más de 2 mm de la punta local es un fragmento a media altura, no un ápice, y se cuenta aparte [S:128-131].
- *¿Sale al exterior?* En cada corte hacia la punta se calcula la **envolvente convexa** del corte (`convex_hull_image`, la "goma elástica" que rodea la sección) y se mira si hay hueco dentro de ella, a ≤ 0,2 mm del eje [S:132-146, S:276]. Un hueco dentro de la envolvente convexa es una muesca abierta al exterior.
- *Qué se guarda:* distancia a la última sección con diente, distancia a la punta local, distancia a la pared externa y longitud del hueco sin etiquetar, por conducto, en `info.json["apice"]` [S:147-154, S:265].
- *Coste:* `convex_hull_image` por corte solo en los cortes apicales (≈ 100–200 por conducto) [S:134-146]; el paso V5 completo del diente 52 (script **más** hashes de 8 ficheros de 1,2–1,4 GB) tardó 81,7 s **[reejecutado hoy]**, así que el script tarda menos (no medido por separado).

**(d) Resultado.** Distancia del final de la máscara a la punta, en mm, a 16 µm **[leído de fichero]** (`results/microct/Diente_XX/info.json`, clave `apice`; el 52 **[reejecutado hoy]**, V5, `json idénticos: True`):

| Diente | Conducto | A la última sección con diente de la raíz | A la punta local |
|---|---|---|---|
| 35 | raíz 1 | 0,912 | 0,864 |
| 35 | raíz 2 | 0,992 | 0,992 |
| 51 | conducto 1 | 0,832 | 0,832 |
| 51 | conducto 2 | 0,864 | 0,720 |
| 51 | conducto 3 | no válido (1,760) | 0,224 |
| 52 | raíz 1 | 0,496 | 0,496 |
| 52 | raíz 2 | 0,352 | 0,352 |

**¿Sale al exterior por el ápice y se pierde ese tramo?** Sí, pero el tramo perdido es corto. Comprobado a ojo en cortes axiales ampliados de 6 de los 7 conductos (el 51/conducto 2 no se miró): hasta el final de la máscara el conducto está encerrado por dentina; ~0,06 mm más abajo el hueco se abre al exterior por una muesca de la raíz (el foramen) y por debajo hay dentina maciza hasta la punta. Se pierden ~0,06–0,2 mm de hueco visible; el resto de la distancia es dentina sin luz visible. En la raíz 1 del 35 el conducto es ancho y ya está abierto en el final de la máscara. Las imágenes se regeneran con `diagnostico/ver_apice.py --diente Diente_52 --raiz 1 --conducto 1`; las del 2026-10-28 para enseñar a Isa están en `results/para_isa/` (`diagnostico/imagen_para_isa.py`). Las imágenes del ápice por conducto que quedan en `results/microct/Diente_XX/` son `control_apice.png` (las genera `S:160-197`). **Isa (endodoncista), de forma informal y relatada por Claudia, dijo que el conducto se mide siempre hasta el foramen apical y que la flecha (final de la máscara) es "perfecta"; solo vio el diente 52 (tres imágenes de ese diente); no vio el 35 ni el 51.** Esto contradice mi lectura de que se pierden 0,06–0,2 mm de hueco y la deja **sin confirmar**. Isa solo vio el diente 52 (tres imágenes de ese diente); el 35 y el 51 no se le enseñaron.
La métrica automática `hueco_sin_etiquetar_hacia_la_punta_mm` (0,22–0,72 mm en `info.json`) **sobreestima**: cuenta también la concavidad natural de la raíz. Para la longitud usar las imágenes, no ese número.

**(e) Límites.** Mi lectura de las imágenes era que la longitud medida sobre esta máscara quedaba ~0,1–0,2 mm corta respecto al foramen (solo 6 conductos mirados). **Isa (endodoncista) lo contradice, de forma informal y relatada por Claudia el 2026-10-28:** el conducto se mide siempre hasta el foramen apical y, en las imágenes de `results/para_isa/`, la flecha (final de la máscara) está "perfecta"; solo vio el diente 52 (tres imágenes de ese diente); no vio el 35 ni el 51. Mi lectura queda **sin confirmar** y la distancia de 0,35–1,0 mm a la punta de la raíz sería la distancia foramen–ápice. El detector de hueco por envolvente convexa tiene falsos positivos en raíces con surcos. La "punta" es el último corte axial con diente, no el punto más lejano a lo largo del eje de la raíz si la raíz está inclinada. El radio de búsqueda (0,2 mm) y la zona apical (2 mm) [S:276, S:201] no se han puesto a prueba.

---
### 3.5 Separar "solo conducto" de "cámara pulpar y cavidades"

**(a) Qué hace.** Decide un **plano** horizontal, el **orificio** (la entrada de los conductos en el suelo de la cámara), y llama "conducto" a todo lo que está por debajo (hacia el ápice) y "cámara y cavidades" a lo que está por encima (hacia la corona) [S:1-26, S:226-233]. La **cámara pulpar** es la sala central de la corona de la que salen los conductos; las **cavidades** son las caries o poros de la corona. Decisión de Claudia: la etiqueta de entrenamiento es solo el conducto, del orificio al ápice.

**(b) Por qué así.**
- *Criterio elegido:* recorrer los cortes axiales desde el ápice hacia la corona y medir en cada uno la **sección 2D más grande** del conducto. En la raíz hay varias secciones pequeñas (3–4 conductos); en la cámara se funden en una sola grande. El orificio es el primer corte con sección mayor ≥ **1,5 mm²** sostenida ≥ 0,1 mm [S:17-21, S:57-62, S:273-274].
- *Evidencia de que separa bien:* la mayor sección en la raíz, medida a ≥ 0,5 mm bajo el orificio, es 1,146 / 1,012 / 1,113 mm² **[leído de fichero]** (`info.json`, clave `separacion.mayor_seccion_en_la_raiz_mm2`; el 52 **[reejecutado hoy]**, V5). El umbral 1,5 queda ~1,3 veces por encima.
- *Alternativas descartadas:*
  - *Radio inscrito (grosor 3D) con un umbral:* el radio del conducto de la raíz llega a 0,34 mm y el de la cámara empieza en 0,39, sin hueco claro; a partir del área sí hay salto (≤ 1,15 → ≥ 1,6 mm² en < 0,6 mm de altura) [S:22-26]. **[reejecutado hoy]** con `diagnostico/secciones_y_orificio.py`, que imprime la sección mayor por corte.
  - *Apertura morfológica 3D (quitar lo "grueso" y reconstruir):* deja flecos finos alrededor de la cámara y depende del grosor del conducto de cada diente. **[razonado, no probado]**.
  - *Marcar el orificio a mano en cada diente:* no escala a ~40 dientes sin intervención.
- *Corona o ápice arriba:* se detecta con el esmalte (que solo está en la corona) en el 20 % superior o inferior de la altura del diente [S:65-73]. En los 3 dientes la corona está hacia z alto: el esmalte del extremo bajo vale 0,0 mm² y el del alto 9485,9 / 8355,8 / 7255,3 mm² (diente 35 / 51 / 52) **[leído de fichero]** (`info.json`, clave `orientacion`).

**(c) Cómo funciona el código.**
- `section_stats` recorre los cortes con conducto y en cada uno etiqueta componentes 2D (`ndi.label`), convierte vóxeles a mm² y se queda con la mayor de las ≥ 0,02 mm² [S:45-54]. Coste: ~1450 etiquetados 2D de 1230×790.
- `find_floor` busca el primer `z` tal que los siguientes `run` cortes (0,1 mm / 0,016 mm ≈ 6) tengan sección mayor ≥ 1,5; devuelve `None` si no hay [S:57-62, S:218-219].
- Trabaja siempre "con la corona hacia z alto": si no lo está, voltea los arrays (`[::-1]`, vistas sin copia) y deshace el volteo al guardar y al convertir índices [S:213-215, S:241-244].
- `solo[z_f:] = False` corta; después quita piezas sueltas < 0,005 mm³ [S:226-232]; `cam = canal & ~solo` [S:233]. Así **solo + cámara = completo y no se solapan** por construcción.
- Guarda `canal_completo.npy` (copia de `canal.npy`, que **no se toca**), `canal_solo_conducto.npy` y `camara_y_cavidades.npy` con nombres fijos [S:41, S:204-207, S:245-246]. Escribe en `info.json` el criterio, los parámetros y avisos [S:252-265].
- *Avisos automáticos* para revisar con ~40 dientes: orientación dudosa (el esmalte no es claramente mayor en un extremo), cámara no detectada, y "margen bajo" si la mayor sección de la raíz supera el 80 % del umbral [S:221-225, S:234-238].
- *Qué pasa si se cambia:* `--area-camara-mm2` entre 1,25 y 2,0: el orificio se mueve −0,11…−0,21 mm / +0,22…+0,59 mm según el diente (35/51/52), menos del 4 % de la longitud del conducto **[reejecutado hoy]** con `diagnostico/secciones_y_orificio.py` (35: −0,11 / +0,59 mm con 1,25 / 2,0 mm²; 51: −0,16 / +0,42; 52: −0,21 / +0,22). `--sostenido-mm`: **[no probado]**.

**(d) Resultado** **[leído de fichero]** (`info.json`; el 52 **[reejecutado hoy]**, V5: hashes SHA-256 de los tres `.npy` y el `info.json` idénticos antes y después de volver a ejecutarlo; el paso entero, con hashes, 81,7 s):

| Diente | Orificio (z, 16 µm) | Completo | Solo conducto | Cámara y cavidades |
|---|---|---|---|---|
| 35 | 786 | 21,199 mm³ | 11,151 | 10,048 |
| 51 | 797 | 23,139 mm³ | 8,001 | 15,138 |
| 52 | 815 | 36,885 mm³ | 11,702 | 25,182 |

Comando: `python 1_Data_Validation/analysis/microct/split_canal.py 1_Data_Validation/results/microct/Diente_52`.

**(e) Límites.** (1) El orificio es **un plano horizontal**; el suelo real de la cámara es irregular, y el corte puede dejar parte de la entrada de un conducto en la "cámara". (2) Ajustado con 3 dientes. (3) Un diente sin cámara detectada queda con **todo** el conducto como etiqueta (con aviso) [S:223-225]. (4) Caries dentro de la raíz, apicales al orificio, se quedarían como conducto (no ocurre en estos 3). (5) La etiqueta nunca ha sido comparada con un experto.

---

### 3.6 Simulador de CBCT a 75 µm

**(a) Qué hace.** De un bloque aleatorio de un µCT ya etiquetado produce un "CBCT falso" de 128×128×128 vóxeles de 75 µm (9,6 mm de lado) más su etiqueta alineada: imagen, **etiqueta blanda** (fracción de conducto por vóxel, entre 0 y 1), etiqueta binaria (≥ 0,5), máscara del diente, **mapa de pesos** para la pérdida y cámara blanda [SIM:1-31, SIM:359-360].

**(b) Por qué así.**
- *Idea central:* el falso sale del mismo volumen que la etiqueta, por lo que **no hay registro** (alinear dos imágenes de distinto origen, que es difícil y propenso a errores).
- *Aleatorización de dominio:* cada parámetro se sortea en un rango, para que la red no se aprenda un escáner concreto [Y:1-6, SIM:10]. Es la idea de SynthSeg (Billot et al., citada en `RD:141`).
- *Alternativas descartadas o no usadas:*
  - *Solo bajar la resolución:* no cierra la diferencia entre µCT y CBCT (borrosidad, ruido, contraste). **[razonado]**.
  - *Simulación física* (proyecciones de rayos X cónicas con ruido y reconstrucción): la más realista, pero requiere la geometría del escáner CBCT y mucho más cómputo. **[no probado]**.
  - *Red generativa que "traduzca" µCT a CBCT (CycleGAN y similares):* necesita muchas imágenes CBCT reales sin pares y puede inventar anatomía. Hay un solo CBCT real. **[no probado]**.
  - *Registrar CBCT reales con µCT:* descartado por falta de pares (los datos de Fer y Pulpy3D están descartados) [CL:9].
- *Mapeo de grises lineal con aire = −1000:* en el µCT el cociente esmalte/dentina es 1,64 (1,6/0,975); en el CBCT de Isa con aire en −1000 es (3339+1000)/(1678+1000) = 1,62 **[leído de fichero]** (`calibration.json`). Casi iguales, así que no hace falta una curva rara [SIM:289-294].
- *Alrededor del diente NO hay aire:* un diente extraído estaba en el aire del escáner, pero en un paciente está rodeado de hueso, médula y tejido blando. El simulador pone un fondo aleatorio con textura y a veces una capa fina de tejido blando pegada a la raíz (el **ligamento periodontal**) [SIM:295-306, Y:27-35]. Medido en Isa: el aire solo es el 0,002 % del recorte alrededor del diente **[leído de fichero]** (`calibration.json`, `grises.fraccion_en_el_minimo`).
- *Conducto, cámara y cavidades rellenos con gris de pulpa:* en un paciente contienen tejido, no aire. Si se rellenaran de "hueso", la red aprendería atajos falsos ("un hueco oscuro dentro del diente = conducto") [SIM:307-319, Y:54-59]. La etiqueta sigue siendo solo el conducto; la cámara va con etiqueta 0 y su gris es el del conducto más un desplazamiento aleatorio (±100 HU) para que no sean idénticos [Y:56].
- *Peso cero (ignorar) en una franja de ±0,15 mm alrededor del orificio:* el corte conducto/cámara es una convención de ±0,1 mm que la red no puede adivinar; ahí un error no debe castigarse. Alternativas: ignorar toda la cámara (la salida quedaría indefinida y habría que posprocesar con la regla del orificio a 75 µm, sin validar) o no ignorar nada [SIM:344-358, Y:57-59]. **[decisión no validada con entrenamiento real]**.
- *Desenfoque ANTES de reducir:* si se redujera primero, se produciría **aliasing** (patrones falsos al muestrear una señal sin filtrar); la PSF gaussiana de la calibración es mucho más ancha (0,14 mm) que el vóxel de 75 µm [SIM:325-329].
- *Ruido Poisson filtrado:* ver "(c)" y el hallazgo 5.2 (el ruido real es casi gaussiano).

**(c) Cómo funciona el código.** Todo ocurre en la GPU con tensores de PyTorch (arrays multidimensionales con soporte de GPU).
- **Entrada.** `Tooth` abre `norm.npy` y las máscaras con `np.load(mmap_mode="r")` [SIM:147-150] y guarda en disco una lista de coordenadas de vóxeles de conducto **submuestreada cada 6 vóxeles** por eje (`coords_*_s6.npy`) para elegir centros rápido [SIM:151-161]. `read` devuelve un bloque cúbico y rellena con ceros lo que cae fuera del volumen [SIM:186-193].
- **Tamaño del parche.** `crop_size(vóxel)` = `2·round(m·0,075/vóxel/2)` vóxeles del µCT: 600 con un µCT de 16 µm (128 vóxeles de 75 µm), 300 con uno de 32 µm; es par para poder promediar de 2 en 2 [SIM:213-218]. **El vóxel se lee del `info.json` de cada diente**, no se supone [SIM:275, SIM:296, SIM:310]; el vóxel de salida efectivo (`n·vóxel/m`) queda en los parámetros del parche [SIM:374]. Con 16 µm sale exacto. Con una copia a 32 µm (`diagnostico/prueba_vox_distinto.py`): mismo tamaño físico (9,599 mm), volumen de la etiqueta dentro de ±0,7 % y diferencia de imagen de 6–8 HU (RMS) **[reejecutado hoy]**. Con los interruptores antiguos el resultado a 16 µm es idéntico al de antes (`diagnostico/regresion_simulador.py`) **[reejecutado hoy]**.
- **Elección del recorte** (`pick_start`): se sortea primero el tercio del conducto (apical, medio, coronal) con las probabilidades `crop.third_weights` (`[1, 1, 1]` por defecto) y el centro sale de un vóxel de conducto de ese tercio más un desplazamiento aleatorio de hasta ±3 mm que no puede sacarlo de él (se vuelve a sortear); con `third_weights: null` el centro sale de un vóxel de conducto cualquiera (el comportamiento anterior). Se vuelve a sortear si el bloque tiene < 0,2 mm³ de conducto, salvo con probabilidad 5 % (parches "vacíos") [SIM:238-269, Y:12-16]. El tercio se define con la extensión del conducto de ese diente (`Tooth.third_of_z`) [SIM:167-184], la misma regla de `MET:31-45` (coinciden en 101 cortes **[reejecutado hoy]**). El cambio de fase respecto a la rejilla de 75 µm sale gratis: el bloque empieza en un vóxel cualquiera del µCT, y cubrir 0–74 µm de desplazamiento es el "medio vóxel aleatorio" [SIM:16-19].
- **Grises** [SIM:289-322]: `rho` = densidad relativa a la dentina; `rho = (min(norm,1) + ganancia_esmalte · max(norm−1, 0)) / 0,975 + (1 − min(norm,1)) · (fondo − aire)/(dentina_hu − aire)`; luego `imagen = aire + (dentina_hu − aire) · rho`. Es una **mezcla lineal** entre el diente y lo que hay detrás en cada vóxel (así el borde queda con gris intermedio).
  - El fondo se genera en una malla gruesa de 64 µm (`smooth_field`: ruido gaussiano filtrado, normalizado y subido a 16 µm con interpolación trilineal) para no gastar memoria ni tiempo [SIM:79-87, SIM:295-299, SIM:306].
  - `dilate_box`: dilatación con cubo de lado 2r+1 en **tres pasadas 1D** de `max_pool3d`; equivale a un máximo 3D de lado 2r+1 pero cuesta O(N·(2r+1)·3) en vez de O(N·(2r+1)³): con r = 9 (franja de 0,15 mm) son 57 operaciones por vóxel en vez de 6859. Equivalencia **[reejecutado hoy]** (E-f) para r = 3 y r = 9 [SIM:90-98].
- **Reducción.** `avg_pool3d(2)` (de 16 a 32 µm), desenfoque gaussiano separable con la PSF (σ_z = σ · factor) y por último `reduce_to`, que por defecto (`crop.reduccion: exacta`) promedia **cajas exactas**: cada vóxel de salida de 75 µm promedia los vóxeles de entrada que solapa con pesos iguales a la fracción solapada (`box_weights`: una matriz m×n por eje, aplicada con tres productos `einsum`; coste O(m·n·n²)) [SIM:101-126, SIM:329]. `blur` aplica un `conv3d` 1D por eje con relleno por replicación; coste O(N·K) por eje (K ≈ 4σ·2) en la malla de 32 µm (300³ = 27 millones de vóxeles) [SIM:54-67].
  - **Por qué se cambió.** La reducción anterior (`crop.reduccion: adaptativa`, `adaptive_avg_pool3d`) usaba ventanas de vóxeles enteros que se **solapan**: al pasar de 300 a 128, cada salida promediaba 3 o 4 vóxeles de 32 µm (media 3,31 = **106 µm**, no 75 µm) con un paso de 2,34 [SIM:112-120] **[reejecutado hoy]** (`diagnostico/comprobaciones.py --que ventanas`: PyTorch coincide con la fórmula floor/ceil; las cajas exactas suman 1 por fila y cubren 75,0 µm). Se conserva como opción para reproducir resultados antiguos.
  - **Efecto de corregirlo sobre la calibración con Isa** (`diagnostico/calibracion_reduccion.py`, mismas semillas y mismos parches, 2 semillas × 36 parches, parámetros en el centro de los rangos): **casi nulo**. Borde dentina–exterior σ 0,142 → 0,141 mm (Isa: 0,140); gris de dentina 1719,6 → 1719,8; ruido en dentina 62,84 → 62,80; autocorrelación 0,816 → 0,817; borde dentina–conducto 0,106 → 0,106 mm. La PSF (σ ≈ 0,14 mm) es mucho mayor que la diferencia entre ventanas de 75 y 106 µm. **No hizo falta retocar los rangos de la PSF** **[reejecutado hoy]**.
- **Ruido** [SIM:331-336]: se genera ruido Poisson (`torch.poisson` con `λ = cuentas · densidad^(2α)`), se centra y normaliza, se **filtra con una gaussiana** (esto introduce la correlación entre vecinos) y se reescala con `noise_gain` para que la desviación típica final sea la deseada. `noise_gain` es la ganancia teórica de un filtro gaussiano sobre ruido blanco; coincide con la medida (0,1488 medida frente a 0,1499 teórica, σ = 1,0; 0,0998/0,1011 con σ = 1,3; 0,0664/0,0676 con σ = 1,7) **[reejecutado hoy]** (E-d) [SIM:70-76].
- **Brillo** suave multiplicativo y aditivo [SIM:337-340, Y:49-52]; la imagen se recorta a ≥ −1000 [SIM:340].
- **Etiqueta** [SIM:123-126, SIM:342-343]: la **misma** función `pool_to` (promedios 2× + reducción exacta) sobre la máscara binaria, **sin** desenfoque, da la fracción de conducto por vóxel. Por usar la misma geometría, imagen y etiqueta quedan alineadas. La suma de la etiqueta blanda × (0,075)³ coincide con el volumen del bloque × (0,016)³ con una diferencia de +0,019 % en los 6 parches (solo el factor 0,015999/0,016: la reducción exacta conserva el volumen; la adaptativa daba −0,72 % … +0,20 %) **[reejecutado hoy]** (`diagnostico/comprobaciones.py --que conservacion`).
- **Alineación imagen–etiqueta (prueba con desenfoque y ruido apagados y el conducto como aire):** la correlación entre imagen y etiqueta desplazada ‑2…+2 vóxeles es la más negativa con desplazamiento 0 en los tres ejes y los tres dientes (p. ej. diente 52 en y: −0,529 a 0, frente a −0,429/−0,226 a ±1/±2) **[reejecutado hoy]** (`diagnostico/prueba_alineacion_y_pesos.py`).
- **Aumentos:** volteos y giro de 90° solo en el plano xy; z no se toca [SIM:361-371, Y:17].
- **Reproducibilidad:** una semilla de NumPy sortea parámetros y recorte; de ella sale la semilla del generador de PyTorch [SIM:279-280]. Misma semilla → imagen, etiqueta y parámetros idénticos (diferencia máxima 0,0); otra semilla → parche distinto **[reejecutado hoy]** (E-g).
- **Rendimiento:** `patch()` tarda 0,41–0,45 s en GPU (la primera llamada, ~1 s, incluye el arranque) y 87,4 s en CPU con 20 hilos: unas **210 veces más lento**; el bucle de la línea de comandos, que además comprime y guarda cada parche, tarda 1,0–1,6 s y llega a 11,3 GB de GPU de pico **[reejecutado hoy]** (`diagnostico/tiempos_simulador.py --que etapas` y `--que cpu_gpu`; `simulate_cbct.py --n 6`). `CUDA_VISIBLE_DEVICES=-1` no sirve para forzar CPU: Python se cae al importar torch.
- *Parámetros y efecto* (los rangos están en `Y`): `psf.sigma_mm` [0,08, 0,175] (más ancho = más borroso); `noise.sd_hu` [20, 95] (desviación típica del ruido); `noise.corr_sigma_vox` [1,0, 1,7] (cuánto se parecen los vecinos); `intensity.canal_hu` [−250, 500] (gris del conducto: es el parámetro más importante para el contraste y el **menos medido**) [Y:24-25]; `labels.weight_mode` `band|chamber|none` y `labels.chamber_fill` `pulp|bg` [Y:55-59]. Efecto de los modos: la franja `band` da peso 0 al 0,23–0,25 % del parche y `chamber` al 1,4–2,8 % **[no reejecutado hoy]** (2026-10-21, `sim_tests2.py`).

**(d) Resultado.** Con los parámetros en el centro de cada rango, el simulador (reducción exacta; la comprobación `--check` usa muestreo proporcional al volumen, como se calibró, porque Isa se midió en un recorte con corona) reproduce las medidas del CBCT de Isa **[reejecutado hoy tras los cambios]** (`python 1_Data_Validation/analysis/microct/simulate_cbct.py <3 carpetas results/microct/Diente_XX> --check --center --n 36 --seed 1 --calibration 1_Data_Validation/results/cbct_calibration/Isa_caso01/calibration.json`):

| Magnitud | Isa (real) | Simulado |
|---|---|---|
| Gris de la dentina (mediana) | 1678 | 1714,9 |
| Gris del esmalte | 3339 | 3393,3 |
| Gris del tejido lejano | 137 | 216,9 |
| Ruido (sd) en dentina | 61,1 | 62,5 |
| Ruido (sd) en tejido blando | 58,5 | 58,3 |
| Autocorrelación entre vecinos | 0,839 | 0,815 |
| Asimetría / curtosis del ruido en dentina | −0,316 / 0,173 | −0,372 / 0,293 |
| Borde dentina–exterior (σ) | 0,140 mm | 0,140 mm |
| Borde esmalte–exterior (σ) | 0,141 mm | 0,133 mm |
| Borde dentina–conducto (σ) | 0,114 mm | 0,105 mm |

**(e) Límites.**
1. **Un solo CBCT real** (un paciente, un escáner). Los rangos se ensancharon a propósito, pero ninguno se ha validado con otro CBCT. Que los grises sean unidades Hounsfield reales no está verificado [Y:6].
2. El **gris del conducto** está adivinado ([−250, 500]) porque el borrador de Isa y los conductos finos no dan un borde limpio [Y:24-25].
3. La textura de fondo es aleatoria, **no anatomía**: no hay dientes vecinos, ni cortical, ni raíces contiguas.
4. **No se simulan** endurecimiento del haz, rayas metálicas, movimiento ni "cupping". La caries abierta al exterior sigue siendo fondo con textura, no pulpa.
5. **El vóxel de cada diente se lee de su `info.json`** [SIM:213-218], ya no se supone 16 µm. Probado con una copia a 32 µm (arriba); **falta probarlo con un µCT real de otra resolución**. Si el vóxel no divide exactamente 75 µm, el vóxel de salida efectivo difiere algo (queda en los parámetros del parche). **[no verificado si los ~40 µCT son todos de 16 µm]**; `export_slicer.py` sí reduce siempre ×2.
6. **Muestreo por tercios (cambiado).** Antes los centros se sorteaban entre vóxeles de conducto sin ponderar y solo el 13,9 % (10–18 % según el diente) caía en el tercio apical, y el 16,8 % fuera del conducto. Ahora, por defecto, se reparten a partes iguales: apical 31,9 %, medio 31,2 %, coronal 35,2 %, fuera 1,7 %; el bloque contiene un 60 % más de conducto apical (0,38 → 0,61 mm³ de media) y la fracción de bloques con ≥ 0,05 mm³ de conducto apical sube del 40–66 % al 73–75 %; ningún recorte dejó de cumplir lo pedido (0/600 por diente) **[reejecutado hoy]** (`diagnostico/muestreo_por_tercios.py --n-centros 600 --n-bloques 100`). **Efecto sobre el entrenamiento:** con 1000 iteraciones (una vuelta, una semilla) el Dice apical final es igual (0,58 con ambos muestreos) y el equilibrado converge antes; no hay evidencia de mejora del ápice (3.12).
7. La comprobación contra Isa tiene un **sesgo de la medida**: con rangos aleatorios el "ruido en dentina" sale hasta 81–86 frente a 61 (comparación en 5.1); con los parámetros en el centro sale 62,6.
8. Los números del simulador reproducen los de Isa en las magnitudes **que se usaron para calibrarlo** (circularidad): no demuestran que el falso "engañe" a una red; eso solo se verá evaluando en CBCT real (etapa 4, pendiente).

---

### 3.7 Calibración con el CBCT de Isa

**(a) Qué hace.** Mide en un CBCT real tres cosas que definen "cómo se ve" un CBCT: **grises** por estructura, **ruido** en zonas uniformes y **desenfoque de los bordes** [C:1-17].

**(b) Por qué así.**
- Son las tres degradaciones que el simulador aplica, así que lo medido fija el centro de los rangos del azar [C:3-8, Y:1-6].
- *Alternativas:* (1) copiar valores de la bibliografía (los CBCT varían mucho por escáner); (2) medir la **PSF/MTF** con un fantoma (objeto de calibración): no hay; (3) no calibrar y sortear rangos enormes: aumenta la brecha de dominio. **[no probado ninguna]**.
- El **borrador de la segmentación de Isa solo se usa para decir dónde está el conducto** (zona de interés) [C:15-17]; su etiqueta es "en curso" (`Segmentation.Status: inprogress`, cabecera del NRRD) **[reejecutado hoy]** (E-a).

**(c) Cómo funciona el código.**
- `load_crop`: lee el NRRD (ejes x, y, z), recorta alrededor de la segmentación con 2,3 mm de margen [C:30-41].
- **Grises** [C:51-75]: esmalte = vóxeles con valor suavizado > 2800 erosionados; dentina = anillo entre 0,2 y 1,5 mm del conducto con valor suavizado entre 1400 y 2000; interior del conducto = vóxeles a ≥ 0,15 mm del borde de la máscara; tejido lejano = a > 3 mm. Ojo: **las ventanas de gris están escritas para este CBCT** [C:54, 58]; con otro escáner habría que cambiarlas.
- **Ruido** [C:79-134]: se parte el volumen en bloques de 9×9×9 vóxeles; en cada bloque cuyo gris medio cae en la clase (esmalte 2900–4500, dentina 1450–1850, blando −300…150) [C:114] se ajusta un **plano** por mínimos cuadrados (4 parámetros: `A @ (pinv @ v)`) y se mide la desviación típica del residuo, corrigiendo los grados de libertad (`sqrt(n/(n−4))`) [C:85-87, C:101-103]. Se rechazan bloques que cruzan un borde (rango p5–p95 > 1500) [C:97-99]. También se mide la correlación entre vecinos a 1 vóxel, la asimetría y la curtosis [C:105-110].
- **Bordes** [C:142-215]: gradiente gaussiano (σ = 1,2 vóxeles), máximos locales del módulo del gradiente (ventana 3×3×3), perfil muestreado en 3D (trilineal) a ±0,6 mm a lo largo del gradiente, aceptado solo si las dos mesetas son planas (< 12 % del contraste) y el contraste > 800; se normaliza a 1→0, se alinea en el 50 % y se ajusta con `least_squares` una **función error** (borde gaussiano emborronado) para obtener σ [C:144-158, C:176-186, C:211]. σ **incluye el vóxel y la interpolación** [C:196-197].
- *Coste:* 9,9 s **[reejecutado hoy]** (V4). Los bucles de bloques son en Python puro (`for i, j, k`), aceptable para un volumen recortado de 175×209×273.

**(d) Resultado** **[reejecutado hoy]** (V4: `calibration.json` idéntico al guardado; `python 1_Data_Validation/analysis/microct/calibrate_cbct.py "<cbct.nrrd>" --seg "<seg.nrrd>" --out <carpeta>`):

| Medida | Valor |
|---|---|
| Gris dentina (mediana; p5–p95) | 1678 (1514–1884) |
| Gris esmalte | 3339 |
| Interior del conducto (vóxeles profundos del borrador) | 433 (mezcla con el borde: poco fiable) |
| Tejido lejano | 137 (muy ancho: hueso + médula) |
| Ruido (sd) dentina / blando / esmalte | 61,1 / 58,5 / 156,6 |
| Autocorrelación a 1 vóxel (dentina, x, y, z) | 0,839 / 0,836 / 0,840 |
| Asimetría / curtosis del ruido (dentina; blando) | −0,316 / 0,173; 0,288 / 0,202 |
| σ borde dentina–exterior (n = 752) | 0,140 mm |
| σ borde esmalte–exterior (n = 142) | 0,141 mm |
| σ borde dentina–conducto (n = 35; **ajuste malo**, error 0,062) | 0,114 mm |

**(e) Límites.** El borde dentina–conducto sale de solo 35 bordes con mal ajuste; el de z de 30 bordes: poco fiables. Las ventanas de gris son específicas de este CBCT. La segmentación de Isa es un borrador. Es **un caso**: no se sabe cuánto varían estos valores de un escáner o paciente a otro.

---

### 3.8 Dataset de entrenamiento por carpetas

**(a) Qué hace.** Entrega lotes de parches simulados a la red y un banco fijo de parches para validar [DS:1-18].

**(b) Por qué así.**
- *Descubrir dientes por carpeta:* un diente es toda subcarpeta con `info.json`, `norm.npy` y `canal_solo_conducto.npy` [DS:34, DS:37-40]. Añadir un µCT es añadir su carpeta, sin editar listas. Sustituye a la forma de Pulpy3D, que necesita `splits.json` y volúmenes `.nii.gz` con el paciente ya dividido [PL:23, PL:31]. **Hay ~40 µCT en traspaso** [CL:17], y la lista cambiará.
- *Generar en la GPU, en el proceso principal, con un hilo de prefetch:* el docstring afirma que "la GPU no se comparte bien con procesos hijos" [DS:9-10]; **[no verificado]**, no se probó con procesos hijos. Tampoco se probó `torchio.Queue` (lo que usa Pulpy [PL:53-68]).
- *Alternativas descartadas o no usadas:* (1) **banco pregenerado en disco** (8 MB por parche comprimido; 4000 parches ≈ 32 GB, poca diversidad entre épocas) **[razonado, no probado]**; (2) **`DataLoader` con varios procesos** (descartado por lo anterior, no verificado).
- *Normalización por parche:* los grises de cada parche se reescalan con los percentiles 1 y 99,5 y se recortan a [−0,5; 1,5], para que la red no dependa de la escala de gris de un escáner [DS:43-46].

**(c) Cómo funciona el código.**
- `SimCBCT.__init__` carga los `Tooth` pedidos o todos, y falla con un mensaje claro si falta alguno [DS:50-67]; `crop_overrides` permite cambiar claves de `crop:` del yaml del simulador (p. ej. `{"third_weights": None}`) [DS:16, DS:61].
- `_one` genera un parche y calcula el **mapa de tercios** de sus cortes con `third_map` [DS:70-75]; `_batch` elige un diente al azar **con reemplazo** para cada parche del lote [DS:77-81].
- `_worker` es un hilo (`threading.Thread`) que produce lotes en una cola acotada (`queue.Queue(maxsize=prefetch)`) y se para con un `Event`; `next_batch` lo arranca la primera vez y hace `get()` [DS:84-107]. Los kernels de CUDA son asíncronos, así que el hilo Python (limitado por el GIL) solo lanza trabajo a la GPU y espera la lectura del disco.
- `fixed_bank(n, seed)`: `n` parches con una semilla fija: la misma validación cada vez [DS:109-112].
- *Parámetros:* `prefetch` (2), `batch_size` (2), `val_patches` (6) [CFG:10-14].

**(d) Resultado.** Con 3 carpetas descubre `Diente_35`, `Diente_51` y `Diente_52` y arranca (V8: "dientes: ['Diente_35', 'Diente_51', 'Diente_52']") **[reejecutado hoy]**. El tiempo de espera de la red al simulador fue 0,89–0,90 s por iteración (2 parches) en las 3 vueltas **[leído de fichero]** (`2_UNet/results/entrenamientos_sim/*/metrics.json`).

**(e) Límites.** El diente de cada parche se elige con probabilidad uniforme, sin tener en cuenta el tamaño del diente. No hay control del equilibrio entre dientes. El muestreo por tercios (3.6) vale para el entrenamiento; la validación usa siempre parches repartidos a partes iguales entre tercios [TR:94-95]. Si `holdout` queda vacío, no hay validación [TR:95].

---

### 3.9 La red: U-Net 3D

**(a) Qué hace.** Recibe un parche 3D de 128³ y devuelve, para cada vóxel, un **logit** (un número real; su sigmoide es la probabilidad de "conducto") [NET:13-43].

**(b) Por qué así.**
- Una **U-Net** es una red convolucional con forma de "U": un **codificador** que reduce la resolución varias veces y extrae rasgos cada vez más globales, un **decodificador** que la recupera, y **conexiones de salto** (*skip connections*) que pasan los detalles finos del codificador al decodificador. Es el estándar en segmentación médica.
- *Por qué no se usó la red de Pulpy tal cual:* los modelos `PosPadUNet3D`/`AttentionPosPadUNet3D` necesitan, además de la imagen, un vector de 6 números con la **posición del parche dentro del volumen** (embedding de posición) [UN:18, UN:39, UN:47, MF:14, MF:18] y devuelven la sigmoide dentro del modelo [UN:63]. Los parches simulados no tienen un volumen global; y con etiquetas blandas conviene BCE con logits, que es más estable. `UNETR` tiene el tamaño de imagen fijado en (80,80,80) [MF:22]. Por eso se escribió `CanalUNet3D`, que **solo reutiliza el bloque de convoluciones** de Pulpy [NET:10, NET:22, UN:67-89] **(esto se aparta de la petición de Luis; ver 7(c))**.
- *Alternativas no usadas:* UNETR, VNet (están en el repo, no probadas aquí), nnU-Net (externo).

**(c) Cómo funciona el código.**
- Cada bloque es conv 3×3×3 → **GroupNorm** (8 grupos) → Dropout3d → ReLU, dos veces [UN:67-89]. GroupNorm normaliza por grupos de canales dentro de cada muestra, no entre muestras del lote: con lote 2 es más estable que BatchNorm (**razón general; no consta que sea la de los autores [no verificado]**). `Dropout3d` apaga canales enteros con probabilidad `p` (0,1 en la configuración) [CFG:20].
- Profundidad 4: 4 bloques de codificación con 16, 32, 64, 128 canales y `MaxPool3d(2)` entre ellos, un cuello con 256 canales y 4 decodificadores con `ConvTranspose3d` (k = 2, paso 2) y concatenación con el salto [NET:21-31, NET:34-43]. Con 4 reducciones por 2, el lado del parche debe ser múltiplo de 16 (128 → 8 en el cuello); el código no lo comprueba.
- 5,65 millones de parámetros **[reejecutado hoy]** (V8). Capa final: `Conv3d` 1×1×1 → logits [NET:32, NET:43].

**(d) Resultado.** Entrena y valida (V8; ver 3.12). Calidad **no evaluada**.

**(e) Límites.** Sin ajuste de tamaño o profundidad; sin supervisión profunda; el cuello de 8³ limita el contexto global; sin pruebas de la capacidad de generalizar. El campo receptivo (cuánto ve cada vóxel de salida) **[no calculado]**.

---

### 3.10 La pérdida: BCE + Dice con etiqueta blanda y pesos

**(a) Qué hace.** Mide el error entre la salida de la red y la etiqueta, ignorando los vóxeles con peso 0 [LOSS:1-5].

**(b) Por qué así.**
- *Etiqueta blanda:* a 75 µm un vóxel puede ser en parte conducto; la fracción (entre 0 y 1) que sale de reducir la máscara de 16 µm conserva esa información en vez de forzar 0/1 [SIM:342; CL:24].
- *BCE con logits (entropía cruzada binaria)* acepta objetivos entre 0 y 1 y es numéricamente estable [LOSS:19]. *Dice blando* (solapamiento entre predicción y objetivo) compensa que el conducto ocupa una fracción pequeña del parche [LOSS:20-25]. La suma de ambas es habitual.
- *Alternativas del repo no usadas:* `Jaccard` (pérdida de Pulpy) y `DiceCrossEntropy` (multiclase) [PC:41-42; `2_UNet/losses/LossFactory.py:17-27`]. **[no se probó una comparación]**.
- *Mapa de pesos:* un peso 0 quita ese vóxel de ambas partes (ignorar) [LOSS:18-19, 22-23]; sirve para la franja del orificio (3.6).

**(c) Cómo funciona el código.** Todo se convierte a `float32` [LOSS:17]; la BCE se calcula por vóxel, se multiplica por el peso y se **normaliza por la suma de pesos** (con mínimo 1) [LOSS:19]; el Dice se calcula **por muestra** sobre las dimensiones no de lote, con `eps = 1e-6`, y se promedia sobre el lote [LOSS:21-25]. Coste: O(N) por parche.

**(d) Resultado.** La pérdida de entrenamiento baja de 1,55 a 0,98 / 0,97 / 1,01 en las 3 vueltas de 200 iteraciones (dejando fuera el 35 / 51 / 52) **[leído de fichero]** (`metrics.json`, clave `historia`; se vio también bajar en V8).

**(e) Límites.** En parches sin conducto (5 % por diseño [Y:16]) el término Dice vale ≈ 1 mientras la red no devuelva exactamente 0, lo que mete ruido en el gradiente **[razonado, no medido]**. No hay pesos por clase. El 0,5 usado para binarizar no se ha optimizado.

---

### 3.11 Métricas: Dice y HD95 por tercios

**(a) Qué hace.** Compara la predicción binaria con la etiqueta binaria **por separado** en el tercio apical, medio y coronal del conducto, y en total [MET:1-10].

**(b) Por qué así.**
- El Dice global lo domina la parte coronal ancha; el ápice, lo más difícil y clínicamente importante, pesa poco. De ahí los **tercios** (regla del proyecto [CL:37]).
- **HD95** (distancia de Hausdorff al percentil 95): la distancia (mm) que no supera el 95 % de los puntos de una superficie a la otra; mide los errores "lejanos" y es menos sensible a un único punto atípico que el Hausdorff máximo.
- *Alternativas no implementadas:* Dice de superficie, error en la línea central, error de cada medida (longitud, curvatura, volumen, diámetro), distancia al ápice real; **pendientes** (`MET:8-9`).

**(c) Cómo funciona el código.**
- `dice` devuelve NaN si ambas máscaras están vacías (no inventa un 1 o un 0) [MET:17-20]. `hd95` usa `medpy` con `voxelspacing` y devuelve NaN si falta alguna máscara [MET:23-28].
- `third_map` toma la **extensión en z del conducto del diente completo** (`canal_solo_z_vox`, del `info.json`) y la orientación; calcula la fracción desde la corona hasta el ápice para cada corte del parche y asigna 0 (apical, ≥ 2/3), 1 (medio) o 2 (coronal); fuera de la extensión = −1 [MET:31-45]. Así "apical" significa lo mismo en todos los parches.
- `metrics_by_thirds` selecciona los cortes de cada tercio y calcula Dice y HD95 sobre ese subvolumen [MET:48-59].
- *Comprobación con un caso conocido* **[reejecutado hoy]** (E-h): con la corona hacia z alto, el tercio apical queda en z bajo (101–298, de 100–700) y se invierte con la corona hacia z bajo; un cuadrado de 4 vóxeles desplazado 1 vóxel da Dice 0,75 y HD95 0,075 mm (un vóxel), como se espera.

**(d) Resultado.** En la validación de la prueba de montaje (dejando fuera el 52, it. 200): Dice apical / medio / coronal / total = 0,621 / 0,778 / 0,820 / 0,775; HD95 total 1,54 mm. Para el 35 fuera: 0,711 / 0,748 / 0,679 / 0,672, HD95 2,03 mm. Para el 51 fuera: 0,316 / 0,538 / 0,560 / 0,538, HD95 2,30 mm **[leído de fichero]** (`metrics.json`). **No son resultados de calidad** (3.12).

**(e) Límites.** Se calculan sobre parches, no sobre el diente entero. `np.nanmean` ignora los HD95 NaN (predicción vacía) [TR:53]: un modelo que no predice nada tiene Dice 0 y HD95 "ausente", no "malo". Los límites entre tercios cortan el parche y el HD95 se calcula sobre el subvolumen. Faltan la distancia al ápice real y el error de las medidas.

---

### 3.12 El bucle de entrenamiento

**(a) Qué hace.** Entrena la red con parches simulados y valida **dejando un diente fuera** [TR:1-13].

**(b) Por qué así.**
- **"Dejar un diente fuera"** (*leave-one-tooth-out*): se entrena con todos los dientes menos uno y se valida en ese; así no hay **fuga de datos** (que el mismo diente aparezca en entrenamiento y validación con otro recorte, lo que inflaría los resultados) [TR:5-6, TR:72, TR:84, CL:36]. Con 3 dientes son 3 vueltas.
- **Adam** con tasa 0,001 [CFG:27-30, TR:100] y **bf16** (números de 16 bits con el rango de los de 32) [TR:116]. El Pulpy original usa SGD con tasa 0,1 y planificador por meseta [PC:48-50]. **[no se comparó]**.
- Validación con **6 parches fijos** (semilla 12345) del diente de fuera, **repartidos a partes iguales entre los tercios** del conducto, sea cual sea el muestreo de entrenamiento [TR:93-95, CFG:14]. `--val-patches`, `--seed` y `--third-weights {config,equilibrado,proporcional}` permiten cambiarlo desde la línea de comandos [TR:64-67].

**(c) Cómo funciona el código.**
- Exige GPU (`assert torch.cuda.is_available()`) [TR:80]: **no se puede entrenar en CPU con este script**.
- Bucle [TR:111-135]: pide un lote (cronometra la espera), hace el paso hacia delante bajo `autocast` bf16, pérdida, `backward`, paso de Adam; sincroniza la GPU para medir tiempos; cada 50 iteraciones valida y apunta tiempos y pico de memoria.
- `validate` pone el modelo en modo `eval` (sin dropout), predice cada parche, binariza con 0,5, ignora los vóxeles con peso 0 en las métricas y promedia sobre los parches [TR:35-55].
- Guarda `config.yml`, `log.txt`, `metrics.json` y `last.pt` (solo el último estado, sin "mejor") [TR:87-90, TR:138-144].
- *Parámetros:* `iterations` 200, `val_every` 50, `batch_size` 2, `amp` bf16 [CFG:10, CFG:33-35].

**(d) Resultado** (prueba de montaje, **[leído de fichero]**, `2_UNet/results/entrenamientos_sim/sim_canal_montaje_fuera_Diente_XX/metrics.json`):

| Fuera | Entrena con | s por iteración (datos / red) | GPU pico | Pérdida entr. (it. 1 → 200) | Dice val. total (it. 200) | HD95 total |
|---|---|---|---|---|---|---|
| 35 | 51, 52 | 1,72 (0,89 / 0,75) | 10,9 GB | 1,545 → 0,997 | 0,672 | 2,03 mm |
| 51 | 35, 52 | 1,66 (0,90 / 0,68) | 10,8 GB | 1,545 → 0,975 | 0,538 | 2,30 mm |
| 52 | 35, 51 | 1,71 (0,90 / 0,74) | 10,9 GB | 1,546 → 1,010 | 0,775 | 1,54 mm |

Cada vuelta: 200 iteraciones ≈ 340 s. El Dice vale 0 hasta la iteración 100 y sube después. Reejecutado hoy con 30 iteraciones (V8): 1,65 s por iteración, 10,74 GB de pico, 49,4 s en total. Comando: `cd 2_UNet` y `python train_sim.py -c configs/sim_canal.yml --holdout Diente_52`.
**No se saca ninguna conclusión de calidad:** 2 dientes de entrenamiento, 200 iteraciones, 6 parches de validación. Solo demuestra que el montaje funciona de extremo a extremo.

**Comparación del muestreo por tercios en entrenamiento (2026-10-28, `2_UNet/results/entrenamientos_sim/sim_canal_montaje_tercios_*`).** 10 entrenamientos de 200 iteraciones, lote 2, validación en 12 parches fijos repartidos entre tercios (igual para ambos muestreos), con el muestreo antiguo (`--third-weights proporcional`) y el equilibrado (`equilibrado`). Dice apical / total al final, **[leído de fichero]** (`metrics.json`):

| Fuera | Semilla | Proporcional | Equilibrado |
|---|---|---|---|
| 35 | 33 | 0,537 / 0,597 | 0,000 / 0,000 |
| 51 | 33 | 0,011 / 0,160 | 0,288 / 0,371 |
| 52 | 33 | 0,000 / 0,000 | 0,000 / 0,000 |
| 52 | 34 | 0,247 / 0,519 | 0,341 / 0,530 |
| 52 | 35 | 0,000 / 0,000 | 0,000 / 0,000 |

**No permite concluir nada:** **5 de los 10 entrenamientos acabaron con Dice 0 en todo** (la red aún no había empezado a predecir tras 200 iteraciones; con la misma configuración, la vuelta del 52 con semilla 33 se queda en 0 y con la 34 llega a 0,5), así que la diferencia entre muestreos queda enterrada en si el entrenamiento "despega" o no. En las dos parejas en que ambos despegaron (51 y 52 con semilla 34) el equilibrado tiene mejor Dice apical (0,288 frente a 0,011 y 0,341 frente a 0,247), y en la del 35 solo despegó el proporcional. Cinco parejas y una semilla por pareja no distinguen un efecto del azar. Tiempo: 1,22–1,36 s por iteración (una vuelta con 2,44 s coincidió con otros cálculos de CPU), GPU 11,9–13,6 GB de pico.

**Comparación larga (1000 iteraciones; una sola vuelta, dejando fuera el 52, semilla 33; 12 parches de validación repartidos entre tercios)** **[leído de fichero]** (`..._largo1000_*/metrics.json`):

| Muestreo de entrenamiento | Empieza a predecir (Dice > 0) | Dice apical it. 1000 | Dice total | HD95 apical / total | Pérdida de validación | s por it. / GPU |
|---|---|---|---|---|---|---|
| proporcional (antiguo) | it. ~700 | 0,580 | 0,699 | 1,74 / 0,48 mm | 0,509 | 1,38 s / 12,0 GB |
| equilibrado por tercios | it. ~300 | 0,584 | 0,727 | 2,56 / 0,48 mm | 0,312 | 1,59 s / 13,5 GB |

**Lectura prudente:** el Dice apical final es **el mismo** (0,58); el muestreo equilibrado empezó a predecir antes y acabó con menor pérdida de validación y un Dice total algo mayor (0,03), pero es **una ejecución con una semilla**, y en las pruebas cortas la mitad de los entrenamientos no llegaron a despegar en 200 iteraciones, así que el adelanto puede ser azar. El HD95 apical oscila entre 0,66 y 2,56 mm en las últimas 400 iteraciones del equilibrado (0,66 en la it. 900 y 2,56 en la 1000): con 12 parches es ruido. **No hay evidencia de que el muestreo equilibrado mejore el ápice**; tampoco de que lo empeore. Para decidirlo hacen falta varias semillas y varias vueltas con entrenamientos largos. **Decisión de Claudia (2026-10-28):** el muestreo equilibrado por tercios se queda (`crop.third_weights: [1,1,1]`); la comparación con varias semillas no se repite ahora, porque con 2 dientes de entrenamiento el resultado sería ruido, y se volverá a hacer cuando lleguen los ~40 dientes, con validación cruzada (igual que en CLAUDE.md).

**(e) Límites.** No hay conjunto de test aparte ni selección del mejor modelo; no hay planificador de la tasa de aprendizaje ni parada temprana; el determinismo de la GPU (mismos resultados con la misma semilla) **[no probado]**; la espera por el simulador es el 52 % del tiempo y la red compite por la GPU con el hilo de simulación (0,28 s por paso de red sin competencia en una prueba corta del 2026-10-21 frente a 0,68–0,75 s con él **[no aislado como causa]**).

---

### 3.13 Exportación a 3D Slicer (para revisar las etiquetas)

**Qué hace.** Escribe, por diente, una imagen `uint8` y 3 máscaras (solo conducto, cámara y cavidades, completa) a 32 µm, recortadas a la caja del diente, comprimidas con gzip en NRRD [X:1-13, X:46-50]. La imagen es `round(norm · 127,5)` saturada a 0–255 (0 = aire, ≈124 = dentina, ≈204 = esmalte) [X:6, X:66]. Las máscaras de 32 µm salen por **mayoría de bloques 2×2×2** (≥ 4 de 8) [X:36-43]; la cámara se deriva como "completa − solo conducto" para que no se solapen en el plano del orificio (un bloque a caballo del plano tiene 4 vóxeles de cada lado y ganaría en los dos) [X:74-77].
**Resultado.** Por diente: imagen 50,24 / 50,21 / 53,16 MB (35 / 51 / 52) y cada máscara 0,12–0,21 MB **[leído de fichero]** (`results/para_slicer/`). Releídos con SimpleITK: tamaño, espaciado (0,032 mm) y sumas correctos, sin solapes; **no se han abierto en Slicer**. Comando: `python 1_Data_Validation/analysis/microct/export_slicer.py <carpetas> --out 1_Data_Validation/results/para_slicer`. Instrucciones en `results/para_slicer/LEEME.txt`.
**Límites.** La orientación izquierda/derecha del escaneo no se conoce; un conducto de menos de ~30 µm puede desaparecer en las copias a 32 µm.

---
## 4. Glosario corto

| Término | Qué es |
|---|---|
| **µCT / CBCT** | Dos tipos de tomografía de rayos X. El **µCT** (micro-TC) escanea piezas pequeñas (aquí dientes extraídos) con vóxeles de 16 µm. El **CBCT** (TC de haz cónico) es el que se usa en la consulta, con vóxeles de unos 75 µm y más ruido y borrosidad. |
| **Vóxel** | El "píxel" 3D: un cubito de la imagen con un valor de gris. Un volumen es un array 3D de vóxeles. 1 µm = 0,001 mm. |
| **memmap** | `np.memmap`: un array NumPy cuyo contenido vive en un fichero y se lee del disco solo cuando se indexa. Permite recorrer volúmenes de 2,9 GB sin cargarlos enteros en RAM [P:56]. |
| **Otsu** | Método clásico que elige el umbral que mejor separa dos "montones" de valores en un histograma (fondo y objeto). Aquí solo se usa para definir las anclas de la normalización [P:73-77]. |
| **Efecto de volumen parcial** | Un vóxel que cubre mitad aire y mitad diente tiene un gris intermedio. Por eso el borde de un objeto se sitúa en el punto medio (0,5 tras normalizar). |
| **Ápice / foramen** | El **ápice** es la punta de la raíz; el **foramen apical** es el agujero por donde el conducto sale al hueso. |
| **Orificio / cámara pulpar** | La **cámara pulpar** es la sala dentro de la corona de la que salen los conductos; su suelo tiene la **entrada (orificio)** de cada conducto. |
| **Etiqueta blanda** | Etiqueta con valores entre 0 y 1 (la fracción de cada vóxel que es conducto), en vez de solo 0 o 1. Sale de reducir la máscara de 16 µm a 75 µm promediando [SIM:342]. |
| **PSF** | *Point spread function*: cómo "esparce" el escáner un punto. Un punto se ve como una mancha; modelarla como una gaussiana de anchura σ es la forma más simple de simular la borrosidad [SIM:325-329]. |
| **Dice** | Medida de solapamiento entre dos máscaras: 2·|A∩B| / (|A|+|B|). 1 = idénticas; 0 = sin solapamiento [MET:17-20]. |
| **HD95** | Distancia de Hausdorff al percentil 95, en mm: cuánto se separan los bordes de dos máscaras sin contar el 5 % de puntos más lejanos [MET:23-28]. |
| **U-Net** | Red convolucional con forma de "U": codificador que reduce la imagen, decodificador que la recupera y conexiones de salto entre ambos [NET:13-43]. |
| **Logit** | Salida de la red antes de la sigmoide; `sigmoide(logit)` es la probabilidad de "conducto". |
| **Dejar un diente fuera** | *Leave-one-tooth-out*: validar en un diente que no se ha usado para entrenar, para evitar fugas de datos [TR:5-6]. |
| **Transformada de distancia (EDT)** | Para cada vóxel de una máscara, la distancia a la pared más cercana; en un conducto da su "radio local" [P:144]. |
| **Hounsfield (HU)** | Escala de densidad en TC (aire ≈ −1000, agua ≈ 0). Que los grises del CBCT de Isa sean HU reales **no está verificado** [Y:6]. |

---

## 5. Hallazgos y correcciones durante el proyecto

### 5.1 Hallazgos (lo que no era como se pensaba)

1. **El diente 51 no se partía por "ruido que abre la pared"**, sino porque el conducto dejaba de estar encerrado en cortes sueltos (z = 361 y 440) y el filtro de longitud mínima tiraba los trozos. La hipótesis inicial del proyecto ("el ruido abre huecos en la pared") **era falsa** [P:116-118]. **[no reproducible desde el repo]**.
2. **La corrección creó un problema nuevo**: la unión de trozos arrastraba 60.837 motas de 1–10 vóxeles (1,4 mm³) al diente 35. Se corrigió quitándolas antes de unir [P:96-107, P:129]. **[no reproducible desde el repo]**.
3. **Las "láminas de caries" del diente 35** (3,9 mm³: rayas de 1–2 vóxeles alrededor del agujero de una caries) pasaban todos los filtros porque medían 6 mm de largo; se quitan por grosor (radio inscrito p90 < 0,05 mm) [P:143-147]. La prolongación apical del 35 (5083 vóxeles) y los conductos finos del 51 se conservan al 100 % **[no reproducible desde el repo]**. Efecto secundario conocido: una pieza de 0,37 mm³ de la cámara del 35 se pierde por estar en el mismo grupo que las rayas.
4. **El ruido del CBCT real es casi gaussiano** (asimetría −0,316, curtosis en exceso 0,173 en dentina; una gaussiana daría 0 y 0) y **muy correlacionado** entre vóxeles vecinos (autocorrelación 0,84) **[reejecutado hoy]** (V4). Esto contradice la hipótesis inicial de "ruido no gaussiano" (CL:11). Solo el esmalte se aparta (asimetría −1,38, curtosis 2,32), pero ahí pesa la estructura. El simulador implementa Poisson filtrado [SIM:331-336], pero tras filtrar es casi gaussiano y reproduce los números reales (V6).
5. **El ruido no crece con la densidad**: 61,1 en dentina frente a 58,5 en tejido blando, de ahí el exponente `density_exponent` casi nulo [Y:46].
6. **La relación µCT→CBCT es lineal**: cociente esmalte/dentina 1,64 en µCT frente a 1,62 en CBCT con aire = −1000 [SIM:289-294].
7. **El conducto sale al exterior por el ápice y se pierde un tramo corto** (0,06–0,2 mm de hueco visible; la máscara queda a 0,22–0,99 mm de la punta de la raíz) (3.4).
8. **Los 3 µCT tienen la corona hacia z alto**; el 35 es un molar de dos raíces; 51 y 52 **no están identificados** [CL:17].
9. **La segmentación de Isa**: una sola etiqueta con 109.890 vóxeles = 46,36 mm³ y **15 componentes** (10 de ≥ 100 vóxeles) y `Segmentation.Status: inprogress` **[reejecutado hoy]** (E-a). Esto confirma lo que CL decía de "46 mm³, 15 componentes", que no había comprobado antes.
10. **Las redes de Pulpy3D no encajan**: necesitan el vector de posición del parche [UN:18, UN:39, MF:14] (3.9).
11. **El cuello de botella era el simulador**: la red tardaba 0,28–0,75 s por iteración y la espera al simulador era 5,97 s (prueba del 2026-10-21, `metrics.json` del primer ensayo, ya borrado). Tras pasar la conversión de tipos a la GPU y usar una dilatación separable, `patch()` pasó de ~3 s a 0,46 s, con la dilatación verificada equivalente (E-f) **[reejecutado hoy]**. La espera por iteración quedó en 0,9 s.
12. **El muestreo favorecía la corona (corregido).** Solo el 13,9 % de los centros de parche caía en el tercio apical y el 16,8 % fuera del conducto; con el muestreo por tercios, 31,9 % y 1,7 % (3.6, límite 6) **[reejecutado hoy]**.
13. **Las ventanas de reducción a 75 µm se solapaban (corregido).** Promediaban 106 µm en vez de 75 µm; la reducción exacta no cambia la calibración con Isa de forma apreciable (σ del borde 0,142 → 0,141 mm) (3.6) **[reejecutado hoy]**.
14. **Sesgo de la medida de ruido**: con rangos aleatorios el "ruido en dentina" simulado sale hasta 81–86 frente a 61; el ruido medido correlaciona 0,65–0,76 con el gris de dentina sorteado, porque la ventana de gris de la medida es fija (1450–1850) [C:114]. Con los parámetros en el centro sale 62,6 (V6). **[el análisis parche a parche no es reproducible desde el repo]**.
15. **No se puede "esconder" la GPU con `CUDA_VISIBLE_DEVICES=-1`**: en esta máquina Python se cae al importar torch (código 0xC0000005) **[reejecutado hoy]**.
16. **Con 200 iteraciones los entrenamientos no sirven para comparar muestreos**: 5 de 10 terminaron con Dice 0 en todo (3.12). Con 1000 iteraciones (una vuelta, una semilla) sí despegan los dos y el Dice apical final coincide (0,58); para decidir hacen falta varias semillas y vueltas.
17. **Isa dijo (informalmente, relatado por Claudia) que el conducto se mide hasta el foramen apical y que la flecha del final de la máscara es "perfecta"** (3.4). Contradice mi lectura de que se pierde un tramo corto; solo vio el diente 52 (tres imágenes); el 35 y el 51 no los vio, así que para ellos sigue sin confirmar.

### 5.2 Correcciones de errores propios durante el trabajo

- Un primer "no se cierra ni con cierre de radio 11" era un error de recorte mío, no un hallazgo.
- Una primera lectura del ápice trataba como conductos distintos los trozos unidos solo por la holgura (el 51 salía con 10 conductos); se corrigió agrupando con la misma holgura y separando los fragmentos a media altura [S:108-113, S:128-131].
- La métrica "margen" de `split_canal.py` daba un falso aviso porque medía justo antes del orificio, donde la sección ya está subiendo; se mide a ≥ 0,5 mm por debajo [S:234-238].
- En la exportación a Slicer las máscaras de 32 µm de solo conducto y cámara se solapaban (~2000 vóxeles en el 51 y el 52); la cámara pasó a derivarse como "completa − solo conducto" [X:74-77].
- Hoy, un script mío imprimía "etiqueta idéntica: True" escrito a mano sin comprobar nada (la comparación CPU/GPU); **no se usa** y la equivalencia CPU/GPU está marcada como no verificada.

### 5.3 Contradicciones entre el código y CLAUDE.md / README (corregidas el 2026-10-28)

1. **Riesgo de sobrescribir las etiquetas finales — CORREGIDO.** `README.md` y `1_Data_Validation/README.md` mandaban ejecutar `preprocess_microct.py ... --downsample 4 --out 1_Data_Validation/results/microct`, que habría pisado `norm.npy`, `canal.npy` e `info.json` finales con una versión a 64 µm y dejado la carpeta incoherente. Ahora ambos README usan el comando seguro; el script escribe por defecto a una carpeta por resolución [P:199-201], aborta si va a pisar etiquetas separadas o una carpeta de otra resolución [P:204-217, P:225] (probado: aborta en 0,8 s y las etiquetas siguen intactas; la salida de prueba a 64 µm fue a `results/microct_64um`, que borré), con `--force` como salida; el `--downsample` por defecto es 1 [P:257]; y el `--out` por defecto ya no depende del directorio de trabajo [P:195].
2. **"No gaussiano" — CORREGIDO.** `CL:11` decía ruido "no gaussiano"; ahora dice "casi gaussiano" y remite a la medida (`CL:99`, `calibration.json`).
3. **Docstring del simulador desactualizado — CORREGIDO.** `SIM:1-25` describe ahora `canal_solo_conducto.npy`, el vóxel por diente, la reducción exacta y los ~0,5 s de `patch()`.
4. **Los centros de los rangos no eran lo calibrado — CORREGIDO (comentarios).** `Y:1-6` y los comentarios de la PSF y de `sd_hu` dicen ahora que los centros se retocaron para que `--check` coincidiera con Isa y cuáles no son el valor medido (PSF 0,1275 frente a ≈ 0,138 medido; `sd_hu` 57,5 frente a 61). Los rangos no se tocaron.
5. **"Parametrizado por carpeta" tenía un límite — CORREGIDO para el vóxel.** El simulador ya no fija 16 µm (3.6). Falta probarlo con un µCT real de otra resolución.
6. **CUDA de `requirements.txt` — CORREGIDO.** Ahora `cu130`, como `RD:101` y `CL:28`.
7. **Métricas pedidas y no implementadas — ACLARADO.** `CL:37` dice ahora que solo están implementados Dice y HD95 por tercios (`MET:8-9`); la distancia al ápice real y el error de cada medida siguen **pendientes**.
8. **Cifras de rendimiento — CORREGIDAS.** `CL:118` y `CL:158` dan `patch()` ≈ 0,46 s (87 s en CPU) y el bucle de la línea de comandos 1,1–1,8 s; las medidas de hoy son 0,41–0,45 s, 1,0–1,6 s y hasta 11,3 GB de GPU (ver 3.6, rendimiento).
9. **Código sin usar — CORREGIDO (`flip_z`).** Eliminado el parámetro `flip_z` de `third_map` [MET:31]. `results/microct_full/` (11 GB, del 20 de octubre, resultados del script antiguo; ningún script lo referenciaba) **borrado el 2026-10-28 por orden de Claudia**.
10. **Resuelto:** `1_Data_Validation/README.md` sí marca lo de Pulpy3D como legado (líneas 3-11); no está desactualizado.

---

## 6. Límites y riesgos honestos

**Datos.**
1. **Solo 3 µCT** (de ~40 en traspaso) y **no está comprobado que 51 y 52 sean molares**; CL dice que el 51 "sale con un solo conducto" [CL:17], pero el análisis del ápice encuentra 3 conductos en él (`info.json`, clave `apice`; 3.4). Toda la ingeniería (parámetros, criterios) se ajustó con esos 3.
2. **Ninguna máscara hecha por un experto** sobre un µCT. Todo lo llamado "etiqueta" es salida de un algoritmo sin validar; su fiabilidad frente a un experto es **desconocida** [CL:19].
3. **Un único CBCT real**, con una segmentación **en curso** (borrador) [E-a]. Solo sirve para calibrar y, con cautela, para una prueba cualitativa. Un solo escáner y un solo paciente: no se conoce cuánto varían los valores calibrados.
4. **Datos de paciente** (el CBCT): no compartir. Hoy el fichero está en esta máquina del laboratorio (lo leen V4 y E-a); el permiso de tenerlo aquí sigue abierto en CL:45.

**Algoritmo de etiquetas.**
5. **Parámetros ajustados con 3 dientes**: 0,13 mm (unión), 0,05 mm (grosor), 0,005 mm³ (motas), 1,5 mm² (orificio) [CL:86]. Con ~40 hay que revisar `avisos` e imágenes.
6. **El ápice podría quedarse ~0,1–0,2 mm corto (sin confirmar; Isa dice que el final de la máscara es correcto, ver 3.4(e))**: la máscara termina donde el conducto deja de estar "encerrado", unos 0,06 mm antes de abrirse al exterior, y hay dentina maciza hasta la punta (3.4). La longitud medida sobre esta máscara es subestimada. Comprobado a ojo en 6 de 7 conductos.
7. **El orificio es un plano horizontal** [S:226-227]; el suelo de la cámara real es irregular.
8. **Un conducto fino (< ~30 µm) no se ve a 16 µm**; no se puede saber si existe.
9. **Las etiquetas no incluyen la parte abierta al exterior** (ápice, caries, fractura): el método exige que el hueco esté encerrado en el corte [P:24-26].

**Simulador.**
10. **No se simulan**: endurecimiento del haz, rayas por restauraciones metálicas, movimiento del paciente, "cupping" (brillo que cambia del centro al borde), dientes vecinos, cortical definida. El fondo es textura aleatoria, no anatomía [SIM:295-306].
11. **El gris del conducto está adivinado** [−250, 500] [Y:24]. Es el parámetro más influyente en el contraste y el menos medido.
12. **La calibración es circular**: el simulador reproduce lo que se midió en Isa porque se ajustó a ello. No demuestra que una red entrenada con él funcione en un CBCT real; eso no se ha probado.
13. **Muestreo por tercios y vóxel por diente: cambiados** (3.6, límites 5 y 6). Falta probar un µCT real de otra resolución (el efecto del muestreo sobre el Dice apical se midió solo en una vuelta de 1000 iteraciones y no se vio diferencia; 3.12).
14. **Un parche cubre 9,6 mm; el diente, 15–22 mm.** Para medir longitud hará falta inferencia por ventanas sobre un volumen completo simulado, que **no existe**.

**Entrenamiento.**
15. **No se ha entrenado en serio**; los números de validación (Dice 0,54–0,78) no significan nada: 2 dientes de entrenamiento, 200 iteraciones, 6 parches.
16. No hay conjunto de test, ni selección del mejor modelo, ni planificador de la tasa de aprendizaje; solo CUDA (`TR:80`); determinismo de la GPU sin probar.
17. **El entorno no está congelado**: `2_UNet/requirements.txt` no fija versiones y no hay `environment.yml` (las versiones usadas hoy están en el Apéndice A).

**Mantenimiento.**
18. Muchos resultados intermedios salen de **scripts de diagnóstico temporales que no están en el repositorio** (marcados "no reproducible desde el repo"); si se quieren conservar hay que copiarlos y parametrizarlos.
19. No hay pruebas automáticas (`pytest`) del código nuevo; las comprobaciones de este documento son manuales (Apéndice A).

---

## 7. Estado frente a lo que pidió Luis

| # | Petición | Estado | Evidencia |
|---|---|---|---|
| a | Empezar por la normalización y la umbralización, con código aplicable a varios µCT | **Hecha** | `preprocess_microct.py` toma una carpeta o varias con `--all` [P:253-276]. Reproducido en 2 dientes hoy (V2, V3: salida idéntica byte a byte). Solo se ha aplicado a 3 µCT. |
| b | Generar un CBCT artificial a partir del µCT | **Hecha, con límites** | `simulate_cbct.py` [SIM:272-376]; con los parámetros en el centro reproduce las medidas de Isa (V6). Calibrado con un solo CBCT; que el falso sirva para entrenar una red que funcione en CBCT real **no está demostrado**. |
| c | Pasárselo a la red de Pulpy para ver qué funciona y que entrena | **Parcial** | Entrena de extremo a extremo (V8; `metrics.json` de 3 vueltas, la pérdida baja de 1,55 a ~1,0). **No con la red de Pulpy**: se escribió `CanalUNet3D` que reutiliza solo su bloque [NET:10, NET:22], porque las de Pulpy piden un vector de posición y devuelven la sigmoide [UN:18, UN:39, UN:63, MF:14]. "Ver qué funciona": sin conclusiones de calidad. **Hay que confirmar con Luis si esta desviación vale.** |
| d | Cuando funcione, usar el resto de dientes (~40, en traspaso) | **Pendiente** | Los µCT no están todavía [CL:17]. La infraestructura lo admite: dataset por carpetas [DS:37-40]; por diente, preprocesado + separación ≈ 129 + 82 s (V2, V5) y ~7,7 GB en `results/microct` (23 GB / 3 dientes, `du`). Extrapolado a 40 dientes: ~2,5 h y ~300 GB, **no medido**. |
| e | Probar CPU contra GPU y dejar el entorno en la máquina del laboratorio | **Parcial** | Entorno `tfm` presente en esta máquina (la del laboratorio, según CL:27 **[inferido: el propietario de los ficheros es `BO-MOBIOS05`]**), con RTX 5080 y torch 2.14.1+cu130 (V1, V8). CPU frente a GPU medido hoy: multiplicación de matrices 4000² 0,13 s frente a 0,063–0,070 s (V1; `GPU:4-14`, una sola llamada sin calentamiento, poco representativo) y un parche del simulador 87,4 s frente a 0,46 s (≈190×). El **entrenamiento en CPU no se probó** y el script lo impide [TR:80]. Falta congelar el entorno (`conda env export` o lista de versiones fijas). |
| f | Segunda red que dé el diámetro; proponemos sustituirla por un cálculo geométrico (línea central + distancia a la pared) | **Pendiente** | No hay código de línea central ni de medidas (búsqueda de `skeleton`, `centerline`, `medial_axis` en los `.py`: solo aparece en una nota de `MET:8-9`). Hay ingredientes: transformada de distancia [P:144, S:123]. El diámetro de referencia saldría del µCT por geometría [CL:14]. **Hay que confirmar con Luis que sustituir la red por geometría le parece bien.** |

---

## 8. Lo que falta por hacer, en orden

*(Los responsables son lo que se deduce de CLAUDE.md; **[no verificado]** quién hace qué en la práctica.)*

1. **Claudia**: revisar en 3D Slicer `results/para_slicer/` (instrucciones en `LEEME.txt`): ¿el plano del orificio cae donde un endodoncista pondría la entrada?, ¿queda cámara o caries dentro de "solo conducto"?, ¿acaba el conducto donde se ve? Decidir qué tipo de diente son 51 y 52. **Hasta entonces no se entrena en serio.**
2. **Claudia → Ana**: pedir una máscara del canal hecha por un experto sobre al menos algunos µCT, después de enseñarle lo que ya se puede hacer [CL:44]. Sin esto no hay forma de saber cuánto se parece la etiqueta a la "verdad".
3. **Luis (o quien haga el traspaso)**: los ~40 µCT y el volumen del 58 (copia mal hecha) [CL:42]. Comprobar su formato: el simulador ya lee el vóxel de cada diente (probado con una copia a 32 µm), pero la separación del conducto y el resto se han probado solo con 3 dientes de 16 µm.
4. **Luis**: confirmar (c) y (f) de la sección 7; autorizar tener el CBCT de Isa en la máquina del laboratorio [CL:45].
5. **Isa**: terminar su segmentación (hoy "en curso") o aportar más CBCT (idealmente de otro escáner) para calibrar.
6. **Código, con el material actual**: repetir `preprocess_microct.py` + `split_canal.py` + `export_slicer.py` por cada diente nuevo, revisar `avisos` e imágenes y volver a barrer los parámetros con los ~40.
7. **Código**: (el sesgo de muestreo hacia la corona ya está corregido; en una vuelta de 1000 iteraciones no mejoró el Dice apical final, falta repetirlo con más semillas); construir el **volumen completo simulado** (a 32 µm para que quepa) con inferencia por ventanas; medir la **distancia al ápice real** y el **error de cada medida** con línea central + distancia a la pared (petición f).
8. **Código**: entrenar en serio con validación por dientes (k pliegues y un test aparte): comparar `weight_mode` (`band`/`chamber`/`none`) y `chamber_fill`, con y sin simulador; añadir planificador y parada temprana; guardar el mejor modelo.
9. **Código y Claudia**: evaluar en el CBCT real de Isa (etapa 4); como su segmentación es un borrador, la comparación es solo cualitativa hasta que esté terminada.
10. **Limpieza**: arreglar las contradicciones de 5.3 (sobre todo el comando del README), congelar el entorno, copiar al repo los scripts de diagnóstico que se quieran conservar, escribir pruebas automáticas.

---

## 9. Dudas y cosas que no he podido comprobar

**No verificado en absoluto:**
- Que 3D Slicer abra los NRRD exportados (solo los releí con SimpleITK).
- Que los grises de Isa sean Hounsfield reales.
- Que los ~40 µCT tengan el mismo formato (el simulador ya admite otros tamaños de vóxel; probado solo con una copia a 32 µm).
- Que 51 y 52 sean molares.
- Que la salida de CPU y GPU del simulador sea equivalente (los generadores aleatorios difieren y no se comparó).
- Que "la GPU no se comparte bien con procesos hijos" (`DS:9-10`).
- Que el plano del orificio coincida con el criterio clínico de un endodoncista; y que la interpretación del final del conducto como "foramen" sea correcta (es una lectura mía de las imágenes, sin experto).
- Memoria de pico (RAM) del preprocesado a 16 µm.
- Quién hace qué entre Claudia, Luis, Ana e Isa; y que esta máquina sea la del laboratorio (inferido).

- Resuelto el 2026-10-28 (Claudia): Isa solo vio el diente 52 (tres imágenes de ese diente), no el 35 ni el 51. Su comentario sobre el final del conducto es informal, relatado y vale solo para el 52.

**Visto a ojo en pocos casos:** el patrón del ápice en 6 de 7 conductos (no se miró el conducto 2 del 51).

**Resultados que siguen sin poder reproducirse desde el repositorio** (los generó un script temporal no copiado): las cifras exactas de la máscara intermedia anterior a los filtros (60.837 motas en el 35, rayas de caries de 3,9 mm³, 5083 vóxeles de la prolongación apical, 100 % de conservación de los conductos finos), las comparaciones "antes/después" de los filtros, y las imágenes ampliadas del ápice del 2026-10-21 (se regeneran con `diagnostico/ver_apice.py`). **Sí se reproducen hoy con los scripts copiados** a `1_Data_Validation/analysis/microct/diagnostico/` (ver su `README.md` y el apéndice A.2): el barrido de la holgura, el Dice entre resoluciones, la sensibilidad del orificio, el relleno de la cámara y la franja de peso cero, la alineación imagen–etiqueta, el sesgo de la medida de ruido, los tiempos, el muestreo por tercios y el efecto de la reducción.

**Dudas de diseño que me gustaría que alguien decidiera:**
1. ¿La desviación de (c) (red propia en lugar de la de Pulpy) vale para Luis?
2. ¿Se prefiere ignorar la franja del orificio (`band`), toda la cámara (`chamber`) o nada (`none`)? Está sin comparar con entrenamiento real.
3. ¿La etiqueta debe llegar hasta el foramen? Isa dice (informal) que el final actual de la máscara ya es el foramen; solo vio el diente 52; falta confirmarlo con el 35 (punta partida) y el 51.
4. (Resuelto en el código) El muestreo ya reparte a partes iguales entre tercios; en una vuelta de 1000 iteraciones no se vio mejora del Dice apical final: ¿vale la pena repetirlo con varias semillas antes de decidir el muestreo definitivo?

---

## Apéndice A. Cómo reproducir cada número

**Entorno** (todo **[reejecutado hoy]**): Windows 11, 127,7 GB de RAM, 20 hilos de CPU de PyTorch, GPU NVIDIA GeForce RTX 5080 (16 GB). Python 3.12; numpy 2.5.4; scipy 1.18.1; scikit-image 0.26.0; torch 2.14.1+cu130; pynrrd 1.1.3; PyYAML 6.0.3; medpy 0.5.2; SimpleITK 2.5.6; matplotlib 3.11.2; torchio 1.2.1; nibabel 5.4.2. `python` = `C:\Users\tfg\.conda\envs\tfm\python.exe`.

**Pasos de verificación** (V = comando del repo; el resultado se comparó con lo guardado):

| Paso | Comando (desde `D:\tfm_claudia`) | Resultado de hoy |
|---|---|---|
| V1 | `python test_gpu.py` (dos veces) | CPU 0,1326 / 0,1297 s; GPU 0,0626 / 0,0700 s |
| V2 | `python 1_Data_Validation/analysis/microct/preprocess_microct.py 1_Data_Validation/datasets/MicroCT/Diente_52 --downsample 1 --out <tmp>` y comparar con `results/microct/Diente_52` | canal.npy y norm.npy idénticos byte a byte; volumen 36,884502 mm³; 129,3 s (incluye la comparación) |
| V3 | igual con `Diente_51 --downsample 2` frente a `results/microct_32um/Diente_51` | idéntico; 23,037843 mm³; 24,5 s (incluye la comparación) |
| V4 | `python 1_Data_Validation/analysis/microct/calibrate_cbct.py "1_Data_Validation/datasets/CBCT_75um/Isa_caso01/1 3D CBCT Image Set.nrrd" --seg "1_Data_Validation/datasets/CBCT_75um/Isa_caso01/Segmentation.seg.nrrd" --out <tmp>` | `calibration.json` idéntico al guardado; 9,9 s |
| V5 | `python 1_Data_Validation/analysis/microct/split_canal.py 1_Data_Validation/results/microct/Diente_52` (copia previa de `info.json` y hash SHA-256 de los `.npy`) | hashes y `info.json` idénticos; 81,7 s (incluye los hashes) |
| V6 | `python 1_Data_Validation/analysis/microct/simulate_cbct.py <las 3 carpetas results/microct/Diente_XX> --check --center --n 36 --seed 1 --calibration 1_Data_Validation/results/cbct_calibration/Isa_caso01/calibration.json` | tabla de 3.6(d) (versión del 2026-10-28, con reducción exacta y muestreo proporcional en `--check`); 59 s |
| V7 | `python .../simulate_cbct.py <3 carpetas> --n 6 --seed 0 --out <tmp>` | 1,0–1,6 s por parche (con guardado), pico 11,3 GB (2026-10-28) |
| V8 | `cd 2_UNet` y `python train_sim.py -c configs/sim_canal.yml --holdout Diente_52 --iterations 30 --tag verif` | 49,4 s, 1,65 s por it., 10,74 GB, 5,65 M parámetros (carpeta borrada después) |

**Comprobaciones auxiliares** (E-x; scripts temporales de la sesión, con el código esencial aquí):

- **E-a (Isa):** `seg, h = nrrd.read(<seg>)`; `m = seg > 0`; `lab, n = scipy.ndimage.label(m)` → etiquetas {0, 1}; 109.890 vóxeles = 46,36 mm³; `n = 15`; tamaños 57402, 27138, 10471, 9560, 2376…; `Segment0_Tags` contiene `Segmentation.Status:inprogress`.
- **E-b (sesgo por tercios):** cargar `results/microct/Diente_XX/coords_canal_solo_conducto_s6.npy` (columna 0 = z), con `z0, z1 = info["separacion"]["canal_solo_z_vox"]`; `f = (z1 - z)/(z1 - z0)` si la corona está arriba; proporción de `f ≥ 2/3`, `1/3 ≤ f < 2/3` y `f < 1/3` → apical 16 / 12 / 14 %, medio 39 / 22 / 27 %, coronal 45 / 65 / 58 % (35 / 51 / 52).
- **E-c (ventanas de reducción):** `w = [ceil((i+1)·300/128) − floor(i·300/128) for i in range(128)]` → {3 vóxeles: 88 ventanas, 4 vóxeles: 40}; media 3,3125 vóxeles de 32 µm = 106 µm; `torch.nn.functional.adaptive_avg_pool3d` coincide con esa fórmula.
- **E-d (ganancia del ruido):** `z = torch.randn((96,96,96))`; `zf = simulate_cbct.blur(z, (s,s,s))`; `zf[20:-20,20:-20,20:-20].std()` frente a `simulate_cbct.noise_gain((s,s,s))`: 0,1488/0,1499 (σ = 1,0), 0,0998/0,1011 (1,3), 0,0664/0,0676 (1,7).
- **E-e (conservación de la etiqueta):** para un parche, `tooth.read(tooth.canal, start, n).sum()·0,015999³` frente a `out["label_soft"].sum()·0,075³`: diferencias de −0,46 % a +0,36 % en 6 parches.
- **E-f (dilatación):** `simulate_cbct.dilate_box(x, r) == (F.max_pool3d(x[None,None], 2r+1, 1, r)[0,0] > 0)` para r = 3 y r = 9 sobre una máscara aleatoria de 60×70×50: iguales.
- **E-g (semilla):** dos llamadas a `Simulator.patch(t, np.random.default_rng(7))` → imagen idéntica (diferencia máxima 0,0), etiqueta y parámetros idénticos; con semilla 8 cambia.
- **E-h (tercios y métricas):** `third_map` y `metrics_by_thirds` con el caso conocido de 3.11(c).
- **CPU frente a GPU del simulador:** cambiar `simulate_cbct.DEV = torch.device("cpu")` (y `torch.set_default_device("cpu")`) antes de crear el `Simulator`; mismo recorte (`start`) y semilla; `patch()` tarda 87,37 s en CPU frente a 0,46 s en GPU. (`CUDA_VISIBLE_DEVICES=-1` no sirve: Python se cae al importar torch.)

**Resultados que se leyeron de fichero (no se volvieron a generar hoy):** `results/microct/Diente_{35,51,52}/info.json` (todas las claves salvo lo que reproducen V2 y V5 para el 52); `results/microct_32um/Diente_{35,52}/info.json`; `results/microct/resumen_etapa1.json`; `2_UNet/results/entrenamientos_sim/sim_canal_montaje_fuera_Diente_{35,51,52}/metrics.json`; `results/para_slicer/*.nrrd` y `resumen_exportacion.json`.


### A.2 Scripts de diagnóstico que reproducen los números (2026-10-28)

Todos en `1_Data_Validation/analysis/microct/diagnostico/` (índice y tiempos en su `README.md`), con el entorno `tfm` y desde cualquier directorio:

| Número del documento | Script | Resultado de hoy |
|---|---|---|
| Dice entre 16 y 32 µm (3.3) | `dice_32_vs_16.py` | 0,9739 / 0,9773 / 0,9888 (rejilla 32 µm), idéntico a `resumen_etapa1.json` |
| Barrido de la holgura (3.3) | `barrido_gap.py --dientes 51 --downsample 1 --gaps 0 0.13 0.3 --min-piece-mm3 0` | 16,85 mm³ en 7 / 23,18 en 1 / 23,24 en 2 |
| Sensibilidad del orificio (3.5) | `secciones_y_orificio.py` | 35: −0,11 … +0,82 mm; 51: −0,16 … +1,25; 52: −0,21 … +0,98 (1,25 … 3,0 mm²) |
| Imágenes del ápice (3.4) | `ver_apice.py --diente Diente_52 --raiz 1 --conducto 1`; `imagen_para_isa.py` | imágenes regeneradas |
| Relleno de la cámara, peso cero, alineación (3.6) | `prueba_alineacion_y_pesos.py` | conducto 299–377 / cámara 258–317 / fondo 856–882; peso 0: 0,23–0,25 % (band), 1,33–2,74 % (chamber); mínimo de correlación en 0 |
| Sesgo de la medida de ruido (5.1) | `ruido_medida_sesgo.py` | suelo de estructura 33,9; centro 62,8; con rangos aleatorios 81,5 (correlación 0,65 con el gris de dentina sorteado) |
| Tiempos y CPU frente a GPU (3.6) | `tiempos_simulador.py --que etapas` / `--que cpu_gpu` | `patch()` 0,43–0,71 s en GPU; 87,4 s en CPU |
| E-a, E-c, E-d, E-e, E-f, E-g, E-h (Isa, ventanas, ganancia del ruido, etiqueta, dilatación, semilla, tercios) | `comprobaciones.py` | los mismos valores que en el apéndice A (comprobaciones E-x); la conservación de la etiqueta con reducción exacta da +0,019 % |
| E-b (reparto por tercios) | `muestreo_por_tercios.py --n-centros 600 --n-bloques 100` y `comprobaciones.py --que muestreo` | apical 13,9 % → 31,9 %; conducto apical por bloque 0,38 → 0,61 mm³ |
| Vóxel por diente (3.6) | `prueba_vox_distinto.py --diente Diente_51` | 9,599 mm en ambos; volumen ±0,7 %; RMS de imagen 6–8 HU |
| Regresión del simulador | `regresion_simulador.py --legacy --compare <línea base>` | diferencia 0 en imagen, etiqueta y pesos |
| Efecto de la reducción exacta sobre la calibración (3.6) | `calibracion_reduccion.py --n 36 --seeds 1 2` | σ del borde dentina–exterior 0,142 → 0,141 mm; el resto, ±3 HU |
| Comparar regenerados con guardados (V2–V5) | `comparar_resultados.py pre|hash|json|calib` | idéntico |