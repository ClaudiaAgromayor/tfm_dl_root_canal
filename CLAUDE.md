# Contexto del proyecto (TFM de Claudia Agromayor, ICAI-Comillas)

Responde siempre en **español**, en un lenguaje sencillo: la autora viene de ingeniería y está empezando en endodoncia. Explica los términos médicos y de imagen la primera vez que aparezcan. No des nada por hecho: si algo no está verificado, dilo.

## Documentación del repositorio (antes de escribir o cambiar un documento)
| Documento | Para qué sirve |
|---|---|
| `README.md` (raíz) | Portada del repositorio (en inglés): qué es, estructura, cómo empezar y un mapa de los documentos. |
| `docs/GUIA.md` | **Manual de uso**: el problema en 5 líneas, camino de los datos, mapa de carpetas, scripts uno por uno, ficheros de salida, cómo repetir con un diente nuevo, preguntas frecuentes y glosario. |
| `docs/PROYECTO.md` | **Cuaderno técnico**: cada paso hecho, por qué, qué números salieron, hallazgos, límites, uso final, qué falta, preguntas abiertas e historial de cambios. Citas fichero:línea y cómo reproducir cada número. |
| `docs/MEDICION.md` | **Cómo se miden** línea central, longitud, curvatura, ángulos y diámetros (explicado desde cero, con figuras y código), su error con conductos artificiales, resultados en los 3 µCT y plan de lo que falta medir. |
| `docs/LITERATURA.md` | Resumen de los 10 papers de `00_Papers` y qué se puede reutilizar. |
| `2_UNet/README.md`, `1_Data_Validation/README.md`, `1_Data_Validation/analysis/microct/diagnostico/README.md` | Notas cortas junto al código (entrenar; legado de Pulpy3D; scripts de diagnóstico). |

**Regla: cada dato vive en UN solo documento.** Antes de añadir información, busca si ya existe y actualízala allí; no copies tablas ni cifras entre documentos, enlaza. Los cambios de estado se anotan en `docs/PROYECTO.md` («Historial de cambios») y, en una línea, en el «Estado actual» de este fichero. Este fichero se lee entero en cada sesión: mantenlo corto (reglas, datos, estado en una página).

## Objetivo
A partir de un CBCT clínico, **segmentar los conductos radiculares del primer molar inferior (36/46)** y medir longitud, curvatura, volumen y, con cautela, diámetro (a 75 µm el diámetro apical es impreciso).

**Uso final (aclarado por Claudia el 2026-11-18 y 19):** el dentista introduce el **CBCT entero** (todos serán de **75 µm**, campo pequeño, como los de Isa) y pide el 36 o el 46. El sistema debe acotar (o prestar atención a) ese diente y devolver la máscara de **solo los conductos de ese diente**, sin los de los vecinos; después se miden. Los CBCT de Isa son volúmenes de ~42×50×52 mm con 4–6 dientes en fila, así que el modelo tiene que distinguir el diente; los modelos de `2_UNet/results/entrenamientos_sim/` son **pruebas de montaje** (un diente aislado y fondo aleatorio), no modelos válidos para esto. Diseño propuesto (pendiente de aprobar): localizar el diente (clic del dentista o modelo de numeración) → recortar → segmentar con el diente marcado, con un simulador con vecinos. Detalle y fuentes públicas: `docs/PROYECTO.md`, apartado 8.

## Enfoque decidido (caso B)
No se usan Pulpy3D ni los datos de Fer: **descartados** (usar sus datos solo para localizar el diente está por decidir; ver PROYECTO.md, apartado 8). Se parte de µCT de dientes extraídos:
1. Normalizar cada µCT y obtener el canal por umbral (`analysis/microct/preprocess_microct.py`).
2. **Fabricar CBCT falsos a 75 µm** a partir del µCT: bajar resolución, desenfoque (PSF), ruido (Poisson filtrado: muy correlacionado entre vecinos y **casi gaussiano**), contraste aleatorio. Como el falso sale del µCT, **la etiqueta ya está alineada y no hay registro**.
3. Entrenar un segmentador 3D con los falsos y la etiqueta del µCT.
4. Aplicarlo a los CBCT reales de 75 µm de Isa. Los reales sirven sobre todo para **calibrar el simulador y como prueba**, no para entrenar con pseudoetiquetas sin revisar.
5. Medir sobre la segmentación: línea central + transformada de distancia (el diámetro de referencia sale del µCT por geometría, no hace falta una segunda red). Hecho sobre los µCT: `docs/MEDICION.md`.

## Datos (nunca se suben a GitHub; carpetas en `.gitignore`)
- `1_Data_Validation/datasets/MicroCT/Diente_35`, `Diente_51`, `Diente_52`: Nikon XT, **vóxel 16 µm isotrópico**, `.raw` uint16 con orden (z, y, x); dimensiones en el `.xtekVolume`. Pesan ~2,4–2,9 GB. Los números son la **numeración de la colección** (confirmado por Claudia), no códigos FDI. El tipo de diente **no está comprobado** (el 35 es un molar de dos raíces; 51 y 52 sin identificar): hay que mirarlos en 3D Slicer. Del diente 58 solo hay proyecciones (copia mal hecha; el volumen existe y se volverá a copiar). Habrá ~40 µCT en total, en traspaso.
- `1_Data_Validation/datasets/CBCT_75um/`: `Isa_caso00` (555×667×688, vóxel 0,075 mm; el que se usó para calibrar; su origen no coincide con ningún zip DICOM: sin confirmar con Isa) e `Isa_caso01 … Isa_caso10` (entrega de WeTransfer del 9–11 nov 2026, ordenada el 2026-11-17 con `analysis/cbct_real/ordenar_cbct_75um.py`; mapeo y hashes en `mapeo_renombrado.csv`, se deshace con `--deshacer`). Cada carpeta: `CBCT.nrrd`, `Segmentation.seg.nrrd` (segmento `PULP`; **todas `inprogress` = borradores**, no referencia fiable; probablemente incluyen la cámara pulpar: 18–59 mm³ frente a 8–12 mm³ de conducto en los µCT), `DICOM_original.zip` y `LEEME.txt`. Los casos **5, 9 y 10 solo tienen DICOM**. El caso 4 trae además `Segmentation_otra_encontrada_en_carpeta_caso3.seg.nrrd` (sin confirmar qué es). **Datos de paciente: no compartir; los zips llevan nombres de pacientes (no abrirlos fuera de la máquina ni subirlos).** No se sabe qué diente es cada caso: Claudia confirmó que el de interés es el primer molar.
- **No existe todavía una máscara del canal hecha por un experto sobre el µCT.** La que se usa es la del script, sin validar.

## Entorno
- Máquina del laboratorio (Mobios): RTX 5080 de 16 GB, driver 581.57, CUDA 13.0. El portátil de Claudia **no tiene GPU NVIDIA**: el entrenamiento va en la máquina del laboratorio.
- Repositorio en `D:\tfm_claudia`. Entorno conda `tfm` (Python 3.12, PyTorch 2.14.1+cu130). Anaconda en `C:\ProgramData\anaconda3`. Ejecutar con `(tfm)` activo o con `C:\Users\tfg\.conda\envs\tfm\python.exe`.
- No tocar `D:\DATA\lfsanchez` (originales de Luis).
- Para usar la GPU por defecto: `torch.set_default_device('cuda')` al inicio.

## Código existente
`2_UNet/` (UNet3D, PosPadUNet3D, Attention UNet, UNETR, VNet3D; pérdidas Dice+CE) viene del proyecto Pulpy3D y de ToothFairy: sus configs y dataloader son de Pulpy3D y no se usan. El código nuevo está en `1_Data_Validation/analysis/microct/` (preprocesado, separación conducto/cámara, simulador, medidas, diagnóstico) y en `2_UNet/` (`train_sim.py`, `models/CanalUNet3D.py`, `losses/SoftDiceBCE.py`, `dataloader/SimCBCT.py`, `metrics_canal.py`). `1_Data_Validation/analysis/` (salvo `microct/` y `cbct_real/`) es análisis antiguo de Pulpy3D, solo de referencia.

## Reglas de trabajo
- Repartir entrenamiento/validación/test **por diente**, nunca por recortes (evita fugas de datos).
- Métricas: Dice y HD95 **por tercios** (coronal, medio, apical), distancia del final del conducto al ápice real, y sobre todo el **error de cada medida** frente al µCT. (Hoy solo están implementados Dice y HD95 por tercios sobre parches, `2_UNet/metrics_canal.py`; la distancia al ápice real y el error de cada medida sobre una predicción están **pendientes**.)
- Todo script nuevo debe poder aplicarse a varios µCT (parametrizado por carpeta).
- No hacer `git commit` ni `git push` sin que Claudia lo pida. No subir datos.
- No dar por buena una cifra que no se ha comprobado; cuando una lectura propia resulte equivocada, decirlo y corregir todos los documentos donde aparezca.

## Estado actual (2026-11-19; detalle y cifras en `docs/PROYECTO.md`)
- **Etiquetas a 16 µm** de los 3 dientes: hechas, en `1_Data_Validation/results/microct/Diente_XX/`. Parámetros por defecto de `preprocess_microct.py`: umbral 0,5; `--gap-mm 0.13`; `--min-piece-mm3 0.005`; `--min-mm3 0.05`; `--min-len-mm 1.5`; `--min-thick-mm 0.05` (ajustados con solo 3 dientes; no usar `--downsample 4`: el filtro de grosor está pensado para vóxel ≤ 32 µm). El script no puede pisar las etiquetas finales sin `--force`. **Falta revisarlas en 3D Slicer** (`results/para_slicer/`, no abiertas aún en Slicer).
- **La etiqueta es SOLO conducto** (del orificio al ápice), sin cámara pulpar ni cavidades: decisión de Claudia. Fichero `canal_solo_conducto.npy`; el orificio es el primer corte con sección mayor ≥ 1,5 mm² durante ≥ 0,1 mm (`split_canal.py`). Isa (endodoncista) comentó, informalmente y **solo para el diente 52**, que el final de la máscara coincide con el foramen apical que ella marcaría; para 35 y 51 no hay opinión.
- **Simulador de CBCT a 75 µm** (`simulate_cbct.py` + `simulate_cbct_config.yaml`): calibrado con un solo CBCT real (`Isa_caso00`); reproduce lo medido (grises, ruido, borde). **No simula** dientes vecinos, cortical, artefactos metálicos ni orientación variable (límite principal para el uso final).
- **Entrenamiento:** solo infraestructura y pruebas de montaje (`python train_sim.py -c configs/sim_canal.yml --holdout Diente_52`, desde `2_UNet/`); **no se ha entrenado en serio y no se saca ninguna conclusión de calidad**. En la prueba con el CBCT de Isa: Dice 0,33 en la zona de conducto y 16,6 mm³ de falsos positivos.
- **Medición** de longitud, curvatura, ángulos y diámetros: hecha y validada con conductos artificiales sobre los 3 µCT (`analysis/microct/medidas/`, `docs/MEDICION.md`). **Los diámetros publicados antes del 2026-11-18 no valen** (regla corregida). κ máximo probablemente inflado: usar mediana o percentil 95. El número de conductos de cada diente (10 líneas: 3, 4, 3) **no está comprobado**.
- **Referencia de la literatura** para el nº de conductos de un primer molar inferior: 3 conductos en el 61 %, 4 en el 36 %, 5 en ≈ 1 % (de Pablo et al., J Endod 2010).

## Pendiente (resumen; la lista ordenada, con responsables, en `docs/PROYECTO.md`, apartados 9 y 10)
1. Revisar las etiquetas en 3D Slicer y confirmar el orificio, el ápice, el tipo de diente de 51 y 52 y el nº de conductos. Hasta entonces no se entrena en serio.
2. Máscara del canal hecha por expertos sobre el µCT: Claudia se la pedirá a Ana **después de enseñarle lo que ya se puede hacer**; mientras tanto se sigue con la umbralización propia.
3. Decidir el diseño de segmentación del uso final (localizar el 36/46 + simulador con vecinos) y si se pueden usar Pulpy3D/ToothFairy3 para localizar el diente.
4. ¿Se puede tener el CBCT de Isa en la máquina del laboratorio (datos de paciente)?
5. Elegir las definiciones de medida (eje de referencia de la inclinación, tipo de longitud): `docs/MEDICION.md`, 12.2.

## Próximos pasos
1. Revisión en Slicer (arriba).
2. Medición, etapa 3: medir los 3 dientes a 16, 32, 64 y 75 µm con etiqueta blanda (no necesita red).
3. Segmentación del 36/46: localizador por clic + recorte, simulador con dientes vecinos y entrada extra de «diente objetivo».
4. Entrenar en serio (validación por dientes) y evaluar en los CBCT de Isa; medir sobre la predicción.
