# Contexto del proyecto (TFM de Claudia Agromayor, ICAI-Comillas)

Responde siempre en **español**, en un lenguaje sencillo: la autora viene de ingeniería y está empezando en endodoncia. Explica los términos médicos y de imagen la primera vez que aparezcan. No des nada por hecho: si algo no está verificado, dilo.

## Objetivo
A partir de un CBCT clínico, **segmentar los conductos radiculares del primer molar inferior (36/46)** y medir longitud, curvatura, volumen y, con cautela, diámetro (a 75 µm el diámetro apical es impreciso).

## Enfoque decidido (caso B)
No se usan Pulpy3D ni los datos de Fer: **descartados**. Se parte de µCT de dientes extraídos:
1. Normalizar cada µCT y obtener el canal por umbral (`analysis/microct/preprocess_microct.py`).
2. **Fabricar CBCT falsos a 75 µm** a partir del µCT: bajar resolución, desenfoque (PSF), ruido (Poisson, correlacionado, no gaussiano), contraste aleatorio. Como el falso sale del µCT, **la etiqueta ya está alineada y no hay registro**.
3. Entrenar un segmentador 3D con los falsos y la etiqueta del µCT.
4. Aplicarlo a los CBCT reales de 75 µm de Isa. Los reales sirven sobre todo para **calibrar el simulador y como prueba**, no para entrenar con pseudoetiquetas sin revisar (heredarían los fallos del primer modelo).
5. Medir sobre la segmentación: línea central + transformada de distancia (el diámetro de referencia sale del µCT por geometría, no hace falta una segunda red).

## Datos (nunca se suben a GitHub; carpetas en `.gitignore`)
- `1_Data_Validation/datasets/MicroCT/Diente_35`, `Diente_51`, `Diente_52`: Nikon XT, **vóxel 16 µm isotrópico**, `.raw` uint16 con orden (z, y, x); dimensiones en el `.xtekVolume`. Pesan ~2,4–2,9 GB. Los números son la **numeración de la colección** (confirmado por Claudia), no códigos FDI. El tipo de diente **no está comprobado**: 35 y 52 parecen molares de dos raíces con varios conductos, pero el 51 sale con un solo conducto; hay que mirarlos en 3D Slicer antes de entrenar. Del diente 58 solo hay proyecciones porque su copia está mal hecha; el volumen existe y se volverá a copiar. Habrá ~40 µCT en total, en proceso de traspaso.
- `1_Data_Validation/datasets/CBCT_75um/Isa_caso01`: CBCT real, 555×667×688, vóxel 0,075 mm, NRRD, más `Segmentation.seg.nrrd` hecha a mano por Isa (46 mm³, 15 componentes) que está marcada **`Segmentation.Status: inprogress`: es un borrador**, no una referencia fiable. Datos de paciente: no compartir.
- **No existe todavía una máscara del canal hecha por un experto sobre el µCT.** La que se usa es la del script, sin validar.

## Lo ya hecho y resultados de referencia
`preprocess_microct.py` normaliza con dos anclas (aire y mediana del tejido), umbral 0,5 (punto medio, correcto con efecto de volumen parcial) y define el canal como hueco encerrado por el diente en los cortes axiales (el eje z es el eje largo). Después quita motas de ruido, une trozos del mismo conducto y descarta grupos cortos o delgados (detalle y parámetros en "Estado al cerrar el día").
Cifras antiguas con `--downsample 4` (script **anterior** a los filtros nuevos, no se han repetido con el script actual): 35 → 20,53 mm³ (2 componentes); 51 → 23,38 mm³ (1); 52 → 35,66 mm³ (2). Las de referencia actuales están en "Estado al cerrar el día".
Hallazgos: a 64 µm se pierden conductos finos (visto en el 52); las etiquetas definitivas deben hacerse a 16 µm y bajarse a 75 µm después, mejor como **etiquetas blandas** (fracción de conducto por vóxel) en el ápice. En la corona aparecen falsos positivos por caries/poros.

## Entorno
- Máquina del laboratorio (Mobios): RTX 5080 de 16 GB, driver 581.57, CUDA 13.0. El portátil de Claudia **no tiene GPU NVIDIA**: el entrenamiento va en la máquina del laboratorio.
- Repositorio en `D:\tfm_claudia`. Entorno conda `tfm` (Python 3.12, PyTorch 2.14.1+cu130). Anaconda en `C:\ProgramData\anaconda3`. Ejecutar con `(tfm)` activo o con `C:\Users\tfg\.conda\envs\tfm\python.exe`.
- No tocar `D:\DATA\lfsanchez` (originales de Luis).
- Para usar la GPU por defecto: `torch.set_default_device('cuda')` al inicio.

## Código existente
`2_UNet/` (UNet3D, PosPadUNet3D, Attention UNet, UNETR, VNet3D; pérdidas Dice+CE) viene del proyecto Pulpy3D y de ToothFairy: sus configs y dataloader son de Pulpy3D y hay que adaptarlos. `1_Data_Validation/analysis/` (salvo `microct/`) es análisis antiguo de Pulpy3D, solo de referencia.

## Reglas de trabajo
- Repartir entrenamiento/validación/test **por diente**, nunca por recortes (evita fugas de datos).
- Métricas: Dice y HD95 **por tercios** (coronal, medio, apical), distancia del final del conducto al ápice real, y sobre todo el **error de cada medida** frente al µCT.
- Todo script nuevo debe poder aplicarse a varios µCT (parametrizado por carpeta).
- No hacer `git commit` ni `git push` sin que Claudia lo pida. No subir datos.

## Pendiente / preguntas abiertas
Respondido por Claudia: habrá ~40 µCT (en traspaso); el volumen del 58 existe pero está mal copiado (se recopiará); los números son numeración de la colección.
Sigue abierto:
1. Máscara del canal del µCT hecha por expertos: Claudia se la pedirá a Ana **después de enseñarle lo que ya se puede hacer**; mientras tanto se sigue con la umbralización propia.
2. ¿Se puede tener el CBCT de Isa en la máquina del laboratorio (datos de paciente)?
3. Confirmar en 3D Slicer qué tipo de diente es cada µCT (¿son primeros molares?).

## Próximos pasos
1. ~~Generar las máscaras a 16 µm de los tres dientes~~ (hecho el 2026-10-21, ver abajo). **Falta revisarlas en 3D Slicer.**
2. **Simulador de CBCT a 75 µm**, parametrizable y aleatorizado, calibrado con el CBCT real de Isa (perfil del borde dentina–canal y desviación típica del ruido en dentina).
3. Adaptar el entrenamiento de `2_UNet` y entrenar en la GPU.
4. Evaluar en el CBCT real de Isa y calcular las medidas.

## Estado al cerrar el día (2026-10-21)

### Hecho: etapa 1, etiquetas fiables a 16 µm
Las máscaras finales de los tres dientes están en `1_Data_Validation/results/microct/Diente_XX/` (16 µm: `canal.npy`, `norm.npy`, `info.json`, imágenes de control) y las de 32 µm, para comparar, en `results/microct_32um/`. El resumen numérico está en `results/microct/resumen_etapa1.json`. Es solo código y resultados locales; nada se ha subido ni commiteado.

**Qué estaba mal y qué se cambió en `preprocess_microct.py`** (verificado mirando los cortes, no supuesto):
1. *Conducto partido en trozos (diente 51).* La hipótesis de "el ruido abre huecos en la pared" **no era la causa**. En unos pocos cortes sueltos (z = 361 y 440 del 51) el conducto deja de estar encerrado; el 3D lo parte en trozos de ~1,25 mm y el filtro de largo mínimo (1,5 mm) los tiraba: el 51 pasaba de 23 mm³ a 16,8 mm³ en 7 trozos. Arreglo: el largo y el volumen se miden **uniendo trozos separados por menos de 0,13 mm en z** (`--gap-mm`). La máscara no se modifica, solo se usa para decidir qué se conserva.
2. *Motas de ruido.* La unión en z arrastraba motas de 1–10 vóxeles (60.837 piezas, 1,4 mm³ en el 35). Se quitan **antes** de unir las piezas de menos de 0,005 mm³ (`--min-piece-mm3`). Si se quitaran después, las motas ya habrían hecho de puente entre el conducto y la caries.
3. *Rayas de caries (diente 35).* Alrededor del agujero de la caries de la corona salían rayas de 1–2 vóxeles de grosor, 3,9 mm³, que pasaban los filtros. Se descartan los grupos unidos cuyo radio inscrito (distancia a la pared) en el percentil 90 es menor que 0,05 mm (`--min-thick-mm`). El grosor se mide **sobre el grupo unido, no sobre trozos sueltos**, para no perder los trozos finos que prolongan el conducto hacia el ápice.

**Parámetros por defecto:** umbral 0,5; `--gap-mm 0.13`; `--min-piece-mm3 0.005`; `--min-mm3 0.05`; `--min-len-mm 1.5`; `--min-thick-mm 0.05`. Se guardan en `info.json` junto con los grupos descartados. El filtro de grosor está pensado para vóxel ≤ 32 µm (a 64 µm una raya de 1 vóxel ya mide 0,064 mm): no usar `--downsample 4`.

**Resultados (volumen del canal y nº de componentes; Dice entre 32 y 16 µm):**

| Diente | 32 µm | 16 µm | Dif. volumen | Dice (rejilla 32 µm) | Dice por tercios z |
|---|---|---|---|---|---|
| 35 | 21,07 mm³ (2) | 21,20 mm³ (2) | 0,6 % | 0,974 | 0,986 / 0,988 / 0,962 |
| 51 | 23,04 mm³ (1) | 23,14 mm³ (1) | 0,4 % | 0,977 | 0,975 / 0,978 / 0,977 |
| 52 | 37,25 mm³ (1) | 36,88 mm³ (1) | 1,0 % | 0,989 | 0,983 / 0,989 / 0,989 |

El Dice compara el 16 µm reducido a la rejilla de 32 µm por mayoría (≥ 4 de 8 vóxeles) con la máscara de 32 µm; con la inversa (32 µm repetido a 16 µm) sale 0,957 / 0,959 / 0,970. Los tercios van en z de menor a mayor. Es una medida de **estabilidad entre resoluciones del mismo método**, no de exactitud frente a un experto (esa referencia no existe).
Mejora respecto a antes de los filtros (16 µm): 35 pasaba de 24,1 mm³ en 3 trozos a 21,20; 51 de 16,8 mm³ en 7 trozos a 23,14 (1); 52 de 36,7 a 36,88 y pasó de 2 a 1 componente (antes eran dos piezas de 28,9 y 7,8 mm³ separadas por solo 3 cortes sin encerrar; la unión en z las junta. No se ha comprobado a ojo que sea un único conducto, aunque el hueco es de 0,05 mm).

**Comprobaciones pedidas (16 µm, antes → después):**
- Prolongación apical del 35 (z 119–134, pieza de 0,021 mm³): 5083 vóxeles → 5083 (100 %); el canal sigue empezando en z = 119.
- Conductos finos del 51 (11 piezas, de 0,012 a 8,9 mm³): 100 % conservados en todas.
- Rayas de caries del 35: la pieza de 3,19 mm³ se elimina por completo (0 % conservada), junto con otra de 0,17 mm³; en total salen 3,906 mm³ (25,11 → 21,20).

### Decisiones abiertas
1. **Cámara pulpar** (la sala central del diente de la que salen los conductos). Las etiquetas la incluyen solo en parte: el método marca "hueco encerrado", y si la cámara se abre a una caries o al exterior deja de estar encerrada y queda sin marcar, cortada por un plano plano (visto en el 35). ¿Se quiere el conducto solo (raíz) o conducto + cámara? Hay que decidirlo con Claudia/Ana antes de entrenar, porque cambia la etiqueta.
2. **Pérdida colateral en el 35:** una pieza de 0,37 mm³ de la cámara (grosor normal) se quita con las rayas de caries porque estaba en su mismo grupo. Es un triángulo rojo en `Diente_35/control_antes_despues.png`. No se ha corregido.
3. **Por el mismo motivo, los conductos abiertos al exterior** (por el ápice, por una caries o por una fractura de raíz) **no se etiquetan**. En el corte coronal del 35 se ve una zona negra grande dentro de una raíz sin etiquetar: no se ha investigado si es conducto abierto o raíz rota.
4. **Los parámetros nuevos están ajustados con solo 3 dientes** (0,13 mm, 0,05 mm, 0,005 mm³). Con ~40 µCT hay que repetir el barrido y ver si siguen valiendo.
5. **Componentes: no todo "componente" es un conducto.** Los números de componentes (2, 1, 1) son estables entre resoluciones, pero el 2.º del 35 (3,9 mm³, 1,95 mm) es la parte baja de la cámara, no un segundo conducto. Y en el 51 el grupo principal parece incluir una cavidad grande de la corona (pieza de ~6 mm³ en z 885–956, vista en el corte z = 930 como un agujero con restos); por volumen y largo coincide con lo que salía ya a 64 µm. No está comprobado y no se ha quitado.
6. **El 35 es un molar de dos raíces** (comprobado en las imágenes de control). Para 51 y 52 sigue sin estar comprobado qué tipo de diente son; hay que mirarlos en Slicer.
7. **Sin verificar:** que las máscaras coincidan con lo que vería un experto. Revisarlas en Slicer (`canal.npy` está en orden (z, y, x), vóxel 0,016 mm; `norm.npy` es float16).

### Etapa 2 (simulador): NO empezada
No se ha escrito `simulate_cbct.py` ni se ha hecho la calibración. Solo se leyó el CBCT de Isa para ver qué hay (datos preliminares, **no son la calibración**): int32, 555×667×688, vóxel 0,075 mm; valores entre −1000 (aire, recortado: más del 25 % de los vóxeles valen −1000) y 17.819, mediana −303, percentil 99 = 2333; la segmentación de Isa tiene una sola etiqueta con 109.890 vóxeles = 46,4 mm³ (coincide con los 46 mm³ de arriba; los "15 componentes" no se han comprobado). Los grises parecen unidades tipo Hounsfield; no verificado.

### Por dónde se retoma mañana
1. Revisar en 3D Slicer las máscaras de `results/microct/Diente_XX/canal.npy` (sobre `norm.npy`), sobre todo la cámara pulpar del 35, la zona negra de la raíz y qué tipo de diente son 51 y 52; decidir la decisión abierta nº 1.
2. Después, etapa 2: calibrar con el CBCT de Isa (perfil del borde dentina–aire y dentina–conducto, ruido en dentina uniforme, grises de dentina/esmalte/fondo; recordar que su segmentación es borrador) y escribir `analysis/microct/simulate_cbct.py` (parametrizado por carpeta y semilla, rangos en un fichero de configuración, en GPU, con imágenes de control µCT / falso / real).
3. Los scripts de diagnóstico de hoy (seguimiento del conducto por cortes, barridos de la holgura, comparación antes/después) están en la carpeta temporal de la sesión y **no están en el repositorio**; si se quieren conservar hay que copiarlos a `analysis/microct/`.
