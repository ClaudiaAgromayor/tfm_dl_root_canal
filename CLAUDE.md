# Contexto del proyecto (TFM de Claudia Agromayor, ICAI-Comillas)

Responde siempre en **español**, en un lenguaje sencillo: la autora viene de ingeniería y está empezando en endodoncia. Explica los términos médicos y de imagen la primera vez que aparezcan. No des nada por hecho: si algo no está verificado, dilo.

## Objetivo
A partir de un CBCT clínico, **segmentar los conductos radiculares del primer molar inferior (36/46)** y medir longitud, curvatura, volumen y, con cautela, diámetro (a 75 µm el diámetro apical es impreciso).

## Enfoque decidido (caso B)
No se usan Pulpy3D ni los datos de Fer: **descartados**. Se parte de µCT de dientes extraídos:
1. Normalizar cada µCT y obtener el canal por umbral (`analysis/microct/preprocess_microct.py`).
2. **Fabricar CBCT falsos a 75 µm** a partir del µCT: bajar resolución, desenfoque (PSF), ruido (Poisson filtrado: muy correlacionado entre vecinos y **casi gaussiano**, no "no gaussiano" como se pensó al principio; ver "Hecho: etapa 2"), contraste aleatorio. Como el falso sale del µCT, **la etiqueta ya está alineada y no hay registro**.
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
- Métricas: Dice y HD95 **por tercios** (coronal, medio, apical), distancia del final del conducto al ápice real, y sobre todo el **error de cada medida** frente al µCT. (Hoy solo están implementados Dice y HD95 por tercios sobre parches, `2_UNet/metrics_canal.py`; la distancia al ápice real y el error de cada medida están **pendientes**.)
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
2. ~~Simulador de CBCT a 75 µm~~ (hecho, calibrado con el CBCT de Isa; ver abajo). Falta decidir la cámara pulpar.
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
1. ~~Cámara pulpar~~ **DECIDIDO por Claudia: la etiqueta es SOLO conducto (del orificio de entrada al ápice), sin cámara pulpar ni cavidades de caries.** Aplicado en la sección "Etiqueta solo conducto" de más abajo; la máscara solo-conducto es `canal_solo_conducto.npy`.
2. **Pérdida colateral en el 35:** una pieza de 0,37 mm³ de la cámara (grosor normal) se quita con las rayas de caries porque estaba en su mismo grupo. Es un triángulo rojo en `Diente_35/control_antes_despues.png`. No se ha corregido.
3. **Por el mismo motivo, los conductos abiertos al exterior** (por el ápice, por una caries o por una fractura de raíz) **no se etiquetan**. La parte del ápice está investigada (ver "Pregunta del ápice"); en el corte coronal del 35 se ve además una zona negra grande dentro de una raíz sin etiquetar que no se ha investigado: ¿conducto abierto o raíz rota?
4. **Los parámetros nuevos están ajustados con solo 3 dientes** (0,13 mm, 0,05 mm, 0,005 mm³). Con ~40 µCT hay que repetir el barrido y ver si siguen valiendo.
5. **Componentes: no todo "componente" es un conducto.** Los números de componentes (2, 1, 1) son estables entre resoluciones, pero el 2.º del 35 (3,9 mm³, 1,95 mm) es la parte baja de la cámara, no un segundo conducto. Y en el 51 el grupo principal parece incluir una cavidad grande de la corona (pieza de ~6 mm³ en z 885–956, vista en el corte z = 930 como un agujero con restos); por volumen y largo coincide con lo que salía ya a 64 µm. No está comprobado y no se ha quitado.
6. **El 35 es un molar de dos raíces** (comprobado en las imágenes de control). Para 51 y 52 sigue sin estar comprobado qué tipo de diente son; hay que mirarlos en Slicer.
7. **Sin verificar:** que las máscaras coincidan con lo que vería un experto. Revisarlas en Slicer (`canal.npy` está en orden (z, y, x), vóxel 0,016 mm; `norm.npy` es float16).

### Hecho: etapa 2, simulador de CBCT a 75 µm (sin entrenar nada todavía)
Tres ficheros en `1_Data_Validation/analysis/microct/`: `calibrate_cbct.py` (mide un CBCT real), `simulate_cbct.py` (el simulador, parametrizado por carpeta de diente y semilla, en GPU) y `simulate_cbct_config.yaml` (los rangos del azar, con comentarios de dónde sale cada centro). Salidas locales: `results/cbct_calibration/Isa_caso01/` (`calibration.json`, `calibration.png`) y `results/sim_cbct/ejemplos_seed0/` (9 parches `.npz` + `.json` con todos los parámetros sorteados + 6 imágenes µCT / simulado / real).
Uso: `python simulate_cbct.py results/microct/Diente_35 results/microct/Diente_51 --n 8 --seed 0 --out results/sim_cbct --figs 4 --real-cbct <cbct.nrrd> --real-seg <seg.nrrd>`. Con `--check --calibration <json>` mide el simulador con las mismas funciones que el CBCT real. Para un dataloader: `Simulator(cfg).patch(Tooth(carpeta), rng)`. Cada parche es de 128³ vóxeles de 75 µm (9,6 mm) y sale como imagen, etiqueta blanda (fracción de conducto por vóxel), etiqueta binaria (≥ 0,5) y máscara del diente.

**Pasos del simulador** (todos con valor aleatorio en un rango): recorte de 600³ vóxeles de 16 µm cerca del conducto → grises (µCT pasado de forma lineal a grises de CBCT con aire = −1000; conducto relleno con gris de pulpa, no aire; alrededor del diente, "hueso/tejido blando" con textura y a veces una capa fina de ligamento periodontal) → emborronado gaussiano (PSF) **antes** de reducir → reducción a 75 µm promediando bloques (el parche empieza en un vóxel de 16 µm aleatorio, así el desplazamiento respecto a la rejilla de 75 µm es aleatorio) → ruido Poisson filtrado (correlacionado) + variación suave de brillo → etiqueta con exactamente la misma geometría que la imagen. Aumentos en el plano xy (volteos, giro de 90°); el eje z (largo del diente) no se toca.

**Lo medido en el CBCT de Isa** (`calibrate_cbct.py`; su segmentación es borrador y solo se usó para decir dónde está el conducto):
- Grises (aire recortado en −1000): dentina mediana 1678 (p5–p95 1514–1884); esmalte 3339; tejido blando/médula ≈ −57; tejido lejano del diente mediana 137 (muy ancho: hueso y médula); interior del conducto en su borrador, vóxeles profundos, mediana 433 (mezcla con el borde, poco fiable).
- Ruido en dentina uniforme (bloques de 9³ vóxeles, tras quitar un plano): sd 61 (cuartiles 44–97); en tejido blando 59; en esmalte 157. **Es casi gaussiano** (asimetría −0,32, curtosis en exceso 0,17 en dentina; una gaussiana daría 0 y 0), así que lo de "no gaussiano" del enfoque inicial **no se confirma en zonas uniformes**; lo que sí es fuerte es la **correlación entre vóxeles vecinos** (autocorrelación a 1 vóxel ≈ 0,84). En el esmalte sí se aparta (asimetría −1,4, curtosis 2,3), pero ahí pesa la estructura y los bordes. Tampoco crece con la densidad: dentina 61 frente a tejido blando 59.
- Anchura del borde (sigma de un borde emborronado, incluye el vóxel de 75 µm): dentina–exterior 0,140 mm (752 bordes), esmalte–exterior 0,141 mm (142), dentina–conducto 0,114 mm pero con ajuste malo (35 bordes, error 0,06): conductos finos y borrador, **poco fiable**. En z: 0,121 mm con solo 30 bordes, poco fiable.
- Hallazgo útil: el esmalte/dentina del µCT (1,6 / 0,975 = 1,64) coincide con el del CBCT con aire = −1000 ((3339+1000)/(1678+1000) = 1,62), así que el paso µCT → CBCT es lineal y no hace falta una curva rara.

**Comprobación del simulador contra Isa** (mismas funciones de medida; 36 parches con rangos completos, dos semillas; entre paréntesis, con los parámetros en el centro de cada rango):

| Magnitud | Isa (real) | Simulado |
|---|---|---|
| Gris dentina (mediana) | 1678 | 1703–1735 (1763) |
| Gris esmalte (mediana) | 3339 | 3414 (centro) / 3465 (24 parches) |
| Ruido sd dentina | 61,1 | 63,6–64,0 (62,7) |
| Ruido sd tejido blando | 58,5 | 63,4–71,1 (59,0) |
| Autocorrelación a 1 vóxel, dentina | 0,839 | 0,800–0,804 (0,815) |
| Asimetría / curtosis del ruido, dentina | −0,32 / 0,17 | −0,31…−0,24 / 0,18…0,22 |
| Borde dentina–exterior, sigma (mm) | 0,140 | 0,135–0,143 (0,144) |
| Borde esmalte–exterior, sigma (mm) | 0,141 | 0,143–0,153 (0,139) |
| Borde dentina–conducto, sigma (mm) | 0,114 | 0,112 (0,124) |

Con 24 parches el ruido salió en 77: era azar del muestreo (con 36 parches y dos semillas vuelve a 64). El tejido lejano sale en −15…250 según la tanda (Isa 137): ese fondo es el menos ajustado.
Pruebas hechas: **(a) alineación**: con el desenfoque y el ruido apagados y el conducto como aire, la correlación imagen–etiqueta es la más negativa con desplazamiento 0 en los tres ejes y en los tres dientes (p. ej. diente 52: −0,833 a 0, frente a −0,818/−0,797 a ±1 y −0,763/−0,736 a ±2 en z); **(b) reproducibilidad**: la misma semilla da imagen, etiqueta y parámetros idénticos (diferencia máxima 0,0); **(c) velocidad**: era ~3 s por parche; ahora `patch()` tarda 0,41–0,45 s en GPU (87 s en CPU) y el bucle de la línea de comandos, que además comprime y guarda cada parche, 1,0–1,6 s con hasta 11,3 GB de GPU de pico (medido el 2026-10-28).

**Dudas y límites del simulador (sin resolver):**
1. **Un solo CBCT real** (un paciente, un escáner): los rangos se ensancharon a propósito (p. ej. dentina 1300–2100, ruido 20–95), pero no se han validado con ningún otro CBCT. Que los grises sean Hounsfield reales no está verificado.
2. **Gris del conducto adivinado:** [−250, 500]. La calibración no lo fija bien (borrador de Isa, conductos finos). Es el parámetro más importante para el contraste del conducto y el menos medido.
3. **Qué rodea al diente es textura aleatoria**, no anatomía de mandíbula: no hay dientes vecinos, cortical definida ni raíces contiguas. (Esto cambió: ahora la cámara y las cavidades quitadas SÍ se rellenan con gris de pulpa; ver "Etiqueta solo conducto". Lo que sigue siendo fondo con textura es la caries abierta al exterior, que no es parte del diente enmascarado.)
4. **No se simulan** el endurecimiento del haz, las rayas metálicas, el movimiento ni el "cupping" (brillo que cambia del centro al borde). El µCT del 35 tiene un pequeño pico de gris por encima de 2 (en unidades normalizadas, donde la dentina es ≈ 1): podría ser una restauración; no se ha mirado.
5. El µCT es de dientes extraídos: el diente sale con la orientación de su escaneo (en el 35 y el 51 la corona queda hacia z alto; el 52 no se ha mirado); un CBCT real puede venir en otra orientación. Los aumentos xy no cubren eso.
6. El parche (9,6 mm) cubre solo parte del diente (15–17 mm de largo): para medir longitud o cumplir el apical hará falta inferencia por ventanas sobre un volumen completo simulado, que **aún no existe** (el simulador solo hace recortes).
7. Con 3 dientes (uno quizá no molar) el simulador da mucha variedad de imagen pero **poca variedad de anatomía**: es el límite real para entrenar.
8. El ruido real se midió con bloques de 9³; el simulado, con exactamente la misma función. Coinciden en lo medido, pero esto no demuestra que el falso "engañe" a una red: eso se verá en la etapa 4 (evaluar en Isa).

### Hecho: etiqueta SOLO conducto, pregunta del ápice, exportación a Slicer y montaje de la etapa 3
(Decisión de Claudia: la etiqueta es el conducto del orificio de entrada al ápice, sin cámara pulpar ni cavidades de caries. Nada se ha commiteado.)

**Pregunta del ápice (`split_canal.py`, por conducto, con la corona hacia z alto en los tres dientes; distancias en mm, 16 µm).** Cada raíz se detecta como componente 3D del diente por debajo de la furca; cada conducto es un grupo de trozos unidos con la misma holgura de 0,13 mm. "Punta local" = último corte con diente a menos de 0,6 mm del eje del conducto (en raíces-lámina anchas, y en el 51 donde las dos láminas salen como una sola "raíz" de 288 mm³, la última sección con diente de toda la raíz no sirve).

| Diente | Raíz / conducto | Fin de la máscara a la última sección con diente de la raíz | a la punta local |
|---|---|---|---|
| 35 | raíz 1 | 0,912 | 0,864 |
| 35 | raíz 2 | 0,992 | 0,992 |
| 51 | conducto 1 | 0,832 | 0,832 |
| 51 | conducto 2 | 0,864 | 0,720 |
| 51 | conducto 3 | (1,760: no válido) | 0,224 |
| 52 | raíz 1 | 0,496 | 0,496 |
| 52 | raíz 2 | 0,352 | 0,352 |

**¿Sale el conducto al exterior por el ápice y se pierde ese tramo? Sí, pero el tramo perdido es corto.** Comprobado a ojo en cortes axiales ampliados (cada 0,13 mm) de 6 de los 7 conductos (35 ×2, 52 ×2, 51 conductos 1 y 3; el conducto 2 del 51 no se miró): hasta el final de la máscara el conducto está encerrado por dentina; ~0,06 mm más abajo el hueco se abre al exterior por una muesca de la superficie de la raíz (foramen), y por debajo hay dentina maciza hasta la punta. El hueco visible que se pierde es de ~0,06–0,2 mm; el resto de la distancia de la tabla (0,2–1,0 mm) es dentina sin luz visible. Excepción: en la raíz 1 del 35 el conducto es ancho y está abierto al exterior ya en el final de la máscara (no es un foramen estrecho). **La métrica automática `hueco_sin_etiquetar_hacia_la_punta_mm` (0,2–0,7 mm) sobreestima** porque cuenta también la concavidad natural de la raíz: usar las imágenes, no ese número. Una luz de menos de ~30 µm no se ve a 16 µm, así que un conducto estrecho puede seguir más allá de donde parece acabar. Implicación para las medidas: la longitud del conducto medida sobre esta máscara se queda corta unos 0,1–0,2 mm respecto al foramen. **Comentario de Isa (endodoncista) sobre las imágenes `results/para_isa/`, 2026-10-28, informal y relatado por Claudia (solo vio el diente 52 (tres imágenes de ese diente); no vio el 35 ni el 51):** el conducto se mide siempre hasta el **foramen apical** (no hasta la punta de la raíz); en la imagen sin marcar se ve el cambio justo donde apunta la flecha, y la flecha es "perfecta": el final de la máscara coincide con el que ella pondría. Por eso la frase anterior (que la máscara "se queda corta 0,1–0,2 mm respecto al foramen") era una lectura mía **sin experto y queda sin confirmar**; la distancia de 0,35–1,0 mm entre el final de la máscara y la punta de la raíz sería, entonces, la distancia foramen–ápice. Su comentario vale solo para el diente 52: pendiente confirmarlo con Isa para el 35 (punta partida, el menos claro) y el 51.

**Criterio solo conducto (`split_canal.py`, parámetros y criterio guardados en `info.json["separacion"]`).** Se recorren los cortes axiales desde el ápice hacia la corona y se mide en cada uno la sección 2D más grande del conducto (restos < 0,02 mm² ignorados). En la raíz son varias secciones pequeñas; en la cámara los conductos se funden en una sola grande. **El orificio es el primer corte con sección mayor ≥ 1,5 mm² durante ≥ 0,1 mm**; lo coronal a ese plano es `camara_y_cavidades`, lo apical es el conducto. La corona se detecta como el extremo con más esmalte (norm > 1,3). Por qué área de sección y no radio inscrito: el radio del conducto de la raíz llega a 0,34 mm y el de la cámara empieza en 0,39 (sin hueco claro), mientras que el área pasa de ≤ 1,15 a ≥ 1,6 mm² en < 0,6 mm de altura. Por qué un plano y no un grosor 3D: una sola cifra interpretable (dónde se funden los conductos), sin flecos finos alrededor de la cámara, vale igual para molares con 3-4 conductos que para raíces únicas, y no depende de que el conducto sea fino o ancho.

| Diente | Orificio (z, 16 µm) | Completo (mm³) | Solo conducto | Cámara y cavidades | Mayor sección en la raíz (≥ 0,5 mm bajo el orificio) |
|---|---|---|---|---|---|
| 35 | 786 | 21,199 | 11,151 | 10,048 | 1,146 mm² |
| 51 | 797 | 23,139 | 8,001 | 15,138 | 1,012 mm² |
| 52 | 815 | 36,885 | 11,702 | 25,182 | 1,113 mm² |

Sensibilidad: con umbral 1,25 mm² el orificio sube −0,11 / −0,16 / −0,21 mm; con 2,0 mm² baja +0,59 / +0,42 / +0,22 mm (35 / 51 / 52). Es < 4 % de la longitud del conducto. **Está ajustado con solo 3 dientes**: con ~40 hay que mirar `info.json["separacion"]["avisos"]` (avisa si la mayor sección de la raíz supera el 80 % del umbral, si no se detecta cámara o si la orientación es dudosa) y las imágenes `control_split.png` / `control_apice.png`. Límites: el orificio es un plano horizontal (el suelo real de la cámara es irregular); un diente sin cámara detectada queda con todo el conducto como etiqueta y un aviso; las cavidades de caries dentro de la raíz, apicales al orificio, quedarían como conducto (no ocurre en estos 3).
Ficheros por diente en `results/microct/Diente_XX/`: `canal_solo_conducto.npy` (etiqueta), `camara_y_cavidades.npy` (lo quitado), `canal_completo.npy` (copia de `canal.npy`, que no se ha tocado), `info.json` (+ `orientacion`, `separacion`, `apice`), `control_split.png`, `control_apice.png`. Se cumple solo conducto + cámara = completo, sin solape.

**Simulador con la etiqueta nueva: la cámara y las cavidades quitadas se rellenan con gris de pulpa y la pérdida las ignora solo en una franja de ±0,15 mm alrededor del orificio** (`labels` en `simulate_cbct_config.yaml`; modos `weight_mode: band | chamber | none` y `chamber_fill: pulp | bg` para compararlos). Por qué: (1) en un paciente la cámara contiene pulpa, no hueso; rellenarla como hueso enseñaría a la red que "un vacío oscuro = fondo" y que "el conducto es el único hueco oscuro del diente", dos atajos falsos. (2) Dejarla como fondo con etiqueta 0 y peso 1 es lo correcto salvo en el orificio, donde el corte es una convención de ±0,1 mm que la red no puede adivinar: ahí un error no debe castigarse. (3) Ignorar toda la cámara dejaría su salida indefinida y obligaría a posprocesar con la misma regla del orificio a 75 µm, que no está validada. Medido: gris medio del conducto 294–370, de la cámara 253–311, del fondo 859–884 (pulpa fija en 100 y hueso en 900 para la prueba); franja de peso 0 = 0,23–0,25 % del parche; ignorar toda la cámara = 1,4–2,8 %. Alineación imagen–etiqueta: mínima en desplazamiento 0 en los tres ejes y dientes. Con parámetros en el centro de los rangos sigue coincidiendo con Isa (dentina 1715 frente a 1678; ruido 62,6 frente a 61,1; borde dentina–exterior 0,141 frente a 0,140 mm). **Con rangos aleatorios la medida de ruido sale hasta 83–86 frente a 61: es un sesgo de la medida** (usa una ventana de gris fija 1450–1850 y la dentina simulada cae en 1300–2100; el ruido medido correlaciona 0,76 con el gris sorteado), no del simulador; la comparación válida es en el centro. La caries abierta al exterior sigue siendo fondo con textura. `patch()` tarda ahora 0,41–0,45 s en GPU (antes ~3 s: conversión de tipos en la GPU y dilatación separable, idéntica al máximo 3D, verificado); el bucle de la línea de comandos, con el guardado, 1,0–1,6 s.

**Exportación para 3D Slicer:** `results/para_slicer/` (instrucciones en `LEEME.txt`), cajas del diente a 32 µm. Por diente: imagen uint8 (gris = norm × 127,5) y tres máscaras (solo conducto, cámara y cavidades, completa). Pesan, por diente, ≈ 50 MB la imagen (35: 50,24; 51: 50,21; 52: 53,16 MB) y 0,12–0,21 MB cada máscara. Releídos con SimpleITK: tamaño y espaciado (0,032 mm) correctos y sumas iguales a lo escrito. **No se han abierto en Slicer.** A 32 µm solo conducto y completa salen por mayoría y la cámara = completa − solo conducto (evita solape en el plano del orificio).

**Etapa 3: SOLO infraestructura; no se ha entrenado en serio (espera la confirmación de las etiquetas).** `2_UNet/dataloader/SimCBCT.py` (descubre los dientes por carpeta con `canal_solo_conducto.npy`, sin `splits.json`; un hilo prepara lotes; banco fijo de validación con semilla), `2_UNet/models/CanalUNet3D.py` (UNet sin el embedding de posición de Pulpy3D y con logits; los UNet antiguos no se tocan), `2_UNet/losses/SoftDiceBCE.py` (etiqueta blanda + mapa de pesos), `2_UNet/metrics_canal.py` (Dice y HD95 por tercios apical/medio/coronal definidos sobre el conducto del diente completo; probado con un caso conocido), `2_UNet/train_sim.py` + `configs/sim_canal.yml` (Adam, bf16, validación dejando un diente fuera, nunca recortes del mismo diente). Se lanza con `python train_sim.py -c configs/sim_canal.yml --holdout Diente_52`.
Prueba de montaje (3 dientes, una vuelta por cada diente fuera, 200 iteraciones, lote 2, 5,65 M parámetros, simulando en la GPU): ≈ 340 s por vuelta = **1,7 s por iteración** (0,9 s esperando al simulador, 0,7 s en la red), **10,9 GB de GPU de pico** (de 16). La pérdida de entrenamiento baja de 1,55 a 0,98–1,01. En validación (6 parches fijos del diente de fuera) el Dice es 0 hasta la it. 100 y a la it. 200 sale 0,67 / 0,54 / 0,78 al dejar fuera el 35 / 51 / 52. **No se saca ninguna conclusión**: 2 dientes de entrenamiento, 200 iteraciones, 6 parches; solo demuestra que el montaje funciona de extremo a extremo. Resultados en `2_UNet/results/entrenamientos_sim/` (no van a git).
Pendiente de la etapa 3 (necesita un volumen completo simulado, que aún no existe): distancia del final del conducto predicho al ápice real y error de cada medida (longitud, curvatura, volumen, diámetro). Con ~40 µCT: k pliegues por diente y un test de dientes completos aparte; hay que revisar el criterio del orificio y los avisos de cada diente.

### Por dónde se retoma
1. Revisar en 3D Slicer `results/para_slicer/` (instrucciones en `LEEME.txt`): sobre todo si el plano del orificio cae donde un endodoncista pondría la entrada de los conductos y qué tipo de diente son 51 y 52. Ahí se confirman o se corrigen las etiquetas; hasta entonces no se entrena en serio.
2. Con las etiquetas confirmadas: entrenar en serio (más iteraciones, comparar `weight_mode` y `chamber_fill`, con y sin simulador) y construir el volumen completo simulado para la distancia al ápice y el error de las medidas.
3. Los scripts de diagnóstico de la etapa 1 están en la carpeta temporal de la sesión y **no están en el repositorio**; los de la etapa 2 y de esta sesión (`calibrate_cbct.py`, `simulate_cbct.py`, `split_canal.py`, `export_slicer.py` y los de `2_UNet/`) sí están en el repositorio, **sin commit**.

### Cambios del 2026-10-28 (sin commit; ver `docs/RESUMEN_PROYECTO.md`, apartado 0b)
- **`preprocess_microct.py` ya no puede pisar las etiquetas finales:** por defecto escribe a 16 µm en `results/microct` y a otras resoluciones en `results/microct_32um`, `microct_64um`…; aborta si va a sobrescribir etiquetas ya separadas por `split_canal.py` o una carpeta de otra resolución (salvo `--force`); `--downsample` por defecto 1; la salida no depende del directorio de trabajo. Los README tenían un comando que habría destrozado las etiquetas (corregido).
- **Simulador:** lee el vóxel de cada diente (ya no supone 16 µm; con los interruptores antiguos el resultado a 16 µm es idéntico, diferencia 0); reducción a 75 µm por cajas exactas (`crop.reduccion: exacta`; la antigua, `adaptativa`, promediaba 106 µm; efecto sobre la calibración con Isa despreciable: σ del borde 0,142 → 0,141 mm); y parches repartidos a partes iguales entre tercios del conducto (`crop.third_weights: [1,1,1]`; antes solo el 13,9 % de los centros caía en el ápice, ahora 31,9 %). `--check` usa muestreo proporcional (como se calibró).
- **Entrenamiento (`train_sim.py`):** `--seed`, `--third-weights {config,equilibrado,proporcional}`, `--val-patches`; la validación siempre reparte los parches entre tercios. Efecto del muestreo: **inconcluso** con 200 iteraciones (5 de 10 entrenamientos sin despegar); con 1000 iteraciones (una vuelta, una semilla) el Dice apical final es igual (0,58) y el equilibrado converge antes.
- **Scripts de diagnóstico** copiados y parametrizados en `1_Data_Validation/analysis/microct/diagnostico/` (con `README.md`); `imagen_para_isa.py` genera las imágenes de `results/para_isa/`.
- **Comentario de Isa** (informal, relatado por Claudia): el conducto se mide hasta el foramen apical y la flecha del final de la máscara es "perfecta" (detalle y cautelas en "Pregunta del ápice"). **Solo vio el diente 52** (tres imágenes de ese diente); el 35 y el 51 no se le enseñaron, así que para ellos sigue sin confirmar.
- `results/microct_full/` (11 GB, del script antiguo; ningún script lo usaba) **borrado el 2026-10-28 por orden de Claudia**.
- **Muestreo de parches:** se deja el equilibrado por tercios (`crop.third_weights: [1,1,1]`). No se repite la comparación con varias semillas ahora: con 2 dientes de entrenamiento el resultado sería ruido. Se volverá a comparar con los ~40 dientes y validación cruzada.
