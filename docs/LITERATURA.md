# Literatura: qué cuentan 10 papers y qué se puede reutilizar

**Para qué sirve este documento.** Es el resumen de los 10 papers de `00_Papers` que importan para este proyecto (medir conductos en CBCT/µCT, segmentarlos con redes y fabricar CBCT sintéticos), con **qué técnica usa cada uno, qué cifras da y qué se puede reutilizar**. Sirve para escribir el estado del arte del TFM y para justificar decisiones. No tiene código ni resultados propios.

**Qué NO está aquí:** cómo se mide en este proyecto (→ [MEDICION.md](MEDICION.md)); lo que se ha hecho (→ [PROYECTO.md](PROYECTO.md)); los términos básicos (→ glosario de [GUIA.md](GUIA.md)).

Escrito el 2026-11-17 para Claudia (ingeniera, empezando en endodoncia).

## 0. Cómo se han leído los papers (y qué NO se ha mirado)

- **Qué papers:** los 9 distintos que se añadieron o modificaron el 17/11/2026 (son 10 archivos, porque «A new methodology for the measurement of the root canal curvature» está dos veces con el mismo hash) más Chen et al. 2025, que es del 15/10.
- **Cómo:** se sacó el texto de cada PDF con PyMuPDF y se leyó entero, página a página. Las **figuras no las he mirado** (salvo la página 9 de Chen, para entender su Tabla 4). Lo que solo salga en una figura (por ejemplo, las curvas de curvatura de Kucher) no está recogido aquí.
- **Páginas:** «p. N» es la página del PDF contando desde la 1 (no el número impreso de la revista). En Chen coinciden.
- **No he comprobado nada por mi cuenta:** todo lo que pone «el paper dice» lo he copiado del texto. No he repetido ningún cálculo ni he mirado sus datos o código.
- **Lo que es mío** (no sale de ningún paper) va marcado como *(propuesta mía)*.

---

## 1. Tabla comparativa

| Paper (año) | Datos | Técnica | Qué mide y cómo | Cifras clave | Qué puedo reutilizar |
|---|---|---|---|---|---|
| **Chen 2025** (segmentación del diente + «longitud del conducto») | 39 CBCT de pacientes vivos, 0,25 mm → 0,4 mm iso (p. 4) | Attention U-Net (detección) + ensemble Attention U-Net + V-Net (segmentación del **diente**) | Longitud = recta corona→ápice; ángulo = Schneider con el punto de máxima desviación entre centros de corte (p. 4) | Dice 96,33 %; error relativo de longitud 3,42 %; ángulo 2,85 ° (p. 6–7) | Su manera de definir el ángulo y de validar la longitud; su aviso de que un modelo falla en datos de otro centro (p. 9) |
| **Benyó 2012** | 25 µCT (17 incisivos, 8 molares) y 36 CBCT, voxel CBCT 100–300 µm (p. 8, 10) | Clustering difuso + esqueleto 3D por campo de potencial (µCT) / contracción de malla (CBCT) | Línea central por esqueleto; sin medida de longitud ni diámetro | «Éxito» 92 % (23/25 µCT, 33/36 CBCT); solo 75 % en molares con µCT (p. 15) | Esqueleto y su defecto en los extremos (se acorta) + cómo corregirlo (p. 8) |
| **Kucher 2020** | 50 molares inferiores extraídos, **raíz distal**; CBCT 0,08 mm y µCT 0,021 mm del mismo diente (p. 2–3) | Umbral → malla → cortes → centroides → LOWESS + spline | κ(s), R(s), longitud de arco, por tercios; **validado con un conducto artificial** (p. 5–6) | CBCT vs µCT: longitud mediana 9,6 vs 10,2 mm; curvatura máxima mediana 0,38 vs 0,47 mm⁻¹ (p. 8–9) | **El más útil**: algoritmo de curvatura, comparación CBCT/µCT y test con conducto artificial |
| **Christodoulou 2018** | 30 molares superiores extraídos, CBCT 0,125 mm (p. 2) | Centro por corte + spline B + curvatura | κ medio antes/después de instrumentar | κ medio 0,26–0,28 mm⁻¹; −30 % tras instrumentar (p. 3) | Poco: **no valida** su medida contra nada (lo dicen en p. 4) |
| **Česaitienė 2019** | 36 conductos de molares extraídos, µCT 8 µm (p. 2) | Centroides por corte + suavizado; ángulo con recta de referencia de 1 mm | Ángulo máximo de curvatura (3D); 5 niveles perpendiculares al conducto; área, perímetro, **diámetros con barrido de 1° a 1°** (p. 3–4) | Cambios de ~8–15 µm de diámetro entre grupos, sin diferencias significativas (p. 7) | **Diámetro por barrido angular** y niveles a 0,5/1 mm del ápice |
| **Zhang 2021** (RCS-Net) | CBCT con vóxel 0,1 mm, «78» imágenes (p. 3) | 3D U-Net con encoder global + decoder local, pérdida Dice + contorno, CLAHE | Solo segmentación del conducto | Dice 0,952; ASD 0,125 mm; MSD 0,332 mm (Tabla I, p. 4) | Preproceso, pérdida de contorno y aviso: su posproceso se queda con **un solo** componente conexo (p. 3) |
| **Ji 2025** (nnU-Net, reto STSR 2025) | 30 CBCT etiquetados + 300 sin etiquetar, 40 de validación (p. 4) | nnU-Net 3D, un solo paso | Solo segmentación | Dice «instancia» 0,669 frente a «imagen» 0,966 (p. 6) | Línea base fuerte (nnU-Net) y la lección: el Dice global engaña |
| **Rossi 2021** (CBCT → CT sintético) | 56 pares CT/CBCT de pelvis, 1 mm (p. 4) | U-Net supervisada vs CycleGAN | Convierte CBCT a CT (HU) | MAE 93,3 → 35,1 HU (supervisado) (p. 9) | Ideas para el simulador: artefactos reales del CBCT y riesgo de que una red «invente» estructuras (p. 9–12) |
| **Choi 2023** (resonancia → CBCT) | 21 pacientes (16/5), CBCT 0,3 mm, MRI 0,35 mm (p. 2) | U-Net 3D con encoder ResNet-50, pérdida smooth L1 | Genera CBCT sintético | Bordes borrosos, mejor ruido (p. 6–7) | Parches por ventana deslizante con peso gaussiano (p. 2) |
| **Chaudhari 2026** (predicción de transportación) | 120 molares inferiores, µCT 15 µm (p. 3) | Gradient boosting sobre 17 medidas del µCT | Transportación del conducto tras instrumentar | R² 0,87; MAE 0,031 mm apical (p. 5) | La **lista de 17 medidas** morfométricas (p. 4); el resultado, con cautela |

---

## 2. Resumen por paper

### 2.1 Chen et al. 2025 — *Tooth image segmentation and root canal measurement based on deep learning* (Front. Bioeng. Biotechnol.)
- **Datos (p. 4):** 39 CBCT de pacientes vivos de un hospital, vóxel 0,25 mm con cortes de 1 mm; se pasan a 0,4 mm isotrópicos. Una GPU RTX 3060. Reparto: detección 27/3/9 casos; segmentación 800/71/275 ROI (recortes de 64×64×96 por diente).
- **Técnica (p. 3):** (1) Attention U-Net detecta un «descriptor» del diente (el diente sin su borde) para localizarlo; (2) se recorta el diente y un **ensemble** de Attention U-Net y V-Net lo segmenta, entrenados por separado y conjuntamente, **10 veces cada uno**, ponderando por una métrica compuesta CMetric = Dice + 1/HD + 0,1/ASD; pérdida Generalized Dice Focal. (3) Se **corrige la posición** del diente (porque «las coronas no se alinean con el plano de corte», p. 3–4) con puntos de referencia de la corona, en el orden Z, X, Y.
- **Qué mide (p. 4):** (a) **Longitud**: «distancia directa» entre el centro de la corona y el centro del ápice de la raíz, es decir, una **recta**, no la longitud del conducto curvo. (b) **Ángulo de Schneider**: entre el vector desde el inicio del conducto al punto de inicio de la curva y el vector desde ese punto al final. El punto de inicio de la curva es el centro de corte donde el ángulo entre vectores consecutivos de centros es **máximo**.
  - **Mi lectura (no está dicho de forma explícita):** el texto de 2.4 habla de «dividir el diente en corona y raíz» y medir sobre la raíz; no dice que segmenten la luz del conducto. Las longitudes de la Tabla 4 (p. 9: 17–25 mm) tienen pinta de longitud corona→ápice, no de longitud de conducto. Si es así, su «root canal measurement» es una medida del diente/raíz.
- **Resultados:** segmentación del diente Dice 96,33 ± 0,89 %, Jaccard 92,94 %, HD 2,04 mm, ASD 0,24 mm (Tabla 2, p. 6); el ensemble supera a un U-Net único, pero tarda 4,72 s por diente frente a ~0,3 s. **Longitud** (p. 7): 9 casos de prueba, 275 dientes, 350 conductos; referencia = **media de 6 estudiantes de medicina** (no expertos); diferencia media 0,615 mm, error relativo 3,42 %; la desviación típica entre estudiantes es 0,406 mm; por eso no hay «gold standard» (lo dicen). **Ángulo** (p. 7): 5 CBCT de 0,125 mm, 17 dientes, 21 conductos; referencia = un médico especialista; diferencia media 2,85°; en 3 grados de dificultad (0–10°, 10–25°, >25°) hay 2 errores de clasificación de 21 (90,48 %).
- **Limitaciones (p. 8–10):** pocos datos de un solo hospital; un conjunto externo de 20 CBCT (Li 2024) hizo que el detector **fallara** al localizar dientes; con artefactos simulados leves (gaussianos y rayas) mejora, con artefactos fuertes hay 1 error; sin gold standard (pacientes vivos, no dientes extraídos); los pasos no están integrados de punta a punta. La Tabla 4 (p. 9) tiene 3 filas por tipo de diente sin explicar qué es cada fila.

### 2.2 Benyó 2012 — *Identification of dental root canals and their medial line from micro-CT and cone-beam CT records* (BioMed. Eng. OnLine)
- **Datos:** 25 µCT (17 incisivos y 8 molares) y 36 CBCT (mitad incisivos, mitad molares) (p. 10–11); voxel de CBCT 100–300 µm, conducto de 1 o pocos vóxeles (p. 8).
- **Técnica µCT (p. 4–8):** filtro de mediana → recorte → clustering difuso (EnFCM) con 4 clases (partición local) y 2 clases (global) → árbol de decisión elige cuál → crecimiento de regiones para encontrar las manchas oscuras (el conducto) → centro por mancha → comprobación de correlación entre cortes vecinos (avisa de bifurcaciones o de que hace falta intervención manual) → reconstrucción 3D → **esqueleto con campo de potencial** → corrección de extremos. **Técnica CBCT (p. 8–10):** filtro adaptativo → crecimiento de regiones difuso → malla → **contracción de malla** (con voxeles tan grandes no sirve el campo de potencial, p. 10).
- **Detalle que importa (p. 8):** el esqueleto **queda más corto** de lo que debería en los extremos del conducto; se corrige alargando virtualmente el conducto con tantos cortes como el radio más corto de la mancha del corte periférico.
- **Resultados (p. 13–15):** «éxito global» 92,0 % en µCT (23 de 25) y 91,7 % en CBCT (33 de 36); incisivos 100 %, molares 75 % (µCT) y 83,3 % (CBCT). Tiempos: 341 s medios por µCT, 0,93 s por CBCT.
- **Limitaciones:** no hay error de longitud, de curvatura ni de diámetro frente a ningún valor de referencia: «éxito» significa que un experto lo consideró correcto. Un solo sistema de µCT (p. 15). **Inconsistencia del propio paper:** la Tabla 3 (p. 14) dice «CBCT records» pero el texto la usa para µCT.

### 2.3 Kucher et al. 2020 — *Continuous Measurement of Three-Dimensional Root Canal Curvature Using CBCT and µCT* (Dentistry Journal)
- **Datos (p. 2–3):** 50 molares inferiores (1.º y 2.º) extraídos; **solo la raíz distal**, sin bifurcaciones ni ramificaciones; excluyen dientes con tratamiento, raíces incompletas, reabsorciones, fracturas o caries. Escaneados con **CBCT** (3D Accuitomo, 0,08 mm/píxel) y **µCT** (FCTS 160-IS, 0,021 mm/píxel). Es la única pareja CBCT–µCT del mismo diente en estos 10 papers.
- **Técnica (p. 3–5):** umbral → modelo 3D (STL) → cortes cada 10 µm a lo largo del eje del conducto → **centroide** de cada corte = línea central; se **interpola** a pasos de 10 µm, se filtra con **LOWESS** (ventana w) y se aproxima con spline B; curvatura κ(s) = 1/R(s) con R(s) = |x′|³ / |x′ × x″| (ecuación 3); longitud de arco = suma de segmentos (ecuación 4). La longitud se mide **del foramen apical al suelo de la cámara** (p. 4), el mismo criterio que el nuestro. Dividen el conducto en tercios apical/medio/coronal (p. 5).
- **Validación con un conducto artificial (p. 6–7):** curva de 20 mm de longitud, radio de 11 mm y de 6 mm en una vista y 2 mm en el extremo en la otra, con ruido gaussiano de ±10 µm en x e y. Con w = 120 µm y σ = 0,035 el error de longitud es 0,7 % y el RMSE de curvatura 0,058 mm⁻¹; con w ≥ 370 µm la longitud se acorta.
- **Resultados (p. 8–9):** longitud mediana 9,6 mm (CBCT) y 10,2 mm (µCT), rango 7,1–13,4 y 7,2–13,9 mm; curvatura mediana global 0,12 mm⁻¹ (R ≈ 8,1–8,5 mm) en ambos; curvatura máxima mediana 0,38 mm⁻¹ (CBCT, R 2,6 mm) y 0,47 mm⁻¹ (µCT, R 2,1 mm): **el CBCT da un 17 % menos**; la curvatura máxima cae con más probabilidad en el tercio apical (56 % CBCT, 68 % µCT).
- **Limitaciones (p. 10–11):** solo un conducto sin ramificaciones por diente; el filtro y la ventana son un compromiso: menos puntos reducen el ruido pero también el radio mínimo medible; «se desea una mejor aproximación» en tramos donde la curvatura es discontinua. Detalle: dicen «spline B cúbico» pero p = 4 (p. 5); un spline cúbico es de grado 3.

### 2.4 Christodoulou et al. 2018 — *A new methodology for the measurement of the root canal curvature and its 3D modification after instrumentation* (Acta Odontol. Scand.)
- **Datos (p. 2):** 30 molares superiores extraídos; CBCT NewTom VGi, vóxel 0,125 mm; clasificados por Schneider en radiografía en 3 grupos (0–20, 21–40, 41–60; la unidad no aparece en el texto extraído).
- **Técnica (p. 2):** centro de cada corte axial → spline B de grado 4 (método de Lee 2006) → curvatura. Dos CBCT por diente: antes y después de instrumentar con Reciproc.
- **Resultado (p. 3):** κ medio 0,2569 / 0,2764 / 0,2634 mm⁻¹ antes (grupos 1–3) y 0,1860 / 0,1967 / 0,1731 después; reducción media 30,23 %.
- **Limitación (p. 4):** los propios autores dicen que hace falta «una evaluación más completa» para decidir si la metodología es válida. No comparan con µCT ni con una referencia; la afirmación de que «cualquier inexactitud del CBCT no puede influir» no está respaldada por datos. **Fiabilidad baja como evidencia de exactitud.**

### 2.5 Česaitienė et al. 2019 — *Micro-CT Evaluation of Effects of Different Rotary Glide Path Techniques on Canal Transportation and Centering in Curved Root Canals* (Med. Sci. Monit.)
- **Datos (p. 2):** 36 conductos (mesiovestibulares/distovestibulares de superiores y mesiovestibulares/mesiolinguales de inferiores) de dientes extraídos, curvatura de 25–30° (Schneider); µCT RayScan 250E con vóxel de **8 µm**; dos escáneres por diente (antes/después).
- **Técnica (p. 3):** línea central con Avizo (centro de masa de cortes perpendiculares) y MATLAB. **Ángulo:** longitud de recta de referencia Dk = 1 mm (≈ 1/10 de la longitud del conducto; compromiso entre ruido y sensibilidad); se toma el punto de la línea central más alejado de la recta que une los extremos del tramo y se calcula el ángulo en 3D entre las dos rectas (ecuación del paper). **Cinco niveles** perpendiculares al conducto: A0 (0,5 mm del ápice), A1 (1 mm), C0 (curvatura máxima), C− y C+ (0,5 mm a cada lado de C0).
- **Diámetro (p. 4):** en cada corte, desde el centroide, **distancia a la pared cada 1°**; diámetro = suma de dos radios opuestos; diámetro mayor = máximo, menor = mínimo; «ratio of diameter ratios» (RDR) antes/después.
- **Resultados (p. 5–7):** sin diferencias significativas entre niveles ni entre grupos (p > 0,05); cambios de diámetro de 8 ± 14 a 15 ± 24 µm (Tabla 1).
- **Limitaciones:** el texto dice «6 niveles» (p. 5) cuando define 5; no hay comparación con ninguna medida de referencia (el trabajo compara antes y después de instrumentar).

### 2.6 Zhang et al. 2021 — *Root Canal Segmentation in CBCT Images by 3D U-Net with Global and Local Combination Loss* (EMBC)
- **Datos (p. 3):** CBCT con vóxel 0,1 mm, tamaño 128×128×(128–384), de un hospital; ground truth dibujado por radiólogos. El texto dice **78** imágenes, pero reparte 51 (entrenamiento) + 17 (test) = 68, sin explicar las 10 restantes. No dice qué diente ni si incluye la cámara pulpar.
- **Técnica (p. 2–3):** CLAHE (ecualización local de contraste), imágenes a 128×128×256; encoder global (localiza el centro del conducto) + decoder local (segmenta); pérdida Dice + pérdida Dice del **contorno** (etiqueta de contorno generada con level set y validada por un estomatólogo), peso 0,5; Adam, 100 épocas, 20 h en una Titan Xp de 12 GB. **Posproceso:** se queda con el **mayor componente conexo** porque «el conducto es un único componente conexo» (p. 3).
- **Resultados (Tabla I, p. 4):** Dice 0,952; VOE 0,077; ASD 0,125 mm; MSD 0,332 mm; mejor que Mask-RCNN 3D (0,897), 3D U-Net (0,915) y V-Net (0,929); 350 ms por volumen.
- **Limitaciones:** su regla del mayor componente conexo **borraría conductos separados** en un molar con varios conductos. Sin medida de longitud ni de curvatura. Datos de un solo centro.

### 2.7 Ji et al. 2025 — *Efficient nnU-Net for Tooth and Root Canal Segmentation in CBCT* (reto MICCAI STSR 2025, tarea 1)
- **Datos (p. 4):** 30 CBCT con máscaras de diente y conducto + 300 sin etiquetar; validación con 40. No dan el tamaño de vóxel. RTX 4090 de 24 GB, 1000 épocas (p. 5).
- **Técnica:** nnU-Net 3D a resolución completa, un solo paso; aceleración en inferencia (sin aumento de datos en el test e interpolación en flotante): el tiempo de inferencia baja de 3257 s a 852 s con Dice casi igual (p. 5–6). Probaron además un esquema de dos pasos (primero el diente, luego su ROI con pseudoetiquetas de los 300 sin etiquetar) y **salió peor** (p. 6).
- **Resultados (Tablas 2–3, p. 6):** Dice «instancia» 0,6691 y «imagen» 0,9659 (con aceleración); un paso 0,6738 / 0,9671 frente a dos pasos 0,626 / 0,9207. El paper no define «instancia»; supongo que es por estructura individual (diente/conducto), pero **no está dicho**. Muestra que el Dice global (0,97) queda dominado por lo grande y no dice nada del conducto.
- **Limitaciones:** no da el Dice del conducto por separado ni el tamaño de vóxel, y no hay medidas clínicas.

### 2.8 Rossi y Cerveri 2021 — *Comparison of Supervised and Unsupervised Approaches for the Generation of Synthetic CT from CBCT* (Diagnostics)
- **Datos (p. 4):** 56 pares CT/CBCT de pelvis de la base pública PRD, píxel de 1 mm, cortes de 3 mm; U-Net 2D de ~2,55 M parámetros, 25 épocas, 4 folds por paciente.
- **Resultado (Tabla 1, p. 9):** MAE respecto al CT: CBCT original 93,3 HU; supervisado 35,1 HU; no supervisado (CycleGAN) 46,4 HU.
- **Lo relevante para nosotros (p. 1, 9, 11–12):** el CBCT tiene dispersión, sombreado, **cupping** (el brillo cambia del centro al borde), reducción de contraste y **endurecimiento del haz**; en los perfiles se ve sobre todo un desplazamiento lineal, pero el endurecimiento da curvas no lineales en los bordes. El modelo supervisado **rellenó una cavidad de aire** distinta entre CT y CBCT y «inventó» valores en zonas sin campo; el CycleGAN preservó mejor los contornos. Para un segmentador que depende de que el conducto se conserve, eso es una advertencia.
- **Limitaciones:** otra anatomía (pelvis) y otra escala; no es dental.

### 2.9 Choi et al. 2023 — *Deep learning synthesis of CBCT from zero echo time MRI* (Sci. Rep.)
- **Datos (p. 2):** 21 pacientes con CBCT (Alphard 3030, 0,3 mm) y MRI ZTE (3 T, 0,35 mm); 16 entrenamiento, 5 test; registro rígido manual+información mutua.
- **Técnica (p. 2):** U-Net 3D con encoder de bloques ResNet-50, sin la última conexión de salto (para no heredar errores de registro), pérdida **smooth L1** (más robusta que MSE), parches de 128×128×16 o 64×64×16, inferencia por **ventana deslizante** con peso gaussiano.
- **Resultados (p. 5, 7):** desviación de superficie del modelo 3D 2,95 ± 0,35 mm (parche grande) y 2,93 ± 0,39 mm (pequeño); en tejido duro, MAE 3,3–3,4 y SSIM ≈ 0,89; los expertos puntúan mejor ruido y artefactos que el CBCT real, pero **peor resolución** (bordes borrosos).
- **Limitaciones (p. 7–8):** pocos pacientes; registro imperfecto; es MRI→CBCT, no µCT→CBCT.

### 2.10 Chaudhari et al. 2026 — *Machine learning prediction of canal transportation using micro-CT data* (Bioinformation)
- **Datos (p. 3):** 120 molares inferiores (240 conductos mesiales), curvatura de 20–45° (Schneider); µCT SkyScan 1275 con vóxel de **15 µm**.
- **Técnica (p. 4–5):** 17 medidas del µCT previo (ángulo de Schneider, radio de curvatura, área, perímetro, índice de redondez, diámetro mayor y menor, grosor mínimo de pared mesial y distal, relación de grosores, posición del centroide, área de la raíz, relación conducto/raíz, volumen y superficie hasta el nivel de medida, índice de conicidad) a 1, 3 y 5 mm de la longitud de trabajo; cuatro modelos (RF, GBM, SVR, ANN) con validación cruzada de 5 pliegues **por diente**.
- **Resultados (Tablas 1–2, p. 5):** transportación media 0,128 ± 0,074 mm en el nivel apical; GBM: R² = 0,87, MAE = 0,031 mm, RMSE = 0,042 mm. El predictor más importante es el ángulo de Schneider; la relación se dispara por encima de unos 30° (p. 5–6).
- **Cautelas:** no he visto declaración de disponibilidad de datos o código en lo que he leído; sin validación externa; no he podido comprobar la calidad de la revista. Lo trato como **fuente de ideas** (la lista de medidas), no como evidencia de rendimiento.

---

## 3. Qué dice la literatura sobre lo que quiere Luis

Mapa a las cuatro peticiones de Luis, con lo que se ha encontrado **en estos 10 papers** (no en toda la literatura):

1. **Red de segmentación aunque haya 3 dientes.** Ji 2025 usó 30 casos etiquetados con nnU-Net y Chen 39 casos; Chen y la versión de dos pasos de Ji descomponen cada caso en recortes por diente para multiplicar las muestras (Chen p. 4; Ji p. 6), aunque la versión de dos pasos de Ji salió peor que la de un paso. Nuestro caso es más extremo (3 dientes, ~10 conductos), pero compensa con el simulador.
2. **Diámetros y ángulos de forma analítica a partir de la reconstrucción 3D.** Es lo habitual: Kucher, Christodoulou y Česaitienė calculan centroides por corte + spline + fórmulas; Benyó usa esqueleto. **Solo Česaitienė calcula diámetros** (barrido angular), y en µCT de 8 µm; Chaudhari lista diámetro mayor/menor y redondez, también en µCT.
3. **Si no se puede, medir a mano y entrenar una red que devuelva las medidas.** **En estos 10 papers no he encontrado ninguno que entrene una red para dar directamente las medidas** (Chen mide de forma analítica sobre la segmentación; Chaudhari usa aprendizaje automático sobre medidas ya calculadas, no sobre imágenes).
4. **Referencia de la inclinación.** No hay consenso en los papers: Schneider (Christodoulou p. 1) usa el eje largo del conducto y la recta foramen→inicio de curva; Chen (p. 3–4) corrige la posición del diente respecto a los ejes de la imagen y mide ángulo con puntos del propio conducto; Kucher y Česaitienė usan la línea central y miden curvatura sin ejes anatómicos; Kucher (p. 5) separa vistas buco-lingual y mesio-distal.

**Dos hechos que conviene tener presentes (p. 8–9 de Kucher):** con el mismo diente, el CBCT da longitudes ~6 % más cortas que el µCT (9,6 frente a 10,2 mm) y curvaturas máximas ~17 % menores. Es un buen orden de magnitud de qué esperar de un CBCT de 80 µm, pero es **una sola raíz, un solo escáner**; no se debe asumir para tu CBCT.

---
