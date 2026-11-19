# Medición del conducto: cómo se hace, qué error tiene y qué falta

**Para qué sirve este documento.** Explica, desde cero y con las figuras y el código reales, **cómo se obtienen la línea central, la longitud, la curvatura, los ángulos y los diámetros de cada conducto** a partir de su máscara 3D; qué error tiene cada medida (validada con conductos artificiales); qué sale en los tres µCT; y qué falta por medir. Es el único documento sobre medidas: no hay otro que repita esto.

**A quién va dirigido.** A una persona con base de ingeniería (derivadas, vectores, matrices) que **no sabe nada de endodoncia ni de este código**. Cada término se explica la primera vez. Lo que no está comprobado lleva la etiqueta **[no verificado]**.

**Qué NO está aquí:** cómo se obtiene la máscara del conducto a partir del µCT (→ [PROYECTO.md](PROYECTO.md), apartados 3.1–3.5); cómo se ejecutan los scripts del proyecto (→ [GUIA.md](GUIA.md)); qué dicen los papers (→ [LITERATURA.md](LITERATURA.md)).

**Referencia:** la máscara que se mide es la `canal_solo_conducto.npy` del µCT, obtenida por umbral y **sin validar por ningún experto**. Todo lo medido es «frente a esa máscara».

**Figuras.** Están en `1_Data_Validation/results/medidas/explicacion/Diente_35/` (carpeta local, no va a git) y se regeneran con `figuras_explicacion.py` (apartado 13). Todas son del diente 35, conducto 2.1 (la rama 1 del grupo 2; ese grupo tiene dos conductos, 2.1 y 2.2), salvo que se diga otra cosa.

---

## Índice

- 0. La receta completa en una página
- 1. Los datos de partida
- 2. Paso 1 — Agrupar los trozos de máscara
- 3. Paso 2 — Dividir el grupo en conductos
- 4. Paso 3 — La línea central: el centro de gravedad de cada sección
- 5. Paso 4 — Suavizar, remuestrear y medir la longitud
- 6. Paso 5 — La curvatura: teoría, matemáticas y cálculo
- 7. Paso 6 — Los diámetros
- 8. Cómo leer las figuras de control
- 9. Validación con conductos artificiales
- 10. Resultados en los 3 dientes (16 µm)
- 11. Problemas que aparecieron y cómo se resolvieron
- 12. Límites y lo que falta
- 13. Cómo ejecutarlo

---

## 0. La receta completa en una página

```
máscara 3D binaria del conducto  (canal_solo_conducto.npy, vóxel 16 µm)
   │
   ├─ 1. AGRUPAR      une trozos separados por una rendija pequeña en z             agrupar_conductos
   │
   ├─ 2. DIVIDIR      un camino «de coste mínimo» del ápice al orificio por cada     ramas_por_apices
   │     en conductos conducto; los caminos que pasan por la misma mancha se fusionan fusionar_ramas_misma_seccion
   │
   ├─ 3. CENTRAR      cada punto del camino se mueve al centro de gravedad de la     refinar_linea_central
   │                  sección PERPENDICULAR al conducto (3 iteraciones)             secciones, centroide_seccion
   │
   ├─ 4. SUAVIZAR     Gauss a lo largo del arco (σ = 0,25 mm) → línea central Q     suavizar, remuestrear
   │      y MEDIR     longitud de arco y cuerda                                      longitud_polilinea
   │
   ├─ 5. CURVATURA    κ = |r' × r''| / |r'|³ sobre Q; ángulos                        curvatura, angulo_schneider_3d, giro_maximo
   │
   └─ 6. DIÁMETROS    en cada sección perpendicular a Q: diámetro equivalente,       diametros_seccion, perfil_secciones
                      mayor/menor (barrido de 1°), esfera inscrita, ancho/espesor
```

Los pasos 1–6 están en [`canal_medidas.py`](../1_Data_Validation/analysis/microct/medidas/canal_medidas.py) (las funciones) y se aplican a un diente con [`medir_conductos.py`](../1_Data_Validation/analysis/microct/medidas/medir_conductos.py) (el programa). [`validar_conducto_artificial.py`](../1_Data_Validation/analysis/microct/medidas/validar_conducto_artificial.py) prueba todo con tubos fabricados de medidas exactas, y [`figuras_explicacion.py`](../1_Data_Validation/analysis/microct/medidas/figuras_explicacion.py) dibuja las figuras de este documento.

---

## 1. Los datos de partida

### 1.1 Qué es una «máscara 3D»
Un **µCT** (micro-tomografía computarizada) es un escáner 3D de rayos X para piezas pequeñas, aquí dientes extraídos. Su resultado es un **volumen**: una caja de cubitos llamados **vóxeles** (el «píxel 3D»), cada uno con un valor de gris. En nuestros tres dientes cada vóxel mide **16 µm** (0,016 mm) de lado.

El conducto radicular (el tubito de la raíz por donde pasan el nervio y los vasos) es un hueco: en el µCT se ve oscuro dentro del diente. Un programa anterior (`preprocess_microct.py` + `split_canal.py`, ver PROYECTO.md) lo marcó como una **máscara**: otro volumen del mismo tamaño con un **1** donde hay conducto y un **0** donde no. Esa máscara es la entrada de todo lo que sigue.

- Los ejes están en orden **(z, y, x)**. **z** es el eje largo del diente: en los tres dientes la corona está hacia z alto y la punta de la raíz (el **ápice**) hacia z bajo.
- El corte de la máscara por arriba es el **plano del orificio**: la «puerta de entrada» de los conductos desde la cámara pulpar (la sala grande de la corona). Por debajo del orificio hay conducto; por encima, cámara, que no se mide.
- Las posiciones se escriben en **milímetros**: el vóxel (i, j, k) está centrado en (i·v, j·v, k·v) con v = 0,016 mm.

![La máscara y tres cortes axiales](../1_Data_Validation/results/medidas/explicacion/Diente_35/fig01_mascara_y_cortes.png)

*Figura 1. A la izquierda, la máscara del diente 35 vista de lado (proyección). Cada línea de color es la altura de un **corte axial** (cortar el volumen con un plano horizontal, perpendicular a z). A la derecha, los tres cortes: en cada uno el conducto aparece como varias **manchas** blancas, una por conducto. Las manchas más pequeñas son la sección de cada conducto a esa altura.*

### 1.2 Dos distancias a lo largo del conducto
Una vez calculada la línea central (apartado 4), cada punto se sitúa de dos maneras:
- **s**: longitud de arco medida desde el **orificio** (arriba) hacia el ápice.
- **d**: distancia medida desde el **extremo apical** (el final de la máscara, que por decisión del proyecto es el **foramen apical**, el agujero por donde el conducto sale al hueso) hacia arriba. Es la que se usa en las figuras de perfil: d = 0 es la punta.

---

## 2. Paso 1 — Agrupar los trozos de máscara

**Problema.** La máscara a veces está partida en trozos por rendijas de uno o pocos cortes (un corte suelto donde el conducto «deja de estar encerrado» por la dentina). Si se tratara cada trozo como un conducto distinto saldrían conductos de 1 mm.

**Solución.** Se hace un **cierre morfológico solo en z**: *dilatar* (cada vóxel de conducto invade a sus vecinos en z) y luego *erosionar* (quitar la misma capa). Un hueco de pocos cortes se rellena; lo demás queda igual. Después se etiquetan las **componentes conexas 3D** (trozos unidos, con vecindad de 26): cada componente es un **grupo**, y los grupos se ordenan por volumen (de ahí los nombres «grupo 1», «grupo 2»; **no se corresponden con raíces concretas**).

```python
# canal_medidas.py :: agrupar_conductos
g = int(round(gap_mm / v))                      # 0,13 mm / 0,016 mm = 8 vóxeles
q = g + 2                                       # margen: sin él, la erosión se come los extremos
closed = ndi.binary_closing(np.pad(m, ((q, q), (0, 0), (0, 0))),
                            structure=np.ones((2 * g + 1, 1, 1), bool))[q:-q]   # línea de 17 vóxeles en z
lab, n = ndi.label(closed, structure=np.ones((3, 3, 3), bool))                  # 26 vecinos
```

Detalles que importan:
- `structure=np.ones((2g+1,1,1))` es una «línea» de 17 vóxeles en z: dilata y erosiona 8 vóxeles (0,13 mm) a cada lado, así que rellena huecos de hasta 16 vóxeles (0,26 mm).
- El **relleno de margen** (`np.pad`) se añadió porque, sin él, el borde del array contaba como «fondo» y la erosión se comía unos 4 cortes (≈ 64 µm) de cada extremo del conducto.
- Se descartan grupos de menos de 0,02 mm³.
- El volumen que se mide después es el de **la máscara cerrada** (el grupo), no el original; los vóxeles añadidos son solo las rendijas.

---

## 3. Paso 2 — Dividir el grupo en conductos

Un grupo puede contener **varios conductos** (por ejemplo, los dos conductos de una misma raíz) unidos por puentes o por el embudo del orificio. Hay que dibujar **un camino por cada conducto**. La idea: un conducto es un tubo; si recorres el interior de un tubo de un extremo al otro por el medio, tu camino es la línea central. Para que el camino vaya por el medio se le pone un **precio** a cada vóxel: barato en el centro, caro cerca de la pared.

### 3.1 Distancia a la pared (EDT)
La **transformada de distancia euclídea (EDT)** asigna a cada vóxel de conducto la **distancia al vóxel de pared más cercano**. En el centro de un tubo vale su radio; en la pared, casi 0.

```python
# canal_medidas.py :: edt_con_extension_superior (resumido)
e = ndi.distance_transform_edt(np.pad(big, 1))[1:-1, 1:-1, 1:-1]       # distancia en vóxeles
```

Una sutileza: el conducto está cortado por el plano del orificio, y para la EDT ese plano sería una «pared» que falsearía las distancias del corte de arriba. Por eso la máscara se **prolonga 1 mm hacia arriba** repitiendo su último corte antes de calcular la EDT (`edt_con_extension_superior`) y, más adelante, antes de muestrear secciones (`extender_arriba`). Esa prolongación solo se usa para calcular; nunca se mide.

![EDT, coste y perfil](../1_Data_Validation/results/medidas/explicacion/Diente_35/fig02_distancia_a_la_pared.png)

*Figura 2. Un corte axial ampliado sobre un conducto. Izquierda: la EDT (amarillo = lejos de la pared). Centro: el coste 1/EDT². Derecha: la EDT a lo largo de la línea blanca; es un «tejado» con el pico en el centro (aquí 0,2 mm = el radio del conducto).*

### 3.2 El precio de pasar por cada vóxel y el grafo
Cada vóxel recibe un coste **1 / EDT²** (con EDT mínimo 1 para no dividir por cero): pegado a la pared (EDT = 1) cuesta 1; a 10 vóxeles de la pared cuesta 0,01. Un camino barato **huye de las paredes**. Para ir más rápido se trabaja en una **rejilla gruesa** (cada vóxel grueso = 2×2×2 finos, 32 µm; está dentro si ≥ la mitad de sus vóxeles finos lo están).

Se construye un **grafo**: cada vóxel grueso es un nodo y cada par de vecinos (26 vecinos) está unido por una arista cuyo precio es

> precio(arista) = distancia entre los centros × media de los costes de sus dos extremos

```python
# canal_medidas.py :: ramas_por_apices
cost = 1.0 / np.maximum(edt[coarse], 1.0) ** 2
...
ws.append(np.sqrt(dz * dz + dy * dy + dx * dx) * 0.5 * (cost[r] + cost[c]))
G = coo_matrix((np.concatenate(ws), (np.concatenate(rows), np.concatenate(cols_))), shape=(n, n)).tocsr()
```

### 3.3 El camino más barato (Dijkstra)
**Dijkstra** es el algoritmo clásico del «camino más corto» en un grafo con precios (el mismo que usa un GPS). Se lanza **desde todos los vóxeles del plano del orificio a la vez** (`min_only=True`): para cada vóxel del grupo da el precio mínimo de llegar a él desde el orificio y de qué vecino vino (`pred`). Con eso, **el camino desde cualquier punta hasta el orificio se reconstruye siguiendo los `pred`**.

```python
fuentes = np.flatnonzero(coords[:, 0] == ztop_c)                  # todo el corte superior
D, pred, _ = dijkstra(G, directed=False, indices=fuentes, min_only=True, return_predecessors=True)
```

### 3.4 Elegir las puntas (los ápices)
Cada conducto acaba en una punta. Se buscan así:
1. Se miran solo los **3 mm más bajos** del grupo (`zona_apical_mm`): ahí están los ápices (banda amarilla de la figura 3).
2. Se toma el vóxel **más bajo** que no esté «ocupado», se reconstruye su camino hasta el orificio.
3. Se **ocupa** (se marca como ya usado) todo lo que queda a menos de `1,5 × EDT + 0,05 mm` del camino: así otros vóxeles de la misma punta no cuentan como ápices distintos.
4. El camino se acepta como **conducto nuevo** solo si aporta **≥ 2 mm** de recorrido que no comparte con los anteriores (si no, es un reborde del mismo conducto). Máximo 4 caminos por grupo.
5. Se repite desde el punto 2 hasta que no queden vóxeles libres.

```python
tip = int(cand[np.argmin(z[cand])])                     # el más bajo no ocupado
...
radios = escala * edt[tuple(coords[p].T)] + margen_mm / fv      # bola alrededor del camino (escala = 1,5)
...
if paths and largo_nuevo < min_rama_mm:                  # min_rama_mm = 2,0
    continue                                             # comparte camino: no es otro conducto
```

![Caminos de coste mínimo](../1_Data_Validation/results/medidas/explicacion/Diente_35/fig03_caminos_de_coste_minimo.png)

*Figura 3. Grupo 2 del diente 35, visto de lado en dos proyecciones. Cuadrado = ápice, círculo = orificio, banda amarilla = donde se buscan ápices. Salen 4 caminos; los del mismo color acabarán siendo el mismo conducto (apartado 3.5): 2 conductos.*

### 3.5 Fusionar los caminos que son el mismo conducto
En un conducto **ancho y plano** (una «hoja») varios ápices dan varios caminos paralelos que son **el mismo conducto**. En cambio, dos conductos distintos corren separados y solo se juntan en el embudo del orificio. El criterio: dos caminos son el mismo conducto si, **en los cortes axiales que comparten, caen en la misma mancha en ≥ 60 % de los casos**. Se agrupan con un «unión-búsqueda» (si A≈B y B≈C, los tres son uno) y de cada grupo se conserva **el camino más largo**.

```python
# canal_medidas.py :: fusionar_ramas_misma_seccion (núcleo)
sim[i, j] = np.mean([ci[z] == cj[z] for z in comunes])      # fracción de cortes en la misma mancha
if sim[i, j] >= umbral:                                     # umbral = 0,6
    padre[raiz(i)] = raiz(j)
```

![Fusión de caminos](../1_Data_Validation/results/medidas/explicacion/Diente_35/fig04_fusion_de_caminos.png)

*Figura 4. Cuatro alturas del grupo 2 del diente 35. Cada color es una mancha del corte axial; el número blanco es el camino que pasa por ese punto. Los caminos 0, 2 y 3 caen en la misma mancha (mancha 1) a 3,5, 6 y 8,5 mm → son el mismo conducto. El camino 1 siempre cae en otra mancha → conducto distinto. Resultado: 4 caminos → 2 conductos (2.1 y 2.2).*

### 3.6 Cómo se llaman y qué límites tiene
Los conductos se llaman **G.B** (grupo.rama): «2.1» es la rama 1 del grupo 2 (las ramas se ordenan de más larga a más corta).

Límites (comprobados, ver apartado 11):
- Si dos conductos comparten la misma mancha a lo largo de todo el recorrido (por ejemplo, un conducto ancho con un istmo muy marcado), se cuentan como **uno**.
- Que el programa encuentre 3, 4 o 3 líneas en los tres dientes **no prueba** que cada diente tenga esos conductos: hay que mirarlo en 3D Slicer.

---

## 4. Paso 3 — La línea central: el centro de gravedad de cada sección

El camino de coste mínimo es **un primer dibujo**: es aproximado (rejilla gruesa), tiene esquinas (viene de saltos entre vóxeles) y puede desviarse. La **línea central** de verdad se define así: **en cada punto, el punto está en el centro de la sección del conducto cortada perpendicularmente a la propia línea**. Esto es una definición «autorreferente» (la línea define los planos y los planos definen la línea), y se resuelve **iterando**.

### 4.1 Qué es el centro de gravedad (centroide)
Para una forma plana, el **centroide** (o centro de gravedad, o centro de masas) es el punto donde se equilibraría un recorte de cartulina con esa forma. Si la forma está hecha de píxeles que valen `w` (1 si hay conducto, 0 si no) con posiciones (xᵢ, yᵢ):

> x̄ = Σ wᵢ·xᵢ / Σ wᵢ         ȳ = Σ wᵢ·yᵢ / Σ wᵢ         Área = (Σ wᵢ)·(tamaño de píxel)²

Con la máscara a 75 µm los valores `w` no son solo 0 y 1 sino **fracciones** (0,3 = «30 % del píxel es conducto»): la **etiqueta blanda**. La fórmula es la misma, y por eso el método sirve para µCT y para CBCT.

### 4.2 El plano perpendicular: tangente y marco (T, U, W)
Sea Pᵢ la lista de puntos de la línea. La **tangente** en cada punto es la dirección de la línea ahí:

```python
def tangentes(P):
    T = np.gradient(P, axis=0)                      # diferencia centrada: (P[i+1] - P[i-1]) / 2
    n = np.linalg.norm(T, axis=1, keepdims=True)
    return T / np.maximum(n, 1e-12)                 # vector unitario
```

El **plano perpendicular** en Pᵢ es el que pasa por Pᵢ y es perpendicular a Tᵢ. Para dibujar una imagen 2D en ese plano hace falta una **base** de dos vectores unitarios U y W perpendiculares entre sí y a T:

- Se parte de un U₀ cualquiera perpendicular a T₀ (producto vectorial con un vector auxiliar).
- En cada punto siguiente se **proyecta el U anterior sobre el nuevo plano** (le quitamos su componente a lo largo de la nueva T) y se normaliza. Así el marco **gira lo mínimo** al recorrer el conducto: es el «transporte paralelo». Sin esto, las imágenes de secciones consecutivas saldrían giradas al azar.
- W = T × U (producto vectorial).

```python
# canal_medidas.py :: _base_perpendicular
u = u - np.dot(u, T[i]) * T[i]        # quitar la componente a lo largo de la nueva tangente
u = u / np.linalg.norm(u)
U[i] = u
W[i] = np.cross(T[i], u)
```

Nota: U y W **no tienen nada que ver con vestibular/lingual ni mesial/distal** (los ejes anatómicos); en el µCT de un diente extraído ni se sabe qué lado era el vestibular.

![El plano perpendicular](../1_Data_Validation/results/medidas/explicacion/Diente_35/fig05_plano_perpendicular.png)

*Figura 5. Izquierda: la línea central (negro), la tangente T (rojo) y el plano perpendicular (naranja) en un punto, con sus ejes U (azul) y W (verde). Derecha: la **sección**: lo que vale el volumen sobre ese plano. El conducto aparece como una mancha; la cruz roja es el punto de la línea.*

### 4.3 Leer el volumen sobre el plano (la «sección»)
La sección es una imagen cuadrada de **251 × 251 píxeles** (ventana de ±2 mm, píxel de 16 µm). El píxel (i, j) corresponde al punto 3D

> X(i, j) = P + (i − c)·h·U + (j − c)·h·W        (c = centro de la imagen, h = tamaño de píxel)

y su valor es el del volumen en X, calculado por **interpolación trilineal** (media ponderada de los 8 vóxeles que rodean a X). Por eso, aunque la máscara sea 0/1, la sección tiene valores intermedios en el borde.

```python
# canal_medidas.py :: secciones
pos = P[sl, None, None, :] + A[None, :, :, None] * U[sl, None, None, :] + B[None, :, :, None] * W[sl, None, None, :]
out[sl] = ndi.map_coordinates(vol, (pos / v).reshape(-1, 3).T, order=1, mode="constant", cval=0.0).reshape(-1, m, m)
```

### 4.4 El centroide de «el» conducto
En la sección puede haber varias manchas (el conducto de la línea y sus vecinos). Se toma **solo la mancha que contiene el punto de la línea** (o, si el punto cae fuera por poco, la más cercana a ≤ 3 píxeles), se ensancha 1 píxel para incluir el borde con valores intermedios y se aplica la fórmula del apartado 4.1 con los valores blandos como pesos:

```python
# canal_medidas.py :: centroide_seccion
R = _region_central(sec, c, max_dist_px=3)              # la mancha del centro
Rw = ndi.binary_dilation(R, iterations=1) & (sec > 0)   # + 1 píxel de borde blando
w = np.where(Rw, sec, 0.0)
return ((w * (ii - c)).sum() / tot * paso,              # desplazamiento en U (mm) respecto al punto actual
        (w * (jj - c)).sum() / tot * paso, ...)         # desplazamiento en W
```

El resultado es **cuánto hay que mover el punto** (en U y en W) para que quede en el centro de gravedad de su sección.

### 4.5 Iterar: mover, suavizar los movimientos, repetir
```python
# canal_medidas.py :: refinar_linea_central (esquema; 3 iteraciones)
for it in range(3):
    T = tangentes(P);  U, W = _base_perpendicular(T)
    sec, paso, c = secciones(vol, v, P, T, U, W, mitad_mm)
    mov = [(clip(du, ±0,3 mm), clip(dw, ±0,3 mm)) for cada punto]     # centroide de cada sección
    disp = mov_u[:, None] * U + mov_w[:, None] * W                    # a 3D
    disp = gaussian_filter1d(disp, 0,2 mm / ds)                       # ¡suavizar el campo de movimientos!
    P = P + disp
    P = remuestrear(suavizar(remuestrear(P, ds), ds, 0,05 mm), ds)    # puntos equiespaciados cada 0,04 mm
```

Tres decisiones importantes, todas aprendidas de fallos reales:
- **Suavizar el campo de movimientos** (σ = 0,2 mm). Si cada punto se mueve por su cuenta, dos vecinos pueden cruzarse y la línea hace **bucles** (visto en el diente 52: curvaturas de miles de mm⁻¹).
- **Limitar cada movimiento a ±0,3 mm** por iteración: en conductos planos, el centroide de la sección puede estar lejos y no queremos saltos.
- **Rehacer la punta.** El último tramo del camino inicial se desvía (la EDT cae junto al final). Se recortan los últimos 0,3 mm y se **vuelve a crecer paso a paso** (0,04 mm cada vez) en la dirección de la línea, corrigiendo con el centroide, hasta que la sección se vacía (área < 4 píxeles²) o se llega a 2 mm.

![Recentrado por centroide](../1_Data_Validation/results/medidas/explicacion/Diente_35/fig06_recentrado_por_centroide.png)

*Figura 6. Izquierda: un punto de la línea y cómo se va moviendo (0: camino de coste mínimo; 1–3: iteraciones) hacia el centroide de su sección (cruz roja); la × amarilla es el punto final de la línea. Derecha: la distancia a la línea final baja de ~34 µm (mediana) y ~114 µm (percentil 90) al empezar a ~19 y ~40 µm tras 3 iteraciones. No llega a 0 porque la línea final tiene un suavizado más (σ = 0,25 mm, apartado 5) y la punta se rehace. El desplazamiento máximo por iteración sale 0,30 mm (= el límite) en las tres: algún punto de un conducto en forma de hoja quiere moverse más; ahí el centro de la sección no es una buena guía (ver apartado 10.3).*

**Cuidado con una idea equivocada:** la línea final **no es** el centroide de cada sección. Sigue el camino del conducto, suavizado, y en secciones largas o curvas se separa del centro de la sección (apartado 10.3 da las distancias medidas).

---

## 5. Paso 4 — Suavizar, remuestrear y medir la longitud

### 5.1 Por qué suavizar
La línea tiene ruido de unas pocas micras (el borde de la máscara está pixelado). Para la **longitud** da igual, pero para la **curvatura** (que usa la segunda derivada, apartado 6) es decisivo.

### 5.2 Suavizado gaussiano con «reflexión impar»
Cada coordenada de la línea se sustituye por una **media ponderada de sus vecinos a lo largo del arco** con pesos gaussianos de anchura **σ = 0,25 mm** (con puntos cada 0,04 mm, σ = 6,25 puntos). El problema son los **extremos**: no hay vecinos de un lado. Si se «repite el último valor» (`mode="nearest"`), el extremo se tira hacia dentro. La solución usada es la **reflexión impar**: se inventan puntos fuera del extremo reflejando los de dentro *a través del propio extremo* (P₋ₖ = 2·P₀ − Pₖ), de modo que una recta con pendiente sigue siendo recta.

```python
# canal_medidas.py :: suavizar
head = 2 * P[0] - P[1:pad + 1][::-1]            # reflexión impar en el extremo inicial
tail = 2 * P[-1] - P[-pad - 1:-1][::-1]
Q = ndi.gaussian_filter1d(np.vstack([head, P, tail]), sig, axis=0, mode="nearest")
return Q[pad:-pad]
```

![Suavizado](../1_Data_Validation/results/medidas/explicacion/Diente_35/fig07_suavizado.png)

*Figura 7. Izquierda: una recta con pendiente y ruido de 10 µm. Con `nearest` (naranja) el extremo se desvía; con reflexión impar (azul, la que se usa) sigue la recta. Derecha: el error de posición de un arco con 5 µm de ruido, sin suavizar (gris), con σ = 0,1 mm y con σ = 0,25 mm: son micras, invisibles en un dibujo, pero (apartado 6.5) enormes para la curvatura.*

### 5.3 Remuestrear y longitud
Después de suavizar se **remuestrea** la línea a puntos exactamente equiespaciados cada 0,04 mm (interpolación lineal por longitud de arco). Con eso:

- **Longitud de arco** L = Σ |Pᵢ₊₁ − Pᵢ| (suma de los segmentos).
- **Cuerda** = distancia en línea recta entre el primer y el último punto. Arco/cuerda ≥ 1; cuanto más grande, más curvo.

```python
def longitud_polilinea(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
```

El suavizado acorta un poco la longitud (corta las esquinas); con σ = 0,25 mm en los conductos artificiales el error de longitud es de −0,5 a −0,3 % (µCT) (apartado 9).

---

## 6. Paso 5 — La curvatura: teoría, matemáticas y cálculo

### 6.1 Idea intuitiva
Imagina que recorres el conducto con un coche. La **curvatura** es **cuánto gira el volante por cada milímetro recorrido**: recta = 0; curva suave = poco; curva cerrada = mucho. Se mide en **mm⁻¹**. Su inversa, el **radio de curvatura R = 1/κ** (en mm), es el radio del círculo que mejor imita la curva en ese punto («círculo osculador»): una curva con κ = 0,167 mm⁻¹ se comporta como un arco de círculo de radio 6 mm.

### 6.2 Definición matemática
Sea r(s) la línea central **parametrizada por longitud de arco** s (s = lo recorrido). Su derivada es la **tangente unitaria** T(s) = r′(s). La curvatura es **cuánto cambia la dirección de la tangente por unidad de arco**:

> κ(s) = |dT/ds| = |r″(s)|          (r″ = segunda derivada respecto a s)

Geométricamente: si T gira un ángulo Δθ mientras recorres una distancia Δs, κ ≈ Δθ/Δs. En un círculo de radio R, Δs = R·Δθ y por tanto κ = 1/R (panel izquierdo de la figura 8).

### 6.3 Fórmula para un parámetro cualquiera
Nuestros puntos no están parametrizados por s «exacto», así que se usa la fórmula general para una curva r(t):

> **κ = |r′ × r″| / |r′|³**         (× = producto vectorial)

*Por qué sale así.* Escribe r′ = v·T (v = |r′| la velocidad). Derivando otra vez: r″ = v′·T + v²·κ·N (N = vector normal unitario, el que apunta hacia donde se curva; v′ = cuánto cambia la velocidad). El producto vectorial de r′ con r″ elimina el término paralelo a T (T × T = 0): r′ × r″ = v·T × (v²κ·N) = v³·κ·(T × N). Su módulo es v³·κ, de donde κ = |r′ × r″| / v³.

*Comprobaciones:*
- **Recta:** r″ = 0 → κ = 0.
- **Círculo de radio R:** r(t) = (R cos t, R sin t) → r′ = (−R sin t, R cos t), r″ = (−R cos t, −R sin t) → |r′ × r″| = R², |r′|³ = R³ → **κ = 1/R** ✔.

### 6.4 Cálculo con puntos: diferencias finitas
Con puntos equiespaciados cada ds, las derivadas se aproximan con diferencias entre vecinos. `np.gradient` usa diferencias **centradas**: r′ᵢ ≈ (rᵢ₊₁ − rᵢ₋₁)/(2·ds). Aplicado dos veces da r″:

```python
# canal_medidas.py :: curvatura
def curvatura(P, ds):
    r1 = np.gradient(P, ds, axis=0)                  # r'  (velocidad)
    r2 = np.gradient(r1, ds, axis=0)                 # r'' (aceleración)
    num = np.linalg.norm(np.cross(r1, r2), axis=1)   # |r' × r''|
    den = np.maximum(np.linalg.norm(r1, axis=1) ** 3, 1e-12)
    return num / den
```

Comprobado también con otra fórmula equivalente (|dT/ds|, línea discontinua azul de la figura 9: coincide).

### 6.5 El gran problema: la segunda derivada amplifica el ruido
Derivar es **amplificar el ruido**: una diferencia (rᵢ₊₁ − rᵢ₋₁)/(2·ds) divide el error de posición por ds = 0,04 mm; la segunda derivada lo divide **otra vez**. Un error de posición de solo **5 µm** (0,005 mm) por coordenada produce un error de curvatura del orden de σ_ruido / ds² ≈ 0,005 / 0,0016 ≈ 3 mm⁻¹: **¡18 veces la curvatura verdadera de un arco de 6 mm (0,167 mm⁻¹)!**

Por eso es obligatorio **suavizar antes** (apartado 5). Medido con un arco de radio 6 mm (κ real = 0,1667 mm⁻¹) con 5 µm de ruido (promedio de 10 repeticiones, ignorando 0,5 mm de cada extremo):

| σ del suavizado | 0 (sin) | 0,02 mm | 0,05 mm | 0,1 mm | 0,15 mm | 0,2 mm | **0,25 mm** | 0,35 mm | 0,5 mm |
|---|---|---|---|---|---|---|---|---|---|
| error típico de κ (mm⁻¹) | 2,60 | 2,01 | 0,60 | 0,117 | 0,048 | 0,024 | **0,015** | 0,010 | 0,013 |

![Curvatura: teoría y ruido](../1_Data_Validation/results/medidas/explicacion/Diente_35/fig08_curvatura_teoria.png)

*Figura 8. Izquierda: la idea (Δθ y Δs). Centro: κ a lo largo del arco sin suavizar (gris, inservible) y con σ = 0,05; 0,15; 0,25 mm (cada vez más cerca de la verdad, línea negra). Derecha: error de κ frente a σ (escala logarítmica). Más suavizado = menos ruido, pero a σ grande se empieza a «aplanar» una curva real (el error sube otra vez a σ = 0,5 mm).*

**El suavizado no es gratis:** suavizar **reduce la curvatura máxima medible**. Un conducto con una curva de radio menor que ~σ queda aplanado (Kucher 2020, p. 10 avisa de lo mismo). Por eso **no se copian los parámetros de otros papers** (el σ de Kucher es para otro tamaño de vóxel y otro ruido): se fijó **σ = 0,25 mm** con los conductos artificiales (apartado 9).

### 6.6 La curvatura de un conducto real
![Curvatura real](../1_Data_Validation/results/medidas/explicacion/Diente_35/fig09_curvatura_real.png)

*Figura 9. Diente 35, conducto 2.1. Arriba izquierda: la línea central coloreada por κ (escala de y exagerada). Arriba derecha: κ a lo largo del conducto (gris: los 0,5 mm de cada extremo, que se ignoran al buscar el máximo porque el suavizado y el corte los distorsionan). Abajo: **vista en el plano osculador** en tres puntos (κ bajo, medio y máximo): la línea central (negro) y el círculo de radio R = 1/κ (discontinuo) que mejor la imita. Donde κ es bajo el círculo es grande y se pega a la línea; en el máximo (κ = 1,63 mm⁻¹, R = 0,61 mm) el círculo es pequeño.*

Lo que se ve: los tres puntos tienen un círculo osculador razonable, pero en el máximo la línea tiene una **ondulación pequeña** (longitud de onda ≈ 1 mm) más que una curva real del conducto. Esto confirma la sospecha: **κ máximo sale inflado** (1,0–1,9 mm⁻¹ frente a 0,38–0,47 que mide Kucher en otro diente y con otra definición), probablemente por ondulaciones de la línea central de ese tamaño. **Usa la mediana (0,13–0,41 mm⁻¹) o el percentil 95**, no el máximo.

### 6.7 Los ángulos
Además de κ(s) se calculan dos ángulos, ambos sobre la línea suavizada Q (cada uno **se calcula con la misma función sobre la línea exacta** en los conductos artificiales, así que la validación mide el error de la cadena de medida, no el de la definición):

- **Ángulo de Schneider en 3D** *(operacionalización propuesta por mí; el original es 2D sobre una radiografía)*. Recta 1 = dirección del tramo coronal recto, de 0,3 a 1,3 mm bajo el orificio. C = primer punto donde la línea se aparta **≥ 0,1 mm** de la recta 1. Ángulo = entre la recta 1 y el segmento C → extremo apical.
- **Giro máximo en cuerdas de 1 mm** (estilo Chen/Česaitienė): el ángulo entre los vectores Q(s) − Q(s − 1 mm) y Q(s + 1 mm) − Q(s), máximo a lo largo del conducto.

```python
# angulo_entre: ángulo entre dos vectores a, b
np.degrees(np.arccos(np.clip(np.dot(a, b) / (|a||b|), -1, 1)))
```

También se da la inclinación de la cuerda respecto al eje z del escáner (no anatómica: el diente extraído cae como cae). Qué eje de referencia usar para la inclinación sigue abierto (apartado 12).

---

## 7. Paso 6 — Los diámetros

### 7.1 Primero: ¿qué es «el diámetro» de algo que no es redondo?
Un círculo tiene un solo diámetro. Pero la sección de un conducto es un óvalo, una lágrima o una hoja plana, y entonces hay **varias definiciones razonables** que dan números distintos. Medimos cuatro, siempre **en la sección perpendicular a la línea central** (la imagen de las figuras 5 y 10):

| Medida | Qué es | Cuándo es útil |
|---|---|---|
| **Diámetro equivalente D_eq** | el diámetro del círculo con la **misma área** que la sección | un solo número, robusto; la única medida fiable a 75 µm |
| **Diámetro mayor / menor** (barrido angular) | el mayor y el menor de los diámetros que pasan por el centro | óvalos; lo que usan Česaitienė y otros (a 1° de paso) |
| **Esfera inscrita** | el diámetro de la esfera más grande que cabe dentro del conducto en ese punto | lo que «pasa» una lima redonda; en µCT es ≈ el diámetro **menor** |
| **Ancho y espesor** (elipse de momentos) | ejes de la elipse con la misma dispersión | hojas planas: ancho ≫ espesor |

### 7.2 Qué sección se mide (y por qué una versión anterior se equivocó)
En la ventana de ±2 mm de la sección puede haber **varios conductos** (los dos del grupo 2 del 35 están a ~1 mm uno de otro). El diámetro de **«el» conducto** se mide solo sobre **la mancha que contiene el punto de la línea central**. Antes se anulaba el diámetro si había **cualquier** trozo de otro conducto en la ventana: los conductos con un vecino cerca salían «sin diámetro» y yo lo atribuí a la anatomía. Era un fallo de la regla, corregido el 2026-11-18.

`diametros_seccion` (en `canal_medidas.py`) hace:
1. Binariza la sección (≥ 0,5) y **cierra huecos de 1 píxel** (fragmentos del mismo conducto por el borde blando; a 75 µm no cierra nada, para no unir conductos vecinos).
2. Toma la mancha que contiene el punto de la línea (o la más cercana a ≤ 0,05 mm; si no hay, **no hay diámetro**: «el centro de la línea cae fuera del conducto»).
3. Si esa mancha **toca el borde de la ventana**, está recortada: no hay diámetro.
4. Cuenta cuántas otras manchas ≥ 0,005 mm² hay (`n_otros`, solo informativo).

### 7.3 Diámetro equivalente (la medida principal)
Área A = suma de los valores blandos de la mancha (ensanchada 1 píxel) × (tamaño de píxel)²; el círculo de esa área tiene diámetro

> **D_eq = 2·√(A / π)**

```python
out["d_eq"] = 2 * np.sqrt(tot * paso ** 2 / np.pi)       # tot = suma de pesos de la mancha
```

### 7.4 Diámetros mayor y menor: el barrido angular
Desde el **centroide de la mancha** se lanzan **360 rayos**, uno por grado. En cada uno se avanza en pasos de 0,25 píxel (interpolando el valor) hasta que el valor baja de 0,5 (**la pared**); el cruce se afina por interpolación lineal para tener el radio R(α) con precisión sub-píxel. El diámetro en la dirección α es

> **D(α) = R(α) + R(α + 180°)**       →       **mayor = máx D(α)**, **menor = mín D(α)**

```python
# canal_medidas.py :: radios_angulares y diametros_seccion
frac = (v0 - umbral) / max(v0 - v1, 1e-9)                # dónde cruza 0,5 entre dos muestras
out[a] = (r[k - 1] + frac * paso_r) * paso               # R(α) en mm
D = rad[:180] + rad[180:]                                # D(α) con α = 0…179°
d_mayor, d_menor = np.nanmax(D), np.nanmin(D)
```

**Cuándo el barrido NO vale.** El barrido supone que desde el centro se «ve» todo el contorno (forma **estrellada**). En secciones curvas (media luna, riñón) o en franjas con huecos, algunos rayos salen por donde pueden y el «contorno» cruza la forma (figura 11). Una comprobación lo detecta: el **área del polígono** que forman los 360 radios,

> A_polígono = ½ · Σ R(α)·R(α+1°)·sen(1°)

debe coincidir con el área de la mancha (A) con error ≤ ±12 % (±24 % con píxel de 75 µm). Si no, **se da D_eq pero no mayor/menor** («forma alargada o curva»).

### 7.5 Esfera inscrita
Con la **EDT 3D** (apartado 3.1), el valor en el centro del conducto es el **radio de la esfera más grande que cabe**. Se toma el **máximo de la EDT en los 27 vóxeles vecinos** al punto de la línea (no se interpola: el centro de un tubo es un pico en forma de cono y la interpolación lo recorta), y se resta 0,2 vóxeles de corrección (comprobada con tubos artificiales: sin ella sale 8–12 % corto):

```python
res["d_edt_mm"] = 2 * np.maximum(best - 0.2, 0.0) * v     # best = máx de la EDT en 3×3×3 vóxeles
```

Precisión ±5 % solo en µCT; **a 75 µm no sirve** (errores de −4 a −39 %).

### 7.6 Ancho, espesor y alargamiento
Para una hoja plana ni D_eq ni la esfera dicen bien «lo ancha que es». Se usa la **matriz de covarianza** de las posiciones de los píxeles de la mancha (cuánto se dispersa en cada dirección):

> C = [[σ²ᵢᵢ, σ²ᵢⱼ], [σ²ᵢⱼ, σ²ⱼⱼ]]        autovalores λ₁ ≥ λ₂

Para una elipse llena de semiejes a y b, la varianza a lo largo de un eje es a²/4, así que **el eje completo vale 4·σ = 4·√λ**:

> ancho = 4·√λ₁·h        espesor = 4·√λ₂·h        **alargamiento = √(λ₁/λ₂)**

(1 = redonda; ≥ 4 = franja, istmo u hoja.) En el perfil, `ancho_mm` y `espesor_mm` describen **todo lo que hay en la ventana** (el sistema), mientras que `alargamiento` es de la mancha del conducto de la línea.

### 7.7 Todo junto en una sección real
![Diámetros en una sección](../1_Data_Validation/results/medidas/explicacion/Diente_35/fig10_diametros_seccion_buena.png)

*Figura 10. Diente 35, conducto 2.1, a 0,52 mm del extremo apical. ① La sección muestreada (rojo: umbral 0,5). ② La mancha con sus pesos blandos, el centroide (+) y el círculo de igual área: D_eq = 0,321 mm. ③ El barrido: rayos cada 10°, el polígono rojo (error de área −0,4 %), el mayor (0,432 mm) y el menor (0,239 mm). ④ La esfera inscrita (0,227 mm: se parece al **menor**) y la elipse de momentos (0,43 × 0,24 mm, alargamiento 1,8).*

Un óvalo da **cuatro números distintos** (0,32; 0,43; 0,24; 0,23 mm) y todos son «el diámetro». Hay que decir siempre **cuál** se usa. Para un conducto más plano (a 1 mm) la diferencia es mayor:

![Sección plana](../1_Data_Validation/results/medidas/explicacion/Diente_35/fig10b_diametros_seccion_plana.png)

*Figura 10b. El mismo conducto a 1,0 mm: D_eq = 0,35 mm pero mayor = 0,68 mm y menor = 0,17 mm (alargamiento 3,8). Para un conducto así, «el diámetro» depende por completo de la definición; para una lima redonda manda el menor.*

### 7.8 Cuando no hay diámetro
![Secciones no válidas](../1_Data_Validation/results/medidas/explicacion/Diente_35/fig11_secciones_no_validas.png)

*Figura 11. Izquierda: diente 51, conducto 2.1 a 3 mm. La sección es una **media luna** de 0,05–0,1 mm de grosor y el punto de la línea (×) cae en el lado cóncavo, fuera del conducto (a más de 0,05 mm): **no hay diámetro**. Derecha: diente 35, conducto 1.1 a 5,5 mm. Un conducto ancho en forma de riñón con un agujero: D_eq = 1,09 mm es válido, pero el contorno del barrido (naranja discontinuo) cruza la forma, su área difiere −46 % de la real y por tanto **no se da mayor/menor**. El borde real de la mancha va en cian.*

---

## 8. Cómo leer las figuras de control

Para cada diente, `medir_conductos.py` escribe en `results/medidas/Diente_XX/` tres imágenes y, por conducto, un CSV con una fila cada 0,04 mm.

**`control_linea_central.png`**: la máscara proyectada (de lado y desde arriba) con la línea de cada conducto (círculo = orificio, cuadrado = extremo apical). Sirve para ver que cada línea va por su conducto.

**`control_perfil.png`**: una fila por conducto. *Izquierda*: el diámetro a lo largo del conducto (x = distancia al extremo apical, 0 = la punta). Las cuatro curvas son las cuatro definiciones del apartado 7.1; la azul (equivalente) es la principal. Las **líneas punteadas** separan los tercios apical, medio y coronal. Las **bandas grises** marcan dónde el barrido no es válido (mayor/menor no se dibujan ahí); la curva naranja que sale del dibujo está recortada a 1,6 mm. *Derecha*: la curvatura κ (0 = recto).

![Perfil anotado](../1_Data_Validation/results/medidas/explicacion/Diente_35/fig12_perfil_anotado.png)

*Figura 12. El perfil del conducto 2.1 del 35 anotado. El diámetro equivalente (azul) va de ~0,3 mm en el ápice a ~0,6 mm en el orificio (el conducto se ensancha hacia arriba). El mayor (naranja) tiene picos donde la mancha tiene un apéndice fino.*

**`control_secciones.png`**: una columna por distancia al ápice (0,5; 1; 2 y 3 mm) y una fila por conducto: la sección perpendicular (ventana de 4×4 mm). Rojo = el contorno que mide el diámetro; **× amarilla** = el punto de la línea; **+ roja** = el centro de la sección; **× celeste** = dónde cruza ese plano la línea de **otro** conducto (que se mide en su propia fila); naranja = borde del trozo cuando el barrido no es válido; **título rojo** = sin diámetro (con el motivo); «sin línea» = una mancha del plano a la que no llega ninguna línea. Las figuras anteriores al 2026-11-18 recortaban la ventana y enseñaban diámetros que el perfil descartaba: no usarlas.

---

## 9. Validación con conductos artificiales

Cuatro conductos con curva de radio conocido (R = 6 y 4 mm), inclinados 0–30° respecto a z, uno oval (aspecto 0,5) y uno recto, estrechándose hacia el ápice. A **15 µm** (binario, como el µCT) y a **75 µm** (etiqueta blanda por promedio de bloques de 5×5×5: lo que daría una segmentación perfecta en CBCT). Error de la medida frente a la verdad analítica (rango sobre los 4 casos; la verdad de los ángulos se calcula aplicando la misma función a la línea central exacta, así que mide el error de la cadena de medida, no el de la definición):

| Magnitud | 15 µm | 75 µm |
|---|---|---|
| Longitud de arco | −0,46 … −0,29 % | −1,8 … −0,5 % |
| κ(s), RMSE (verdad 0,17–0,25 mm⁻¹) | 0,004 … 0,022 mm⁻¹ | 0,011 … 0,026 mm⁻¹ |
| Ángulo de Schneider 3D | −0,55 … 0,0 ° | −0,74 … 0,0 ° |
| Giro máximo en 1 mm | −0,0 … +0,6 ° | +0,06 … +0,64 ° |
| Línea central a la verdad (media) | 1 … 7 µm | 2 … 7 µm |
| Diámetro equivalente | −0,5 … +0,8 % | −5,4 … +0,5 % |
| Diámetro mayor / menor (barrido) | −0,6…+3,8 % / −3,7…+4,2 % | −17,3…+1,2 % / −10,4…−0,9 % |
| Esfera inscrita (EDT) | −6,5 … +1,0 % | −38,6 … −4,4 % |

(Diámetros con σ = 0,15 mm; el suavizado casi no les afecta.) Lectura: la longitud, la curvatura y los ángulos se miden bien con una segmentación perfecta, incluso a 75 µm. **El diámetro equivalente es el único diámetro fiable a 75 µm**; el barrido angular se degrada y la esfera inscrita deja de servir (en µCT sí). Límite de la prueba: son tubos de sección elíptica constante, sin hojas ni istmos. A 75 µm el conducto oval y fino (semieje menor de 1 vóxel en el ápice) se rompió en 6 trozos al binarizar a 0,5: **un conducto de menos de ~2 vóxeles de espesor se fragmenta**, y eso es un límite de resolución, no del código.

Además, `diametros_seccion` se probó con secciones sintéticas (un disco de 0,3 mm solo, con un vecino a 0,9 mm, con un vóxel suelto en la esquina, una franja que toca el borde, una media luna) a 16 y 75 µm: el diámetro del disco sale exacto (0,300 mm a 16 µm; 0,306 a 75 µm) con y sin vecino, y la franja y la media luna se rechazan con su motivo. El experimento de ruido de la curvatura (apartado 6.5) es un tubo de radio 6 mm con ruido de 5 µm.

---

## 10. Resultados en los 3 dientes (16 µm)

### 10.1 Longitud, curvatura y ángulos

Línea central de arriba (plano del orificio) al extremo apical de la máscara. Salen **10 líneas** (35: 3, 51: 4, 52: 3). Los «7 conductos» que contó `split_canal.py` son raíces o grupos del análisis del ápice, no conductos: no son comparables (ver 12.1).

| Diente | Cond. | Arco (mm) | Cuerda (mm) | Arco/cuerda | κ máx / p95 (mm⁻¹) | R mín (mm) | Schneider 3D (°) | Giro en 1 mm (°) | Cortes con varias secciones |
|---|---|---|---|---|---|---|---|---|---|
| 35 | 1.1 | 11,17 | 10,82 | 1,03 | 1,48 / 1,08 | 0,68 | 18,2 | 23,7 | 30 % |
| 35 | 2.1 | 10,70 | 9,65 | 1,11 | 1,63 / 1,40 | 0,61 | 38,3 | 59,4 | 48 % |
| 35 | 2.2 | 10,37 | 9,62 | 1,08 | 1,57 / 0,61 | 0,63 | 46,5 | 24,3 | 51 % |
| 51 | 1.1 | 12,06 | 10,84 | 1,11 | 5,70 / 2,00 | 0,18 | 34,7 | 64,2 | 81 % |
| 51 | 1.2 | 11,73 | 10,93 | 1,07 | 1,87 / 1,65 | 0,53 | 38,8 | 45,0 | 81 % |
| 51 | 2.1 | 11,55 | 11,21 | 1,03 | 1,07 / 0,33 | 0,93 | 19,6 | 20,4 | 22 % |
| 51 | 2.2 | 11,66 | 11,12 | 1,05 | 1,70 / 0,46 | 0,59 | 33,0 | 23,0 | 21 % |
| 52 | 1.1 | 11,81 | 11,59 | 1,02 | 1,56 / 0,69 | 0,64 | 9,9 | 21,3 | 5 % |
| 52 | 2.1 | 12,80 | 11,93 | 1,07 | 1,22 / 0,80 | 0,82 | 36,4 | 19,7 | 82 % |
| 52 | 2.2 | 11,99 | 11,44 | 1,05 | 1,03 / 0,76 | 0,97 | 32,8 | 21,4 | 80 % |

### 10.2 Diámetros

**Diámetros** (mm, medidos en la sección perpendicular **del conducto por el que pasa la línea**; los conductos vecinos de la ventana no cuentan; eq = de su área, may/men = barrido angular, edt = esfera inscrita; «—» = no hay valor válido). Todos los conductos a 0,5 y 1 mm del ápice (mediana de ±0,1 mm; `medidas.json` tiene también 2, 3 y 5 mm):

| Diente | Cond. | Cortes con Deq | a 0,5 mm: eq / may / men / edt | a 1 mm: eq / may / men / edt | a 2 mm: eq / may | a 3 mm: eq / may |
|---|---|---|---|---|---|---|
| 35 | 1.1 | 100 % | 0,52 / 1,03 / 0,32 / 0,27 | 0,66 / 0,85 / 0,45 / 0,44 | 0,53 / 0,63 | 0,67 / 0,85 |
| 35 | 2.1 | 100 % | 0,32 / 0,42 / 0,24 / 0,22 | 0,35 / 0,59 / 0,15 / 0,17 | 0,30 / 0,45 | 0,46 / 0,65 |
| 35 | 2.2 | 100 % | 0,31 / 0,52 / 0,16 / 0,15 | 0,28 / 0,43 / 0,17 / 0,17 | 0,37 / 0,69 | 0,34 / 0,47 |
| 51 | 1.1 | 99 % | 0,23 / 0,28 / 0,19 / 0,16 | 0,25 / 0,37 / 0,16 / 0,15 | 0,55 / — | 0,31 / 0,56 |
| 51 | 1.2 | 100 % | 0,19 / 0,34 / 0,11 / 0,06 | 0,33 / 0,60 / 0,20 / 0,19 | 0,49 / 1,01 | 0,36 / 0,61 |
| 51 | 2.1 | 98 % | 0,30 / 0,33 / 0,27 / 0,25 | 0,27 / 0,30 / 0,24 / 0,23 | 0,33 / 0,62 | — / — |
| 51 | 2.2 | 100 % | 0,29 / 0,37 / 0,22 / 0,19 | 0,23 / 0,31 / 0,17 / 0,17 | 0,25 / 0,35 | 0,23 / 0,28 |
| 52 | 1.1 | 100 % | 0,30 / 0,33 / 0,27 / 0,25 | 0,25 / 0,32 / 0,20 / 0,20 | 0,49 / 1,07 | 0,63 / 1,42 |
| 52 | 2.1 | 100 % | 0,33 / 0,35 / 0,31 / 0,29 | 0,32 / 0,36 / 0,28 / 0,28 | 0,40 / 0,46 | 0,45 / 0,56 |
| 52 | 2.2 | 99 % | 0,30 / 0,43 / 0,23 / 0,20 | 0,34 / 0,42 / 0,27 / 0,27 | 0,38 / 0,43 | 0,31 / 0,34 |

**Corrección del 2026-11-18.** La versión anterior de este documento decía que el diámetro «solo está definido donde el corte tiene una única sección» y daba valores solo para 4 conductos a 0,5–1 mm. Era un fallo de mi regla: cualquier trozo de ≥ 1 vóxel de otro conducto en la ventana de medida (±2 mm) anulaba el diámetro, así que los conductos con un vecino cerca (por ejemplo, los dos conductos de un mismo grupo) salían «sin diámetro» y yo lo había atribuido a la anatomía. Ahora hay Deq en el 98–100 % de los cortes de los 10 conductos. Lo que sigue sin definirse bien: **el barrido angular (mayor/menor)** solo es válido si el contorno que dibuja reproduce el área de la sección (tolerancia ±12 %); en secciones curvas o en franja con huecos no lo hace y entonces solo se da el diámetro equivalente. Cortes con mayor/menor válido: 35 → 62 / 98 / 94 % (conductos 1.1 / 2.1 / 2.2); 51 → 58 / 63 / 63 / 99 %; 52 → 49 / 97 / 96 %. Y **el diámetro equivalente de una sección alargada (alargamiento ≥ 4: istmo u hoja) mezcla ancho y espesor**: es el 73 % de los cortes del 52 1.1, el 38–39 % de los del 51 1.1 y 1.2, ≤ 8 % en el resto (columna `alargamiento` de los CSV). Sobre ellas dar anchura y espesor, no un diámetro.

### 10.3 Distancia entre el punto de la línea y el centro de su sección

**Distancia entre el punto de la línea y el centro de su sección** (comprobado el 2026-11-19; `+` roja frente a × amarilla): mediana de 0,01 mm (≈ 1 vóxel) en los 10 conductos salvo el 51 1.1 y 1.2 (0,03–0,04 mm); percentil 90 de 0,03–0,06 mm en los conductos de sección redonda u oval, de 0,18 y 0,31 mm en el 51 1.2 y 1.1 (franjas de istmo) y de 0,11 mm en el 35 1.1 (luz ancha en media luna). La línea no es el centroide de cada sección a propósito: sigue el camino del conducto, suavizado 0,25 mm, y en secciones largas o curvas se separa del centro de la sección.

### 10.4 ¿Queda algún conducto sin línea?

**¿Queda algún conducto sin línea?** Comprobación independiente (`cobertura_cortes_axiales` en `medidas.json`): en cortes axiales cada 0,25 mm, cada trozo de conducto de ≥ 0,01 mm² debe tener un punto de alguna línea a menos de 0,1 mm. Sin línea quedan 33 de 150 trozos en el 35 (3,5 % del área), 6 de 150 en el 51 (1,5 %; los grandes están justo bajo el orificio, en el embudo) y 23 de 160 en el 52 (1,4 %). La mayoría son trozos pequeños (< 0,05 mm²); en el 35 varios se repiten en cortes consecutivos (y ≈ 11 mm, x 3,2–6,3 mm; 0,01–0,04 mm² cada uno), lo que parece una luz muy fina continua sin línea (¿conducto accesorio o aleta?; no se ha mirado). **Los trozos más grandes sin línea están en el 35 (raíz del conducto 1.1), y mirados en el corte axial (2026-11-19) NO son un segundo conducto:** el de 0,43 mm² a 7,15 mm del extremo es la mitad de una sola luz ancha en media luna (~1,8 × 0,7 mm) a la que un agujero interior de la máscara (zona negra) parte en dos trozos; y el de 0,16 mm² a 2,4 mm es uno de dos trozos separados por una rendija finísima que ya no existe a 3,2 mm. (Una versión anterior de este párrafo decía que era «compatible con una segunda luz distal»: era una lectura apresurada mía, sin mirar los cortes.) El mismo efecto da algún trozo en el 52 (0,12 mm² a 2,9 mm). Por eso **no se añaden líneas por cobertura**: la métrica señala flancos y fragmentos de conductos anchos, no conductos que falten. Para decidir cuántos conductos tiene cada diente hace falta mirar el volumen en 3D Slicer. En las figuras de secciones, «sin línea» marca estos trozos.

### 10.5 Control a ojo

**Control a ojo** (con las figuras nuevas, que muestran la ventana completa de 4×4 mm, la misma sección que usa el perfil, y el motivo cuando no hay diámetro): las líneas centrales son suaves y cada una sigue su conducto; las secciones del 52 2.1 y 2.2 y del 35 2.1 y 2.2 son óvalos limpios de 0,3–0,45 mm con el contorno pegado al borde. En el 51, 2.1 a 3 mm la sección es una media luna muy fina (0,05–0,1 mm de grosor) y la línea cae 0,05 mm fuera: no hay diámetro y la figura lo dice. Las figuras de secciones y de perfil anteriores al 2026-11-18 recortaban la ventana a ±0,5 mm y enseñaban diámetros que el perfil descartaba: no usarlas.

### 10.6 Qué se puede creer

| | Qué | Por qué |
|---|---|---|
| **Fiable** | Longitud de arco y cuerda; posición de la línea central; ángulos (con la definición elegida) | Error < 0,5 % y < 1° en los artificiales a 15 µm; a ojo coincide con la máscara |
| **Con cautela** | κ máx | Salen 1,0–1,9 mm⁻¹ (R 0,5–1,0 mm), 2–5× lo que mide Kucher (0,38–0,47; otro diente y otra definición de «máximo»). Casi seguro está inflado por ruido de la línea central, sobre todo cerca del ápice. Usar la mediana (0,13–0,41) o el p95. El 51 1.1 (5,7) es un valor aislado |
| **Con cautela** | Diámetro equivalente (de la sección del conducto de la línea) | Validado en tubos (±1 % a 15 µm). En µCT real puede ser más fino que el de la sección real si el umbral es el de mi script, y en secciones alargadas (istmo, hoja) mezcla ancho y espesor |
| **Con cautela** | Diámetro mayor / menor (barrido angular) | Solo en los cortes donde el contorno del barrido reproduce el área (49–99 % según el conducto) |
| **No definido** | Diámetro de una hoja ancha o de un istmo | Es una lámina, no un tubo. Dar anchura y espesor |
| **No usar a 75 µm** | La esfera inscrita | Error de −4 a −39 % en los artificiales |
| **No comprobado** | Que cada línea sea un conducto real, y que no falte alguno (3, 4 y 3 líneas en 35, 51, 52) | Cortes axiales independientes dan 3–4 secciones grandes por diente a media raíz, en línea con los 3–4 conductos de un primer molar inferior (61 % tres, 36 % cuatro, 1 % cinco; de Pablo 2010). No se sabe si son primeros molares. Los trozos sin línea más grandes del 35 son partes de una luz ancha, no un conducto más (10.4) |

---

## 11. Problemas que aparecieron y cómo se resolvieron

Los ocho están corregidos y los números de arriba son los de después. Quedan aquí porque una medida «aparentemente buena» podía haber salido mal:

1. **El cierre en z comía los extremos** (`agrupar_conductos`): el borde del array contaba como fondo y la erosión quitaba cortes a cada extremo (8 cortes en el conducto artificial, que lo destapó; en los dientes reales calculo ~4 cortes = ~64 µm, **estimado, no medido**). Se rellena con margen antes de cerrar.
2. **El plano del orificio actuaba de pared.** La transformada de distancia del corte superior valía ~1 vóxel y el punto de arranque caía fuera de sitio (264 µm); y en conductos inclinados el corte perpendicular quedaba cortado a la mitad. Se prolonga el conducto hacia arriba (repitiendo el último corte) para la distancia y para muestrear secciones.
3. **La esfera inscrita salía 8–12 % corta** por interpolar el pico de la transformada. Ahora se toma el máximo local de 27 vóxeles con −0,2 vóxeles de corrección (±5 % solo en µCT).
4. **Bucles en la línea central:** recentrar cada punto por separado cruzaba vecinos y dejaba picos de κ de miles de mm⁻¹. Se suaviza el campo de desplazamientos.
5. **Detección de conductos:** el primer método (camino a lo más lejano, con ramas laterales) daba 28 «conductos» falsos en el embudo del orificio; el eje único por grupo caía en la dentina entre dos conductos paralelos. Se sustituyó por ápices + fusión por sección.
6. **(2026-11-18) Regla del diámetro demasiado estricta:** cualquier trozo de otro conducto en la ventana (±2 mm) anulaba el diámetro. Ahora se mide solo el trozo que contiene la línea (`diametros_seccion`). Comprobado con secciones sintéticas (un vecino a 0,9 mm, un vóxel suelto, una franja que toca el borde, una media luna) y con los conductos artificiales (la validación de conductos sueltos da lo mismo que antes).
7. **(2026-11-18) Figuras de control que no coincidían con el perfil:** `control_secciones.png` recortaba a ±0,5 mm de una ventana de ±2 mm (las franjas largas salían cortadas con una «raya» roja falsa) y ponía un Deq que el perfil descartaba. Ahora figura y perfil salen de la misma función, se ve la ventana completa y el motivo si no hay diámetro, y una × celeste marca dónde cruzan el plano las líneas de los otros conductos (para ver que cada conducto de la sección tiene su fila).
8. **(2026-11-18) El barrido angular no sirve para secciones curvas o en franja con huecos:** el «contorno» cruzaba la forma y daba picos (mayor hasta 2,3 mm) en el perfil. Ahora solo se da mayor/menor si el contorno reproduce el área de la sección (±12 %; el doble con píxel ≥ 75 µm). En los conductos artificiales (óvalos, siempre válidos) no cambia nada.

---

## 12. Límites y lo que falta

### 12.1 Límites

- **Número de conductos (10 líneas).** Los «7 conductos» de `split_canal.py` son raíces/grupos del análisis del ápice, no conductos: no son comparables con las 10 líneas (35: 3, 51: 4, 52: 3). Referencia de la literatura para un primer molar inferior: 3 conductos en el 61,3 %, 4 en el 35,7 %, 5 en ≈ 1 % (revisión sistemática de de Pablo et al., J Endod 2010: 41 estudios, 18.781 dientes); en la raíz mesial 2 conductos en el 94,4 % y un istmo en el 54,8 %. Es un dato de referencia, **no una comprobación** (no se sabe si 35, 51 y 52 son primeros molares). Lo que sí se comprobó, sin usar las líneas: cortes axiales cada 1 mm dan 3–4 trozos grandes por diente a media raíz. Pendiente: mirar en 3D Slicer el volumen de cada diente para saber cuántos conductos tiene de verdad.
- **Hojas anchas:** varios conductos son planos (hasta ~3 mm de ancho por ~0,3 mm de espesor); ahí el diámetro equivalente mezcla ancho y espesor (ver «Corrección del 2026-11-18»).
- **Línea central fuera de la luz en secciones finas y curvas:** en el 51 el 20–28 % de los puntos de las líneas 1.1, 1.2 y 2.1 caen a 1–4 vóxeles (≤ 0,07 mm) fuera de la máscara (2.2: 2 %; 35 y 52: ≈ 0–2 %, 1 vóxel de redondeo). Es el centroide de una media luna fina (0,05–0,1 mm de grosor), que cae en el lado cóncavo. Efecto en longitud y ángulos: despreciable (< 0,1 mm de desplazamiento sobre 11 mm); en diámetro: no hay valor en esos cortes. **Corrige la cifra «4–8 % cerca del orificio» de la versión anterior de este documento, que era incorrecta.**
- **κ máximo inflado** (apartado 6.6): la línea central tiene ondulaciones de ~1 mm que el máximo recoge. Tres cosas por hacer: más validación con tubos de curvatura realista (R 2–11 mm, como Kucher), elegir el suavizado por conducto y comparar contra Kucher con la misma definición de «máximo».
- **Solo conducto:** el extremo superior es el plano del orificio (convención de ±0,1 mm); el extremo apical es el final de la máscara (sin confirmar con Isa en 35 y 51).

### 12.2 Lo que falta medir, por etapas

Estado de las etapas del plan de medición (los papers en que se apoya cada una están en [LITERATURA.md](LITERATURA.md); las preguntas para Luis, Isa y Ana están **todas** en [PROYECTO.md](PROYECTO.md), apartado 10):

| Etapa | Qué | Estado |
|---|---|---|
| 0 | Definiciones (longitud, ángulo, eje de referencia, dónde acaba) | **Parcial**: se calculan las variantes (abajo) y falta que Luis e Isa elijan |
| 1 | Herramienta de medida analítica sobre el µCT | **Hecha** (este documento) |
| 2 | Validación con conductos artificiales | **Hecha** (apartado 9) |
| 3 | Degradación con la resolución: medir los 3 dientes a 16, 32, 64 y 75 µm (máscara bajada con etiqueta blanda) | **Pendiente** (es el siguiente paso natural: no necesita red) |
| 4 | Red de segmentación + error de cada medida sobre su predicción frente al µCT | **Pendiente** (ver PROYECTO.md; el diseño de dos etapas para el 36/46 está allí) |
| 5 | CBCT reales de Isa: medir sobre la predicción y pedir a Isa que mida a mano una muestra | **Pendiente** |
| 6 | (solo si hace falta) red que devuelva directamente las medidas | **Pendiente, sin empezar** |

**Etapa 0, definiciones.**
- **Longitud.** Tres definiciones en uso: (i) **arco** de la línea central, del orificio al foramen (Kucher p. 4); (ii) **cuerda** recta (Chen p. 4); (iii) **longitud de trabajo clínica** (desde un punto de referencia de la corona; en el laboratorio se resta 0,5 mm del punto donde se ve la lima en el foramen, Christodoulou p. 2). Se guardan **(i) y (ii)**; (iii) espera a saber qué usa Isa.
- **Ángulo.** Schneider es 2D (radiografía). Hoy se calcula un ángulo **en el espacio** (apartado 6.7). Los ángulos **por planos** (buco-lingual y mesio-distal) **no se calculan**: en un µCT de diente extraído no se sabe qué lado era el vestibular (habría que inferirlo de la corona o anotarlo en la colección).
- **Dónde acaba el conducto.** Ya decidido: del plano del orificio (arriba) al final de la máscara (el foramen). Isa dijo, informalmente y solo para el diente 52, que el final de la máscara coincide con el foramen que ella marcaría; no se confirmó para el 35 ni el 51.
- **Eje de referencia de la inclinación** (petición 4 de Luis):

| Opción | Cómo se define | A favor | En contra |
|---|---|---|---|
| **Eje del escáner (z)** | El eje z del µCT/CBCT | Trivial (es lo que da hoy `inclinacion_cuerda_vs_z_deg`) | En el µCT el diente cae como cae; en el CBCT depende de la cabeza del paciente: no es anatómico |
| **Eje largo del diente** | Recta corona→ápice o eje principal (PCA) de la máscara del diente | Referencia propia del diente (parecida a la de Chen p. 4) | Necesita segmentar el diente completo, que ahora no se hace; en molares de 2 raíces es ambiguo |
| **Tramo recto del conducto** | Dirección del conducto en el tercio coronal | Es la definición original de Schneider (Christodoulou p. 1) | Hay que decidir qué tramo es «recto» (es lo que hace `angulo_schneider_3d`) |
| **Planos anatómicos del paciente** | Buco-lingual y mesio-distal | Es lo que ve el clínico | En µCT no se conoce el lado vestibular |

*Recomendación mía (a discutir con Luis):* medir **κ(s)** (no depende de ningún eje) y los ángulos respecto al **eje largo del diente**, y calcular en paralelo la versión Schneider (tramo recto) para comparar.

**Etapa 3, degradación con la resolución.** Medir cada conducto a 16 µm (referencia), 32 y 64 µm y 75 µm (máscara bajada con etiqueta blanda) y dar el error por medida y por tercio. El tercio apical es el más duro: a 75 µm un conducto de 0,2 mm ocupa unos 2,7 vóxeles. Es una comprobación **sin red**: dice cuánto se puede esperar medir incluso con una segmentación perfecta. La validación con tubos (apartado 9) ya da el orden de magnitud: longitud −1,8…−0,5 %, ángulos < 1°, D_eq −5…+0,5 %; la esfera inscrita no sirve.

**Etapa 6, plan B de Luis (medir a mano y entrenar una red que devuelva las medidas).** *(Propuesta mía.)* No hace falta medir a mano: como el simulador parte de un µCT con medidas analíticas conocidas, esas medidas serían las etiquetas de entrenamiento. Riesgo grande: con 3 dientes la red memorizaría ~10 formas; haría falta variedad (por ejemplo, insertar conductos sintéticos de medidas exactas, cuyo realismo tendría que revisar Isa). Solo merece la pena si la vía analítica no alcanza la precisión necesaria a 75 µm. En los 10 papers revisados no hay ninguno que entrene una red para dar directamente las medidas.

### 12.3 Riesgos específicos de la medición

1. **A 75 µm el diámetro apical es impreciso** y un conducto estrecho puede no verse (una luz de < ~30 µm no se ve ni a 16 µm). La longitud medida sobre la máscara puede quedarse corta respecto al foramen (sin confirmar en 35 y 51).
2. **El CBCT mide menos curvatura y menos longitud que el µCT:** Kucher (p. 8–9, una sola raíz y un solo escáner) da ~−17 % de curvatura máxima y ~−6 % de longitud. La etapa 3 debe cuantificar este sesgo para este caso; no se debe asumir.
3. **El suavizado lo decide todo en la curvatura** (apartado 6.5): no copiar parámetros de otros papers.
4. **Evidencia limitada en los papers:** la mayoría valida con un solo tipo de conducto (distal de molar inferior), con dientes extraídos y sin referencia independiente, o con una referencia hecha por estudiantes (Chen). Ninguno resuelve conductos finos de molares inferiores con varios conductos a 75 µm.
5. **La referencia es la máscara por umbral sin validar por un experto**: el error «frente al µCT» es error frente a esa máscara.

---

## 13. Cómo ejecutarlo

Desde `D:\tfm_claudia\1_Data_Validation`, con el entorno `tfm` (`C:\Users\tfg\.conda\envs\tfm\python.exe`):

```
# medir uno o varios dientes (necesita results/microct/Diente_XX/ con canal_solo_conducto.npy e info.json)
python analysis/microct/medidas/medir_conductos.py results/microct/Diente_35 results/microct/Diente_51 results/microct/Diente_52

# validar con tubos artificiales (varios σ de suavizado)
python analysis/microct/medidas/validar_conducto_artificial.py --suav-mm 0.1 0.15 0.25

# dibujar las figuras de este documento para un diente y un conducto (necesita haber ejecutado medir_conductos antes)
python analysis/microct/medidas/figuras_explicacion.py results/microct/Diente_35 --canal 2.1 --ejemplos Diente_51:2.1:3.0 Diente_35:1.1:5.5
```

| Fichero | Para qué |
|---|---|
| `canal_medidas.py` | Las funciones. Sirven para un volumen binario (µCT) o **blando** (fracción de conducto por vóxel, CBCT): todas reciben el volumen y el tamaño de vóxel `v`. |
| `medir_conductos.py` | El programa: se aplica a una o varias carpetas de diente. Parámetros principales: `--gap-mm 0.13`, `--min-mm3 0.02`, `--coarse 2`, `--min-rama-mm 2.0`, `--max-ramas 4`, `--umbral-fusion 0.6`, `--zona-apical-mm 3.0`, `--ds-mm 0.04`, `--suav-mm 0.25`, `--mitad-mm 2.0`, `--Dk-mm 1.0`, `--tol-mm 0.1`, `--ext-max-mm 2.0`. |
| `validar_conducto_artificial.py` | Fabrica tubos de medidas exactas (a 15 µm binario y a 75 µm blando) y compara. Escribe `results/medidas/validacion_conducto_artificial.json`. |
| `figuras_explicacion.py` | Dibuja las figuras 1–12 de este documento en `results/medidas/explicacion/<diente>/`. |

Salida de `medir_conductos.py` (en `results/medidas/Diente_XX/`, no va a git): `medidas.json` (todo, con los parámetros usados y `cobertura_cortes_axiales`), un `perfil_conducto_G_B.csv` por conducto (una fila cada 0,04 mm: posición, κ, área, diámetros, alargamiento…), y las tres imágenes de control del apartado 8. Cada conducto se llama `G.B` (grupo.rama; los grupos van por volumen, no por raíz).

**Para un diente nuevo** basta con que su carpeta tenga `canal_solo_conducto.npy` e `info.json`; no hay listas que editar. Con ~40 µCT hay que repetir la validación y revisar los parámetros ajustados con solo 3 dientes (0,13 mm de unión, 3 mm de zona apical, 2 mm de rama mínima, 0,6 de fusión, σ = 0,25 mm).
