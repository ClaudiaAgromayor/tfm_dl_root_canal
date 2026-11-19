"""
Medidas analíticas de un conducto a partir de su máscara 3D (sin redes): línea central, longitud, curvatura,
ángulos y diámetro. Funciona sobre una máscara binaria (µCT, 16 µm) o blanda (fracción de conducto por vóxel,
CBCT 75 µm): todas las funciones reciben el volumen y el tamaño de vóxel `v` (mm).

Convenciones
  * Volumen en orden (z, y, x); la corona hacia z ALTO (el llamador voltea si hace falta).
  * Coordenadas de puntos en mm, orden (z, y, x), con el vóxel (i, j, k) centrado en (i·v, j·v, k·v).
  * `s` = longitud de arco desde el ORIFICIO (arriba) hacia el ápice; `d` = distancia a lo largo del conducto
    desde el EXTREMO APICAL (final de la máscara = foramen) hacia arriba.

Qué hace cada paso (explicado con figuras y código en docs/MEDICION.md)
  1. agrupar_conductos: une trozos separados menos de `gap_mm` en z (igual que split_canal.py) y descarta motas.
  2. ramas_por_apices + fusionar_ramas_misma_seccion: un camino de coste mínimo (coste 1/EDT²) del ápice al orificio
     por conducto, en una rejilla gruesa; los caminos que caen en la misma mancha axial se fusionan.
     (ramas_por_grosor es el primer método, TEASAR, descartado: dio ramas falsas en el embudo del orificio.)
  3. refinar_linea_central: recentra cada punto con el centroide de la sección PERPENDICULAR al conducto (secciones,
     centroide_seccion) y rehace la punta.
  4. suavizar (Gauss con reflexión impar) y remuestrear; longitud de arco y cuerda.
  5. curvatura / angulo_schneider_3d / giro_maximo sobre la línea central suavizada.
  6. perfil_secciones / diametros_seccion: diámetro equivalente, barrido angular (cada 1°), esfera inscrita y
     ancho/espesor en cada sección perpendicular.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


# ------------------------------------------------------------------------------------------ geometría básica
def longitud_polilinea(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def remuestrear(P, ds):
    """Polilínea (n,3) a puntos equiespaciados cada `ds` (incluye el último punto)."""
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    keep = np.r_[True, seg > 1e-9]                      # quita puntos repetidos
    s, P = s[keep], P[keep]
    n = max(int(np.ceil(s[-1] / ds)), 1)
    t = np.linspace(0.0, s[-1], n + 1)
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], axis=1)


def suavizar(P, ds, sigma_mm):
    """Gauss a lo largo del arco con reflexión IMPAR en los extremos (conserva tendencias lineales; no encoge
    los extremos como el 'nearest'). P equiespaciada a `ds`."""
    if sigma_mm <= 0 or len(P) < 5:
        return P.copy()
    sig = sigma_mm / ds
    pad = min(int(4 * sig) + 1, len(P) - 1)
    head = 2 * P[0] - P[1:pad + 1][::-1]
    tail = 2 * P[-1] - P[-pad - 1:-1][::-1]
    Q = ndi.gaussian_filter1d(np.vstack([head, P, tail]), sig, axis=0, mode="nearest")
    return Q[pad:-pad]


def tangentes(P):
    T = np.gradient(P, axis=0)
    n = np.linalg.norm(T, axis=1, keepdims=True)
    return T / np.maximum(n, 1e-12)


def curvatura(P, ds):
    """κ(s) = |r' × r''| / |r'|³ en mm⁻¹ (P equiespaciada a ds)."""
    r1 = np.gradient(P, ds, axis=0)
    r2 = np.gradient(r1, ds, axis=0)
    num = np.linalg.norm(np.cross(r1, r2), axis=1)
    den = np.maximum(np.linalg.norm(r1, axis=1) ** 3, 1e-12)
    return num / den


def _base_perpendicular(T):
    """Marco (u, w) perpendicular a cada tangente, por transporte paralelo (continuo a lo largo del conducto)."""
    n = len(T)
    U = np.zeros_like(T)
    W = np.zeros_like(T)
    helper = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(T[0], helper)
    u /= np.linalg.norm(u)
    for i in range(n):
        u = u - np.dot(u, T[i]) * T[i]
        nu = np.linalg.norm(u)
        if nu < 1e-9:                                   # tangente giró 180°: arranque nuevo
            u = np.cross(T[i], helper)
            nu = np.linalg.norm(u)
        u = u / nu
        U[i] = u
        W[i] = np.cross(T[i], u)
    return U, W


# ------------------------------------------------------------------------------------------ 1. agrupar
def agrupar_conductos(mask, v, gap_mm=0.13, min_mm3=0.02):
    """Grupos de trozos unidos (cierre en z de `gap_mm` por cada lado, como split_canal.py).
    Devuelve [{vol (float32 recortado, 0–1), origen (z0, y0, x0 en vóxeles), vol_mm3, z_top, ...}]."""
    sl = ndi.find_objects(mask.astype(np.uint8))[0]
    mrg = 4
    sl = tuple(slice(max(s.start - mrg, 0), min(s.stop + mrg, n)) for s, n in zip(sl, mask.shape))
    m = mask[sl] > 0.5
    g = int(round(gap_mm / v))
    if g > 0:
        # el cierre (dilatar y erosionar) necesita margen de al menos g cortes a cada lado en z: sin él, la erosión
        # se come los extremos del conducto (el borde del array cuenta como fondo)
        q = g + 2
        closed = ndi.binary_closing(np.pad(m, ((q, q), (0, 0), (0, 0))), structure=np.ones((2 * g + 1, 1, 1), bool))[q:-q]
    else:
        closed = m
    lab, n = ndi.label(closed, structure=np.ones((3, 3, 3), bool))
    out = []
    for i in range(1, n + 1):
        gm = lab == i
        vol_mm3 = float((gm & m).sum() * v ** 3)
        if vol_mm3 < min_mm3:
            continue
        zz = np.flatnonzero(gm.any(axis=(1, 2)))
        out.append({"mask": gm, "origen": (sl[0].start, sl[1].start, sl[2].start), "vol_mm3": vol_mm3,
                    "z_top": int(zz.max()), "z_bot": int(zz.min())})
    out.sort(key=lambda d: -d["vol_mm3"])
    return out



def edt_con_extension_superior(mask, ext_vox, ztop=None):
    """Distancia al fondo (en vóxeles) con el conducto PROLONGADO ext_vox cortes por encima del plano del orificio
    (se repite el último corte). Sin esto el plano de corte actúa de pared y la EDT del corte superior vale ~1.
    Devuelve la EDT con la forma de `mask`."""
    m = np.asarray(mask, bool)
    if ztop is None:
        ztop = int(np.flatnonzero(m.any(axis=(1, 2))).max())
    big = np.zeros((m.shape[0] + ext_vox + 1,) + m.shape[1:], bool)
    big[:m.shape[0]] = m
    top = m[ztop]
    for k in range(1, ext_vox + 1):
        if ztop + k < big.shape[0]:
            big[ztop + k] = top
    e = ndi.distance_transform_edt(np.pad(big, 1))[1:-1, 1:-1, 1:-1]
    return e[:m.shape[0]]


def extender_arriba(vol, z_top, mm, v):
    """Copia de `vol` con el último corte (z_top) repetido `mm` hacia arriba. Se usa SOLO para muestrear secciones:
    el plano del orificio es una convención (no una pared) y, si el conducto está inclinado, la sección
    perpendicular cruza ese plano y quedaría cortada a la mitad (centroide sesgado)."""
    k = int(np.ceil(mm / v))
    out = np.zeros((max(vol.shape[0], z_top + 1 + k),) + vol.shape[1:], vol.dtype)
    out[:vol.shape[0]] = vol
    out[z_top + 1:z_top + 1 + k] = vol[z_top]
    return out

# ------------------------------------------------------------------------------------------ 2. ramas (TEASAR)
def _vecinos_mitad():
    offs = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dz, dy, dx) > (0, 0, 0):
                    offs.append((dz, dy, dx))
    return offs                                          # 13 desplazamientos (la mitad de los 26)


def ramas_por_grosor(gm, v, f=2, min_rama_mm=2.0, escala=1.5, margen_mm=0.05, min_descenso_mm=1.0, max_ramas=4):
    """Ramas (orificio → punta) del grupo `gm` (bool recortado), por camino de coste mínimo con coste 1/EDT².
    Trabaja en una rejilla `f` veces más gruesa (rápido). Devuelve lista de polilíneas (n,3) en mm (z,y,x),
    ordenadas de más larga a más corta, y un dict de diagnóstico.
    La primera rama (la más lejana del orificio) se acepta siempre. Una rama lateral solo cuenta como otro conducto
    si su tramo NUEVO mide ≥ min_rama_mm y su punta queda ≥ min_descenso_mm POR DEBAJO del punto donde se separa
    (un conducto baja hacia el ápice; el embudo plano del orificio da ramas laterales que no bajan)."""
    # rejilla gruesa: un vóxel grueso está dentro si ≥ la mitad de sus vóxeles finos lo están
    pad = [(0, (-s) % f) for s in gm.shape]
    g = np.pad(gm, pad)
    cs = (g.shape[0] // f, g.shape[1] // f, g.shape[2] // f)
    coarse = g.reshape(cs[0], f, cs[1], f, cs[2], f).mean(axis=(1, 3, 5)) >= 0.5
    ztop_c = int(np.flatnonzero(coarse.any(axis=(1, 2))).max())
    edt = edt_con_extension_superior(coarse, int(np.ceil(1.0 / (f * v))), ztop_c)   # en vóxeles gruesos
    n = int(coarse.sum())
    idx = -np.ones(coarse.shape, np.int32)
    idx[coarse] = np.arange(n, dtype=np.int32)
    coords = np.argwhere(coarse)                                                  # (n, 3) vóxeles gruesos
    cost = 1.0 / np.maximum(edt[coarse], 1.0) ** 2
    rows, cols, ws = [], [], []
    for dz, dy, dx in _vecinos_mitad():
        a = coarse[max(0, -dz):cs[0] - max(0, dz), max(0, -dy):cs[1] - max(0, dy), max(0, -dx):cs[2] - max(0, dx)]
        b = coarse[max(0, dz):cs[0] - max(0, -dz) or None, max(0, dy):cs[1] - max(0, -dy) or None,
                   max(0, dx):cs[2] - max(0, -dx) or None]
        ia = idx[max(0, -dz):cs[0] - max(0, dz), max(0, -dy):cs[1] - max(0, dy), max(0, -dx):cs[2] - max(0, dx)]
        ib = idx[max(0, dz):cs[0] - max(0, -dz) or None, max(0, dy):cs[1] - max(0, -dy) or None,
                 max(0, dx):cs[2] - max(0, -dx) or None]
        both = a & b
        r, c = ia[both], ib[both]
        step = np.sqrt(dz * dz + dy * dy + dx * dx)
        rows.append(r)
        cols.append(c)
        ws.append(step * 0.5 * (cost[r] + cost[c]))
    G = coo_matrix((np.concatenate(ws), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    # raíz: vóxel del corte superior con mayor EDT
    ztop = coords[:, 0].max()
    top = np.flatnonzero(coords[:, 0] == ztop)
    root = int(top[np.argmax(edt[tuple(coords[top].T)])])
    D, pred = dijkstra(G, directed=False, indices=root, return_predecessors=True)
    ok = np.isfinite(D)
    to_mm = lambda c: (np.asarray(c, float) * f + (f - 1) / 2.0) * v                # centro del bloque grueso → mm
    tree = cKDTree(coords)
    claimed = ~ok.copy()
    on_path = np.zeros(n, bool)
    paths, info = [], {"n_nodos_gruesos": n, "ramas_descartadas_cortas": 0, "ramas_descartadas_sin_descenso": 0,
                       "ramas_descartadas_por_limite": 0}
    fv = f * v
    while True:
        cand = np.flatnonzero(~claimed)
        if len(cand) == 0:
            break
        far = int(cand[np.argmax(D[cand])])
        p = [far]
        while p[-1] != root and pred[p[-1]] >= 0:
            p.append(int(pred[p[-1]]))
        p = np.array(p)
        nuevo = p[~on_path[p]]
        largo_nuevo = longitud_polilinea(coords[nuevo].astype(float)) * fv if len(nuevo) > 1 else 0.0
        radios = (escala * edt[tuple(coords[p].T)] + margen_mm / fv) * 1.0
        for lst in tree.query_ball_point(coords[p].astype(float), r=radios):
            claimed[lst] = True
        claimed[p] = True
        if paths:
            if largo_nuevo < min_rama_mm:
                info["ramas_descartadas_cortas"] += 1
                continue
            hit = np.flatnonzero(on_path[p])
            z_union = coords[p[hit[0]], 0] if len(hit) else coords[p[-1], 0]
            if (z_union - coords[p[0], 0]) * fv < min_descenso_mm:
                info["ramas_descartadas_sin_descenso"] += 1
                continue
            if len(paths) >= max_ramas:
                info["ramas_descartadas_por_limite"] += 1
                continue
        on_path[p] = True
        paths.append(p[::-1])                              # orificio → punta
    out = []
    for p in paths:
        P = to_mm(coords[p])
        out.append(P)
    out.sort(key=lambda P: -longitud_polilinea(P))
    info["paso_grueso_mm"] = fv
    return out, info



def ramas_por_apices(gm, v, f=2, min_rama_mm=2.0, escala=1.5, margen_mm=0.05, zona_apical_mm=3.0, max_ramas=4):
    """Un conducto = un camino ÁPICE → orificio de coste mínimo (coste 1/EDT², así sigue el medio de la luz).
    Los ápices son los puntos más bajos (corona en z alto) del grupo, dentro de los `zona_apical_mm` más apicales.
    Se toma el más bajo, se traza su camino hasta el plano superior (fuente = todo el corte superior), se «ocupa» una
    bola alrededor del camino y se repite. Un ápice cuenta como conducto propio solo si su camino aporta ≥ min_rama_mm
    que NO comparte con los anteriores (si no, es un reborde del mismo conducto). Devuelve polilíneas (n,3) en mm
    (z,y,x) de ARRIBA (orificio) a ABAJO (ápice), la más larga primero, y un dict de diagnóstico."""
    pad = [(0, (-s) % f) for s in gm.shape]
    g = np.pad(gm, pad)
    cs = (g.shape[0] // f, g.shape[1] // f, g.shape[2] // f)
    coarse = g.reshape(cs[0], f, cs[1], f, cs[2], f).mean(axis=(1, 3, 5)) >= 0.5
    ztop_c = int(np.flatnonzero(coarse.any(axis=(1, 2))).max())
    edt = edt_con_extension_superior(coarse, int(np.ceil(1.0 / (f * v))), ztop_c)
    n = int(coarse.sum())
    idx = -np.ones(coarse.shape, np.int32)
    idx[coarse] = np.arange(n, dtype=np.int32)
    coords = np.argwhere(coarse)
    cost = 1.0 / np.maximum(edt[coarse], 1.0) ** 2
    rows, cols_, ws = [], [], []
    for dz, dy, dx in _vecinos_mitad():
        sa = tuple(slice(max(0, -d), cs[k] - max(0, d)) for k, d in enumerate((dz, dy, dx)))
        sb = tuple(slice(max(0, d), cs[k] - max(0, -d)) for k, d in enumerate((dz, dy, dx)))
        both = coarse[sa] & coarse[sb]
        r, c = idx[sa][both], idx[sb][both]
        rows.append(r)
        cols_.append(c)
        ws.append(np.sqrt(dz * dz + dy * dy + dx * dx) * 0.5 * (cost[r] + cost[c]))
    G = coo_matrix((np.concatenate(ws), (np.concatenate(rows), np.concatenate(cols_))), shape=(n, n)).tocsr()
    fuentes = np.flatnonzero(coords[:, 0] == ztop_c)
    D, pred, _ = dijkstra(G, directed=False, indices=fuentes, min_only=True, return_predecessors=True)
    ok = np.isfinite(D)
    fv = f * v
    z = coords[:, 0].astype(float)
    zmin = z[ok].min()
    claimed = ~ok | (z > zmin + zona_apical_mm / fv)
    tree = cKDTree(coords)
    on_path = np.zeros(n, bool)
    paths = []
    info = {"n_nodos_gruesos": n, "paso_grueso_mm": fv, "apices_rechazados_comparten_camino": 0,
            "apices_rechazados_por_limite": 0, "apices_examinados": 0}
    to_mm = lambda cc: (np.asarray(cc, float) * f + (f - 1) / 2.0) * v
    while True:
        cand = np.flatnonzero(~claimed)
        if len(cand) == 0:
            break
        tip = int(cand[np.argmin(z[cand])])
        info["apices_examinados"] += 1
        p = [tip]
        while pred[p[-1]] >= 0:
            p.append(int(pred[p[-1]]))
        p = np.array(p)                                     # ápice → orificio
        hit = np.flatnonzero(on_path[p])
        prefijo = p[:hit[0] + 1] if len(hit) else p
        largo_nuevo = longitud_polilinea(coords[prefijo].astype(float)) * fv if len(prefijo) > 1 else 0.0
        radios = escala * edt[tuple(coords[p].T)] + margen_mm / fv
        for lst in tree.query_ball_point(coords[p].astype(float), r=radios):
            claimed[lst] = True
        claimed[p] = True
        if paths and largo_nuevo < min_rama_mm:
            info["apices_rechazados_comparten_camino"] += 1
            continue
        if len(paths) >= max_ramas:
            info["apices_rechazados_por_limite"] += 1
            continue
        on_path[p] = True
        paths.append(p[::-1])
    out = [to_mm(coords[p]) for p in paths]
    out.sort(key=lambda P: -longitud_polilinea(P))
    info["n_ramas"] = len(out)
    return out, info


def fusionar_ramas_misma_seccion(gm, v, ramas, umbral=0.6):
    """Une los caminos que son el MISMO conducto: en una hoja ancha (una sola luz) cada ápice elige un camino distinto
    a lo ancho, pero todos pasan por la misma sección conexa del corte axial; dos conductos paralelos (dos patas
    unidas por puentes) solo coinciden en el embudo y en los puentes. Dos caminos se funden si, en los cortes que
    comparten, caen en la misma sección conexa en ≥ `umbral` de los casos. Devuelve (lista de grupos de índices,
    similitud máxima entre conductos distintos)."""
    n = len(ramas)
    etiquetas = {}
    conj = []
    for P in ramas:
        pts = np.round(P / v).astype(int)
        s_ = set()
        for z, y, x in pts:
            if z not in etiquetas:
                if not (0 <= z < gm.shape[0]):
                    continue
                etiquetas[z] = ndi.label(gm[z], structure=np.ones((3, 3)))[0]
            L = etiquetas[z]
            y0, y1, x0, x1 = max(y - 2, 0), y + 3, max(x - 2, 0), x + 3
            vec = L[y0:y1, x0:x1]
            vec = vec[vec > 0]
            if len(vec):
                s_.add((int(z), int(np.bincount(vec).argmax())))
        conj.append(s_)
    padre = list(range(n))

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            zi = {z for z, _ in conj[i]}
            zj = {z for z, _ in conj[j]}
            comunes = zi & zj
            if not comunes:
                continue
            ci = {z: l for z, l in conj[i]}
            cj = {z: l for z, l in conj[j]}
            sim[i, j] = sim[j, i] = np.mean([ci[z] == cj[z] for z in comunes])
            if sim[i, j] >= umbral:
                padre[raiz(i)] = raiz(j)
    grupos = {}
    for i in range(n):
        grupos.setdefault(raiz(i), []).append(i)
    out = list(grupos.values())
    entre = max([sim[i, j] for g1 in out for g2 in out if g1 is not g2 for i in g1 for j in g2] or [0.0])
    return out, float(entre)

# ------------------------------------------------------------------------------------------ 3. secciones
def secciones(vol, v, P, T, U, W, mitad_mm=1.0, paso_mm=None):
    """Muestrea el volumen en planos perpendiculares a cada tangente. Devuelve (n, m, m) y el paso (mm).
    El píxel (i, j) está en P + (i-c)·paso·U + (j-c)·paso·W."""
    paso = paso_mm or v
    c = int(round(mitad_mm / paso))
    m = 2 * c + 1
    ax = (np.arange(m) - c) * paso
    A, B = np.meshgrid(ax, ax, indexing="ij")
    out = np.zeros((len(P), m, m), np.float32)
    for k0 in range(0, len(P), 40):
        sl = slice(k0, min(k0 + 40, len(P)))
        pos = P[sl, None, None, :] + A[None, :, :, None] * U[sl, None, None, :] + B[None, :, :, None] * W[sl, None, None, :]
        pos = (pos / v).reshape(-1, 3).T                   # en vóxeles
        out[sl] = ndi.map_coordinates(vol, pos, order=1, mode="constant", cval=0.0).reshape(-1, m, m)
    return out, paso, c


def _region_central(sec, c, umbral=0.5, max_dist_px=None):
    """Componente (≥ umbral) que contiene el centro (o la más cercana al centro). Devuelve máscara bool o None."""
    b = sec >= umbral
    if not b.any():
        return None
    lab, n = ndi.label(b)
    l0 = lab[c, c]
    if l0 == 0:
        d = ndi.distance_transform_edt(lab == 0, return_indices=True)
        dist, (iy, ix) = d[0], d[1]
        l0 = lab[iy[c, c], ix[c, c]]
        if max_dist_px is not None and dist[c, c] > max_dist_px:
            return None
    return lab == l0


def centroide_seccion(sec, c, paso):
    """Centroide (en mm, relativo al centro) y área (mm²) de la región central, ponderando con el valor blando."""
    R = _region_central(sec, c, max_dist_px=3)
    if R is None:
        return None
    Rw = ndi.binary_dilation(R, iterations=1) & (sec > 0)
    w = np.where(Rw, sec, 0.0)
    tot = w.sum()
    if tot <= 0:
        return None
    ii, jj = np.indices(sec.shape)
    return ((w * (ii - c)).sum() / tot * paso, (w * (jj - c)).sum() / tot * paso, tot * paso ** 2, R)


def radios_angulares(sec, c, paso, centro_ij, n_ang=360, umbral=0.5):
    """Radio (mm) hasta la pared cada 360/n_ang grados desde `centro_ij` (en píxeles, puede ser fraccionario).
    NaN si el rayo sale de la imagen antes de cruzar el umbral."""
    ang = np.deg2rad(np.arange(n_ang) * 360.0 / n_ang)
    paso_r = 0.25
    r = np.arange(0.0, c * 1.4, paso_r)
    ci, cj = centro_ij
    I = ci + np.outer(np.cos(ang), r)
    J = cj + np.outer(np.sin(ang), r)
    val = ndi.map_coordinates(sec, [I.ravel(), J.ravel()], order=1, mode="constant", cval=0.0).reshape(I.shape)
    inside = val >= umbral
    out = np.full(n_ang, np.nan)
    if val[0, 0] < umbral:                 # el centro cae fuera de la sección (forma en herradura): barrido no válido
        return out
    for a in range(n_ang):
        below = np.flatnonzero(~inside[a])
        if len(below) == 0:
            continue
        k = below[0]
        if k == 0:
            out[a] = 0.0
            continue
        v0, v1 = val[a, k - 1], val[a, k]
        frac = (v0 - umbral) / max(v0 - v1, 1e-9)
        out[a] = (r[k - 1] + frac * paso_r) * paso
    return out



def diametros_seccion(sec, c, paso, cierre_mm=0.016, max_dist_mm=0.05, min_otro_mm2=0.005, tol_area_barrido=0.12):
    """Diámetros del conducto que pasa por el centro de la sección (el punto de la línea central). Los otros conductos
    de la ventana no cuentan: antes (y por error) cualquier trozo ≥ 1 vóxel en la ventana (±2 mm por defecto) anulaba el
    diámetro, lo que descartaba conductos con un vecino cerca (p. ej. los dos mesiales, a 1–2 mm uno de otro).
    El trozo del conducto es la componente (≥ 0,5) que contiene el centro; se cierran huecos de ≤ 2·cierre_mm (fragmentos
    del mismo conducto por efecto de volumen parcial; a 16 µm es 1 píxel, a 75 µm ninguno: no hay que unir conductos
    vecinos). Si el centro cae a más de max_dist_mm de cualquier conducto, o el trozo toca el borde de la ventana
    (recortado), no hay diámetro y `motivo` lo dice.
    Devuelve dict: motivo (None = válida), d_eq, d_mayor, d_menor (mm, NaN si no), centro_ij (px, fraccionario), rad (360 radios
    en mm), region (bool), n_otros (trozos ≥ min_otro_mm2 distintos del central)."""
    out = {"motivo": None, "d_eq": np.nan, "d_mayor": np.nan, "d_menor": np.nan, "centro_ij": None, "rad": None,
           "region": None, "n_otros": 0, "alargamiento": np.nan, "error_area_barrido": np.nan}
    b = sec >= 0.5
    if not b.any():
        out["motivo"] = "sin conducto en la ventana"
        return out
    cierre_px = int(round(cierre_mm / paso))
    pad = cierre_px + 1
    bc = ndi.binary_closing(np.pad(b, pad), iterations=cierre_px)[pad:-pad, pad:-pad] | b if cierre_px > 0 else b
    lab, n = ndi.label(bc)
    l0 = lab[c, c]
    if l0 == 0:
        dist, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        if dist[c, c] * paso > max_dist_mm + 1e-9:
            out["motivo"] = "el centro de la línea cae fuera del conducto (> 0,05 mm)"
            return out
        l0 = lab[iy[c, c], ix[c, c]]
    R = lab == l0
    out["region"] = R
    areas = ndi.sum(b, lab, range(1, n + 1)) * paso ** 2
    out["n_otros"] = int(sum(1 for k, ar in enumerate(areas, 1) if k != l0 and ar >= min_otro_mm2))
    if R[0].any() or R[-1].any() or R[:, 0].any() or R[:, -1].any():
        out["motivo"] = "el conducto toca el borde de la ventana (recortado)"
        return out
    Rw = ndi.binary_dilation(R, iterations=1) & (sec > 0)
    w = np.where(Rw, sec, 0.0)
    tot = w.sum()
    if tot <= 0:
        out["motivo"] = "sin conducto en la ventana"
        return out
    ii, jj = np.indices(sec.shape)
    di, dj = (w * (ii - c)).sum() / tot, (w * (jj - c)).sum() / tot
    out["d_eq"] = 2 * np.sqrt(tot * paso ** 2 / np.pi)
    out["centro_ij"] = (c + di, c + dj)
    # alargamiento = cociente de los ejes de la elipse con la misma dispersión (1 = redonda; ≥ 4: franja, istmo u hoja)
    pts = np.vstack(np.nonzero(R)).astype(float)
    ev = np.linalg.eigvalsh(np.cov(pts)) if pts.shape[1] > 5 else np.array([0.0, 0.0])
    out["alargamiento"] = float(np.sqrt(ev[1] / max(ev[0], 1e-12)))
    rad = radios_angulares(w, c, paso, out["centro_ij"])
    out["rad"] = rad
    if np.isfinite(rad).sum() >= 300:
        # El barrido solo vale si el contorno que dibuja reproduce el área de la propia sección. En franjas finas o curvas
        # los rayos salen por donde pueden y el «contorno» cruza la forma (comprobado en el 51): ahí no se da mayor/menor.
        a_poli = 0.5 * np.nansum(rad * np.roll(rad, -1)) * np.sin(np.deg2rad(360.0 / len(rad)))
        a_reg = tot * paso ** 2                                  # área blanda de la sección (válida también a 75 µm)
        out["error_area_barrido"] = float(a_poli / a_reg - 1)
        # con píxeles gruesos (≥ 75 µm) un círculo de 0,3 mm ya da −11 % (4 píxeles de ancho): tolerancia doble
        if abs(out["error_area_barrido"]) > tol_area_barrido * (1 if paso <= 0.032 else 2):
            out["motivo"] = (f"forma alargada o curva: el barrido angular no reproduce el área ({100 * out['error_area_barrido']:+.0f} %); "
                             "Deq sí es válido, mayor/menor no")
        else:
            D = rad[:180] + rad[180:]
            out["d_mayor"], out["d_menor"] = float(np.nanmax(D)), float(np.nanmin(D))
    else:
        out["motivo"] = "el centro de la sección cae fuera de la forma (conducto curvo): Deq válido, mayor/menor no"
    return out


def linea_central_por_cortes(gm, v):
    """Eje del SISTEMA de conductos de un grupo: centroide de TODO el grupo en cada corte axial, de arriba (orificio)
    a abajo (ápice). Polilínea (n,3) en mm (z,y,x). Sirve para una raíz con hoja ancha o varios conductos juntos,
    donde el camino de coste mínimo recorre el borde y no el medio."""
    zs = np.flatnonzero(gm.any(axis=(1, 2)))[::-1]
    pts = []
    for z in zs:
        yy, xx = np.nonzero(gm[z])
        pts.append((z * v, yy.mean() * v, xx.mean() * v))
    return np.array(pts, float)


def momentos_seccion(sec, c, paso):
    """Todas las componentes de la sección a la vez (peso = valor blando). Devuelve
    (di, dj, área mm², anchura mm, espesor mm, nº de componentes): centroide relativo al centro (mm); anchura y
    espesor = ejes de la elipse con la misma distribución de masa (4·σ; para una elipse llena son sus ejes)."""
    w = sec.astype(np.float64)
    tot = w.sum()
    if tot < 1e-9:
        return None
    ii, jj = np.indices(sec.shape)
    ci, cj = (w * ii).sum() / tot, (w * jj).sum() / tot
    di, dj = ii - ci, jj - cj
    sxx, syy, sxy = (w * di * di).sum() / tot, (w * dj * dj).sum() / tot, (w * di * dj).sum() / tot
    ev = np.linalg.eigvalsh(np.array([[sxx, sxy], [sxy, syy]]))
    ncomp = int(ndi.label(sec >= 0.5)[1])
    return ((ci - c) * paso, (cj - c) * paso, tot * paso ** 2, 4 * np.sqrt(max(ev[1], 0)) * paso,
            4 * np.sqrt(max(ev[0], 0)) * paso, ncomp)


def sin_retrocesos_z(P, tol_mm=1e-9):
    """Camino orificio → ápice sin tramos que suben (z crece): un conducto baja hacia el ápice; los bucles del
    camino inicial no son anatomía. Solo se conservan los puntos que alcanzan un z nuevo mínimo. Devuelve (P, fracción
    de puntos eliminados). Limitación: un gancho apical de más de 90° (el ápice más alto que un punto anterior) se
    recortaría; se avisa si se elimina más del 5 %."""
    if len(P) < 3:
        return P, 0.0
    zmin_prev = np.minimum.accumulate(P[:-1, 0])
    keep = np.r_[True, P[1:, 0] <= zmin_prev + tol_mm]
    return P[keep], float(1 - keep.mean())

# ------------------------------------------------------------------------------------------ 3. refinar
def refinar_linea_central(vol, v, P0, ds_mm=0.04, suav_inicial_mm=0.1, iteraciones=3, mitad_mm=1.0,
                          ext_apical_max_mm=1.0, z_top_mm=None, trim_apical_mm=0.3, todo=False, desp_suav_mm=0.2):
    """Recentra la línea central con los centroides de las secciones perpendiculares (iterando) y extiende el
    extremo apical hasta que la sección se vacía. Devuelve (P, info)."""
    P0, frac_retro = sin_retrocesos_z(P0)
    P = remuestrear(P0, ds_mm)
    P = suavizar(P, ds_mm, suav_inicial_mm)
    P = remuestrear(P, ds_mm)
    desplaz = []
    for it in range(iteraciones):
        T = tangentes(P)
        U, W = _base_perpendicular(T)
        sec, paso, c = secciones(vol, v, P, T, U, W, mitad_mm)
        mov = np.zeros((len(P), 2))
        for i in range(len(P)):
            r = momentos_seccion(sec[i], c, paso) if todo else centroide_seccion(sec[i], c, paso)
            if r is not None:
                mov[i] = (np.clip(r[0], -0.3, 0.3), np.clip(r[1], -0.3, 0.3))
        disp = mov[:, :1] * U + mov[:, 1:] * W
        # el campo de desplazamientos se suaviza a lo largo del conducto: si cada punto se mueve por su cuenta,
        # los vecinos se cruzan y aparecen bucles (comprobado en el 52)
        disp = ndi.gaussian_filter1d(disp, desp_suav_mm / ds_mm, axis=0, mode="nearest")
        P = P + disp
        desplaz.append(float(np.abs(mov).max()))
        P = remuestrear(suavizar(remuestrear(P, ds_mm), ds_mm, suav_inicial_mm * 0.5), ds_mm)
    # el último tramo del camino se desvía (la EDT cae junto al final): se recorta y se rehace paso a paso,
    # con la dirección tomada de un tramo más largo
    k_trim = int(round(trim_apical_mm / ds_mm))
    if len(P) > k_trim + 12:
        P = P[:-k_trim]
    k_dir = min(int(round(0.4 / ds_mm)), len(P) - 1)
    ext = 0.0
    paso_ext = ds_mm
    ult = P[-1].copy()
    tan = (P[-1] - P[-1 - k_dir])
    tan = tan / np.linalg.norm(tan)
    nuevos = []
    while ext < ext_apical_max_mm:
        cand = ult + tan * paso_ext
        Tt = tan[None, :]
        Uu, Ww = _base_perpendicular(Tt)
        sec, paso, c = secciones(vol, v, cand[None, :], Tt, Uu, Ww, mitad_mm)
        r = momentos_seccion(sec[0], c, paso) if todo else centroide_seccion(sec[0], c, paso)
        if r is None or r[2] < (2 * paso) ** 2:
            break
        cand = cand + np.clip(r[0], -0.05, 0.05) * Uu[0] + np.clip(r[1], -0.05, 0.05) * Ww[0]
        paso_dir = (cand - ult) / np.linalg.norm(cand - ult)
        tan = 0.85 * tan + 0.15 * paso_dir
        tan = tan / np.linalg.norm(tan)
        ult = cand
        nuevos.append(cand)
        ext += paso_ext
    if nuevos:
        P = np.vstack([P, np.array(nuevos)])
    return P, {"desplazamiento_max_por_iteracion_mm": desplaz, "extension_apical_mm": ext, "tramo_apical_rehecho_mm": trim_apical_mm,
               "fraccion_puntos_con_retroceso_eliminados": frac_retro}


# ------------------------------------------------------------------------------------------ 4. perfil
def perfil_secciones(vol, v, P, mitad_mm=1.0, edt3=None, paso_mm=None):
    """Para cada punto de P: área, diámetro equivalente, diámetros mayor/menor por barrido angular (1°),
    diámetro de la esfera inscrita (EDT 3D) y avisos. P debe ser la línea central suavizada y equiespaciada."""
    T = tangentes(P)
    U, W = _base_perpendicular(T)
    sec, paso, c = secciones(vol, v, P, T, U, W, mitad_mm, paso_mm)
    n = len(P)
    res = {k: np.full(n, np.nan) for k in ("area_mm2", "d_eq_mm", "d_mayor_mm", "d_menor_mm", "d_edt_mm", "ancho_mm",
                                          "espesor_mm", "n_componentes")}
    res["toca_borde"] = np.zeros(n, bool)
    res["n_otros_trozos"] = np.zeros(n)
    res["alargamiento"] = np.full(n, np.nan)
    res["motivo_diametro"] = [None] * n
    for i in range(n):
        mo = momentos_seccion(sec[i], c, paso)
        if mo is None:
            res["motivo_diametro"][i] = "sin conducto en la ventana"
            continue
        # área, anchura y espesor describen TODO lo que hay en la ventana (el sistema); n_componentes = trozos ≥ 1 vóxel
        res["area_mm2"][i], res["ancho_mm"][i], res["espesor_mm"][i], res["n_componentes"][i] = mo[2], mo[3], mo[4], mo[5]
        b = sec[i] >= 0.5
        # ¿algo de la sección toca el borde de la imagen? (sección recortada: las medidas del sistema no son fiables)
        res["toca_borde"][i] = bool(b[0].any() or b[-1].any() or b[:, 0].any() or b[:, -1].any())
        # los diámetros son solo los del conducto que pasa por el centro (ver diametros_seccion)
        dd = diametros_seccion(sec[i], c, paso)
        res["d_eq_mm"][i], res["d_mayor_mm"][i], res["d_menor_mm"][i] = dd["d_eq"], dd["d_mayor"], dd["d_menor"]
        res["n_otros_trozos"][i], res["motivo_diametro"][i], res["alargamiento"][i] = dd["n_otros"], dd["motivo"], dd["alargamiento"]
    if edt3 is not None:
        # esfera inscrita: máximo de la EDT 3D en los 27 vóxeles vecinos a la línea central. NO se interpola: el centro
        # de un tubo es un pico (cono) y la interpolación lineal lo recorta ~0,5 vóxeles (sesgo −10 % comprobado con
        # conductos artificiales). Corrección de −0,2 vóxeles: la distancia a los centros de los vóxeles de fondo
        # sobrestima el radio ~0,5 y el vóxel más cercano al centro está ~0,3 desviado. Precisión ±5 % solo en µCT.
        base = np.round(P / v).astype(int)
        best = np.zeros(len(P), np.float32)
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    idx = np.clip(base + np.array([dz, dy, dx]), 0, np.array(edt3.shape) - 1)
                    best = np.maximum(best, edt3[idx[:, 0], idx[:, 1], idx[:, 2]])
        res["d_edt_mm"] = 2 * np.maximum(best - 0.2, 0.0) * v
    res["_sec"], res["_c"], res["_paso"] = sec, c, paso
    return res


# ------------------------------------------------------------------------------------------ 5. ángulos
def angulo_entre(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))


def angulo_schneider_3d(Q, ds, s_ini_mm=0.3, tramo_mm=1.0, tol_mm=0.1):
    """Ángulo de Schneider en 3D (propuesta de operacionalización; Schneider original es 2D):
      recta 1 = dirección del tramo coronal recto: de s_ini a s_ini+tramo (desde el orificio);
      C = primer punto donde la línea central se aparta más de `tol_mm` de la recta 1;
      ángulo = entre la recta 1 (apuntando al ápice) y el segmento C → extremo apical.
    Devuelve dict con el ángulo (°) y la posición de C (mm desde el orificio) o None si el conducto es recto."""
    s = np.arange(len(Q)) * ds
    L = s[-1]
    if L < s_ini_mm + tramo_mm + 2 * ds:
        return {"angulo_deg": None, "motivo": "conducto demasiado corto"}
    a = int(round(s_ini_mm / ds))
    b = int(round((s_ini_mm + tramo_mm) / ds))
    d1 = Q[b] - Q[a]
    d1 /= np.linalg.norm(d1)
    rel = Q[b:] - Q[a]
    dist = np.linalg.norm(rel - np.outer(rel @ d1, d1), axis=1)
    sobre = np.flatnonzero(dist >= tol_mm)
    if len(sobre) == 0:
        return {"angulo_deg": 0.0, "s_C_mm": None, "motivo": "no se aparta de la recta 1 más de tol: conducto recto"}
    k = b + int(sobre[0])
    return {"angulo_deg": angulo_entre(d1, Q[-1] - Q[k]), "s_C_mm": float(s[k]), "tol_mm": tol_mm,
            "tramo_recto_mm": [s_ini_mm, s_ini_mm + tramo_mm]}


def giro_maximo(Q, ds, Dk_mm=1.0):
    """Giro máximo entre dos cuerdas de longitud Dk (estilo Chen / Česaitienė): ángulo entre Q(s)-Q(s-Dk) y
    Q(s+Dk)-Q(s). Devuelve (ángulo máximo °, s del máximo en mm desde el orificio)."""
    k = int(round(Dk_mm / ds))
    if len(Q) < 2 * k + 2:
        return None, None
    ang = np.array([angulo_entre(Q[i] - Q[i - k], Q[i + k] - Q[i]) for i in range(k, len(Q) - k)])
    j = int(np.argmax(ang))
    return float(ang[j]), float((j + k) * ds)
