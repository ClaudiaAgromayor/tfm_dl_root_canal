"""Imágenes para Isa, versión 2: cortes en los que se ve cada conducto entero, de la entrada al final.

NO sustituye a `imagen_para_isa.py` (esa versión sigue generando results/para_isa/). Esta escribe en results/para_isa_v2/.

Qué cambia respecto a la versión 1 (en la 1 el plano se elegía solo por la máscara de la "raíz" del conducto más largo y el corte
perpendicular era simplemente el primero girado 90°, así que en el 35 casi no tocaba conducto):
  1. Se separa la máscara `canal_solo_conducto` en CONDUCTOS COMPLETOS (trozos unidos con la misma holgura en z de 0,13 mm que
     usa split_canal.py). Cada conducto se trata por separado; "A" es el más largo (acaba más cerca de la punta de la raíz).
  2. Para cada conducto se prueba todo plano vertical (que contiene el eje z, corona arriba) girado de 0 a 180° con paso
     `--paso-angulo` (2° por defecto). Todos pasan a menos de `--desplazamiento-mm` (0,1 mm) del final del conducto, para que la
     flecha marque el final de verdad. Criterio de elección (más estricto que "el de más píxeles", que es lo que se pidió): se
     toman los 40 planos con más píxeles de máscara de ESE conducto entre los que cubren al menos el 90 % (`--min-cobertura-rel`)
     de la mayor cobertura de altura posible, y de ellos se elige el que deja más máscara en UN solo trozo conexo. Sin eso, el
     máximo de píxeles favorece planos que recogen la parte ancha cercana a la entrada y rompen en trozos la parte fina (se vio
     en el 35, conducto B: con "más píxeles a secas" salían 7 trozos). El plano de "más píxeles a secas" se imprime también.
  3. Segunda imagen (`_90`): el mejor plano entre los que forman 60°-120° con el primero, solo si contiene al menos
     `--min-frac-90` (30 %) de los píxeles del primero. Si no, no se genera (y se avisa).
  4. Si hay dos conductos: un plano oblicuo con los dos. Se prueban (a) el plano por mínimos cuadrados a la entrada y el final de
     los dos (solo si ningún punto queda a más de `--tol-oblicuo-mm`, 0,3 mm) y (b) los planos que pasan exactamente por los dos
     finales girados hasta ±45° de la vertical. Se genera `_oblicuo.png` solo si el mejor conserva al menos `--min-frac-oblicuo`
     (60 %) de los píxeles del mejor corte vertical de CADA conducto; si no, se avisa y quedan los cortes separados.
  5. Formato de siempre: dos paneles (izquierda sin marcas; derecha con la máscara en azul y flecha amarilla en el final),
     corona arriba, barra de 1 mm, solo el título "Diente X". Recorte: ~2 mm de margen alrededor de la raíz y bajo la punta.
  6. Una imagen ampliada (`_ampliado`) de los últimos 3 mm de cada conducto, en el mismo plano que su primer corte.
  7. `resumen_v2.json` y la tabla de la salida estándar: píxeles de máscara del conducto en cada corte, píxeles que tenía el
     corte equivalente de la versión 1, número de trozos separados (componentes) y cobertura (qué parte de la altura del
     conducto aparece en el corte). Una imagen en la que la máscara sale "a trozos" tiene varios componentes y cobertura baja.

Los píxeles de un conducto = píxeles del plano (1 píxel = 1 vóxel de 16 µm) cuyo vóxel más cercano pertenece a ese conducto.

    python imagen_para_isa_v2.py [--resultados results/microct] [--out results/para_isa_v2] [--antiguo results/para_isa]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--resultados", type=Path, default=HERE.parents[2] / "results" / "microct")
ap.add_argument("--out", type=Path, default=HERE.parents[2] / "results" / "para_isa_v2")
ap.add_argument("--antiguo", type=Path, default=HERE.parents[2] / "results" / "para_isa",
                help="carpeta de la versión 1 (se lee distancias.json para reconstruir sus planos)")
ap.add_argument("--dientes", nargs="*", default=["Diente_35", "Diente_51", "Diente_52"])
ap.add_argument("--paso-angulo", type=float, default=2.0)
ap.add_argument("--desplazamiento-mm", type=float, default=0.1, help="cuánto puede alejarse el plano del final del conducto")
ap.add_argument("--gap-mm", type=float, default=0.13, help="holgura en z para unir trozos en un conducto (como split_canal.py)")
ap.add_argument("--margen-mm", type=float, default=2.0, help="margen alrededor de la raíz y bajo la punta")
ap.add_argument("--semiancho-mm", type=float, default=8.0, help="semiancho del plano muestreado (para contar píxeles)")
ap.add_argument("--ultimos-mm", type=float, default=3.0)
ap.add_argument("--min-cobertura-rel", type=float, default=0.9,
                help="solo se consideran planos que cubren al menos esta fracción de la mejor cobertura de altura posible")
ap.add_argument("--min-frac-90", type=float, default=0.30)
ap.add_argument("--tol-oblicuo-mm", type=float, default=0.3)
ap.add_argument("--min-frac-oblicuo", type=float, default=0.6, help="fracción de píxeles que cada conducto debe conservar respecto a su mejor corte vertical")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------------------------ muestreo de planos
def muestrear(vol, o, u, w, us, ws, modo="lineal"):
    """Valores de `vol` (Z,Y,X) en el plano o + us*u + ws*w (vectores (z,y,x) en vóxeles); filas = ws, columnas = us.

    modo "lineal": interpolación trilineal (vol float o bool -> 0..1); "cercano": etiqueta del vóxel más cercano.
    Fuera del volumen vale 0.
    """
    W_, U_ = np.meshgrid(ws, us, indexing="ij")
    Z = o[0] + W_ * w[0] + U_ * u[0]
    Y = o[1] + W_ * w[1] + U_ * u[1]
    X = o[2] + W_ * w[2] + U_ * u[2]
    sh = vol.shape
    if modo == "cercano":
        zi, yi, xi = (np.rint(q).astype(int) for q in (Z, Y, X))
        ok = (zi >= 0) & (zi < sh[0]) & (yi >= 0) & (yi < sh[1]) & (xi >= 0) & (xi < sh[2])
        out = np.zeros(Z.shape, vol.dtype)
        out[ok] = vol[zi[ok], yi[ok], xi[ok]]
        return out
    z0, y0, x0 = (np.floor(q).astype(int) for q in (Z, Y, X))
    fz, fy, fx = Z - z0, Y - y0, X - x0
    out = np.zeros(Z.shape, np.float32)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                zi, yi, xi = z0 + dz, y0 + dy, x0 + dx
                ok = (zi >= 0) & (zi < sh[0]) & (yi >= 0) & (yi < sh[1]) & (xi >= 0) & (xi < sh[2])
                wt = (fz if dz else 1 - fz) * (fy if dy else 1 - fy) * (fx if dx else 1 - fx)
                v = np.zeros(Z.shape, np.float32)
                v[ok] = vol[zi[ok], yi[ok], xi[ok]]
                out += wt * v
    return out


def vertical(theta):
    """Dirección horizontal (z,y,x) del plano vertical girado `theta` (rad): y = sin, x = cos (como la versión 1)."""
    return np.array([0.0, np.sin(theta), np.cos(theta)])


# ------------------------------------------------------------------------------------------------ dibujo
def barra(ax, px_mm, shape, color="white"):
    L = 1.0 / px_mm
    x0, y0 = 0.05 * shape[1], 0.95 * shape[0]
    ax.plot([x0, x0 + L], [y0, y0], color=color, lw=3, solid_capstyle="butt")
    ax.text(x0 + L / 2, y0 - 0.02 * shape[0], "1 mm", color=color, ha="center", va="bottom", fontsize=11)


def dibujar(path, titulo, img, msk, px_mm, flechas):
    """Dos paneles: izquierda sin marcas; derecha con la máscara semitransparente y una flecha por final de conducto (col, fila)."""
    H, W = img.shape
    fig, axs = plt.subplots(1, 2, figsize=(2 * 6.0 * W / max(H, W) + 0.4, 6.0 * H / max(H, W) + 0.9))
    fig.patch.set_facecolor("black")
    for k, ax in enumerate(axs):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1.3, origin="upper", interpolation="bicubic", aspect="equal")
        if k == 1:
            rgba = np.zeros(img.shape + (4,), np.float32)
            rgba[..., :3] = (0.0, 0.85, 1.0)
            rgba[..., 3] = 0.55 * np.clip(msk, 0, 1)
            ax.imshow(rgba, origin="upper", interpolation="bicubic", aspect="equal")
            for cx, cy in flechas:
                lado = 1 if cx + 1.6 / px_mm < W - 2 else -1          # la flecha viene de la derecha; si no cabe, de la izquierda
                ax.annotate("", xy=(cx + lado * 0.25 / px_mm, cy), xytext=(cx + lado * 1.6 / px_mm, cy),
                            arrowprops=dict(arrowstyle="-|>", color="#ffd400", lw=2.2, shrinkA=0, shrinkB=0))
                ax.plot([cx - 0.9 / px_mm, cx + 0.9 / px_mm], [cy, cy], color="#ffd400", lw=0.8, ls=(0, (4, 3)), alpha=0.85)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
    barra(axs[0], px_mm, img.shape)
    fig.suptitle(titulo, color="white", fontsize=16, y=0.985)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.03)
    fig.savefig(path, dpi=170, facecolor="black")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ medidas de un corte
def medir(lab_pl, g, nlong_vox, v):
    """Píxeles del conducto `g` en un corte (lab_pl = etiquetas del plano, filas = alturas) y qué tan 'a trozos' sale."""
    m = lab_pl == g
    n = int(m.sum())
    cl, k = ndi.label(m, structure=np.ones((3, 3)))
    tam = np.bincount(cl.ravel())[1:] if k else np.array([0])
    grandes = int((tam >= 20).sum())
    filas = int(m.any(axis=1).sum())
    return {"pixeles": n, "componentes_ge20px": grandes, "pct_en_el_mayor": round(100.0 * tam.max() / max(n, 1), 1),
            "cobertura_altura_pct": round(100.0 * min(filas / nlong_vox, 1.0), 1)}


resumen = {}
tabla = []
antiguo = {}
if (a.antiguo / "distancias.json").exists():
    antiguo = json.loads((a.antiguo / "distancias.json").read_text(encoding="utf-8"))

for d in a.dientes:
    folder = a.resultados / d
    info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
    V = info["voxel_mm"]
    sg = 1 if info["orientacion"]["corona_en_z"] == "alto" else -1
    z_or = info["separacion"]["z_orificio_vox"]
    n = d.replace("Diente_", "Diente ")
    print(f"=== {d}", flush=True)
    solo_full = np.load(folder / "canal_solo_conducto.npy")
    sl = ndi.find_objects(solo_full.astype(np.uint8))[0]
    sub = solo_full[sl]
    g8 = int(round(a.gap_mm / V))
    unido = ndi.binary_dilation(sub, structure=np.ones((2 * g8 + 1, 1, 1), bool))
    lab_sub, ng = ndi.label(unido)
    lab_sub = (lab_sub * sub).astype(np.int32)
    del unido
    # --- conductos completos: ordenar del más largo (final más lejos del orificio) al más corto
    grupos = []
    for i in range(1, ng + 1):
        if not (lab_sub == i).any():
            continue
        zz = np.flatnonzero((lab_sub == i).any(axis=(1, 2))) + sl[0].start
        grupos.append({"id_orig": i, "zmin": int(zz.min()), "zmax": int(zz.max()), "vox": int((lab_sub == i).sum())})
    if sg < 0:
        raise SystemExit(f"{d}: corona hacia z bajo; esta versión solo está probada con la corona hacia z alto (los 3 dientes actuales)")
    grupos.sort(key=lambda gg: gg["zmin"])                          # A = el que acaba más abajo (más largo)
    nombres = "ABCDEFGH"
    remap = np.zeros(ng + 1, np.uint8)
    for k, gg in enumerate(grupos):
        gg["nombre"] = nombres[k]
        gg["g"] = k + 1                                            # etiqueta 1..n en el volumen recortado
        remap[gg["id_orig"]] = k + 1
    # --- puntos de entrada/final de cada conducto y finales de cada rama (info.json)
    finales_info = [(r, c) for r in info["apice"]["raices"] for c in r["conductos"]]
    for gg in grupos:
        m = lab_sub == gg["id_orig"]
        for nombre, zsl in (("fin", slice(gg["zmin"] - sl[0].start, gg["zmin"] - sl[0].start + 3)),
                            ("entrada", slice(gg["zmax"] - sl[0].start - 2, gg["zmax"] - sl[0].start + 1))):
            cz = np.argwhere(m[zsl])
            gg[nombre] = np.array([sl[0].start + zsl.start + cz[:, 0].mean(), sl[1].start + cz[:, 1].mean(), sl[2].start + cz[:, 2].mean()])
        gg["finales"] = []                                         # ramas: conductos de info.json cuyo final cae en este grupo
        gg["z_punta_local"] = None
    for r, c in finales_info:
        zf, (yf, xf) = c["z_fin_vox"], c["yx_fin_vox"]
        corte = lab_sub[zf - sl[0].start]
        yy, xx = np.nonzero(corte)
        if len(yy) == 0:
            continue
        k = int(np.argmin((yy + sl[1].start - yf) ** 2 + (xx + sl[2].start - xf) ** 2))
        orig = int(corte[yy[k], xx[k]])
        gg = next((q for q in grupos if q["id_orig"] == orig), None)
        if gg is None:
            continue
        gg["finales"].append(np.array([zf, yf, xf], float))
        tl = zf - sg * int(round(c["mm_a_la_punta_local"] / V))
        gg["z_punta_local"] = tl if gg["z_punta_local"] is None else (min(gg["z_punta_local"], tl) if sg > 0 else max(gg["z_punta_local"], tl))
        gg["mm_punta"] = c["mm_a_la_punta_local"]
    for gg in grupos:
        if not gg["finales"]:
            gg["finales"] = [gg["fin"]]
            gg["z_punta_local"] = int(gg["zmin"] - 0.8 / V)
        gg["finales"].sort(key=lambda q: q[0])
        gg["fin_info"] = gg["finales"][0]                           # final más profundo, como lo da info.json
    # --- recorte de trabajo: z entre la punta más baja - margen y el orificio + margen; yx = todo lo que pueda tocar un plano
    m_mar = int(round(a.margen_mm / V))
    T = int(round(a.semiancho_mm / V))
    z_a = max(min(gg["z_punta_local"] for gg in grupos) - m_mar, 0)
    z_b = z_or + m_mar
    norm = np.load(folder / "norm.npy", mmap_mode="r")
    z_b = min(z_b, norm.shape[0] - 1)
    y_lo, y_hi = max(sl[1].start - T - 4, 0), min(sl[1].stop + T + 4, norm.shape[1])
    x_lo, x_hi = max(sl[2].start - T - 4, 0), min(sl[2].stop + T + 4, norm.shape[2])
    nv = np.asarray(norm[z_a:z_b + 1, y_lo:y_hi, x_lo:x_hi])
    labv = np.zeros(nv.shape, np.uint8)
    zi0, zi1 = max(sl[0].start, z_a), min(sl[0].stop, z_b + 1)
    labv[zi0 - z_a:zi1 - z_a, sl[1].start - y_lo:sl[1].stop - y_lo, sl[2].start - x_lo:sl[2].stop - x_lo] = \
        remap[lab_sub[zi0 - sl[0].start:zi1 - sl[0].start]]
    bin_tot = labv > 0
    off = np.array([z_a, y_lo, x_lo], float)                        # coordenadas completas -> locales: restar `off`
    del solo_full, sub, lab_sub
    esz = np.array([1.0, 0, 0]) * sg                                # sentido "hacia la corona"
    ts = np.arange(-T, T + 1, dtype=np.float32)
    ws_all = np.arange(0, z_b - z_a + 1, dtype=np.float32)          # alturas locales (z) de menor a mayor
    resumen[d] = {"voxel_mm": V, "conductos": {}, "imagenes": []}
    for gg in grupos:
        resumen[d]["conductos"][gg["nombre"]] = {
            "volumen_mm3": round(gg["vox"] * V ** 3, 3), "z_fin_vox": gg["zmin"], "z_entrada_vox": gg["zmax"],
            "finales_yx_vox": [[int(q[1]), int(q[2])] for q in gg["finales"]]}
        print(f"  conducto {gg['nombre']}: {gg['vox'] * V ** 3:.2f} mm³, z {gg['zmin']}-{gg['zmax']}, {len(gg['finales'])} final(es)", flush=True)

    # ---------- búsqueda del mejor plano vertical por conducto (transformada de Radon sobre los recuentos por columna)
    angulos = np.arange(0, 180, a.paso_angulo)
    offs = np.arange(-a.desplazamiento_mm, a.desplazamiento_mm + 1e-9, 0.05) / V if a.desplazamiento_mm > 0 else np.array([0.0])
    tt = np.arange(-T, T + 1)

    def candidatos(gg):
        """Puntuación de cada (ángulo, desplazamiento) con el plano pasando junto al final principal del conducto."""
        mg = labv == gg["g"]
        C = mg.sum(axis=0).astype(np.int32)
        niv = mg[::4]
        z_has = niv.any(axis=(1, 2))
        ey, ex = gg["fin_info"][1] - off[1], gg["fin_info"][2] - off[2]
        out = []
        ang = list(angulos)
        if len(gg["finales"]) >= 2:                                 # conducto con dos ramas: el plano que pasa por los dos finales
            dyx = gg["finales"][1][1:] - gg["finales"][0][1:]
            ang.append(float(np.rad2deg(np.arctan2(dyx[0], dyx[1])) % 180))
        for deg in ang:
            th = np.deg2rad(deg)
            dy, dx = np.sin(th), np.cos(th)
            for o in offs:
                py, px = ey + o * np.cos(th), ex - o * np.sin(th)
                iy, ix = np.rint(py + tt * dy).astype(int), np.rint(px + tt * dx).astype(int)
                ok = (iy >= 0) & (iy < C.shape[0]) & (ix >= 0) & (ix < C.shape[1])
                iy, ix = iy[ok], ix[ok]
                out.append((int(C[iy, ix].sum()), float(niv[:, iy, ix].any(axis=1)[z_has].mean()), float(deg), float(o), (py, px)))
        return out

    def plano_vertical(deg, centro_yx):
        """Parámetros (o, u, w) del plano vertical de ángulo `deg` por `centro_yx` (coordenadas locales)."""
        th = np.deg2rad(deg)
        return np.array([0.0, centro_yx[0], centro_yx[1]]), vertical(th), esz

    def cortar(o, u, w, us, ws):
        """Imagen de grises, máscara total (0..1) y etiquetas del plano; filas ordenadas de arriba (corona) a abajo."""
        img = muestrear(nv, o, u, w, us, ws, "lineal")
        mt = muestrear(bin_tot, o, u, w, us, ws, "lineal")
        lb = muestrear(labv, o, u, w, us, ws, "cercano")
        return img[::-1], mt[::-1], lb[::-1]

    guardadas = {}                                                  # plano elegido por conducto
    for gg in grupos:
        cand = candidatos(gg)
        cand.sort(key=lambda q: -q[0])

        def mayor_trozo(deg, centro, g):
            """Píxeles del trozo conexo más grande del conducto g en el plano (deg, centro)."""
            o_, u_, w_ = plano_vertical(deg, centro)
            lb_ = muestrear(labv, o_, u_, w_, ts, ws_all, "cercano")
            cl_, k_ = ndi.label(lb_ == g, structure=np.ones((3, 3)))
            return int(np.bincount(cl_.ravel())[1:].max()) if k_ else 0

        def elegir(lista, g, k=40):
            """Entre los `k` planos con más píxeles que cubren al menos el 90 % de la mayor cobertura de altura posible,
            el que deja más máscara en UN solo trozo (la máscara no sale a trozos)."""
            if not lista:
                return None
            cmax = max(q[1] for q in lista)
            top = sorted((q for q in lista if q[1] >= a.min_cobertura_rel * cmax), key=lambda q: -q[0])[:k]
            return max(top, key=lambda q: (mayor_trozo(q[2], q[4], g), q[0]))

        mejor = elegir(cand, gg["g"])
        pura = cand[0]                                               # el de más píxeles sin más (para informar)
        # segundo plano ~90°: entre 60° y 120° respecto al primero
        c90 = [q for q in cand if 60 <= ((q[2] - mejor[2]) % 180) <= 120]
        mejor90 = elegir(c90, gg["g"])
        # reparos: el mejor plano de todos, aunque no pase por el final (para informar de lo que se pierde)
        gg["mejor"], gg["mejor90"], gg["pura"] = mejor, mejor90, pura
        gg["cobertura_alternativa"] = max(cand, key=lambda q: (q[1], q[0]))
        print(f"  {gg['nombre']}: más píxeles a secas {pura[2]:.0f}° ({pura[0]} px, cob. {pura[1]*100:.0f}%) | elegido {mejor[2]:.0f}° ({mejor[0]} px, cobertura {mejor[1]*100:.0f}%) | ~90°: "
              + (f"{mejor90[2]:.0f}° ({mejor90[0]} px)" if mejor90 else "ninguno") +
              f" | mayor cobertura posible {gg['cobertura_alternativa'][1]*100:.0f}% ({gg['cobertura_alternativa'][0]} px a {gg['cobertura_alternativa'][2]:.0f}°)", flush=True)

    def recorte_ancho(img, lb, gg):
        """Columnas a mostrar: de la máscara del conducto a la raíz que la rodea (diente contiguo en la mitad baja) + margen."""
        mcols = np.flatnonzero((lb == gg["g"]).any(axis=0))
        if len(mcols) == 0:
            mcols = np.flatnonzero((lb > 0).any(axis=0))
        c_lo, c_hi = int(mcols.min()), int(mcols.max())
        raiz = (img[int(0.55 * img.shape[0]):] > 0.5).any(axis=0)       # filas bajas = raíces
        tol = int(round(0.3 / V))
        lo, hi = c_lo, c_hi
        while lo > 0 and raiz[max(lo - tol, 0):lo].any():
            lo -= 1
        while hi < len(raiz) - 1 and raiz[hi + 1:hi + 1 + tol].any():
            hi += 1
        marg = int(round(a.margen_mm / V))
        return max(lo - marg, 0), min(hi + marg, img.shape[1] - 1)

    # ---------- imágenes por conducto
    r0, c0_ = max(finales_info, key=lambda rc: sg * (z_or - rc[1]["z_fin_vox"]))
    ang_old = antiguo.get(d, {}).get("angulo_corte_grados")
    yx_old = np.array(c0_["yx_fin_vox"], float) - off[1:]
    cuenta_old = {}
    if ang_old is not None:
        for tag, deg in (("corte", ang_old), ("corte_perpendicular", ang_old + 90)):
            o, u, w = plano_vertical(deg, yx_old)
            ws_o = np.arange(0, z_b - z_a + 1, dtype=np.float32)
            ts_o = np.arange(-int(round(8.0 / V)), int(round(8.0 / V)) + 1, dtype=np.float32)
            lb_o = muestrear(labv, o, u, w, ts_o, ws_o, "cercano")[::-1]
            cuenta_old[tag] = {"lb": lb_o, "ts": ts_o}
    else:
        print("  (sin distancias.json de la versión 1: no se comparan píxeles)")

    def contar_viejo(tag, gg, zlim_loc=None, ancho=None):
        """Píxeles del conducto gg en el corte equivalente de la versión 1 (misma definición de píxel)."""
        if tag not in cuenta_old:
            return None
        lb_o = cuenta_old[tag]["lb"]
        filas_z = np.arange(lb_o.shape[0])[::-1]                     # índice de fila -> altura local (z)
        sel = np.ones(lb_o.shape[0], bool)
        cols = np.ones(lb_o.shape[1], bool)
        if zlim_loc is not None:
            sel = (filas_z >= zlim_loc[0]) & (filas_z <= zlim_loc[1])
        if ancho is not None:
            tcol = cuenta_old[tag]["ts"]
            cols = np.abs(tcol) <= ancho / V
        nl = gg["zmax"] - gg["zmin"] + 1
        m = (lb_o[sel][:, cols] == gg["g"])
        cl, k = ndi.label(m, structure=np.ones((3, 3)))
        filas = int(m.any(axis=1).sum())
        return {"pixeles": int(m.sum()), "componentes_ge20px": int((np.bincount(cl.ravel())[1:] >= 20).sum()) if k else 0,
                "cobertura_altura_pct": round(100.0 * min(filas / nl, 1.0), 1)}

    for gg in grupos:
        nm = gg["nombre"]
        gl = gg["g"]
        nl = gg["zmax"] - gg["zmin"] + 1
        for tag, sel in (("corte", gg["mejor"]), ("corte_90", gg["mejor90"])):
            if sel is None:
                resumen[d]["imagenes"].append({"fichero": f"{d}_{nm}_{tag}.png", "generada": False, "motivo": "ningún plano a 60-120° del primero"})
                print(f"  {nm} {tag}: NO generada (ningún plano a 60-120°)")
                continue
            cnt, cob, deg, offset, (py, px) = sel
            if tag == "corte_90" and cnt < a.min_frac_90 * gg["mejor"][0]:
                resumen[d]["imagenes"].append({"fichero": f"{d}_{nm}_{tag}.png", "generada": False, "pixeles_mejor_plano_a_90": cnt,
                                               "motivo": f"menos del {a.min_frac_90 * 100:.0f} % de los píxeles del primero ({gg['mejor'][0]})"})
                print(f"  {nm} {tag}: NO generada ({cnt} px < {a.min_frac_90 * 100:.0f} % de {gg['mejor'][0]})")
                continue
            o, u, w = plano_vertical(deg, (py, px))
            ws = np.arange(0, z_b - z_a + 1, dtype=np.float32)
            # alturas mostradas: punta local - margen ... orificio + margen
            lo_z = int(max(gg["z_punta_local"] - m_mar - z_a, 0))
            ws = ws[lo_z:]
            img, mt, lb = cortar(o, u, w, ts, ws)
            lo, hi = recorte_ancho(img, lb, gg)
            med = medir(lb, gl, nl, V)                                # sobre el plano completo (antes de recortar)
            # flechas: final de cada rama del conducto que quede a <= 0.15 mm del plano
            flechas = []
            for fz in gg["finales"]:
                loc = fz - off
                dist = abs(np.dot(loc[1:] - np.array([py, px]), np.array([np.cos(np.deg2rad(deg)), -np.sin(np.deg2rad(deg))]))) * V
                if dist <= 0.2:
                    col = float(np.dot(loc[1:] - np.array([py, px]), np.array([np.sin(np.deg2rad(deg)), np.cos(np.deg2rad(deg))])) - ts[0])
                    fila = float((ws[-1] - loc[0]) if sg > 0 else (loc[0] - ws[0]))
                    flechas.append((col - lo, fila))
            fich = a.out / f"{d}_{nm}_{tag}.png"
            dibujar(fich, n, img[:, lo:hi + 1], mt[:, lo:hi + 1], V, flechas)
            old = contar_viejo("corte" if tag == "corte" else "corte_perpendicular", gg)
            resumen[d]["imagenes"].append({"fichero": fich.name, "generada": True, "conducto": nm, "angulo_grados": deg,
                                           "desplazamiento_mm_del_final": round(offset * V, 3), "flechas": len(flechas),
                                           **med, "version_1_mismo_conducto": old})
            tabla.append((d, fich.name, med, old))
            print(f"  -> {fich.name}: {med['pixeles']} px, {med['componentes_ge20px']} trozos, {med['pct_en_el_mayor']} % en el mayor, "
                  f"cobertura {med['cobertura_altura_pct']} % | versión 1: {old['pixeles'] if old else '-'} px", flush=True)
            if tag == "corte":
                guardadas[nm] = (deg, (py, px), flechas, lo, hi, ws)
        # ---------- ampliado: últimos 3 mm de este conducto, en el plano de su primer corte
        if nm in guardadas:
            deg, (py, px), _, lo_p, hi_p, ws_full = guardadas[nm]
            o, u, w = plano_vertical(deg, (py, px))
            zf_loc = gg["zmin"] - z_a
            tl_loc = (gg["z_punta_local"] - z_a)
            z_lo = min(zf_loc - int(round(0.8 / V)), tl_loc - int(round(0.8 / V)))
            z_hi = zf_loc + int(round(a.ultimos_mm / V))
            ws = np.arange(max(z_lo, 0), min(z_hi, z_b - z_a) + 1, dtype=np.float32)
            # centro de la imagen: la columna del final del conducto en este plano
            fin_loc = gg["fin_info"] - off
            col_fin = float(np.dot(fin_loc[1:] - np.array([py, px]), np.array([np.sin(np.deg2rad(deg)), np.cos(np.deg2rad(deg))])) - ts[0])
            img, mt, lb = cortar(o, u, w, ts, ws)
            lo = int(max(round(col_fin - 3.0 / V), 0)); hi = int(min(round(col_fin + 3.0 / V), img.shape[1] - 1))
            fila = float(ws[-1] - fin_loc[0])
            med = medir(lb[:, lo:hi + 1], gl, nl, V)
            fich = a.out / f"{d}_{nm}_ampliado.png"
            dibujar(fich, n, img[:, lo:hi + 1], mt[:, lo:hi + 1], V, [(col_fin - lo, fila)])
            # versión 1: mismas alturas, ±3 mm alrededor del plano que pasaba por SU final (t = 0)
            old = contar_viejo("corte", gg, zlim_loc=(max(z_lo, 0), min(z_hi, z_b - z_a)), ancho=3.0)
            resumen[d]["imagenes"].append({"fichero": fich.name, "generada": True, "conducto": nm, "angulo_grados": deg,
                                           "ultimos_mm": a.ultimos_mm, **{k: med[k] for k in ("pixeles", "componentes_ge20px", "pct_en_el_mayor")},
                                           "version_1_mismo_conducto": ({"pixeles": old["pixeles"]} if old else None)})
            tabla.append((d, fich.name, med, old))
            print(f"  -> {fich.name}: {med['pixeles']} px, {med['componentes_ge20px']} trozos | versión 1: {old['pixeles'] if old else '-'} px", flush=True)

    # ---------- plano oblicuo con los dos conductos (A y B). Dos familias; se queda la que conserve más de cada conducto:
    #   (1) el plano ajustado por mínimos cuadrados a la entrada y el final de los dos (4 puntos);
    #   (2) el que pasa exactamente por los dos finales y se gira (hasta ±45° de la vertical) para contener más máscara.
    if len(grupos) >= 2:
        A, B = grupos[0], grupos[1]
        PA_e, PA_f, PB_e, PB_f = A["entrada"] - off, A["fin_info"] - off, B["entrada"] - off, B["fin_info"] - off
        ref = {A["g"]: A["mejor"][0], B["g"]: B["mejor"][0]}      # píxeles del mejor corte vertical de cada uno

        def plano_desde_normal(cen, nrm, puntos):
            wdir = esz - (esz @ nrm) * nrm
            wdir /= np.linalg.norm(wdir)
            udir = np.cross(nrm, wdir)
            udir /= np.linalg.norm(udir)
            pw, pu = (puntos - cen) @ wdir, (puntos - cen) @ udir
            ws_ = np.arange(np.floor(pw.min() - m_mar), np.ceil(pw.max() + m_mar) + 1, dtype=np.float32)
            us_ = np.arange(np.floor(pu.min() - m_mar - 2.0 / V), np.ceil(pu.max() + m_mar + 2.0 / V) + 1, dtype=np.float32)
            return udir, wdir, us_, ws_

        P4 = np.array([PA_e, PA_f, PB_e, PB_f])
        cen4 = P4.mean(axis=0)
        nrm4 = np.linalg.svd(P4 - cen4)[2][2]
        dist4 = np.abs((P4 - cen4) @ nrm4) * V
        opciones = []
        if dist4.max() <= a.tol_oblicuo_mm:
            opciones.append(("4 puntos (mínimos cuadrados)", cen4, nrm4))
        e = PB_f - PA_f
        e = e / np.linalg.norm(e)
        n0 = np.cross(e, esz)
        if np.linalg.norm(n0) > 1e-6:
            n0 /= np.linalg.norm(n0)
            n1 = np.cross(e, n0)
            for fi in np.deg2rad(np.arange(-45, 46, 3.0)):
                opciones.append((f"por los dos finales, girado {np.rad2deg(fi):+.0f}°", (PA_f + PB_f) / 2, n0 * np.cos(fi) + n1 * np.sin(fi)))
        evals = []
        for nombre_op, cen, nrm in opciones:
            udir, wdir, us_, ws_ = plano_desde_normal(cen, nrm, P4)
            lbq = muestrear(labv, cen, udir, wdir, us_, ws_, "cercano")
            pa, pb = int((lbq == A["g"]).sum()), int((lbq == B["g"]).sum())
            evals.append((min(pa / max(ref[A["g"]], 1), pb / max(ref[B["g"]], 1)), nombre_op, cen, nrm, pa, pb))
        evals.sort(key=lambda q: -q[0])
        info_obl = {"distancias_de_los_4_puntos_al_plano_mm": [round(float(x), 3) for x in dist4],
                    "mejor_opcion": evals[0][1] if evals else None,
                    "fraccion_minima_conservada": round(float(evals[0][0]), 3) if evals else None,
                    "px_A_B_mejor_opcion": [evals[0][4], evals[0][5]] if evals else None,
                    "px_A_B_mejores_cortes_verticales": [ref[A["g"]], ref[B["g"]]]}
        if evals and evals[0][0] >= a.min_frac_oblicuo:
            _, nombre_op, cen, nrm, _, _ = evals[0]
            udir, wdir, us_, ws_ = plano_desde_normal(cen, nrm, P4)
            img, mt, lb = cortar(cen, udir, wdir, us_, ws_)
            medA = medir(lb, A["g"], A["zmax"] - A["zmin"] + 1, V)
            medB = medir(lb, B["g"], B["zmax"] - B["zmin"] + 1, V)
            flechas = [(float(((pt - cen) @ udir) - us_[0]), float(ws_[-1] - ((pt - cen) @ wdir))) for pt in (PA_f, PB_f)]
            fich = a.out / f"{d}_oblicuo.png"
            dibujar(fich, n, img, mt, V, flechas)
            oldA, oldB = contar_viejo("corte", A), contar_viejo("corte", B)
            resumen[d]["imagenes"].append({"fichero": fich.name, "generada": True, "plano_oblicuo": info_obl, "opcion": nombre_op,
                                           "conducto_A": medA, "conducto_B": medB, "version_1_corte_A": oldA, "version_1_corte_B": oldB})
            tabla.append((d, fich.name + " [A]", medA, oldA))
            tabla.append((d, fich.name + " [B]", medB, oldB))
            print(f"  -> {fich.name} ({nombre_op}): A {medA['pixeles']} px ({medA['componentes_ge20px']} trozos), "
                  f"B {medB['pixeles']} px ({medB['componentes_ge20px']} trozos)", flush=True)
        else:
            resumen[d]["imagenes"].append({"fichero": f"{d}_oblicuo.png", "generada": False, "plano_oblicuo": info_obl,
                                           "motivo": f"ningún plano oblicuo conserva al menos el {a.min_frac_oblicuo * 100:.0f} % de los píxeles "
                                                     f"del mejor corte vertical de cada conducto; se dejan los cortes separados"})
            print(f"  oblicuo: NO generado (mejor opción: {info_obl['mejor_opcion']}, conserva {info_obl['fraccion_minima_conservada']} "
                  f"de lo mejor de cada conducto; px A,B = {info_obl['px_A_B_mejor_opcion']} frente a {info_obl['px_A_B_mejores_cortes_verticales']}); "
                  f"4 puntos a {info_obl['distancias_de_los_4_puntos_al_plano_mm']} mm del plano")
    del nv, labv

(a.out / "resumen_v2.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
print("\nTABLA: píxeles de máscara del conducto en el corte (nuevo | versión 1), trozos (≥20 px) y cobertura de la altura del conducto")
for d, f, med, old in tabla:
    print(f"{f:44s} nuevo {med['pixeles']:6d} px, {med['componentes_ge20px']:2d} trozos"
          + (f", cob. {med['cobertura_altura_pct']:5.1f} %" if 'cobertura_altura_pct' in med else "")
          + (f" | v1 {old['pixeles']:6d} px, {old['componentes_ge20px']:2d} trozos, cob. {old['cobertura_altura_pct']:5.1f} %" if old else " | v1 -"))
