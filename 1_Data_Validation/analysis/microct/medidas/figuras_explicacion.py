"""
Figuras para docs/EXPLICACION_MEDIDAS_DETALLADA.md: cada paso de la medida del conducto dibujado con los datos reales de
UN diente (la máscara solo-conducto del µCT, `canal_solo_conducto.npy`) y de UN conducto de ese diente.

Uso (después de haber ejecutado medir_conductos.py sobre el mismo diente, porque se leen sus CSV):
  python figuras_explicacion.py results/microct/Diente_35 --canal 2.1 \
         --ejemplos Diente_51:2.1:3.0 Diente_35:1.1:5.5
Salida: results/medidas/explicacion/Diente_35/figNN_*.png (no va a git: es un resultado local).

`--canal G.B` es el identificador de medir_conductos.py (grupo.rama). `--ejemplos diente:G.B:d` son secciones «malas» que
se dibujan aparte (d = mm desde el extremo apical). Todo se calcula con las MISMAS funciones de canal_medidas.py que las
medidas: aquí solo se dibuja.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import canal_medidas as cm  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Ellipse, Circle  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
RESULTS = HERE.parents[2] / "results"


# ------------------------------------------------------------------------------------------ carga
def preparar(folder: Path, cid: str, medidas_dir: Path, con_ramas: bool, a) -> dict:
    """Reconstruye lo que hace medir_conductos.procesar para el conducto `cid` (grupo.rama)."""
    info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
    v = float(info["voxel_mm"])
    solo = np.load(folder / "canal_solo_conducto.npy")
    if info.get("orientacion", {}).get("corona_en_z", "alto") == "bajo":
        solo = solo[::-1]
    gi, bi = (int(t) for t in cid.split("."))
    grupos = cm.agrupar_conductos(solo, v, 0.13, 0.02)
    g = grupos[gi - 1]
    org = np.array(g["origen"]) * v
    csv_path = medidas_dir / folder.name / f"perfil_conducto_{gi}_{bi}.csv"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    tabla = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    Qt = np.stack([tabla["z_mm"], tabla["y_mm"], tabla["x_mm"]], axis=1)       # línea final, coordenadas del diente
    ctx = {"folder": folder, "dte": folder.name, "cid": cid, "v": v, "solo": solo, "g": g, "org": org, "tabla": tabla,
           "Qg": Qt - org, "d": tabla["d_desde_apice_mm"], "z_top_mm": g["z_top"] * v}
    ctx["vol"] = cm.extender_arriba(g["mask"].astype(np.float32), g["z_top"], 1.0, v)   # como medir_rama
    if con_ramas:
        ramas, rd = cm.ramas_por_apices(g["mask"], v, f=a.coarse, min_rama_mm=2.0, zona_apical_mm=3.0, max_ramas=4)
        conj = None
        if len(ramas) > 1:
            conj, _ = cm.fusionar_ramas_misma_seccion(g["mask"], v, ramas, 0.6)
        finales = ramas if conj is None else [max((ramas[i] for i in gr), key=cm.longitud_polilinea) for gr in conj]
        orden = sorted(range(len(finales)), key=lambda i: -cm.longitud_polilinea(finales[i]))
        ctx.update({"ramas": ramas, "conj": conj, "P0": finales[orden[bi - 1]], "rd": rd})
    return ctx


def seccion_en(ctx, d0, mitad=2.0):
    """Sección perpendicular a la línea en el punto a d0 mm del extremo apical (mismo marco que el perfil)."""
    Q = ctx["Qg"]
    T = cm.tangentes(Q)
    U, W = cm._base_perpendicular(T)
    k = int(np.argmin(np.abs(ctx["d"] - d0)))
    sec, paso, c = cm.secciones(ctx["vol"], ctx["v"], Q[k:k + 1], T[k:k + 1], U[k:k + 1], W[k:k + 1], mitad)
    return k, sec[0], paso, c, (T, U, W)


def guardar(fig, out: Path, nombre: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / nombre, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("   ", nombre)


# ------------------------------------------------------------------------------------------ fig 01: la máscara
def fig01(ctx, out):
    solo, v = ctx["solo"], ctx["v"]
    m = solo.astype(bool)
    zs = np.flatnonzero(m.any(axis=(1, 2)))
    ys = np.flatnonzero(m.any(axis=(0, 2)))
    xs = np.flatnonzero(m.any(axis=(0, 1)))
    zb, zt = zs.min(), zs.max()
    alturas = [1.5, 5.0, 9.0]
    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1, 1, 1])
    ax0 = fig.add_subplot(gs[0])
    proj = m.any(axis=2)[zb:zt + 1, ys.min():ys.max() + 1]
    ax0.imshow(proj, cmap="gray_r", origin="lower", extent=(ys.min() * v, (ys.max() + 1) * v, zb * v, (zt + 1) * v), aspect="equal")
    for k, h in enumerate(alturas):
        ax0.axhline(zb * v + h, color=f"C{k}", lw=1.5)
        ax0.text(ys.max() * v, zb * v + h + 0.1, f"{h:g} mm", color=f"C{k}", ha="right")
    ax0.axhline((zt + 1) * v, color="k", ls=":")
    ax0.text(ys.min() * v, (zt + 1) * v + 0.1, "plano del orificio", fontsize=8)
    ax0.set_xlabel("y (mm)")
    ax0.set_ylabel("z (mm): eje largo del diente")
    ax0.set_title("Máscara del conducto vista de lado\n(cada línea de color = un corte axial)")
    for k, h in enumerate(alturas):
        zi = int(round(zb + h / v))
        sl = solo[zi, max(ys.min() - 12, 0):ys.max() + 13, max(xs.min() - 12, 0):xs.max() + 13]
        n = ndi.label(sl)[1]
        ax = fig.add_subplot(gs[k + 1])
        y0, x0 = max(ys.min() - 12, 0), max(xs.min() - 12, 0)
        ax.imshow(sl, cmap="gray", origin="lower", extent=(x0 * v, (x0 + sl.shape[1]) * v, y0 * v, (y0 + sl.shape[0]) * v), vmin=0, vmax=1)
        ax.set_title(f"Corte axial a {h:g} mm del ápice\n({n} trozos de conducto; cada píxel = {v * 1000:.0f} µm)", color=f"C{k}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    fig.suptitle(f"{ctx['dte']}: la máscara 3D es un cubo de vóxeles 0/1; cortarla a una altura da una imagen 2D con varias manchas", y=1.02)
    guardar(fig, out, "fig01_mascara_y_cortes.png")


# ------------------------------------------------------------------------------------------ fig 02: distancia a la pared
def fig02(ctx, out, h_mm=6.0):
    g, v = ctx["g"], ctx["v"]
    e = cm.edt_con_extension_superior(g["mask"], int(np.ceil(1.0 / v)), g["z_top"])
    zi = g["z_bot"] + int(round(h_mm / v))
    sl, es = g["mask"][zi], e[zi]
    # se amplía el conducto más ancho de este corte (el de mayor EDT)
    iy, ix = np.unravel_index(np.argmax(np.where(sl, es, 0)), sl.shape)
    r = int(0.7 / v)
    y0, y1 = max(iy - r, 0), min(iy + r, sl.shape[0])
    x0, x1 = max(ix - r, 0), min(ix + r, sl.shape[1])
    emm = np.where(sl, es * v, np.nan)[y0:y1, x0:x1]
    cost = np.where(sl, 1.0 / np.maximum(es, 1.0) ** 2, np.nan)[y0:y1, x0:x1]
    ext = (x0 * v, x1 * v, y0 * v, y1 * v)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"wspace": 0.55})
    im = ax[0].imshow(emm, origin="lower", extent=ext, cmap="viridis")
    plt.colorbar(im, ax=ax[0], label="distancia a la pared (mm)", shrink=0.8)
    ax[0].set_title("EDT: para cada píxel de conducto,\ndistancia a la pared más cercana")
    im = ax[1].imshow(np.log10(cost), origin="lower", extent=ext, cmap="magma_r")
    plt.colorbar(im, ax=ax[1], label="log10 del coste 1/EDT²", shrink=0.8)
    ax[1].set_title("Coste de pasar por cada píxel\n(oscuro = caro = pegado a la pared)")
    ax[0].axhline(iy * v, color="w", ls="--", lw=0.8)
    fila = np.where(sl, es * v, 0.0)[iy, x0:x1]
    xx = np.arange(x0, x1) * v
    ax[2].plot(xx, fila, "k.-")
    ax[2].set_xlabel("x (mm) a lo largo de la línea blanca del primer panel")
    ax[2].set_ylabel("EDT (mm)")
    ax[2].set_title("Perfil de la EDT a través del conducto:\nes máxima en el medio y 0 en la pared")
    ax[2].grid(alpha=0.3)
    for a_ in ax[:2]:
        a_.set_xlabel("x (mm)")
        a_.set_ylabel("y (mm)")
    fig.suptitle(f"{ctx['dte']}, grupo {ctx['cid'].split('.')[0]}: corte axial a {h_mm:g} mm del ápice del grupo (ampliado sobre su conducto más ancho)", y=1.04)
    guardar(fig, out, "fig02_distancia_a_la_pared.png")


# ------------------------------------------------------------------------------------------ fig 03: caminos
def fig03(ctx, out):
    g, v, ramas, conj = ctx["g"], ctx["v"], ctx["ramas"], ctx["conj"]
    gm = g["mask"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 7.4))
    zb = g["z_bot"] * v
    for k, (nom, proj, idx) in enumerate((("y", gm.any(axis=2), 1), ("x", gm.any(axis=1), 2))):
        ys = np.flatnonzero(proj.any(axis=0))
        sub = proj[:, ys.min():ys.max() + 1]
        ax[k].imshow(sub, cmap="gray_r", alpha=0.35, origin="lower", aspect="equal",
                     extent=(ys.min() * v, (ys.max() + 1) * v, 0, sub.shape[0] * v))
        ax[k].axhspan(zb, zb + 3.0, color="gold", alpha=0.2, lw=0)
        ax[k].axhline(g["z_top"] * v, color="k", ls=":")
        for i, P in enumerate(ramas):
            gr = [j for j, lst in enumerate(conj) if i in lst][0] if conj is not None else 0
            ax[k].plot(P[:, idx], P[:, 0], color=f"C{gr}", lw=1.2, alpha=0.85)
            ax[k].plot(P[-1, idx], P[-1, 0], "s", color=f"C{gr}", ms=6)
            ax[k].plot(P[0, idx], P[0, 0], "o", color=f"C{gr}", ms=5)
        ax[k].set_xlabel(f"{nom} (mm)")
        ax[k].set_ylabel("z (mm)")
    n_final = len(conj) if conj is not None else 1
    fig.suptitle(f"{ctx['dte']}, grupo {ctx['cid'].split('.')[0]}: {len(ramas)} caminos ápice→orificio (cuadrado = ápice, círculo = orificio);\n"
                 f"el mismo color = el mismo conducto tras fusionar → {n_final} conducto(s)", y=0.97)
    fig.text(0.5, 0.02, "banda amarilla = zona donde se buscan ápices (los 3 mm más bajos del grupo); línea de puntos = plano del orificio;\n"
             "gris = máscara del grupo proyectada", ha="center", fontsize=9)
    guardar(fig, out, "fig03_caminos_de_coste_minimo.png")


# ------------------------------------------------------------------------------------------ fig 04: fusión
def fig04(ctx, out):
    g, v, ramas, conj = ctx["g"], ctx["v"], ctx["ramas"], ctx["conj"]
    gm = g["mask"]
    alturas = (1.5, 3.5, 6.0, 8.5)
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
    marcas = ["o", "s", "^", "D", "v", "P"]
    for a_, h in zip(axs.ravel(), alturas):
        z = g["z_bot"] + int(round(h / v))
        lab, n = ndi.label(gm[z], structure=np.ones((3, 3)))
        ys, xs = np.nonzero(gm[z])
        y0, y1 = max(ys.min() - 12, 0), min(ys.max() + 13, gm.shape[1])
        x0, x1 = max(xs.min() - 12, 0), min(xs.max() + 13, gm.shape[2])
        a_.imshow(np.ma.masked_where(lab == 0, lab)[y0:y1, x0:x1], origin="lower", cmap="tab10", vmin=0, vmax=9,
                  extent=(x0 * v, x1 * v, y0 * v, y1 * v))
        a_.set_facecolor("0.15")
        for kk in range(1, n + 1):
            yy_, xx_ = np.nonzero(lab == kk)
            a_.text(xx_.mean() * v, yy_.mean() * v - 0.22, f"mancha {kk}", color="w", fontsize=9, ha="center")
        mapa = {}
        for i, P in enumerate(ramas):
            pts = np.round(P / v).astype(int)
            sel = pts[pts[:, 0] == z]
            if len(sel):
                yy, xx = sel[0, 1], sel[0, 2]
                a_.plot(xx * v, yy * v, marcas[i % 6], mfc="w", mec="k", ms=10, mew=1.5)
                a_.text(xx * v + 0.08, yy * v + 0.08, f"{i}", fontsize=11, weight="bold", color="w")
                comp = int(lab[min(yy, lab.shape[0] - 1), min(xx, lab.shape[1] - 1)])
                mapa.setdefault(comp, []).append(i)
        txt = "; ".join(f"mancha {c_}: camino(s) {', '.join(map(str, p_))}" for c_, p_ in sorted(mapa.items()))
        a_.set_title(f"a {h:g} mm del ápice del grupo: {n} manchas separadas", fontsize=10)
        a_.set_xlabel(f"x (mm)\n{txt}")
        a_.set_ylabel("y (mm)")
    fig.suptitle(f"{ctx['dte']}: dos caminos son el MISMO conducto si en ≥ 60 % de los cortes caen en la misma mancha\n"
                 "(cada color = una mancha; el número blanco = el camino que pasa por ese punto)", y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    guardar(fig, out, "fig04_fusion_de_caminos.png")


# ------------------------------------------------------------------------------------------ fig 05: plano perpendicular
def fig05(ctx, out, d0=2.0, mitad=2.0):
    Q = ctx["Qg"]
    k, sec, paso, c, (T, U, W) = seccion_en(ctx, d0, mitad)
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    sel = np.abs(ctx["d"] - ctx["d"][k]) < 3.5
    ax.plot(Q[sel, 2], Q[sel, 1], Q[sel, 0], "k-", lw=2)
    p = Q[k]
    esc = 0.9
    for vec, col, nom in ((T[k], "C3", "T (tangente)"), (U[k], "C0", "U"), (W[k], "C2", "W")):
        ax.quiver(p[2], p[1], p[0], vec[2] * esc, vec[1] * esc, vec[0] * esc, color=col, lw=2.5, arrow_length_ratio=0.2)
        ax.text(p[2] + vec[2] * esc * 1.15, p[1] + vec[1] * esc * 1.15, p[0] + vec[0] * esc * 1.15, nom, color=col)
    L = 1.0
    cor = [p + a * L * U[k] + b * L * W[k] for a, b in ((-1, -1), (-1, 1), (1, 1), (1, -1))]
    ax.add_collection3d(Poly3DCollection([[(q[2], q[1], q[0]) for q in cor]], alpha=0.25, color="C1"))
    ax.plot([p[2]], [p[1]], [p[0]], "ro")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("La línea central (negro), su tangente T en un punto y el plano\nperpendicular (naranja) generado por U y W")
    rng = 2.2
    ax.set_xlim(p[2] - rng, p[2] + rng)
    ax.set_ylim(p[1] - rng, p[1] + rng)
    ax.set_zlim(p[0] - rng, p[0] + rng)
    ax.set_box_aspect((1, 1, 1))
    ax2 = fig.add_subplot(1, 2, 2)
    m = mitad
    ax2.imshow(sec, origin="lower", extent=(-m, m, -m, m), cmap="gray", vmin=0, vmax=1)
    ax2.plot(0, 0, "r+", ms=14, mew=2)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_xlabel("W (mm)")
    ax2.set_ylabel("U (mm)")
    ax2.set_title(f"La sección: el volumen leído sobre ese plano\n(a {ctx['d'][k]:.2f} mm del ápice; rojo = el punto de la línea)")
    guardar(fig, out, "fig05_plano_perpendicular.png")


# ------------------------------------------------------------------------------------------ fig 06: centroide iterativo
def fig06(ctx, out, a):
    g, v = ctx["g"], ctx["v"]
    z_top = ctx["z_top_mm"]
    P0 = np.vstack([[z_top, ctx["P0"][0, 1], ctx["P0"][0, 2]], ctx["P0"]])
    Ps, desp = [], None
    for k in range(4):
        P, info = cm.refinar_linea_central(ctx["vol"], v, P0, ds_mm=0.04, mitad_mm=a.mitad_mm, ext_apical_max_mm=2.0,
                                           z_top_mm=z_top, iteraciones=k)
        Ps.append(P)
        if k == 3:
            desp = info["desplazamiento_max_por_iteracion_mm"]
    Q = ctx["Qg"]
    T = cm.tangentes(Q)
    U, W = cm._base_perpendicular(T)
    tree = [__import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(P) for P in Ps]
    mejor, kk = -1, 0
    for k in range(10, len(Q) - 10, 12):
        if ctx["d"][k] < 0.8 or ctx["d"][k] > ctx["d"].max() - 1.0:
            continue
        dist, j = tree[0].query(Q[k])
        off = Ps[0][j] - Q[k]
        off_plano = np.hypot(off @ U[k], off @ W[k])
        if off_plano > mejor:
            mejor, kk = off_plano, k
    k = kk
    sec, paso, c = cm.secciones(ctx["vol"], v, Q[k:k + 1], T[k:k + 1], U[k:k + 1], W[k:k + 1], a.mitad_mm)
    cen = cm.centroide_seccion(sec[0], c, paso)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
    m = a.mitad_mm
    ax[0].imshow(sec[0], origin="lower", extent=(-m, m, -m, m), cmap="gray", vmin=0, vmax=1)
    cols = ["0.55", "C1", "C2", "C0"]
    etiquetas = ["0: camino de coste mínimo (suavizado)", "1: tras la 1.ª iteración", "2: tras la 2.ª", "3: tras la 3.ª"]
    pts = []
    for i, P in enumerate(Ps):
        j = tree[i].query(Q[k])[1]
        off = P[j] - Q[k]
        pts.append((off @ W[k], off @ U[k]))
        ax[0].plot(pts[-1][0], pts[-1][1], "o", color=cols[i], ms=9, mec="k", label=etiquetas[i])
    for i in range(3):
        ax[0].annotate("", xy=pts[i + 1], xytext=pts[i], arrowprops=dict(arrowstyle="->", color="y", lw=1.5))
    if cen is not None:
        ax[0].plot(cen[1], cen[0], "r+", ms=16, mew=2.5, label="centroide de esta sección (objetivo)")
    ax[0].plot(0, 0, "x", color="yellow", ms=11, mew=2.5, label="punto de la línea FINAL")
    ax[0].set_xlim(-0.8, 0.8)
    ax[0].set_ylim(-0.8, 0.8)
    ax[0].legend(fontsize=7, loc="lower left")
    ax[0].set_xlabel("W (mm)")
    ax[0].set_ylabel("U (mm)")
    ax[0].set_title(f"Un punto de la línea a {ctx['d'][k]:.1f} mm del ápice: cómo se va centrando")
    # convergencia: distancia media de cada línea a la final
    from scipy.spatial import cKDTree
    tf = cKDTree(Q)
    med = [np.median(tf.query(P)[0]) * 1000 for P in Ps]
    p90 = [np.percentile(tf.query(P)[0], 90) * 1000 for P in Ps]
    ax[1].plot(range(4), med, "o-", label="mediana")
    ax[1].plot(range(4), p90, "s-", label="percentil 90")
    ax[1].set_xticks(range(4))
    ax[1].set_xlabel("nº de iteraciones de recentrado")
    ax[1].set_ylabel("distancia a la línea final (µm)")
    ax[1].set_title("La línea deja de moverse tras 2–3 iteraciones\n(desplazamiento máx. por iteración, mm: "
                    + ", ".join(f"{x:.2f}" for x in desp) + ")")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    guardar(fig, out, "fig06_recentrado_por_centroide.png")


# ------------------------------------------------------------------------------------------ fig 07: suavizado
def fig07(out):
    rng = np.random.default_rng(3)
    ds = 0.04
    s = np.arange(0, 4.0 + ds / 2, ds)
    ver = np.stack([0.4 * s + rng.normal(0, 0.01, len(s)), np.zeros_like(s), np.zeros_like(s)], axis=1)   # recta con pendiente + ruido de 10 µm
    sig = 0.25
    nearest = ndi.gaussian_filter1d(ver, sig / ds, axis=0, mode="nearest")
    odd = cm.suavizar(ver, ds, sig)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ax[0].plot(s, ver[:, 0], ".", color="0.6", label="puntos con ruido (10 µm)")
    ax[0].plot(s, 0.4 * s, "k-", lw=1, label="verdad (recta)")
    ax[0].plot(s, nearest[:, 0], "C1-", lw=2, label="Gauss repitiendo el último valor (mode='nearest')")
    ax[0].plot(s, odd[:, 0], "C0-", lw=2, label="Gauss con reflexión impar (el que usa el código)")
    ax[0].set_xlabel("posición a lo largo del conducto (mm)")
    ax[0].set_ylabel("coordenada (mm)")
    ax[0].set_title("Extremos: 'nearest' tira de los extremos hacia dentro;\nla reflexión impar conserva la pendiente")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[0].set_xlim(-0.05, 1.2)
    ax[0].set_ylim(-0.05, 0.6)
    R = 6.0
    s2 = np.arange(0, 8.0 + ds / 2, ds)
    arc = np.stack([R * np.sin(s2 / R), R * (1 - np.cos(s2 / R)), np.zeros_like(s2)], axis=1)
    arcn = arc + rng.normal(0, 0.005, arc.shape)
    for sg, col in ((0.0, "0.6"), (0.1, "C1"), (0.25, "C0")):
        Qs = cm.suavizar(arcn, ds, sg)
        err = np.linalg.norm(Qs - arc, axis=1) * 1000
        rm = np.sqrt(np.mean(err[10:-10] ** 2))
        ax[1].plot(s2, err, "-", color=col, lw=1.3, label=(f"σ = {sg:g} mm: error de posición típico {rm:.1f} µm" if sg else f"sin suavizar: {rm:.1f} µm"))
    ax[1].set_xlabel("posición a lo largo del arco (mm)")
    ax[1].set_ylabel("distancia a la verdad (µm)")
    ax[1].set_title("Arco de radio 6 mm con ruido de 5 µm por coordenada: error de posición.\nSon micras: invisible en un dibujo, pero κ usa la SEGUNDA derivada (ver fig08)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    guardar(fig, out, "fig07_suavizado.png")


# ------------------------------------------------------------------------------------------ fig 08: curvatura, teoría
def fig08(out):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    R = 6.0
    th1, th2 = 0.25, 0.85
    t = np.linspace(0, 1.3, 100)
    ax[0].plot(R * np.sin(t), R * (1 - np.cos(t)), "k", lw=2)
    for th, col in ((th1, "C0"), (th2, "C3")):
        P = np.array([R * np.sin(th), R * (1 - np.cos(th))])
        Tv = np.array([np.cos(th), np.sin(th)])
        ax[0].annotate("", xy=P + Tv * 1.5, xytext=P, arrowprops=dict(arrowstyle="->", color=col, lw=2.5))
        ax[0].plot(*P, "o", color=col)
    A = np.array([R * np.sin(th1), R * (1 - np.cos(th1))])
    B = np.array([R * np.sin(th2), R * (1 - np.cos(th2))])
    ax[0].plot([A[0], R * np.sin(th1) * 0 + 0], [A[1], R], "--", color="C0", lw=0.8)
    ax[0].plot([B[0], 0], [B[1], R], "--", color="C3", lw=0.8)
    ax[0].plot(0, R, "k+", ms=12)
    ax[0].text(0.2, R + 0.15, "centro del círculo\n(osculador)", fontsize=8)
    ax[0].text(2.1, 0.55, "T en A", color="C0")
    ax[0].text(5.1, 2.1, "T en B", color="C3")
    ax[0].text(2.3, 1.55, "Δs = R·Δθ", fontsize=10)
    ax[0].text(0.3, 3.1, "Δθ = ángulo que gira T\nκ = Δθ/Δs = 1/R\n(R = 6 mm → κ = 0,167 mm⁻¹)", fontsize=10,
               bbox=dict(fc="w", ec="0.6"))
    ax[0].set_aspect("equal")
    ax[0].set_xlim(-0.5, 7)
    ax[0].set_ylim(-0.5, 7)
    ax[0].set_title("Idea: curvatura = cuánto gira la dirección (T) por mm recorrido")
    ax[0].set_xlabel("mm")
    ax[0].set_ylabel("mm")
    # numérica
    rng = np.random.default_rng(1)
    ds = 0.04
    s = np.arange(0, 8.0 + ds / 2, ds)
    arc = np.stack([R * np.sin(s / R), R * (1 - np.cos(s / R)), np.zeros_like(s)], axis=1)
    arcn = arc + rng.normal(0, 0.005, arc.shape)
    kt = 1 / R
    for sg, col in ((0.0, "0.6"), (0.05, "C2"), (0.15, "C1"), (0.25, "C0")):
        Qs = cm.suavizar(arcn, ds, sg)
        kap = cm.curvatura(Qs, ds)
        sel = slice(int(0.5 / ds), -int(0.5 / ds))
        rmse = np.sqrt(np.mean((kap[sel] - kt) ** 2))
        ax[1].plot(s, kap, color=col, lw=1.4, label=(f"σ = {sg:g} mm: error típico {rmse:.3f} mm⁻¹" if sg else f"sin suavizar: error típico {rmse:.2f} mm⁻¹"))
    ax[1].axhline(kt, color="k", ls="--", label="verdad: 1/6 = 0,167 mm⁻¹")
    ax[1].set_ylim(-0.05, 1.6)
    ax[1].set_xlabel("posición a lo largo del arco (mm)")
    ax[1].set_ylabel("κ (mm⁻¹)")
    ax[1].set_title("Con ruido de solo 5 µm, sin suavizar κ sale\nmuy mal; con σ = 0,25 mm sale casi exacto")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].grid(alpha=0.3)
    # error vs sigma (promedio de 20 semillas)
    sigs = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5]
    errs = []
    for sg in sigs:
        e = []
        for sd in range(10):
            r = np.random.default_rng(100 + sd)
            Pn = arc + r.normal(0, 0.005, arc.shape)
            kap = cm.curvatura(cm.suavizar(Pn, ds, sg), ds)
            sel = slice(int(0.5 / ds), -int(0.5 / ds))
            e.append(np.sqrt(np.mean((kap[sel] - kt) ** 2)))
        errs.append(np.mean(e))
    ax[2].semilogy(sigs, errs, "o-")
    ax[2].axhline(kt, color="r", ls=":", label="κ verdadero (0,167)")
    ax[2].set_xlabel("σ del suavizado (mm)")
    ax[2].set_ylabel("error típico de κ (mm⁻¹, escala log)")
    ax[2].set_title("Cuanto más se suaviza, menos ruido…\nsiempre que el conducto no tenga curvas más cerradas que σ")
    ax[2].legend()
    ax[2].grid(alpha=0.3, which="both")
    print("      error de κ por σ:", ", ".join(f"σ={s_:g}: {e_:.3f}" for s_, e_ in zip(sigs, errs)))
    guardar(fig, out, "fig08_curvatura_teoria.png")


# ------------------------------------------------------------------------------------------ fig 09: curvatura real
def fig09(ctx, out, suav_mm=0.25):
    Q = ctx["Qg"]
    t = ctx["tabla"]
    kap = t["kappa_mm-1"]
    d = ctx["d"]
    ds = np.linalg.norm(np.diff(Q, axis=0), axis=1).mean()
    T = cm.tangentes(Q)
    dT = np.gradient(T, ds, axis=0)
    kT = np.linalg.norm(dT, axis=1)
    marg = int(round(max(0.3, 2 * suav_mm) / ds))
    inter = np.arange(marg, len(Q) - marg)
    ordenados = inter[np.argsort(kap[inter])]
    elegidos = [("κ bajo (tramo casi recto)", ordenados[len(ordenados) // 5]), ("κ medio", ordenados[len(ordenados) // 2]),
                ("κ máximo del interior", ordenados[-1])]
    fig = plt.figure(figsize=(17, 8))
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    seg = np.stack([Q[:-1, [1, 0]], Q[1:, [1, 0]]], axis=1)
    lc = LineCollection(seg, array=kap[:-1], cmap="viridis", norm=plt.Normalize(0, max(kap[inter].max(), 0.5)), linewidths=3)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("auto")
    plt.colorbar(lc, ax=ax, label="κ (mm⁻¹)", shrink=0.8)
    for (nom, i), col in zip(elegidos, ("C3", "C1", "C0")):
        ax.plot(Q[i, 1], Q[i, 0], "o", mfc="none", mec=col, ms=11, mew=2)
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title("La línea central vista de lado\n(escala de y exagerada), coloreada por κ")
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.plot(d, kap, "k-")
    ax2.plot(d, kT, "C9--", lw=1, label="|dT/ds| (otra fórmula; debe coincidir)")
    for (nom, i), col in zip(elegidos, ("C3", "C1", "C0")):
        ax2.plot(d[i], kap[i], "o", mfc="none", mec=col, ms=11, mew=2)
    ax2.axvspan(0, 0.5, color="0.9")
    ax2.axvspan(d.max() - 0.5, d.max(), color="0.9")
    ax2.text(0.25, ax2.get_ylim()[1] * 0.92, "márgenes\n(se ignoran\npara κ máx)", fontsize=7, ha="center")
    ax2.set_xlabel("distancia al extremo apical (mm)")
    ax2.set_ylabel("κ (mm⁻¹)")
    ax2.set_title(f"κ(s) del conducto {ctx['cid']} (σ del suavizado = {suav_mm} mm)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    for n, ((nom, i), col) in enumerate(zip(elegidos, ("C3", "C1", "C0"))):
        a_ = fig.add_subplot(gs[1, n])
        sel = np.abs(np.arange(len(Q)) - i) * ds <= 1.8
        Ti = T[i]
        Ni = dT[i] / max(np.linalg.norm(dT[i]), 1e-12)
        rel = Q[sel] - Q[i]
        a_.plot(rel @ Ti, rel @ Ni, "k-", lw=2, label="línea central")
        if kT[i] > 1e-6:
            Rr = 1 / kap[i]
            ph = np.linspace(-1.2, 1.2, 200)
            a_.plot(Rr * np.sin(ph), Rr * (1 - np.cos(ph)), "--", color=col, lw=2, label=f"círculo osculador, R = 1/κ = {Rr:.2f} mm")
        a_.plot(0, 0, "o", color=col)
        a_.set_aspect("equal")
        a_.set_xlim(-1.8, 1.8)
        a_.set_ylim(-0.6, 2.0)
        a_.set_xlabel("a lo largo de T (mm)")
        a_.set_ylabel("hacia donde se curva, N (mm)")
        a_.set_title(f"{nom}: κ = {kap[i]:.2f} mm⁻¹")
        a_.legend(fontsize=7, loc="upper center")
        a_.grid(alpha=0.3)
    fig.suptitle(f"{ctx['dte']}, conducto {ctx['cid']}: curvatura medida sobre la línea central real", y=0.99)
    fig.tight_layout()
    guardar(fig, out, "fig09_curvatura_real.png")


# ------------------------------------------------------------------------------------------ fig 10: diámetros
def fig10(ctx, out, d0=1.0, nombre="fig10_diametros_seccion_buena.png", mitad=2.0):
    k, sec, paso, c, _ = seccion_en(ctx, d0, mitad)
    dd = cm.diametros_seccion(sec, c, paso)
    R = dd["region"]
    t = ctx["tabla"]
    m = mitad
    fig, ax = plt.subplots(2, 2, figsize=(13, 11.5))
    ax = ax.ravel()
    lim = 0.75
    ext = (-m, m, -m, m)
    ci, cj = dd["centro_ij"]
    xc, yc = (cj - c) * paso, (ci - c) * paso
    for a_ in ax:
        a_.set_xlim(-lim, lim)
        a_.set_ylim(-lim, lim)
        a_.set_xlabel("W (mm)")
        a_.set_ylabel("U (mm)")
    ax[0].imshow(sec, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    xs = (np.arange(sec.shape[1]) - c) * paso
    ax[0].contour(xs, xs, sec, levels=[0.5], colors="r", linewidths=1)
    ax[0].plot(0, 0, "x", color="yellow", ms=11, mew=2)
    ax[0].set_title("① La sección tal como se muestrea: brillo = fracción de conducto\ndel píxel (0 a 1). Rojo = umbral 0,5 (el borde). Cada píxel = 16 µm")
    # B: región y área
    ax[1].imshow(np.where(R, sec, np.nan), origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax[1].set_facecolor("0.93")
    ax[1].plot(xc, yc, "r+", ms=16, mew=2.5, label="centroide (centro de gravedad)")
    ax[1].plot(0, 0, "x", color="orange", ms=11, mew=2, label="punto de la línea central")
    ax[1].add_patch(Circle((xc, yc), dd["d_eq"] / 2, fill=False, ls="--", ec="C3", lw=2, label=f"círculo de igual área: D_eq = {dd['d_eq']:.3f} mm"))
    area = (dd["d_eq"] / 2) ** 2 * np.pi
    ax[1].set_title(f"② Trozo del conducto (el que contiene la línea) y su área\nÁrea = Σ(valor del píxel)·(paso)² = {area:.3f} mm²;   D_eq = 2·√(Área/π) = {dd['d_eq']:.3f} mm")
    ax[1].legend(fontsize=7, loc="lower left")
    # C: barrido
    ax[2].imshow(np.where(R, 1.0, np.nan), origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=2, interpolation="nearest")
    ax[2].set_facecolor("0.93")
    rad = dd["rad"]
    ang = np.deg2rad(np.arange(360))
    for a in range(0, 360, 10):
        ax[2].plot([xc, xc + rad[a] * np.sin(ang[a])], [yc, yc + rad[a] * np.cos(ang[a])], "-", color="0.5", lw=0.6)
    ax[2].plot(xc + rad * np.sin(ang), yc + rad * np.cos(ang), "r-", lw=1.5)
    if np.isfinite(dd["d_mayor"]):
        D = rad[:180] + rad[180:]
        for idx, col, nom, val in ((int(np.argmax(D)), "C1", "mayor", dd["d_mayor"]), (int(np.argmin(D)), "C2", "menor", dd["d_menor"])):
            p1 = (xc + rad[idx] * np.sin(ang[idx]), yc + rad[idx] * np.cos(ang[idx]))
            p2 = (xc + rad[idx + 180] * np.sin(ang[idx + 180]), yc + rad[idx + 180] * np.cos(ang[idx + 180]))
            ax[2].plot([p1[0], p2[0]], [p1[1], p2[1]], "-", color=col, lw=3, label=f"D {nom} = {val:.3f} mm (ángulo {idx}°)")
    ax[2].plot(xc, yc, "r+", ms=14, mew=2)
    ax[2].set_title("③ Barrido angular: 360 radios (se dibuja 1 de cada 10). D(α) = R(α) + R(α+180°);\nmayor = máx D, menor = mín D "
                    f"(el polígono rojo reproduce el área con error {100 * dd['error_area_barrido']:+.1f} %)")
    ax[2].legend(fontsize=7, loc="lower left")
    # D: esfera inscrita + elipse
    ax[3].imshow(np.where(R, 1.0, np.nan), origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=2, interpolation="nearest")
    ax[3].set_facecolor("0.93")
    kk = int(np.argmin(np.abs(t["d_desde_apice_mm"] - ctx["d"][k])))
    r_edt = t["d_edt_mm"][kk] / 2
    ax[3].add_patch(Circle((0, 0), r_edt, fill=False, ec="C4", lw=2.5, label=f"esfera inscrita (EDT): D = {2 * r_edt:.3f} mm"))
    pts = np.vstack(np.nonzero(R)).astype(float)
    mu = pts.mean(axis=1)
    cov = np.cov(pts)
    ev, vec = np.linalg.eigh(cov)
    ancho, esp = 4 * np.sqrt(ev[1]) * paso, 4 * np.sqrt(ev[0]) * paso
    ang_el = np.degrees(np.arctan2(vec[0, 1], vec[1, 1]))              # eje mayor: componente (i, j) → (y, x)
    ax[3].add_patch(Ellipse(((mu[1] - c) * paso, (mu[0] - c) * paso), ancho, esp, angle=ang_el, fill=False, ec="C3", lw=2, ls="--",
                            label=f"elipse de igual dispersión: {ancho:.2f} × {esp:.2f} mm\n(alargamiento {dd['alargamiento']:.1f})"))
    ax[3].plot(0, 0, "x", color="orange", ms=11, mew=2)
    ax[3].set_title("④ Otras dos medidas del mismo trozo: la esfera inscrita (EDT)\ny la elipse de igual dispersión (ancho × espesor)")
    ax[3].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{ctx['dte']}, conducto {ctx['cid']}, sección a {ctx['d'][k]:.2f} mm del extremo apical (ventana de medida: {2 * m:g}×{2 * m:g} mm; "
                 f"aquí se amplía ±{lim} mm)", y=0.97)
    guardar(fig, out, nombre)


# ------------------------------------------------------------------------------------------ fig 11: secciones malas
def fig11(ejemplos, out):
    fig, ax = plt.subplots(1, len(ejemplos), figsize=(6.2 * len(ejemplos), 6.4), squeeze=False)
    for a_, ctx_, d0 in zip(ax[0], *zip(*ejemplos)):
        k, sec, paso, c, _ = seccion_en(ctx_, d0)
        dd = cm.diametros_seccion(sec, c, paso)
        m = 2.0
        a_.imshow(sec, origin="lower", extent=(-m, m, -m, m), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        a_.plot(0, 0, "x", color="yellow", ms=12, mew=2.5, label="punto de la línea central")
        if dd["centro_ij"] is not None:
            xc, yc = (dd["centro_ij"][1] - c) * paso, (dd["centro_ij"][0] - c) * paso
            a_.plot(xc, yc, "r+", ms=16, mew=2.5, label="centroide del trozo")
            if dd["rad"] is not None:
                ang = np.deg2rad(np.arange(360))
                a_.plot(xc + dd["rad"] * np.sin(ang), yc + dd["rad"] * np.cos(ang), "--", color="orange", lw=2, label="contorno que dibujaría el barrido")
        if dd["region"] is not None:
            xs = (np.arange(sec.shape[1]) - c) * paso
            a_.contour(xs, xs, dd["region"].astype(float), levels=[0.5], colors="cyan", linewidths=1.5)
        a_.set_xlim(-1.0, 1.0)
        a_.set_ylim(-1.0, 1.0)
        a_.legend(fontsize=7, loc="lower left")
        d_eq = "—" if not np.isfinite(dd["d_eq"]) else f"{dd['d_eq']:.2f} mm"
        a_.set_title(f"{ctx_['dte']} conducto {ctx_['cid']} a {ctx_['d'][k]:.1f} mm del ápice\nD_eq = {d_eq}; mayor/menor: "
                     f"{'sí' if np.isfinite(dd['d_mayor']) else 'NO'}\n" + textwrap.fill(dd["motivo"] or "válida", 62), fontsize=8.5, color="C3" if dd["motivo"] else "k")
        a_.set_xlabel("W (mm)")
        a_.set_ylabel("U (mm)")
    fig.suptitle("Cuando una sección NO se puede medir con el barrido angular (cian = borde real del trozo)", y=1.0)
    guardar(fig, out, "fig11_secciones_no_validas.png")


# ------------------------------------------------------------------------------------------ fig 12: perfil anotado
def fig12(ctx, out):
    t = ctx["tabla"]
    d = ctx["d"]
    L = d.max()
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for k, lab in (("d_eq_mm", "equivalente (de su área)"), ("d_mayor_mm", "mayor (barrido)"), ("d_menor_mm", "menor (barrido)"),
                   ("d_edt_mm", "esfera inscrita")):
        ax[0].plot(d, t[k], label=lab)
    ax[0].set_ylim(0, 1.2)
    for x in (L / 3, 2 * L / 3):
        ax[0].axvline(x, color="k", ls=":", lw=0.8)
        ax[1].axvline(x, color="k", ls=":", lw=0.8)
    for nom, x in (("tercio apical", L / 6), ("tercio medio", L / 2), ("tercio coronal", 5 * L / 6)):
        ax[0].text(x, 1.14, nom, ha="center", fontsize=8)
    ax[0].annotate("extremo apical\n(final de la máscara, d = 0)", xy=(0.02, t["d_eq_mm"][np.argmin(d)]), xytext=(0.5, 0.55),
                   arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax[0].annotate("el conducto se ensancha\nhacia el orificio", xy=(L - 0.3, np.nanmax(t["d_eq_mm"][d > L - 0.6])), xytext=(L - 5.5, 0.9),
                   arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax[0].set_xlabel("distancia al extremo apical (mm)")
    ax[0].set_ylabel("diámetro (mm)")
    ax[0].set_title("Diámetro a lo largo del conducto")
    ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    ax[0].grid(alpha=0.3)
    ax[1].plot(d, t["kappa_mm-1"], "C3")
    ax[1].set_xlabel("distancia al extremo apical (mm)")
    ax[1].set_ylabel("κ = 1/R (mm⁻¹)")
    ax[1].set_title("Curvatura a lo largo del conducto (0 = recto)")
    ax[1].grid(alpha=0.3)
    fig.suptitle(f"{ctx['dte']}, conducto {ctx['cid']}: así se lee control_perfil.png", y=1.02)
    guardar(fig, out, "fig12_perfil_anotado.png")


# ------------------------------------------------------------------------------------------ main
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="results/microct/Diente_XX")
    ap.add_argument("--canal", default="2.1", help="conducto G.B de medir_conductos.py")
    ap.add_argument("--medidas", type=Path, default=RESULTS / "medidas", help="carpeta con los CSV de medir_conductos.py")
    ap.add_argument("--out", type=Path, default=None, help="por defecto results/medidas/explicacion/<diente>")
    ap.add_argument("--ejemplos", nargs="*", default=[], help="diente:G.B:d (mm del ápice) de secciones no válidas")
    ap.add_argument("--coarse", type=int, default=2)
    ap.add_argument("--mitad-mm", type=float, default=2.0)
    ap.add_argument("--d-seccion", type=float, default=0.5, help="mm del ápice de la sección de las figuras 5 y 10")
    ap.add_argument("--solo", nargs="*", default=None, help="números de figura a dibujar (por defecto todas)")
    a = ap.parse_args()
    out = a.out or (a.medidas / "explicacion" / a.folder.name)
    quiere = lambda n: a.solo is None or str(n) in a.solo
    print(f"[{a.folder.name}] canal {a.canal} → {out}")
    necesita_ramas = any(quiere(n) for n in (3, 4, 6))
    ctx = preparar(a.folder, a.canal, a.medidas, necesita_ramas, a)
    if quiere(1):
        fig01(ctx, out)
    if quiere(2):
        fig02(ctx, out)
    if quiere(3):
        fig03(ctx, out)
    if quiere(4):
        fig04(ctx, out)
    if quiere(5):
        fig05(ctx, out, a.d_seccion, a.mitad_mm)
    if quiere(6):
        fig06(ctx, out, a)
    if quiere(7):
        fig07(out)
    if quiere(8):
        fig08(out)
    if quiere(9):
        fig09(ctx, out)
    if quiere(10):
        fig10(ctx, out, a.d_seccion, mitad=a.mitad_mm)
        fig10(ctx, out, 1.0, nombre="fig10b_diametros_seccion_plana.png", mitad=a.mitad_mm)
    if quiere(12):
        fig12(ctx, out)
    if quiere(11) and a.ejemplos:
        ej = []
        for spec in a.ejemplos:
            dte, cid, d0 = spec.split(":")
            ej.append((preparar(a.folder.parent / dte, cid, a.medidas, False, a), float(d0)))
        fig11(ej, out)


if __name__ == "__main__":
    main()
