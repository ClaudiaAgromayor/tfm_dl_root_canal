"""Imágenes para enseñar a una endodoncista dónde termina la máscara del conducto respecto a la punta de la raíz.

Por diente genera, en results/para_isa/ (sin tocar nada de results/microct):
  Diente_XX_corte.png                  dos paneles del MISMO corte longitudinal: izquierda sin marcas, derecha con la máscara
                                       `canal_solo_conducto` en color semitransparente y una flecha donde termina la máscara
  Diente_XX_corte_perpendicular.png    lo mismo en el plano perpendicular (girado 90° alrededor del eje largo)
  Diente_XX_ampliado.png               el primer corte ampliado, solo los últimos 3 mm del conducto y la punta
  (y distancias.json con las distancias de info.json, para quien las quiera)

Corte: se elige el conducto MÁS LARGO (el que acaba más cerca del ápice) de info.json["apice"]; el plano contiene el eje
largo (z) y pasa por el final de ese conducto; se prueba un giro cada 5° y se queda el que contiene más máscara de la raíz
de ese conducto. Corona arriba, corona recortada (2 mm sobre el orificio), 2 mm de margen bajo la punta de la raíz.
Escala de grises = µCT normalizado (norm: aire 0, dentina ≈ 1). Sin texto técnico: "Diente X" y una barra de 1 mm.

    python imagen_para_isa.py [--resultados results/microct] [--out results/para_isa] [--dientes Diente_35 Diente_51 Diente_52]

Hace falta haber ejecutado split_canal.py (usa su info.json). La orientación izquierda/derecha del escaneo no se conoce.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--resultados", type=Path, default=HERE.parents[2] / "results" / "microct")
ap.add_argument("--out", type=Path, default=HERE.parents[2] / "results" / "para_isa")
ap.add_argument("--dientes", nargs="*", default=["Diente_35", "Diente_51", "Diente_52"])
ap.add_argument("--margen-punta-mm", type=float, default=2.0)
ap.add_argument("--sobre-orificio-mm", type=float, default=2.0, help="cuánto se deja de corona sobre el orificio")
ap.add_argument("--semiancho-mm", type=float, default=8.0, help="semiancho máximo del plano muestreado")
ap.add_argument("--ultimos-mm", type=float, default=3.0, help="tramo de conducto de la imagen ampliada")
ap.add_argument("--paso-angulo", type=float, default=5.0)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)


def trilinear(vol, z, y, x):
    """Interpolación trilineal de vol (Z,Y,X) en coordenadas reales; fuera del volumen vale 0."""
    z0, y0, x0 = (np.floor(q).astype(int) for q in (z, y, x))
    fz, fy, fx = z - z0, y - y0, x - x0
    out = np.zeros(z.shape, np.float32)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                zi, yi, xi = z0 + dz, y0 + dy, x0 + dx
                ok = (zi >= 0) & (zi < vol.shape[0]) & (yi >= 0) & (yi < vol.shape[1]) & (xi >= 0) & (xi < vol.shape[2])
                w = (fz if dz else 1 - fz) * (fy if dy else 1 - fy) * (fx if dx else 1 - fx)
                v = np.zeros(z.shape, np.float32)
                v[ok] = vol[zi[ok], yi[ok], xi[ok]]
                out += w * v
    return out


def plano(vol, z0, y0, x0, theta, zs, ts):
    """Imagen (len(zs), len(ts)) del plano que contiene el eje z y pasa por (y0, x0), girado `theta` (rad) en el plano xy."""
    Z, T = np.meshgrid(zs, ts, indexing="ij")
    return trilinear(vol, Z - z0, y0 + T * np.sin(theta), x0 + T * np.cos(theta))


def barra(ax, px_mm, shape, color="white"):
    """Barra de escala de 1 mm abajo a la izquierda."""
    L = 1.0 / px_mm
    x0, y0 = 0.05 * shape[1], 0.95 * shape[0]
    ax.plot([x0, x0 + L], [y0, y0], color=color, lw=3, solid_capstyle="butt")
    ax.text(x0 + L / 2, y0 - 0.02 * shape[0], "1 mm", color=color, ha="center", va="bottom", fontsize=11)


def dibujar(path, titulo, img, msk, px_mm, fin_xy, crown_up=True, flecha=True):
    """Dos paneles: izquierda sin marcas; derecha con máscara semitransparente y flecha en `fin_xy` (columna, fila)."""
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
            if flecha:
                cx, cy = fin_xy
                ax.annotate("", xy=(cx + 0.25 / px_mm, cy), xytext=(cx + 1.6 / px_mm, cy),
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


resumen = {}
for d in a.dientes:
    folder = a.resultados / d
    info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
    V = info["voxel_mm"]
    crown_high = info["orientacion"]["corona_en_z"] == "alto"
    sg = 1 if crown_high else -1                         # sentido corona -> en z
    norm = np.load(folder / "norm.npy", mmap_mode="r")
    solo = np.load(folder / "canal_solo_conducto.npy", mmap_mode="r")
    # conducto más largo = el que acaba más cerca del ápice (más lejos del orificio)
    cand = [(r, c) for r in info["apice"]["raices"] for c in r["conductos"]]
    r, c = max(cand, key=lambda rc: sg * (info["separacion"]["z_orificio_vox"] - rc[1]["z_fin_vox"]))
    z_end, (yc, xc) = c["z_fin_vox"], c["yx_fin_vox"]
    z_tip_root = r["z_punta_vox"]
    z_or = info["separacion"]["z_orificio_vox"]
    m_tip, m_up, T = (int(round(q / V)) for q in (a.margen_punta_mm, a.sobre_orificio_mm, a.semiancho_mm))
    z_a, z_b = sorted((z_tip_root - sg * m_tip, z_or + sg * m_up))          # tramo vertical a mostrar
    z_a, z_b = max(z_a, 0), min(z_b, norm.shape[0] - 1)
    (by0, by1), (bx0, bx1) = r["bbox_yx_vox"]
    # recorte en memoria alrededor del eje: [z_a, z_b] x [yc±T, xc±T]
    y_lo, y_hi, x_lo, x_hi = max(yc - T - 2, 0), min(yc + T + 2, norm.shape[1]), max(xc - T - 2, 0), min(xc + T + 2, norm.shape[2])
    nv = np.asarray(norm[z_a:z_b + 1, y_lo:y_hi, x_lo:x_hi])
    mv = np.asarray(solo[z_a:z_b + 1, y_lo:y_hi, x_lo:x_hi]).astype(np.uint8)
    # solo la máscara de la raíz elegida (para puntuar el plano)
    mroot = np.zeros_like(mv)
    mroot[:, max(by0 - y_lo, 0):max(by1 - y_lo, 0), max(bx0 - x_lo, 0):max(bx1 - x_lo, 0)] = mv[:, max(by0 - y_lo, 0):max(by1 - y_lo, 0), max(bx0 - x_lo, 0):max(bx1 - x_lo, 0)]
    zs_full = np.arange(0, z_b - z_a + 1, dtype=np.float32)               # índices locales de z
    ts = np.arange(-T, T + 1, dtype=np.float32)
    y0l, x0l, z_end_l = yc - y_lo, xc - x_lo, z_end - z_a
    # mejor plano: el que contiene más máscara de la raíz (paso 2 para ir rápido)
    mejor, mejor_s = 0.0, -1
    for deg in np.arange(0, 180, a.paso_angulo):
        s = plano(mroot, 0, y0l, x0l, np.deg2rad(deg), zs_full[::2], ts[::2]).sum()
        if s > mejor_s:
            mejor, mejor_s = deg, s
    info_d = {"conducto_elegido": {"raiz": r["raiz"], **{k: c[k] for k in ("z_fin_vox", "mm_a_la_ultima_seccion_con_diente_de_la_raiz", "mm_a_la_punta_local")}},
              "angulo_corte_grados": float(mejor), "todos_los_conductos": [
                  {"raiz": rr["raiz"], "conducto": k + 1, "mm_a_la_ultima_seccion_con_diente_de_la_raiz": cc["mm_a_la_ultima_seccion_con_diente_de_la_raiz"],
                   "mm_a_la_punta_local": cc["mm_a_la_punta_local"]} for rr in info["apice"]["raices"] for k, cc in enumerate(rr["conductos"])]}

    def imagen(deg, zi=None, tj=None):
        th = np.deg2rad(deg)
        z_rng = zs_full if zi is None else zi
        t_rng = ts if tj is None else tj
        img = plano(nv, 0, y0l, x0l, th, z_rng, t_rng)
        msk = plano(mv, 0, y0l, x0l, th, z_rng, t_rng)
        return img, msk, z_rng, t_rng

    def preparar(deg, zlim=None, ancho_mm=None):
        """Plano -> (img, msk, fin_xy) con la corona arriba, recortado a lo ancho al diente (o a `ancho_mm`)."""
        img, msk, z_rng, t_rng = imagen(deg)
        if zlim is not None:
            sel = (z_rng >= zlim[0]) & (z_rng <= zlim[1])
            img, msk, z_rng = img[sel], msk[sel], z_rng[sel]
        if ancho_mm is None:
            cols = np.flatnonzero((img > 0.5).any(axis=0))
            lo, hi = max(cols.min() - int(1.0 / V), 0), min(cols.max() + int(1.0 / V), len(t_rng) - 1)
        else:
            c0 = int(np.argmin(np.abs(t_rng))); w = int(ancho_mm / V)
            lo, hi = max(c0 - w, 0), min(c0 + w, len(t_rng) - 1)
        img, msk, t_rng = img[:, lo:hi + 1], msk[:, lo:hi + 1], t_rng[lo:hi + 1]
        if crown_high:                                                   # corona arriba: z mayor arriba
            img, msk, z_rng = img[::-1], msk[::-1], z_rng[::-1]
        fila = int(np.argmin(np.abs(z_rng - z_end_l)))
        col = int(np.argmin(np.abs(t_rng - 0)))
        return img, msk, (col, fila)

    n = d.replace("Diente_", "Diente ")
    img, msk, fin = preparar(mejor)
    dibujar(a.out / f"{d}_corte.png", n, img, msk, V, fin)
    img, msk, fin = preparar(mejor + 90)
    dibujar(a.out / f"{d}_corte_perpendicular.png", n, img, msk, V, fin)
    zlim = (z_end_l - sg * int(round(a.ultimos_mm / V)) if not crown_high else z_end_l - int(round(0.8 / V)), z_end_l + int(round(a.ultimos_mm / V)))
    if not crown_high:
        zlim = (z_end_l - int(round(a.ultimos_mm / V)), z_end_l + int(round(0.8 / V)))
    tip_loc = z_end_l - sg * int(round(c["mm_a_la_punta_local"] / V))
    zlim = (min(zlim[0], tip_loc - int(round(0.8 / V))), zlim[1]) if crown_high else (zlim[0], max(zlim[1], tip_loc + int(round(0.8 / V))))
    img, msk, fin = preparar(mejor, zlim=zlim, ancho_mm=3.0)
    dibujar(a.out / f"{d}_ampliado.png", n, img, msk, V, fin)
    resumen[d] = info_d
    print(d, "-> conducto elegido: raíz", r["raiz"], "| fin de la máscara a la punta local %.3f mm, a la última sección con diente de la raíz %.3f mm" %
          (c["mm_a_la_punta_local"], c["mm_a_la_ultima_seccion_con_diente_de_la_raiz"]), "| corte a", mejor, "grados")
(a.out / "distancias.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
