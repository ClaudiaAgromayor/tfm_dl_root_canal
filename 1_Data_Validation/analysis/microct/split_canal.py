"""Etiqueta SOLO conducto: separa el conducto de la cámara pulpar y de las cavidades de la corona, y mide
dónde acaba el conducto en el ápice. Se aplica a una carpeta results/microct/Diente_XX (la salida de
preprocess_microct.py a 16 µm).

    python split_canal.py results/microct/Diente_35 [--area-camara-mm2 1.5 ...]

Escribe en la carpeta del diente (nombres fijos):
    canal_completo.npy        copia de canal.npy (la máscara completa de preprocess_microct.py); canal.npy no se toca
    canal_solo_conducto.npy   del orificio de entrada al ápice  <- la ETIQUETA
    camara_y_cavidades.npy    lo quitado: cámara pulpar y cavidades de la corona (solo_conducto + esto = completo)
    info.json                 + "orientacion", "separacion" (criterio y parámetros) y "apice" (por raíz)
    control_split.png, control_apice.png

CRITERIO (reproducible, sin intervención manual). Las tres cifras salen de los 3 µCT medidos; con ~40 hay que
revisar los avisos de info.json["separacion"]["avisos"]:
 1. Orientación: la corona es el extremo en z con más esmalte (vóxeles de norm > 1.3 dentro del diente).
 2. Se recorren los cortes axiales desde el ápice hacia la corona y se mide en cada uno la SECCIÓN MÁS GRANDE
    del conducto (componente 2D, ignorando restos < 0.02 mm²). En la raíz son varias secciones pequeñas
    (≤ 1.13 mm² en los tres dientes); en la cámara los conductos se funden en una sección grande (> 1.6 mm²).
 3. El orificio es el primer corte con sección mayor ≥ --area-camara-mm2 (1.5) durante ≥ --sostenido-mm (0.1).
    Todo lo que está de ese corte hacia la corona es "camara_y_cavidades"; lo apical es el conducto.
Por qué área de sección y no radio inscrito: en los tres dientes el radio del conducto de la raíz llega a 0.34 mm
y el de la cámara empieza en 0.39 (no hay hueco claro), mientras que el área de la mayor sección pasa de ≤ 1.13 a
≥ 1.6 mm² en < 0.6 mm de altura. Por qué un plano y no un grosor 3D: el corte queda definido por una sola cifra
interpretable (donde los conductos se funden), no deja flecos finos alrededor de la cámara y sirve igual para
molares con 3-4 conductos que para raíces únicas.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_microct import largest_component  # noqa: E402

NAMES = {"completo": "canal_completo.npy", "solo": "canal_solo_conducto.npy", "camara": "camara_y_cavidades.npy"}


# ------------------------------------------------------------------------------------ geometría
def section_stats(canal, voxel, min_mm2=0.02):
    """Por corte z: área de la mayor sección 2D del conducto (mm²) y nº de secciones >= min_mm2."""
    big = np.zeros(canal.shape[0])
    nsec = np.zeros(canal.shape[0], int)
    for z in np.flatnonzero(canal.any(axis=(1, 2))):
        lab, n = ndi.label(canal[z])
        a = np.bincount(lab.ravel())[1:] * voxel ** 2
        a = a[a >= min_mm2]
        nsec[z], big[z] = len(a), (a.max() if len(a) else 0.0)
    return big, nsec


def find_floor(big, z_lo, z_hi, a_c, run):
    """Primer z (de z_lo hacia z_hi) con sección mayor >= a_c durante `run` cortes seguidos; None si no hay."""
    for z in range(z_lo, z_hi - run + 2):
        if (big[z:z + run] >= a_c).all():
            return z
    return None


def orientation(tooth, norm, voxel):
    """Corona = extremo con más esmalte. Devuelve (corona_en_z_alto: bool, info)."""
    tz = np.flatnonzero(tooth.any(axis=(1, 2)))
    k = max(len(tz) // 5, 1)
    enam = np.array([((norm[z] > 1.3) & tooth[z]).sum() for z in range(tz.min(), tz.max() + 1)]) * voxel ** 2
    low, high = float(enam[:k].sum()), float(enam[-k:].sum())
    return high >= low, {"esmalte_mm2_20pct_bajo": round(low, 1), "esmalte_mm2_20pct_alto": round(high, 1),
                         "metodo": "extremo en z con más esmalte (norm > 1.3 dentro del diente), 20 % de la altura del diente",
                         "dudosa": bool(min(low, high) > 0.33 * max(low, high))}


def roots_and_apex(tooth, canal_solo, z_f, voxel, root_min_mm2, lumen_radius_mm, gap_mm=0.13, apical_zone_mm=2.0):
    """Por raíz: dónde acaba el conducto frente a la punta de la raíz y si sale al exterior por el ápice.

    Trabaja con la corona hacia z ALTO (el llamador voltea si hace falta). Las raíces son las componentes 3D
    del diente apical al último corte con ≥ 2 secciones de diente separadas (la furca).
    """
    v = voxel
    z_lo = int(np.flatnonzero(tooth.any(axis=(1, 2))).min())
    cnt = np.zeros(z_f + 1, int)
    for z in range(z_lo, z_f + 1):
        lab, n = ndi.label(tooth[z])
        cnt[z] = int((np.bincount(lab.ravel())[1:] * v ** 2 >= root_min_mm2).sum())
    multi = np.flatnonzero(cnt >= 2)
    z_split = int(multi.max()) if len(multi) else z_f
    lab, n = ndi.label(tooth[z_lo:z_split + 1])
    vols = np.bincount(lab.ravel())[1:] * v ** 3
    out = {"z_furca_vox": z_split, "raices": []}
    rr = lumen_radius_mm / v
    for i in np.flatnonzero(vols >= 1.0) + 1:
        sl = ndi.find_objects((lab == i).astype(np.uint8))[0]
        mask = lab[sl] == i
        z0 = z_lo + sl[0].start
        tip = int(np.flatnonzero(mask.any(axis=(1, 2))).min())          # índice local del último corte con diente
        filled = np.stack([ndi.binary_fill_holes(s) for s in mask])
        canal_in = canal_solo[z0:z0 + mask.shape[0], sl[1], sl[2]] & filled
        r = {"raiz": int(len(out["raices"]) + 1), "volumen_raiz_mm3": round(float(vols[i - 1]), 2),
             "z_punta_vox": int(z0 + tip), "bbox_yx_vox": [[int(s.start), int(s.stop)] for s in sl[1:]]}
        if not canal_in.any():
            r["conductos"] = []
            r["aviso"] = "sin conducto etiquetado en esta raíz"
            out["raices"].append(r)
            continue
        # cada conducto = componente 3D del conducto dentro de la raíz (conductos que confluyen cuentan como uno)
        g = int(round(gap_mm / v))
        joined = ndi.binary_dilation(canal_in, structure=np.ones((2 * g + 1, 1, 1), bool)) if g > 0 else canal_in
        cl, cn = ndi.label(joined)
        cl = cl * canal_in                                  # etiquetas solo sobre vóxeles de conducto
        cvol = np.bincount(cl.ravel(), minlength=cn + 1)[1:] * v ** 3
        r["fragmentos_no_apicales"] = {"n": 0, "volumen_mm3": 0.0}
        yy, xx = np.mgrid[:mask.shape[1], :mask.shape[2]]
        r["conductos"] = []
        for j in np.flatnonzero(cvol >= 0.02) + 1:
            cm = cl == j
            end = int(np.flatnonzero(cm.any(axis=(1, 2))).min())
            cy0, cx0 = ndi.center_of_mass(cm[end])
            cy, cx = cy0, cx0
            dz = (end - tip) * v
            d_out = float(ndi.distance_transform_edt(filled[end])[int(round(cy)), int(round(cx))] * v)
            # punta local: último corte con diente a menos de 0.6 mm del eje del conducto (en una raíz-lámina
            # ancha la punta del otro lado puede estar más abajo)
            disc = ((yy - cy0) ** 2 + (xx - cx0) ** 2) <= (0.6 / v) ** 2
            tip_loc = next((z for z in range(0, end + 1) if (mask[z] & disc).any()), end)
            if (end - tip_loc) * v > apical_zone_mm:        # extremo de un trozo a media altura, no es el ápice
                r["fragmentos_no_apicales"]["n"] += 1
                r["fragmentos_no_apicales"]["volumen_mm3"] = round(r["fragmentos_no_apicales"]["volumen_mm3"] + float(cvol[j - 1]), 3)
                continue
            # ¿sigue el hueco hacia la punta sin estar encerrado? hueco dentro de la envolvente convexa del corte
            gap, last, lum_tip = 0, end, False
            for z in range(end - 1, tip - 1, -1):
                if not mask[z].any():
                    break
                notch = convex_hull_image(mask[z]) & ~mask[z] & (((yy - cy) ** 2 + (xx - cx) ** 2) <= rr ** 2)
                a = notch.sum() * v ** 2
                if 0.005 <= a <= 0.5:
                    cy, cx = ndi.center_of_mass(notch)
                    gap, last = 0, z
                    lum_tip = z <= tip_loc + 1
                else:
                    gap += 1
                    if gap > 3:
                        break
            r["conductos"].append({
                "volumen_mm3": round(float(cvol[j - 1]), 3), "z_fin_vox": int(z0 + end),
                "yx_fin_vox": [int(sl[1].start + round(cy0)), int(sl[2].start + round(cx0))],
                "mm_a_la_ultima_seccion_con_diente_de_la_raiz": round(dz, 3),
                "mm_a_la_punta_local": round((end - tip_loc) * v, 3),
                "mm_a_la_pared_externa_en_el_plano": round(d_out, 3),
                "hueco_sin_etiquetar_hacia_la_punta_mm": round((end - last) * v, 3),
                "hueco_llega_a_la_punta_local": bool(lum_tip)})
        out["raices"].append(r)
    return out


# ------------------------------------------------------------------------------------- control
def figures(folder, norm, tooth, solo, cam, z_f, apex, crown_high, voxel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    full = solo | cam
    cz, cy, cx = [int(round(c)) for c in ndi.center_of_mass(full)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 12))
    for a, (img, s, c, t) in zip(ax, ((norm[:, :, cx], solo[:, :, cx], cam[:, :, cx], f"sagital x={cx}"),
                                      (norm[:, cy, :], solo[:, cy, :], cam[:, cy, :], f"coronal y={cy}"))):
        a.imshow(np.asarray(img[::3, ::3], np.float32), cmap="gray", vmin=0, vmax=1.6, aspect="equal")
        rgb = np.zeros(s[::3, ::3].shape + (4,))
        rgb[c[::3, ::3] > 0] = (0, 0.8, 1, 0.8)
        rgb[s[::3, ::3] > 0] = (0, 1, 0, 0.9)
        a.imshow(rgb, aspect="equal")
        a.axhline(z_f / 3, color="yellow", lw=0.8)
        a.set_title(f"{folder.name} {t}: verde = solo conducto, cian = cámara y cavidades, amarillo = orificio z={z_f}")
        a.axis("off")
    plt.tight_layout(); plt.savefig(folder / "control_split.png", dpi=50); plt.close(fig)
    raices = [dict(c, raiz=r["raiz"], z_punta_vox=r["z_punta_vox"]) for r in apex["raices"] for c in r["conductos"]]
    if not raices:
        return
    fig, ax = plt.subplots(len(raices), 2, figsize=(10, 6 * len(raices)), squeeze=False)
    for k, r in enumerate(raices):
        z_end, z_tip = r["z_fin_vox"], r["z_punta_vox"]
        yc, xc = r["yx_fin_vox"]
        za, zb = max(z_tip - 40, 0), min(z_end + 160, norm.shape[0])
        h = 90
        for j, (img, s, nm) in enumerate(((norm[za:zb, yc - h:yc + h, xc], solo[za:zb, yc - h:yc + h, xc], "sagital"),
                                          (norm[za:zb, yc, xc - h:xc + h], solo[za:zb, yc, xc - h:xc + h], "coronal"))):
            a = ax[k, j]
            a.imshow(np.asarray(img, np.float32), cmap="gray", vmin=0, vmax=1.6, origin="lower", aspect="equal")
            a.contour(np.asarray(s), [0.5], colors="lime", linewidths=0.8)
            a.axhline(z_tip - za, color="red", lw=0.8)
            a.axhline(z_end - za, color="yellow", lw=0.8)
            a.set_title(f"raíz {r['raiz']} conducto {k + 1} {nm}: rojo = último corte con diente de la raíz, amarillo = fin del "
                        f"conducto ({r['mm_a_la_ultima_seccion_con_diente_de_la_raiz']} mm; punta local {r['mm_a_la_punta_local']} mm)",
                        fontsize=7)
    plt.tight_layout(); plt.savefig(folder / "control_apice.png", dpi=50); plt.close(fig)


# --------------------------------------------------------------------------------------- main
def process(folder: Path, a_c, sostenido_mm, root_min_mm2, lumen_radius_mm, min_piece_mm3=0.005, gap_mm=0.13, apical_zone_mm=2.0):
    info = json.loads((folder / "info.json").read_text())
    v = float(info["voxel_mm"])
    src = folder / NAMES["completo"]
    if not src.exists():
        shutil.copy2(folder / "canal.npy", src)          # canal.npy queda intacto
    canal = np.load(src).astype(bool)
    norm = np.load(folder / "norm.npy", mmap_mode="r")
    tooth = largest_component(norm > 0.5)
    crown_high, ori = orientation(tooth, norm, v)
    ori["corona_en_z"] = "alto" if crown_high else "bajo"
    nz = canal.shape[0]
    if not crown_high:                                    # trabajamos siempre con la corona hacia z alto
        canal, tooth = canal[::-1], tooth[::-1]
        norm = norm[::-1]
    big, nsec = section_stats(canal, v)
    zs = np.flatnonzero(canal.any(axis=(1, 2)))
    run = max(int(round(sostenido_mm / v)), 1)
    z_f = find_floor(big, int(zs.min()), int(zs.max()), a_c, run)
    avisos = []
    if ori["dudosa"]:
        avisos.append("orientación dudosa: el esmalte no es claramente mayor en un extremo")
    if z_f is None:
        avisos.append("no se detectó cámara (ninguna sección llega al umbral): todo el conducto completo queda como conducto")
        z_f = int(zs.max()) + 1
    solo = canal.copy()
    solo[z_f:] = False
    pl, pn = ndi.label(solo)
    if pn:
        small = np.bincount(pl.ravel(), minlength=pn + 1) * v ** 3 < min_piece_mm3
        small[0] = False
        solo &= ~small[pl]
    cam = canal & ~solo
    # margen: mayor sección en la raíz, medida a >= 0.5 mm por debajo del orificio (justo antes la sección ya está subiendo)
    z_marg = z_f - int(round(0.5 / v))
    root_max = float(big[int(zs.min()):z_marg].max()) if z_marg > zs.min() else 0.0
    if root_max > 0.8 * a_c:
        avisos.append(f"margen bajo: la mayor sección de la raíz ({root_max:.2f} mm²) está cerca del umbral ({a_c})")
    apex = roots_and_apex(tooth, solo, z_f, v, root_min_mm2, lumen_radius_mm, gap_mm, apical_zone_mm)
    zz = np.flatnonzero(solo.any(axis=(1, 2)))
    to_orig = (lambda z: int(z)) if crown_high else (lambda z: int(nz - 1 - z))
    solo_z = sorted([to_orig(zz.min()), to_orig(zz.max())])
    figures(folder, norm, tooth, solo, cam, z_f, apex, crown_high, v)
    unflip = (lambda x: np.ascontiguousarray(x)) if crown_high else (lambda x: np.ascontiguousarray(x[::-1]))
    np.save(folder / NAMES["solo"], unflip(solo).astype(np.uint8))
    np.save(folder / NAMES["camara"], unflip(cam).astype(np.uint8))
    for r in apex["raices"]:                              # z de las raíces en índices del volumen original
        r["z_punta_vox"] = to_orig(r["z_punta_vox"])
        for c in r["conductos"]:
            c["z_fin_vox"] = to_orig(c["z_fin_vox"])       # y, x no cambian al voltear z
    apex["z_furca_vox"] = to_orig(apex["z_furca_vox"])
    info["orientacion"] = ori
    info["separacion"] = {
        "criterio": ("orificio = primer corte, desde el ápice, cuya mayor sección 2D del conducto es >= area_camara_mm2 "
                     "durante >= sostenido_mm; lo coronal a ese corte es camara_y_cavidades. Ver docstring de split_canal.py."),
        "parametros": {"area_camara_mm2": a_c, "sostenido_mm": sostenido_mm, "seccion_minima_mm2": 0.02,
                       "min_piece_mm3": min_piece_mm3, "root_min_mm2": root_min_mm2, "lumen_radius_mm": lumen_radius_mm,
                       "gap_mm_union_de_trozos": gap_mm, "zona_apical_mm": apical_zone_mm},
        "z_orificio_vox": to_orig(z_f),
        "canal_solo_z_vox": solo_z,
        "mayor_seccion_en_la_raiz_mm2": round(root_max, 3),
        "volumen_mm3": {"completo": round(float(canal.sum() * v ** 3), 3), "solo_conducto": round(float(solo.sum() * v ** 3), 3),
                        "camara_y_cavidades": round(float(cam.sum() * v ** 3), 3)},
        "archivos": NAMES, "avisos": avisos}
    info["apice"] = apex
    (folder / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({"orientacion": ori, "separacion": info["separacion"], "apice": apex}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", type=Path, nargs="+", help="results/microct/Diente_XX")
    ap.add_argument("--area-camara-mm2", type=float, default=1.5)
    ap.add_argument("--sostenido-mm", type=float, default=0.1)
    ap.add_argument("--root-min-mm2", type=float, default=0.3, help="sección mínima de diente para contar una raíz en un corte")
    ap.add_argument("--lumen-radius-mm", type=float, default=0.2)
    a = ap.parse_args()
    for f in a.folders:
        process(f, a.area_camara_mm2, a.sostenido_mm, a.root_min_mm2, a.lumen_radius_mm)
