"""Calibración del simulador de CBCT con un CBCT real (vóxel ~75 µm).

Mide, sobre un CBCT real, las tres cosas que fijan los rangos del azar del simulador:
  1. grises por estructura (esmalte, dentina, contenido del conducto, tejido de alrededor);
  2. ruido en zonas uniformes (desviación típica, forma, correlación entre vóxeles vecinos);
  3. perfil del borde (cuánto emborrona el escáner): ancho gaussiano sigma, por tipo de borde
     (esmalte–exterior, dentina–exterior, dentina–conducto) y por dirección (en el plano / en z).

Las mismas funciones (`measure`) se aplican después a CBCT simulados para comprobar que el
simulador reproduce lo medido (simulate_cbct.py --check), por eso aceptan una lista de volúmenes.

Uso:
    python calibrate_cbct.py <cbct.nrrd> --seg <Segmentation.seg.nrrd> --out results/cbct_calibration/caso01

La segmentación solo sirve para decir DÓNDE está el conducto (zona de interés); si es un borrador,
los grises de su interior son aproximados (se usan solo vóxeles profundos). Nada de esto se copia
tal cual al simulador: son los centros de los rangos (ver simulate_cbct_config.yaml).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.special import erfc


# ----------------------------------------------------------------------------- lectura
def load_crop(cbct: Path, seg: Path | None, margin_mm: float = 2.3):
    """Lee el NRRD (ejes x,y,z) y recorta alrededor de la segmentación. Devuelve (img, seg, vóxel mm)."""
    import nrrd
    img, h = nrrd.read(str(cbct))
    voxel = float(np.abs(np.asarray(h["space directions"], float)).max())
    m = nrrd.read(str(seg))[0] > 0 if seg else None
    if m is not None:
        idx = np.nonzero(m)
        mg = int(round(margin_mm / voxel))
        sl = tuple(slice(max(a.min() - mg, 0), min(a.max() + mg + 1, s)) for a, s in zip(idx, img.shape))
        img, m = img[sl], m[sl]
    return img.astype(np.float32), m, voxel


def pct(x):
    p = np.percentile(x, [5, 25, 50, 75, 95])
    return {"p5": float(p[0]), "p25": float(p[1]), "mediana": float(p[2]), "p75": float(p[3]),
            "p95": float(p[4]), "n": int(x.size)}


# ------------------------------------------------------------------------------ grises
def _grey_one(img, seg, voxel):
    sm = ndi.gaussian_filter(img, 1.0)
    out = {}
    en = ndi.binary_erosion(sm > 2800, iterations=2)
    out["esmalte"] = pct(img[en]) if en.any() else None
    if seg is not None and seg.any():
        d = ndi.distance_transform_edt(~seg) * voxel
        ring = ndi.binary_erosion((d > 0.2) & (d < 1.5) & (sm > 1400) & (sm < 2000), iterations=2)
        out["dentina"] = pct(img[ring]) if ring.any() else None
        deep = ndi.distance_transform_edt(seg) * voxel >= 0.15
        out["interior_conducto_vox_profundos"] = pct(img[deep]) if deep.any() else None
        far = d > 3.0
        out["tejido_lejano"] = pct(img[far]) if far.any() else None
    return out


def grey_levels(imgs, segs, voxel):
    """Percentiles de gris por estructura. Con varios volúmenes, mediana de los resultados de cada uno."""
    per = [_grey_one(i, s, voxel) for i, s in zip(imgs, segs)]
    out = {"min": float(min(i.min() for i in imgs)),
           "fraccion_en_el_minimo": float(np.mean([(i <= i.min() + 0.5).mean() for i in imgs]))}
    for k in ("esmalte", "dentina", "interior_conducto_vox_profundos", "tejido_lejano"):
        v = [p[k] for p in per if p.get(k)]
        out[k] = {q: float(np.median([x[q] for x in v])) for q in v[0] if q != "n"} | {"n_volumenes": len(v)} if v else None
    return out


# -------------------------------------------------------------------------------- ruido
def block_noise(img, lo, hi, b=9):
    """Ruido en bloques bxbxb con gris medio en [lo,hi]: residuo tras ajustar un plano (4 parámetros).

    Devuelve listas por bloque aceptado: media, sd, autocorrelación lag 1 (x,y,z), (asimetría, curtosis
    en exceso). Se rechazan los bloques que cruzan un borde (rango p5–p95 > 1500).
    """
    zz, yy, xx = np.meshgrid(*[np.arange(b) - (b - 1) / 2] * 3, indexing="ij")
    A = np.stack([np.ones(b ** 3), xx.ravel(), yy.ravel(), zz.ravel()], 1)
    pinv = np.linalg.pinv(A)
    means, sds, ac, sk = [], [], [], []
    nx, ny, nz = [s // b for s in img.shape]
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                blk = img[i*b:(i+1)*b, j*b:(j+1)*b, k*b:(k+1)*b]
                mu = blk.mean()
                if not (lo <= mu <= hi):
                    continue
                p5, p95 = np.percentile(blk, [5, 95])
                if p95 - p5 > 1500:
                    continue
                v = blk.ravel()
                r = v - A @ (pinv @ v)
                sd = r.std()
                sds.append(float(sd * np.sqrt(b ** 3 / (b ** 3 - 4))))
                means.append(float(mu))
                r3 = r.reshape(b, b, b)
                ac.append([float((r3[1:] * r3[:-1]).mean() / r3.var()),
                           float((r3[:, 1:] * r3[:, :-1]).mean() / r3.var()),
                           float((r3[:, :, 1:] * r3[:, :, :-1]).mean() / r3.var())])
                z = r / sd
                sk.append([float((z ** 3).mean()), float((z ** 4).mean() - 3)])
    return means, sds, ac, sk


NOISE_CLASSES = {"esmalte": (2900, 4500), "dentina": (1450, 1850), "blando": (-300, 150)}


def noise_stats(imgs):
    """Ruido por clase de gris (esmalte, dentina, tejido blando/médula), juntando todos los volúmenes."""
    out = {}
    for name, (lo, hi) in NOISE_CLASSES.items():
        M, S, AC, SK = [], [], [], []
        for img in imgs:
            m, s, ac, sk = block_noise(img, lo, hi)
            M += m; S += s; AC += ac; SK += sk
        if len(S) >= 5:
            AC, SK = np.array(AC), np.array(SK)
            out[name] = {"n_bloques": len(S), "gris_medio": float(np.median(M)), "sd_mediana": float(np.median(S)),
                         "sd_p25": float(np.percentile(S, 25)), "sd_p75": float(np.percentile(S, 75)),
                         "autocorr_lag1_x_y_z": [float(np.median(AC[:, k])) for k in range(3)],
                         "asimetria_mediana": float(np.median(SK[:, 0])),
                         "curtosis_exceso_mediana": float(np.median(SK[:, 1]))}
        else:
            out[name] = {"n_bloques": len(S)}
    return out


# -------------------------------------------------------------------------------- bordes
def esf(t, s):
    return 0.5 * erfc(t / (s * np.sqrt(2)))


def _edge_one(img, seg, voxel, n_cand, half_mm, rng, tt):
    """Perfiles normalizados y alineados de un volumen, por clase. Devuelve dict clase -> lista de perfiles."""
    g = np.stack([ndi.gaussian_filter(img, 1.2, order=o) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))])
    gm = np.sqrt((g ** 2).sum(0))
    cand = (gm >= ndi.maximum_filter(gm, size=3)) & (gm > np.percentile(gm, 97))
    pts = np.argwhere(cand)
    pts = pts[rng.permutation(len(pts))[:n_cand]]
    u = -g[:, pts[:, 0], pts[:, 1], pts[:, 2]].T                # hacia el lado bajo
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
    h = half_mm / voxel
    t = np.arange(-h, h + 1e-6, 0.25)
    coords = pts[:, None, :] + t[None, :, None] * u[:, None, :]
    prof = ndi.map_coordinates(img, coords.reshape(-1, 3).T, order=1, mode="nearest").reshape(len(pts), len(t))
    hi_w, lo_w = t <= -0.75 * h, t >= 0.75 * h
    H, L = prof[:, hi_w].mean(1), prof[:, lo_w].mean(1)
    contrast = H - L
    flat = (prof[:, hi_w].std(1) < 0.12 * contrast) & (prof[:, lo_w].std(1) < 0.12 * contrast) & (contrast > 800)
    if seg is not None and seg.any():
        near = (ndi.distance_transform_edt(~seg) * voxel)[pts[:, 0], pts[:, 1], pts[:, 2]] < 0.6
    else:
        near = np.zeros(len(pts), bool)
    az = np.abs(u[:, 2])
    base = {"esmalte_exterior": (H > 2600) & (L < 600) & ~near,
            "dentina_exterior": (H > 1300) & (H < 2200) & (L < 500) & ~near,
            "dentina_conducto": (H > 1300) & (H < 2200) & (L < 700) & near}
    sel = {}
    for c, m in base.items():
        sel[c] = m
        sel[c + "|plano_xy"] = m & (az < 0.4)
        sel[c + "|eje_z"] = m & (az > 0.85)
    res = {}
    for c, m in sel.items():
        ps = []
        for i in np.flatnonzero(m & flat):
            p = (prof[i] - L[i]) / contrast[i]
            if not (p[0] > 0.8 and p[-1] < 0.2):
                continue
            k = np.flatnonzero((p[:-1] >= 0.5) & (p[1:] < 0.5))
            if len(k) == 0:
                continue
            k = k[np.argmin(np.abs(t[k]))]
            t0 = t[k] + (p[k] - 0.5) / (p[k] - p[k + 1]) * (t[k + 1] - t[k])
            if abs(t0) > 2.0:
                continue
            ps.append(np.interp(tt / voxel + t0, t, p))
        res[c] = ps
    return res


def edge_profiles(imgs, segs, voxel, n_cand=30000, half_mm=0.6, seed=0):
    """Anchura del borde (sigma, mm) por clase, juntando los perfiles de todos los volúmenes.

    Un candidato es un máximo local del gradiente; su perfil se muestrea en 3D (trilineal) de alto
    (t<0) a bajo (t>0) y se acepta solo si ambas mesetas son planas (borde limpio, no una estructura
    fina). Se normaliza a 1->0, se alinea en el 50 % y se ajusta una función de error (borde
    gaussiano emborronado): sigma es la anchura medida e INCLUYE el vóxel y la interpolación.
    """
    rng = np.random.default_rng(seed)
    tt = np.arange(-0.45, 0.4501, 0.0125)
    allp = {}
    for img, seg in zip(imgs, segs):
        for c, ps in _edge_one(img, seg, voxel, n_cand, half_mm, rng, tt).items():
            allp.setdefault(c, []).extend(ps)
    res = {}
    for c, ps in allp.items():
        if len(ps) < 30:
            res[c] = {"n": len(ps)}
            continue
        med = np.median(np.array(ps), axis=0)
        fit = least_squares(lambda s: esf(tt, s[0]) - med, [0.1], bounds=(0.01, 0.6))
        res[c] = {"n": len(ps), "sigma_mm": float(fit.x[0]), "fwhm_mm": float(2.355 * fit.x[0]),
                  "rms_ajuste": float(np.sqrt(np.mean(fit.fun ** 2))),
                  "perfil_t_mm": tt[::4].round(4).tolist(), "perfil": med[::4].round(4).tolist()}
    return res


# ----------------------------------------------------------------------------------- main
def measure(imgs, segs, voxel):
    """Todas las medidas sobre una lista de volúmenes (con su máscara de conducto, o None)."""
    return {"voxel_mm": voxel, "n_volumenes": len(imgs), "grises": grey_levels(imgs, segs, voxel),
            "ruido": noise_stats(imgs), "bordes": edge_profiles(imgs, segs, voxel)}


def figure(img, seg, res, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    cx = int(np.nonzero(seg)[0].mean()) if seg is not None else img.shape[0] // 2
    ax[0].imshow(img[cx].T, cmap="gray", origin="lower", vmin=-500, vmax=3500)
    if seg is not None:
        ax[0].contour(seg[cx].T, [0.5], colors="r", linewidths=0.5)
    ax[0].set_title(f"corte sagital x={cx}: borrador de segmentación en rojo")
    for c, r in res["bordes"].items():
        if "perfil" in r and "|" not in c:
            ax[1].plot(r["perfil_t_mm"], r["perfil"], label=f"{c}: sigma={r['sigma_mm']*1000:.0f} µm (n={r['n']})")
    ax[1].set_xlabel("distancia al borde (mm); alto -> bajo"); ax[1].set_title("perfil medio del borde"); ax[1].legend(fontsize=7)
    names = [k for k in res["ruido"] if "sd_mediana" in res["ruido"][k]]
    ax[2].bar(names, [res["ruido"][k]["sd_mediana"] for k in names])
    ax[2].set_title("sd del ruido por clase (mediana de bloques)")
    plt.tight_layout(); plt.savefig(out, dpi=60); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cbct", type=Path)
    ap.add_argument("--seg", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    img, seg, vox = load_crop(a.cbct, a.seg)
    res = measure([img], [seg], vox)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "calibration.json").write_text(json.dumps(res, indent=1))
    figure(img, seg, res, a.out / "calibration.png")
    print(json.dumps({k: v for k, v in res.items() if k != "bordes"}, indent=1))
    for c, r in res["bordes"].items():
        print(c, {k: v for k, v in r.items() if not k.startswith("perfil")})
