"""PRUEBA DE MONTAJE de extremo a extremo: red entrenada con CBCT simulados -> CBCT real de Isa -> comparación con su segmentación.

NO es un resultado: la red se entrenó con muy pocos dientes (3 µCT) y la segmentación de Isa es un BORRADOR
(`Segmentation.Status: inprogress`) que además incluye la cámara pulpar. La comparación es sobre todo cualitativa.

    python prueba_cbct_isa.py --checkpoint results/entrenamientos_sim/<carpeta>/last.pt [--out ../1_Data_Validation/results/prueba_cbct_isa]

Pasos:
  1. Lee el CBCT (NRRD, ejes x,y,z, vóxel 0,075 mm) y la segmentación de Isa y los pasa al orden (z,y,x) del simulador.
  2. Caja de análisis = caja de la segmentación + `--margen-mm` (3 mm). Para inferir se añade `--contexto-vox` (32 vóxeles) de
     contexto extra (si el volumen lo permite) para que los bordes de la caja no sufran efectos de borde; se mide solo dentro de la
     caja de análisis.
  3. Gris: la misma normalización que ve la red al entrenar: `normalize_patch` de dataloader/SimCBCT.py (percentiles 1 y 99,5 de
     cada ventana de 128³ -> 0..1). Las unidades crudas del CBCT de Isa (tipo Hounsfield, aire = -1000) son las que usa el simulador.
  4. Ventanas deslizantes de 128³ con paso 64 (50 % de solape) y fusión con peso suave (más peso en el centro de la ventana).
  5. Guarda probabilidad y máscara (umbral 0,5) en .npy y .nrrd; compara con Isa: Dice, HD95, volúmenes, por zona (conducto / cámara)
     y por componente; hace cuatro imágenes (3 axiales y 1 longitudinal).
  6. `--grises-sim N`: compara las estadísticas de gris que ve la red en N parches simulados frente a las ventanas del CBCT real.

Los datos del paciente no salen de la máquina: la carpeta de resultados está ignorada por git.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import scipy.ndimage as ndi
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
from dataloader.SimCBCT import normalize_patch  # noqa: E402  (misma función que ve la red al entrenar)
from metrics_canal import dice, hd95  # noqa: E402
from models.CanalUNet3D import CanalUNet3D  # noqa: E402

DATA = ROOT / "1_Data_Validation" / "datasets" / "CBCT_75um" / "Isa_caso00"
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", type=Path, required=True)
ap.add_argument("--cbct", type=Path, default=DATA / "CBCT.nrrd")
ap.add_argument("--seg", type=Path, default=DATA / "Segmentation.seg.nrrd")
ap.add_argument("--out", type=Path, default=ROOT / "1_Data_Validation" / "results" / "prueba_cbct_isa")
ap.add_argument("--margen-mm", type=float, default=3.0)
ap.add_argument("--contexto-vox", type=int, default=32)
ap.add_argument("--ventana", type=int, default=128)
ap.add_argument("--paso", type=int, default=64)
ap.add_argument("--umbral", type=float, default=0.5)
ap.add_argument("--z-camara", type=int, default=511,
                help="z global (en vóxeles del CBCT) desde el que la segmentación de Isa es 'cámara' (a ojo; ver el informe)")
ap.add_argument("--grises-sim", type=int, default=0, help="nº de parches simulados para comparar el gris de entrada de la red")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda")
torch.set_default_device("cuda")


def save_nrrd(path, vol_zyx, origin_xyz, voxel, kind):
    h = {"space": "left-posterior-superior", "space directions": np.diag([voxel] * 3), "space origin": np.asarray(origin_xyz, float),
         "kinds": ["domain"] * 3, "encoding": "gzip"}
    nrrd.write(str(path), np.ascontiguousarray(vol_zyx.transpose(2, 1, 0)), h)


# ------------------------------------------------------------------------------------------- 1. datos
t0 = time.time()
img_xyz, hdr = nrrd.read(str(a.cbct))
seg_xyz, _ = nrrd.read(str(a.seg))
voxel = float(np.abs(np.asarray(hdr["space directions"], float)).max())
origin = np.asarray(hdr["space origin"], float)
img = img_xyz.transpose(2, 1, 0).astype(np.float32)
seg = seg_xyz.transpose(2, 1, 0) > 0
del img_xyz, seg_xyz
nz = np.nonzero(seg)
mg = int(round(a.margen_mm / voxel))
lo = np.array([max(q.min() - mg, 0) for q in nz])
hi = np.array([min(q.max() + mg + 1, s) for q, s in zip(nz, img.shape)])
lo_i = np.maximum(lo - a.contexto_vox, 0)
hi_i = np.minimum(hi + a.contexto_vox, np.array(img.shape))
sl_a = tuple(slice(l, h) for l, h in zip(lo, hi))
sl_i = tuple(slice(l, h) for l, h in zip(lo_i, hi_i))
vol_i = img[sl_i]
print(f"CBCT {img.shape} (z,y,x) vóxel {voxel} mm | caja de análisis (z,y,x) {lo.tolist()}..{hi.tolist()} = {(hi - lo).tolist()} vóxeles "
      f"({', '.join(f'{(h - l) * voxel:.1f}' for l, h in zip(lo, hi))} mm) | caja de inferencia {(hi_i - lo_i).tolist()}", flush=True)

# ------------------------------------------------------------------------------------------- 2. red
ck = torch.load(a.checkpoint, map_location="cuda", weights_only=False)
m = ck["cfg"]["model"]
model = CanalUNet3D(1, 1, m["init_features"], m["depth"], m["dropout"]).to(DEV)
model.load_state_dict(ck["model"])
model.eval()
print(f"modelo {a.checkpoint.parent.name} | dientes fuera en el entrenamiento: {ck.get('holdout')}", flush=True)

# ------------------------------------------------------------------------------------------- 3. ventanas deslizantes
W, S = a.ventana, a.paso
pad = [max(W - s, 0) for s in vol_i.shape]
vol_p = np.pad(vol_i, [(0, p) for p in pad], mode="edge") if any(pad) else vol_i


def starts(n):
    s = list(range(0, n - W + 1, S))
    if s[-1] != n - W:
        s.append(n - W)
    return s


w1 = np.sin(np.pi * (np.arange(W) + 0.5) / W) ** 2
w1 = np.maximum(w1, 1e-3).astype(np.float32)
wt = torch.from_numpy(w1[:, None, None] * w1[None, :, None] * w1[None, None, :]).to(DEV)
acc = torch.zeros(vol_p.shape, dtype=torch.float32, device=DEV)
den = torch.zeros(vol_p.shape, dtype=torch.float32, device=DEV)
win_stats = []                                      # p1, p99.5 crudos y cuantiles del gris que ve la red
with torch.no_grad():
    for z in starts(vol_p.shape[0]):
        for y in starts(vol_p.shape[1]):
            for x in starts(vol_p.shape[2]):
                win = torch.from_numpy(vol_p[z:z + W, y:y + W, x:x + W]).to(DEV)
                xin = normalize_patch(win)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit = model(xin[None, None])
                p = torch.sigmoid(logit.float())[0, 0]
                acc[z:z + W, y:y + W, x:x + W] += p * wt
                den[z:z + W, y:y + W, x:x + W] += wt
                q = torch.quantile(win.flatten(), torch.tensor([0.01, 0.995], device=DEV)).tolist()
                nq = torch.quantile(xin.flatten()[::7], torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=DEV)).tolist()
                win_stats.append({"inicio_zyx": [z, y, x], "p1_crudo": q[0], "p99_5_crudo": q[1], "normalizado_p5_25_50_75_95": nq})
prob_i = (acc / den.clamp(min=1e-6)).cpu().numpy()[:vol_i.shape[0], :vol_i.shape[1], :vol_i.shape[2]]
del acc, den
print(f"inferencia: {len(win_stats)} ventanas de {W}^3, paso {S} | {time.time() - t0:.0f} s en total", flush=True)

# recorte a la caja de análisis
off = lo - lo_i
prob = prob_i[off[0]:off[0] + (hi - lo)[0], off[1]:off[1] + (hi - lo)[1], off[2]:off[2] + (hi - lo)[2]]
pred = prob >= a.umbral
gt = seg[sl_a]
crop = img[sl_a]
origin_box = origin + voxel * lo[::-1]                  # origen (x,y,z) de la caja de análisis
np.save(a.out / "prediccion_probabilidad.npy", prob.astype(np.float16))
np.save(a.out / "prediccion_mascara_umbral05.npy", pred.astype(np.uint8))
save_nrrd(a.out / "prediccion_probabilidad.nrrd", prob.astype(np.float32), origin_box, voxel, "prob")
save_nrrd(a.out / "prediccion_mascara_umbral05.nrrd", pred.astype(np.uint8), origin_box, voxel, "mask")
save_nrrd(a.out / "segmentacion_isa_recortada.nrrd", gt.astype(np.uint8), origin_box, voxel, "mask")
save_nrrd(a.out / "cbct_recortado.nrrd", crop.astype(np.float32), origin_box, voxel, "img")

# ------------------------------------------------------------------------------------------- 4. comparación
v3 = voxel ** 3
zg = np.arange(lo[0], hi[0])[:, None, None]
zona_canal = zg < a.z_camara                           # sólo conducto (según el z fijado a ojo); la cámara es lo de arriba
zona_camara = ~zona_canal


def comp(p, g, voxel_mm=voxel):
    return {"vol_pred_mm3": float(p.sum() * v3), "vol_isa_mm3": float(g.sum() * v3), "dice": dice(p, g), "hd95_mm": hd95(p, g, voxel_mm),
            "solape_mm3": float((p & g).sum() * v3)}


res = {"aviso": "PRUEBA DE MONTAJE: modelo con 3 dientes de entrenamiento y segmentación de Isa en borrador (incluye la cámara). No es un resultado.",
       "modelo": str(a.checkpoint), "voxel_mm": voxel, "caja_analisis_zyx": [lo.tolist(), hi.tolist()], "umbral": a.umbral,
       "z_camara_vox_global": a.z_camara}
res["todo"] = comp(pred, gt)
res["solo_zona_conducto"] = comp(pred & zona_canal, gt & zona_canal)
res["solo_zona_camara"] = comp(pred & zona_camara, gt & zona_camara)
res["prediccion_fuera_de_la_segmentacion_de_Isa_mm3"] = float((pred & ~gt).sum() * v3)
res["isa_sin_prediccion_mm3"] = float((gt & ~pred).sum() * v3)

# por componente de Isa (zona conducto)
lab_g, ng = ndi.label(gt & zona_canal, structure=np.ones((3, 3, 3)))
lab_p, npr = ndi.label(pred & zona_canal, structure=np.ones((3, 3, 3)))
comps = []
usados = set()
for i in range(1, ng + 1):
    mg_ = lab_g == i
    n_i = int(mg_.sum())
    if n_i < 200:
        continue
    zz = np.nonzero(mg_)[0]
    ids = [int(k) for k in np.unique(lab_p[mg_]) if k > 0]
    mp = np.isin(lab_p, ids) if ids else np.zeros_like(mg_)
    usados |= set(ids)
    zp = np.nonzero(mp)[0] if mp.any() else None
    comps.append({"componente_isa": i, "vol_isa_mm3": n_i * v3, "z_isa_vox_global": [int(zz.min() + lo[0]), int(zz.max() + lo[0])],
                  "extension_z_isa_mm": float((zz.max() - zz.min() + 1) * voxel),
                  "componentes_pred_que_lo_tocan": ids, "vol_pred_que_lo_toca_mm3": float(mp.sum() * v3),
                  "z_pred_vox_global": [int(zp.min() + lo[0]), int(zp.max() + lo[0])] if zp is not None else None,
                  "extension_z_pred_mm": float((zp.max() - zp.min() + 1) * voxel) if zp is not None else 0.0,
                  "isa_cubierto_por_pred_pct": float(100.0 * (mg_ & mp).sum() / n_i),
                  "dice": dice(mp, mg_), "hd95_mm": hd95(mp, mg_, voxel)})
res["componentes_isa_conducto"] = comps
fp = []
for k in range(1, npr + 1):
    if k in usados:
        continue
    mk = lab_p == k
    if mk.sum() * v3 >= 0.05:
        zz = np.nonzero(mk)[0]
        fp.append({"componente_pred": k, "vol_mm3": float(mk.sum() * v3), "z_vox_global": [int(zz.min() + lo[0]), int(zz.max() + lo[0])]})
res["componentes_pred_en_zona_conducto"] = int(npr)
res["componentes_pred_sin_solape_con_isa_ge_0.05mm3"] = fp
res["longitud_del_conducto"] = "no calculada: todavía no hay código de línea central; solo la extensión en z de cada componente"

# ------------------------------------------------------------------------------------------- 5. gris
res["gris"] = {"cbct_caja_percentiles_0.1_1_5_25_50_75_95_99_99.5_99.9": np.percentile(crop, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.5, 99.9]).round(0).tolist(),
               "cbct_caja_min_max": [float(crop.min()), float(crop.max())],
               "ventanas_real": {"n": len(win_stats), "p1_crudo": np.percentile([w["p1_crudo"] for w in win_stats], [0, 50, 100]).round(0).tolist(),
                                 "p99_5_crudo": np.percentile([w["p99_5_crudo"] for w in win_stats], [0, 50, 100]).round(0).tolist(),
                                 "normalizado_mediana_de_p5_25_50_75_95": np.median([w["normalizado_p5_25_50_75_95"] for w in win_stats], axis=0).round(3).tolist()}}
if a.grises_sim:
    import yaml
    from dataloader.SimCBCT import SimCBCT, discover_teeth
    cfg_tr = ck["cfg"]
    base = HERE
    root = (base / cfg_tr["data"]["root"]).resolve()
    simcfg = (base / cfg_tr["data"]["simulator_config"]).resolve()
    names = [p.name for p in discover_teeth(root)]
    src = SimCBCT(root, names, simcfg, seed=7, prefetch=0, crop_overrides={"third_weights": [1, 1, 1]})
    rng = np.random.default_rng(7)
    sp = []
    raw_min, raw_max = [], []
    for i in range(a.grises_sim):
        out, _ = src.sim.patch(src.teeth[i % len(src.teeth)], rng)
        im = out["image"]
        q = torch.quantile(im.flatten(), torch.tensor([0.01, 0.995], device=im.device)).tolist()
        xn = normalize_patch(im)
        nq = torch.quantile(xn.flatten()[::7], torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=im.device)).tolist()
        sp.append({"p1_crudo": q[0], "p99_5_crudo": q[1], "normalizado_p5_25_50_75_95": nq})
        raw_min.append(float(im.min()))
        raw_max.append(float(im.max()))
    res["gris"]["parches_simulados"] = {"n": len(sp), "p1_crudo": np.percentile([w["p1_crudo"] for w in sp], [0, 50, 100]).round(0).tolist(),
                                        "p99_5_crudo": np.percentile([w["p99_5_crudo"] for w in sp], [0, 50, 100]).round(0).tolist(),
                                        "min_crudo_percentiles_0_50_100": np.percentile(raw_min, [0, 50, 100]).round(0).tolist(),
                                        "max_crudo_percentiles_0_50_100": np.percentile(raw_max, [0, 50, 100]).round(0).tolist(),
                                        "normalizado_mediana_de_p5_25_50_75_95": np.median([w["normalizado_p5_25_50_75_95"] for w in sp], axis=0).round(3).tolist()}
(a.out / "resultados.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

# ------------------------------------------------------------------------------------------- 6. imágenes
LEYENDA = "PRUEBA DE MONTAJE — verde: segmentación de Isa (borrador, incluye cámara) · rojo: predicción de la red (3 dientes) · amarillo: coinciden"
VMIN, VMAX = -300, 2700


def overlay(ax, g, p):
    rgba = np.zeros(g.shape + (4,), np.float32)
    rgba[g & ~p] = (0.1, 0.9, 0.1, 0.55)
    rgba[p & ~g] = (1.0, 0.1, 0.1, 0.55)
    rgba[p & g] = (1.0, 0.9, 0.0, 0.6)
    ax.imshow(rgba, origin="lower", interpolation="nearest")


def figura(path, im, g, p, titulo, extent_mm=None):
    H, Wd = im.shape
    fig, axs = plt.subplots(1, 2, figsize=(2 * 5.0 * Wd / max(H, Wd) + 0.4, 5.0 * H / max(H, Wd) + 1.0))
    for k, ax in enumerate(axs):
        ax.imshow(im, cmap="gray", vmin=VMIN, vmax=VMAX, origin="lower", interpolation="bilinear")
        if k == 1:
            overlay(ax, g, p)
        ax.axis("off")
    L = 1.0 / voxel
    axs[0].plot([0.05 * Wd, 0.05 * Wd + L], [0.05 * H] * 2, color="white", lw=3)
    axs[0].text(0.05 * Wd + L / 2, 0.05 * H + 0.015 * H, "1 mm", color="white", ha="center", va="bottom", fontsize=9)
    fig.suptitle(titulo + "\n" + LEYENDA, fontsize=8.5)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


zc = np.nonzero(gt & zona_canal)[0] + lo[0]
niveles = [int(np.percentile(zc, q)) for q in (15, 50, 85)] if len(zc) else [int((lo[0] + hi[0]) / 2)] * 3
for k, zg_ in enumerate(niveles):
    zl = zg_ - lo[0]
    figura(a.out / f"axial_{k + 1}_z{zg_}.png", crop[zl], gt[zl], pred[zl], f"Corte axial z = {zg_} (vóxel del CBCT)")

# corte longitudinal: plano vertical (contiene z) que más conducto de Isa contiene
C = (gt & zona_canal).sum(axis=0).astype(np.float32)
cy, cx = ndi.center_of_mass(C) if C.sum() else (C.shape[0] / 2, C.shape[1] / 2)
T = max(C.shape)
tt = np.arange(-T, T + 1)
best = (-1, 0.0, 0.0)
for deg in np.arange(0, 180, 3.0):
    th = np.deg2rad(deg)
    for o in np.arange(-30, 31, 3.0):
        py, px = cy + o * np.cos(th), cx - o * np.sin(th)
        iy, ix = np.rint(py + tt * np.sin(th)).astype(int), np.rint(px + tt * np.cos(th)).astype(int)
        ok = (iy >= 0) & (iy < C.shape[0]) & (ix >= 0) & (ix < C.shape[1])
        s = float(C[iy[ok], ix[ok]].sum())
        if s > best[0]:
            best = (s, deg, o)
_, deg, o = best
th = np.deg2rad(deg)
py, px = cy + o * np.cos(th), cx - o * np.sin(th)
zs = np.arange(crop.shape[0])
Z, Tg = np.meshgrid(zs, tt, indexing="ij")
Y, X = py + Tg * np.sin(th), px + Tg * np.cos(th)


def muestra(v, order=1):
    return ndi.map_coordinates(v.astype(np.float32), [Z, Y, X], order=order, mode="constant", cval=0.0)


imp = muestra(crop)
gp = muestra(gt) >= 0.5
pp = muestra(pred) >= 0.5
cols = np.flatnonzero((imp > 500).any(axis=0))
c0, c1 = max(cols.min() - 10, 0), min(cols.max() + 10, imp.shape[1] - 1)
figura(a.out / "longitudinal.png", imp[:, c0:c1 + 1], gp[:, c0:c1 + 1], pp[:, c0:c1 + 1],
       f"Corte longitudinal (plano vertical a {deg:.0f}°, corona arriba)")
res["corte_longitudinal"] = {"angulo_grados": float(deg), "desplazamiento_vox": float(o), "pixeles_isa_en_el_corte": int(gp.sum()),
                             "pixeles_prediccion_en_el_corte": int(pp.sum()), "pixeles_solape": int((gp & pp).sum())}
res["axiales_z_global"] = niveles
(a.out / "resultados.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("todo", "solo_zona_conducto", "solo_zona_camara")}, indent=1))
print("componentes de Isa (zona conducto):")
for c in comps:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()})
print("predicción sin solape con Isa:", fp)
print(f"total {time.time() - t0:.0f} s")
