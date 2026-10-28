"""Métricas del conducto por tercios (apical, medio, coronal) para parches simulados.

Los tercios se definen sobre la extensión en z del conducto (solo conducto) del DIENTE COMPLETO, no del
parche: así "apical" quiere decir lo mismo en todos los parches. Hace falta que el info.json del diente
tenga `orientacion.corona_en_z` ("alto"/"bajo") y `separacion.canal_solo_z_vox` ([zmin, zmax], en vóxeles
del µCT), que escribe analysis/microct/split_canal.py.

Pendiente (necesita un volumen completo simulado, que aún no existe): distancia del final del conducto
predicho al ápice real y error de cada medida (longitud, curvatura, volumen, diámetro).
"""
import numpy as np
from medpy.metric.binary import hd95 as _hd95

THIRDS = ("apical", "medio", "coronal")


def dice(pred, gt, eps=1e-8):
    pred, gt = pred.astype(bool), gt.astype(bool)
    s = pred.sum() + gt.sum()
    return float(2 * (pred & gt).sum() / (s + eps)) if s > 0 else float("nan")


def hd95(pred, gt, voxel_mm):
    """HD95 en mm; NaN si falta alguna de las dos máscaras (no se inventa un valor)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    return float(_hd95(pred, gt, voxelspacing=voxel_mm))


def third_map(info, start_z_vox, n_vox, m):
    """Tercio de cada corte axial (eje 0) de un parche: 0 apical, 1 medio, 2 coronal, -1 fuera del conducto.

    start_z_vox: z (en vóxeles del µCT) de la esquina del parche; n_vox: lado del parche en vóxeles del µCT
    (600 con un µCT de 16 µm); m: lado en vóxeles de 75 µm (128). Misma regla que `Tooth.third_of_z` del simulador.
    """
    z0, z1 = info["separacion"]["canal_solo_z_vox"]
    crown_high = info["orientacion"]["corona_en_z"] == "alto"
    z = start_z_vox + (np.arange(m) + 0.5) * n_vox / m
    # fracción del conducto medida desde la corona (0) hacia el ápice (1)
    apical_frac = (z1 - z) / max(z1 - z0, 1) if crown_high else (z - z0) / max(z1 - z0, 1)
    t = np.full(m, -1, int)
    inside = (z >= z0) & (z <= z1)
    t[inside] = np.where(apical_frac[inside] >= 2 / 3, 0, np.where(apical_frac[inside] >= 1 / 3, 1, 2))
    return t


def metrics_by_thirds(pred, gt, thirds, voxel_mm):
    """pred, gt: (D,H,W) binarias; thirds: (D,) de third_map. Devuelve {tercio: {dice, hd95_mm, n_gt}} y 'total'."""
    out = {}
    for k, name in enumerate(THIRDS):
        sel = thirds == k
        if not sel.any():
            out[name] = {"dice": float("nan"), "hd95_mm": float("nan"), "n_gt": 0}
            continue
        p, g = pred[sel], gt[sel]
        out[name] = {"dice": dice(p, g), "hd95_mm": hd95(p, g, voxel_mm), "n_gt": int(g.sum())}
    out["total"] = {"dice": dice(pred, gt), "hd95_mm": hd95(pred, gt, voxel_mm), "n_gt": int(gt.sum())}
    return out
