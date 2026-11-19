"""
Valida las medidas (medir_conductos.py / canal_medidas.py) con conductos ARTIFICIALES cuya forma conocemos:
un tramo recto en el orificio + un arco de radio R, sección circular u oval, diámetro que se estrecha hacia el
ápice y una inclinación respecto a z. Cada caso se mide a 15 µm (binario, como el µCT) y a 75 µm (etiqueta
BLANDA obtenida promediando bloques de 5×5×5: es lo que vería un CBCT con segmentación perfecta).

    python validar_conducto_artificial.py [--suav-mm 0.05 0.1 0.15 0.25] [--out results/medidas]

Qué se compara (verdad analítica frente a medida):
  * longitud de arco desde el plano del orificio hasta el final;   * κ(s) a lo largo del conducto;
  * diámetros (equivalente, mayor, menor, esfera inscrita) a 0,5 / 1 / 2 / 3 mm del extremo apical;
  * ángulos: la MISMA función aplicada a la línea central exacta (mide solo el error de la cadena de medida,
    no el de la definición).
Esto valida el método, no el diente: los conductos artificiales son más sencillos que uno real.
"""
import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):                       # consola de Windows: cp1252 no tiene κ ni ° ni ⁻¹
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(HERE))
import canal_medidas as cm  # noqa: E402
import medir_conductos as mc  # noqa: E402

V_FINO = 0.015            # mm: 75 µm = 5 × 15 µm, así el promedio por bloques es exacto


# ------------------------------------------------------------------------------------------ conducto analítico
def linea_central_analitica(L_recta, R, L_total, tilt_deg, ds=0.004, s_ini=-1.0):
    """Puntos (z, y, x) en mm, tangentes y κ de un tramo recto (hacia abajo, -z) seguido de un arco de radio R,
    todo girado `tilt_deg` alrededor de y. s=0 es el plano del orificio (z=0)."""
    s = np.arange(s_ini, L_total + ds / 2, ds)
    a = np.zeros_like(s)                                  # lateral (en el plano de curvatura)
    b = -s.copy()                                         # eje del conducto (z decrece al bajar)
    kap = np.zeros_like(s)
    if R is not None:
        phi = np.clip((s - L_recta) / R, 0, None)
        arc = s > L_recta
        a[arc] = R - R * np.cos(phi[arc])
        b[arc] = -L_recta - R * np.sin(phi[arc])
        kap[arc] = 1.0 / R
    be = np.deg2rad(tilt_deg)
    z = b * np.cos(be) - a * np.sin(be)
    x = b * np.sin(be) + a * np.cos(be)
    P = np.stack([z, np.zeros_like(z), x], axis=1)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    return s, P, T, kap


def voxelizar(s, P, T, r_top, r_ap, L_total, aspecto, v):
    """Máscara 0/1 a tamaño de vóxel v. Sección elíptica: semieje r en el plano de curvatura y r·aspecto fuera."""
    r = r_ap + (r_top - r_ap) * np.clip(1 - s / L_total, 0, 1)
    rmax = max(r_top, r_ap)
    lo = P.min(axis=0) - rmax - 3 * v
    hi = P.max(axis=0) + rmax + 3 * v
    hi[0] = 0.0                                           # corte por el plano z = 0 (orificio)
    n = np.ceil((hi - lo) / v).astype(int) + 1
    zz, yy, xx = np.meshgrid(*[lo[k] + np.arange(n[k]) * v for k in range(3)], indexing="ij")
    X = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
    tree = cKDTree(P)
    dist, idx = tree.query(X, distance_upper_bound=rmax + 2 * v)
    ok = np.isfinite(dist)
    idx = np.where(ok, idx, 0)
    rel = X - P[idx]
    yhat = np.array([0.0, 1.0, 0.0])
    U = np.cross(T[idx], yhat)
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    du = (rel * U).sum(axis=1)
    dy = rel[:, 1]
    dt = (rel * T[idx]).sum(axis=1)
    dentro = ok & ((du / r[idx]) ** 2 + (dy / (r[idx] * aspecto)) ** 2 <= 1.0) & (X[:, 0] <= 0.0 + 1e-9)
    dentro &= ~((idx == len(P) - 1) & (dt > 0))           # sin tapa más allá del extremo apical
    m = dentro.reshape(n).astype(np.uint8)
    return m, lo


def a_75um(m):
    """Promedia bloques 5×5×5 → etiqueta blanda a 75 µm (fracción de conducto por vóxel)."""
    f = 5
    pad = [(0, (-s) % f) for s in m.shape]
    g = np.pad(m.astype(np.float32), pad)
    return g.reshape(g.shape[0] // f, f, g.shape[1] // f, f, g.shape[2] // f, f).mean(axis=(1, 3, 5))


# ------------------------------------------------------------------------------------------ medir y comparar
def medir(vol, v, a):
    """vol: 0/1 o blanda. Devuelve (res, tabla, Q) del conducto más grande."""
    gr = cm.agrupar_conductos((vol >= 0.5).astype(np.uint8), v, a.gap_mm, 0.0)
    g = gr[0]
    o = g["origen"]
    sl = tuple(slice(o[k], o[k] + g["mask"].shape[k]) for k in range(3))
    vol_g = vol[sl].astype(np.float32)
    edt3 = cm.edt_con_extension_superior(g["mask"], int(np.ceil(1.0 / v)), g["z_top"]).astype(np.float32)
    ramas, _ = cm.ramas_por_apices(g["mask"], v, f=max(1, int(round(a.coarse * 0.016 / v))) if v < 0.05 else 1, min_rama_mm=a.min_rama_mm,
                                   zona_apical_mm=3.0, max_ramas=a.max_ramas)
    res, tabla, Q, perf = mc.medir_rama(vol_g, edt3, v, ramas[0], a, g["z_top"] * v)
    res["_n_ramas"] = len(ramas)
    # origen del recorte (mm) para volver a coordenadas del volumen
    return res, tabla, Q, np.array(o) * v


def nombre_caso(c):
    return f"R{c['R']}_incl{c['tilt']}_asp{c['aspecto']}"


_CACHE = {}


def evaluar(caso, a, suav):
    s, P, T, kap = linea_central_analitica(caso["L_recta"], caso["R"], caso["L"], caso["tilt"])
    clave = nombre_caso(caso)
    if clave not in _CACHE:
        m, lo = voxelizar(s, P, T, caso["r_top"], caso["r_ap"], caso["L"], caso["aspecto"], V_FINO)
        _CACHE[clave] = (m, lo, a_75um(m))
    m, lo, m75 = _CACHE[clave]
    resultados = {}
    for etiqueta, vol, v in (("15um_binario", m, V_FINO), ("75um_blanda", m75, V_FINO * 5)):
        args = Namespace(**{**vars(a), "suav_mm": suav})
        t0 = time.time()
        res, tabla, Q, org = medir(vol, v, args)
        # el volumen se voxelizó con origen `lo`: Q está en mm del recorte; verdad en mm absolutos del generador
        # el centro del vóxel grueso i está en lo + (5·i + 2)·15 µm: la rejilla de 75 µm queda desplazada 2 vóxeles finos
        Qabs = Q + org + lo + (2 * V_FINO if v > V_FINO * 2 else 0.0)
        # --- verdad
        sel0 = s >= 0
        Pt = P[sel0]
        Lt = caso["L"]
        r = lambda d: caso["r_ap"] + (caso["r_top"] - caso["r_ap"]) * (1 - (Lt - d) / Lt)
        asp = caso["aspecto"]
        err = {"tiempo_s": round(time.time() - t0, 1), "n_ramas": res["_n_ramas"]}
        err["longitud_mm"] = {"verdad": Lt, "medida": res["longitud_arco_mm"], "error_%": 100 * (res["longitud_arco_mm"] - Lt) / Lt}
        # κ: verdad en el punto más cercano de la línea central exacta
        d_ap = tabla["d_desde_apice_mm"]
        s_med = Lt - d_ap
        k_true = np.interp(s_med, s, kap)
        k_med = tabla["kappa_mm-1"]
        fuera = (s_med > 0.4) & (s_med < Lt - 0.4)
        # ± 0,3 mm alrededor del cambio recto→arco la verdad es discontinua: no se evalúa
        if caso["R"] is not None:
            fuera &= np.abs(s_med - caso["L_recta"]) > 0.3
        err["kappa_rmse_mm-1"] = float(np.sqrt(np.mean((k_med[fuera] - k_true[fuera]) ** 2)))
        if caso["R"] is not None:
            en_arco = (s_med > caso["L_recta"] + 0.4) & (s_med < Lt - 0.4)
            err["kappa_mediana_en_arco"] = {"verdad": 1 / caso["R"], "medida": float(np.median(k_med[en_arco]))}
        # diámetros a distancias fijas
        dd = {}
        for d0 in (0.5, 1.0, 2.0, 3.0):
            clave = f"{d0:g}_mm_del_apice"
            if clave not in res["distancias_fijas"]:
                continue
            m_ = res["distancias_fijas"][clave]
            rr = r(d0)
            ver = {"d_eq_mm": 2 * rr * np.sqrt(asp), "d_mayor_mm": 2 * rr * max(1, asp), "d_menor_mm": 2 * rr * min(1, asp),
                   "d_edt_mm": 2 * rr * min(1, asp)}
            dd[clave] = {k: {"verdad": round(ver[k], 4), "medida": m_[k], "error_%": None if m_[k] is None else round(100 * (m_[k] - ver[k]) / ver[k], 1)}
                         for k in ver}
        err["diametros"] = dd
        # ángulos: misma función sobre la línea central exacta (remuestreada igual)
        Qt = cm.remuestrear(Pt[Pt[:, 0] <= 1e-9 + 0.0], a.ds_mm)
        Qt = Qt[: int(np.searchsorted(np.cumsum(np.r_[0, np.linalg.norm(np.diff(Qt, axis=0), axis=1)]), Lt)) + 1]
        ds_t = cm.longitud_polilinea(Qt) / (len(Qt) - 1)
        sch_t = cm.angulo_schneider_3d(Qt, ds_t, tol_mm=a.tol_mm)
        giro_t, _ = cm.giro_maximo(Qt, ds_t, Dk_mm=a.Dk_mm)
        err["schneider_3d_deg"] = {"verdad": sch_t.get("angulo_deg"), "medida": res["schneider_3d"].get("angulo_deg")}
        err["giro_max_Dk_deg"] = {"verdad": giro_t, "medida": res["giro_max_en_Dk"]["angulo_deg"]}
        err["distancia_de_la_linea_central_a_la_verdad_mm"] = {
            "media": float(np.mean(cKDTree(Pt).query(Qabs)[0])), "max": float(np.max(cKDTree(Pt).query(Qabs)[0]))}
        err["avisos"] = res["avisos"]
        resultados[etiqueta] = err
    return resultados


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suav-mm", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.25])
    ap.add_argument("--out", type=Path, default=HERE.parents[2] / "results" / "medidas")
    ap.add_argument("--casos", nargs="*", help="subconjunto por nombre (por defecto todos)")
    a0 = Namespace(gap_mm=0.13, min_mm3=0.0, coarse=2, min_rama_mm=2.0, min_descenso_mm=1.0, max_ramas=4, ds_mm=0.04, suav_mm=0.15, mitad_mm=1.0, Dk_mm=1.0,
                   tol_mm=0.1, ext_max_mm=2.0, metodo="ramas")
    a = ap.parse_args()
    casos = [
        {"R": 6.0, "L_recta": 3.0, "L": 11.0, "tilt": 0, "aspecto": 1.0, "r_top": 0.35, "r_ap": 0.12},
        {"R": 4.0, "L_recta": 2.0, "L": 8.0, "tilt": 30, "aspecto": 1.0, "r_top": 0.35, "r_ap": 0.12},
        {"R": 6.0, "L_recta": 3.0, "L": 11.0, "tilt": 15, "aspecto": 0.5, "r_top": 0.4, "r_ap": 0.15},
        {"R": None, "L_recta": 0.0, "L": 10.0, "tilt": 20, "aspecto": 1.0, "r_top": 0.3, "r_ap": 0.12},
    ]
    if a.casos:
        casos = [c for c in casos if nombre_caso(c) in a.casos]
    salida = []
    for c in casos:
        for sg in a.suav_mm:
            t0 = time.time()
            r = evaluar(c, a0, sg)
            salida.append({"caso": nombre_caso(c), "parametros": c, "suav_mm": sg, "resultados": r})
            print(f"=== {nombre_caso(c)}  suav {sg} mm  ({time.time() - t0:.0f} s)")
            for et, e in r.items():
                L = e["longitud_mm"]
                print(f"  [{et}] longitud verdad {L['verdad']:.3f} medida {L['medida']} ({L['error_%']:+.2f} %); "
                      f"κ RMSE {e['kappa_rmse_mm-1']:.3f} mm⁻¹; línea central a {e['distancia_de_la_linea_central_a_la_verdad_mm']['media'] * 1000:.0f} µm de media "
                      f"(máx {e['distancia_de_la_linea_central_a_la_verdad_mm']['max'] * 1000:.0f}); ramas {e['n_ramas']}; "
                      f"Schneider {e['schneider_3d_deg']}; giro {e['giro_max_Dk_deg']}")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "validacion_conducto_artificial.json").write_text(json.dumps(salida, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("guardado en", a.out / "validacion_conducto_artificial.json")


if __name__ == "__main__":
    main()
