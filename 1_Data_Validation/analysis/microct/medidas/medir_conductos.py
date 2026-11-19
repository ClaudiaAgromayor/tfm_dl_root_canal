"""
Mide cada conducto de un diente (µCT) sobre `canal_solo_conducto.npy`: línea central, longitud (arco y cuerda),
curvatura κ = 1/R, ángulos y diámetro (barrido angular y esfera inscrita), por tercios y a distancias fijas del
ápice. Se aplica a cualquier carpeta results/microct/Diente_XX (parametrizado por carpeta).

    python medir_conductos.py ../../../results/microct/Diente_35 ../../../results/microct/Diente_51 ...
    python medir_conductos.py results/microct/Diente_52 --out results/medidas --suav-mm 0.15

Salida por diente en <out>/Diente_XX/: medidas.json, perfil_conducto_<N>.csv, control_linea_central.png,
control_perfil.png, control_secciones.png. Qué significa cada cosa: docs/MEDICION.md.

Referencia: la máscara del µCT es la del script (umbral), SIN validar por un experto. Un conducto = una rama
orificio → punta de un grupo de trozos unidos (ver agrupar_conductos).
"""
import argparse
import csv
import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):                       # consola de Windows: cp1252 no tiene κ ni ° ni ⁻¹
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(HERE))
import canal_medidas as cm  # noqa: E402

DIST_FIJAS_MM = (0.5, 1.0, 2.0, 3.0, 5.0)


def _mediana_ventana(y, d_arr, d0, semi=0.1):
    sel = (d_arr >= d0 - semi) & (d_arr <= d0 + semi) & np.isfinite(y)
    return float(np.median(y[sel])) if sel.any() else None


def _r(x, nd=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


def medir_rama(vol, edt3, v, P0, a, z_top_mm):
    """Todas las medidas de un conducto a partir de su polilínea inicial P0 (mm, z-y-x del recorte)."""
    ds = a.ds_mm
    P0 = np.vstack([[z_top_mm, P0[0, 1], P0[0, 2]], P0])                 # empieza en el plano del orificio
    vol = cm.extender_arriba(vol, int(round(z_top_mm / v)), 1.0, v)         # solo para muestrear las secciones
    P, rinfo = cm.refinar_linea_central(vol, v, P0, ds_mm=ds, mitad_mm=a.mitad_mm, ext_apical_max_mm=a.ext_max_mm,
                                        z_top_mm=z_top_mm, todo=(a.metodo == 'sistema'))
    Q = cm.remuestrear(cm.suavizar(P, ds, a.suav_mm), ds)
    n = len(Q)
    L = cm.longitud_polilinea(Q)
    s = np.arange(n) * ds
    s = s * (L / s[-1]) if s[-1] > 0 else s
    d = L - s                                                              # distancia desde el extremo apical
    kap = cm.curvatura(Q, L / (n - 1))
    perf = cm.perfil_secciones(vol, v, Q, mitad_mm=a.mitad_mm, edt3=edt3)
    ok = ~perf["toca_borde"]
    # área, anchura y espesor son del conjunto de la ventana (se anulan si algo la toca); los diámetros son solo los del
    # conducto que pasa por la línea y ya son NaN cuando ese trozo está recortado
    cols = {k: np.where(ok, perf[k], np.nan) for k in ("area_mm2", "ancho_mm", "espesor_mm")}
    cols.update({k: perf[k] for k in ("d_eq_mm", "d_mayor_mm", "d_menor_mm")})
    cols["n_componentes"] = perf["n_componentes"]
    cols["n_otros_trozos"] = perf["n_otros_trozos"]
    cols["alargamiento"] = perf["alargamiento"]
    cols["d_edt_mm"] = perf["d_edt_mm"]
    cuerda = float(np.linalg.norm(Q[-1] - Q[0]))
    sch = cm.angulo_schneider_3d(Q, L / (n - 1), tol_mm=a.tol_mm)
    giro, s_giro = cm.giro_maximo(Q, L / (n - 1), Dk_mm=a.Dk_mm)
    incl = cm.angulo_entre(Q[-1] - Q[0], np.array([-1.0, 0.0, 0.0]))        # eje z del escáner (corona arriba)
    marg = max(int(round(max(0.3, 2 * a.suav_mm) / (L / (n - 1)))), 2)       # extremos: el suavizado y el corte los distorsionan
    interior = np.zeros(n, bool)
    interior[marg:n - marg] = True
    jmax = int(np.flatnonzero(interior)[np.argmax(kap[interior])]) if interior.sum() > 3 else int(np.argmax(kap))
    avisos = []
    if rinfo["extension_apical_mm"] >= a.ext_max_mm - 1e-6:
        avisos.append("la extensión apical llegó al límite: el conducto puede seguir más abajo")
    if (~ok).mean() > 0.1:
        avisos.append(f"en el {100 * (~ok).mean():.0f} % de los cortes algo de la ventana toca el borde: área, anchura y espesor del "
                      "conjunto se excluyen en esos cortes")
    motivos = {}
    for m in perf["motivo_diametro"]:
        if m is not None:
            motivos[m] = motivos.get(m, 0) + 1
    sin_d = float((~np.isfinite(cols["d_eq_mm"])).mean())
    if sin_d > 0.3:
        avisos.append(f"no hay diámetro en el {100 * sin_d:.0f} % de los cortes; motivos: "
                      + "; ".join(f"{m} ({k})" for m, k in sorted(motivos.items(), key=lambda t: -t[1])))
    if rinfo["fraccion_puntos_con_retroceso_eliminados"] > 0.05:
        avisos.append(f"el camino inicial tenía un {100 * rinfo['fraccion_puntos_con_retroceso_eliminados']:.0f} % de puntos que suben (bucles o "
                      "gancho apical > 90°): se eliminaron")
    if L < 3.0:
        avisos.append("conducto muy corto (< 3 mm): ángulos y tercios poco fiables")
    frac_multi = float(np.nanmean(perf["n_componentes"] > 1))
    if frac_multi > 0.3:
        avisos.append(f"en el {100 * frac_multi:.0f} % de los cortes hay más de un trozo de conducto en la ventana de ±{a.mitad_mm:g} mm "
                      "(conducto vecino o istmo): los diámetros son solo los del trozo que contiene la línea; área, anchura y "
                      "espesor describen el conjunto")
    res = {
        "longitud_arco_mm": _r(L), "longitud_arco_sin_suavizar_mm": _r(cm.longitud_polilinea(P)),
        "cuerda_mm": _r(cuerda), "arco_entre_cuerda": _r(L / cuerda, 4),
        "kappa_max_mm-1": _r(kap[jmax]), "R_min_mm": _r(1 / kap[jmax]), "kappa_max_a_mm_del_apice": _r(d[jmax], 3),
        "kappa_mediana_mm-1": _r(np.median(kap[interior])), "kappa_p95_interior_mm-1": _r(np.percentile(kap[interior], 95)),
        "schneider_3d": sch, "giro_max_en_Dk": {"Dk_mm": a.Dk_mm, "angulo_deg": _r(giro, 2), "a_mm_del_orificio": _r(s_giro, 3)},
        "inclinacion_cuerda_vs_z_deg": _r(incl, 2),
        "extension_apical_mm": _r(rinfo["extension_apical_mm"], 3), "puntos_con_retroceso_eliminados": _r(rinfo["fraccion_puntos_con_retroceso_eliminados"], 3), "fraccion_cortes_con_varias_secciones": _r(frac_multi, 3),
        "fraccion_cortes_con_diametro": _r(1 - sin_d, 3), "motivos_sin_diametro": motivos,
        "desplazamiento_max_por_iteracion_mm": [_r(x, 4) for x in rinfo["desplazamiento_max_por_iteracion_mm"]],
        "distancias_fijas": {}, "tercios": {}, "avisos": avisos,
    }
    for d0 in DIST_FIJAS_MM:
        if d0 + 0.1 > L:
            continue
        res["distancias_fijas"][f"{d0:g}_mm_del_apice"] = {
            "d_eq_mm": _r(_mediana_ventana(cols["d_eq_mm"], d, d0), 4), "d_mayor_mm": _r(_mediana_ventana(cols["d_mayor_mm"], d, d0), 4),
            "d_menor_mm": _r(_mediana_ventana(cols["d_menor_mm"], d, d0), 4), "d_edt_mm": _r(_mediana_ventana(cols["d_edt_mm"], d, d0), 4),
            "anchura_mm": _r(_mediana_ventana(cols["ancho_mm"], d, d0), 4), "espesor_mm": _r(_mediana_ventana(cols["espesor_mm"], d, d0), 4),
            "n_componentes_mediana": _r(_mediana_ventana(cols["n_componentes"], d, d0), 1),
            "kappa_mm-1": _r(_mediana_ventana(kap, d, d0), 4)}
    for nombre, lo, hi in (("apical", 0, L / 3), ("medio", L / 3, 2 * L / 3), ("coronal", 2 * L / 3, L + 1e-9)):
        sel = (d >= lo) & (d < hi)
        t = {"desde_mm": _r(lo, 3), "hasta_mm": _r(hi, 3), "longitud_arco_mm": _r(hi - lo if nombre != "coronal" else L - lo, 3)}
        for k in ("d_eq_mm", "d_mayor_mm", "d_menor_mm", "d_edt_mm", "ancho_mm", "espesor_mm", "n_componentes"):
            t[k + "_mediana"] = _r(np.nanmedian(cols[k][sel]), 4) if np.isfinite(cols[k][sel]).any() else None
        kk = kap[sel & interior] if (sel & interior).any() else kap[sel]
        t["kappa_mediana_mm-1"] = _r(np.median(kk), 4)
        t["kappa_max_mm-1"] = _r(kk.max(), 4)
        res["tercios"][nombre] = t
    tabla = {"s_desde_orificio_mm": s, "d_desde_apice_mm": d, "z_mm": Q[:, 0], "y_mm": Q[:, 1], "x_mm": Q[:, 2],
             "kappa_mm-1": kap, **cols, "seccion_toca_borde": perf["toca_borde"].astype(int)}
    return res, tabla, Q, perf


def procesar(folder: Path, out_root: Path, a):
    t0 = time.time()
    info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
    v = float(info["voxel_mm"])
    solo = np.load(folder / "canal_solo_conducto.npy")
    corona = info.get("orientacion", {}).get("corona_en_z", "alto")
    if corona == "bajo":
        solo = solo[::-1]
    nombre = folder.name
    out = out_root / nombre
    out.mkdir(parents=True, exist_ok=True)
    grupos = cm.agrupar_conductos(solo, v, a.gap_mm, a.min_mm3)
    z_max = max(g["origen"][0] + g["z_top"] for g in grupos)
    print(f"[{nombre}] vóxel {v * 1000:.1f} µm, {len(grupos)} grupos (≥ {a.min_mm3} mm³), corona en z {corona}")
    parametros = {k: (str(x) if isinstance(x, Path) else x) for k, x in vars(a).items() if k != "folders"}
    salida = {"diente": nombre, "voxel_mm": v, "parametros": parametros, "corona_en_z": corona,
              "aviso_referencia": "máscara del µCT por umbral, sin validar por un experto", "grupos": []}
    lineas = []
    for gi, g in enumerate(grupos, 1):
        vol = g["mask"].astype(np.float32)
        edt3 = cm.edt_con_extension_superior(g["mask"], int(np.ceil(1.0 / v)), g["z_top"]).astype(np.float32)
        if a.metodo == "sistema":
            ramas, rd = [cm.linea_central_por_cortes(g["mask"], v)], {"metodo": "centroide de todo el grupo por corte axial"}
        elif a.metodo == "ramas":
            ramas, rd = cm.ramas_por_apices(g["mask"], v, f=a.coarse, min_rama_mm=a.min_rama_mm,
                                            zona_apical_mm=a.zona_apical_mm, max_ramas=a.max_ramas)
            if len(ramas) > 1:
                conj, entre = cm.fusionar_ramas_misma_seccion(g["mask"], v, ramas, a.umbral_fusion)
                rd["caminos_antes_de_fusionar"] = len(ramas)
                rd["similitud_maxima_entre_conductos_distintos"] = round(entre, 3)
                # de cada conducto se conserva el camino más largo (el que llega más lejos hacia el ápice)
                ramas = [max((ramas[i] for i in g_), key=cm.longitud_polilinea) for g_ in conj]
                ramas.sort(key=lambda P: -cm.longitud_polilinea(P))
        else:
            ramas, rd = cm.ramas_por_grosor(g["mask"], v, f=a.coarse, min_rama_mm=a.min_rama_mm,
                                            min_descenso_mm=a.min_descenso_mm, max_ramas=a.max_ramas)
        z_top_mm = g["z_top"] * v
        gout = {"grupo": gi, "volumen_mm3": round(g["vol_mm3"], 3), "z_top_vox": g["origen"][0] + g["z_top"],
                "z_bot_vox": g["origen"][0] + g["z_bot"], "origen_vox_zyx": list(g["origen"]),
                "ramas_detectadas": len(ramas), "diagnostico_ramas": rd, "conductos": []}
        if g["origen"][0] + g["z_top"] < z_max - 5:
            gout["aviso"] = "el grupo no llega al plano del orificio de los demás (¿trozo de otro conducto?)"
        for bi, P0 in enumerate(ramas, 1):
            res, tabla, Q, perf = medir_rama(vol, edt3, v, P0, a, z_top_mm)
            res["id"] = f"{gi}.{bi}"
            gout["conductos"].append(res)
            lineas.append((res["id"], Q, g["origen"], tabla, perf, vol, edt3))
            org = np.array(g["origen"]) * v                                  # recorte → coordenadas del diente (mm)
            desplaz = {"z_mm": org[0], "y_mm": org[1], "x_mm": org[2]}
            with open(out / f"perfil_conducto_{gi}_{bi}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(list(tabla.keys()))
                for fila in zip(*[np.round(tabla[k] + desplaz.get(k, 0.0), 5) for k in tabla]):
                    w.writerow(fila)
            print(f"   conducto {res['id']}: arco {res['longitud_arco_mm']} mm, cuerda {res['cuerda_mm']} mm, κmax {res['kappa_max_mm-1']} mm⁻¹, "
                  f"Schneider3D {res['schneider_3d'].get('angulo_deg')}°, avisos {res['avisos']}")
        salida["grupos"].append(gout)
    salida["cobertura_cortes_axiales"] = cobertura(solo, v, lineas)
    cb = salida["cobertura_cortes_axiales"]
    print(f"   cobertura: {cb['trozos_sin_linea']} de {cb['trozos_totales']} trozos de conducto de los cortes axiales sin ninguna línea cerca "
          f"({cb['porcentaje_area_sin_linea']} % del área)")
    (out / "medidas.json").write_text(json.dumps(salida, indent=1, ensure_ascii=False), encoding="utf-8")
    figuras(out, solo, v, lineas, nombre, a)
    print(f"[{nombre}] hecho en {time.time() - t0:.0f} s → {out}")


def cobertura(solo, v, lineas, paso_mm=0.25, min_mm2=0.01, tol_mm=0.1):
    """¿Queda algún conducto sin línea? En cortes axiales cada `paso_mm`, cada trozo de conducto (≥ `min_mm2`) debe tener
    al menos un punto de alguna línea a menos de `tol_mm`. Es independiente de
    cómo se trazaron las líneas. Devuelve el recuento, el área sin línea y la lista (todos los de ≥ 0,05 mm² y los primeros 15)."""
    pts = []
    for cid, Q, org, *_ in lineas:
        pts.append(Q + np.array(org) * v)                                   # a coordenadas del diente (mm, z-y-x)
    pts = np.vstack(pts)
    zs = np.flatnonzero(solo.any(axis=(1, 2)))
    total = sin = 0
    area_tot = area_sin = 0.0
    lista = []
    for z in np.arange(zs.min() * v, zs.max() * v, paso_mm):
        zi = int(round(z / v))
        sl = solo[zi].astype(bool)
        if not sl.any():
            continue
        sl = ndi.binary_closing(np.pad(sl, 2), iterations=1)[2:-2, 2:-2] | sl   # une fragmentos de 1 vóxel de un mismo conducto
        lab, n = ndi.label(sl)
        areas = ndi.sum(sl, lab, range(1, n + 1)) * v * v
        cerca = pts[np.abs(pts[:, 0] - zi * v) <= v * 1.5]                  # puntos de línea en este corte
        pix = [(int(round(p[1] / v)), int(round(p[2] / v))) for p in cerca]
        pix = [(y, x) for y, x in pix if 0 <= y < sl.shape[0] and 0 <= x < sl.shape[1]]
        for k, ar in enumerate(areas, 1):
            if ar < min_mm2:
                continue
            total += 1
            area_tot += ar
            dk = ndi.distance_transform_edt(lab != k) * v                   # distancia (mm) a este trozo
            if not any(dk[y, x] <= tol_mm for y, x in pix):
                sin += 1
                area_sin += ar
                if len(lista) < 15 or ar >= 0.05:
                    yy, xx = np.nonzero(lab == k)
                    lista.append({"z_mm": round(zi * v, 2), "area_mm2": round(float(ar), 3), "y_mm": round(float(yy.mean() * v), 2),
                                  "x_mm": round(float(xx.mean() * v), 2)})
    return {"paso_mm": paso_mm, "area_minima_mm2": min_mm2, "distancia_maxima_mm": tol_mm, "trozos_totales": total, "trozos_sin_linea": sin,
            "porcentaje_area_sin_linea": round(100 * area_sin / max(area_tot, 1e-9), 1), "trozos_sin_linea_lista": lista}


def figuras(out, solo, v, lineas, nombre, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colores = plt.cm.tab10(np.arange(10))
    m = solo.astype(bool)
    proj = {"y-z": m.max(axis=2), "x-z": m.max(axis=1), "x-y": m.max(axis=0)}
    fig, ax = plt.subplots(1, 3, figsize=(16, 9))
    for k, (nom, im) in enumerate(proj.items()):
        ax[k].imshow(im, cmap="gray_r", origin="lower", aspect="equal", alpha=0.5)
        ax[k].set_title(f"{nombre}: proyección {nom} del conducto + línea central")
    for i, (cid, Q, org, tabla, perf, vol, edt3) in enumerate(lineas):
        zz = Q[:, 0] / v + org[0]
        yy = Q[:, 1] / v + org[1]
        xx = Q[:, 2] / v + org[2]
        c = colores[i % 10]
        ax[0].plot(yy, zz, color=c, lw=1.5, label=f"conducto {cid}")
        ax[1].plot(xx, zz, color=c, lw=1.5)
        ax[2].plot(xx, yy, color=c, lw=1.5)
        for k in range(3):
            pt = [(yy[0], zz[0]), (xx[0], zz[0]), (xx[0], yy[0])][k]
            pa = [(yy[-1], zz[-1]), (xx[-1], zz[-1]), (xx[-1], yy[-1])][k]
            ax[k].plot(*pt, "o", color=c, ms=5)
            ax[k].plot(*pa, "s", color=c, ms=5)
    ax[0].legend()
    fig.suptitle("círculo = orificio (arriba), cuadrado = extremo apical")
    fig.tight_layout()
    fig.savefig(out / "control_linea_central.png", dpi=90)
    plt.close(fig)
    # perfil
    nrow = len(lineas)
    fig, ax = plt.subplots(nrow, 2, figsize=(13, 3.2 * nrow), squeeze=False)
    for i, (cid, Q, org, tabla, perf, vol, edt3) in enumerate(lineas):
        d = tabla["d_desde_apice_mm"]
        for k, lab in (("d_eq_mm", "equivalente (de su área)"), ("d_mayor_mm", "mayor (barrido)"), ("d_menor_mm", "menor (barrido)"),
                       ("d_edt_mm", "esfera inscrita")):
            ax[i, 0].plot(d, tabla[k], label=lab)
        ax[i, 0].set_ylabel(f"conducto {cid}\ndiámetro (mm)")
        ax[i, 0].set_xlabel("distancia al extremo apical (mm)")
        tope = 1.6                                                           # los picos de hoja/istmo aplastarían la escala
        sin_barrido = ~np.isfinite(tabla["d_mayor_mm"])                      # el barrido no es válido (forma curva o en franja)
        if sin_barrido.any():
            ax[i, 0].fill_between(d, 0, tope, where=sin_barrido, color="0.85", alpha=0.7, step="mid", zorder=0,
                                  label="gris: sin mayor/menor (forma curva o en franja)")
        ax[i, 0].legend(fontsize=7, loc="upper left")
        ax[i, 0].grid(alpha=0.3)
        ax[i, 0].set_ylim(0, tope)
        if np.nanmax(np.r_[tabla["d_mayor_mm"], tabla["d_eq_mm"]]) > tope:
            ax[i, 0].text(0.99, 0.97, f"valores > {tope} mm recortados", transform=ax[i, 0].transAxes, ha="right", va="top", fontsize=6)
        ax[i, 1].plot(d, tabla["kappa_mm-1"], color="C3")
        ax[i, 1].set_ylabel("κ = 1/R (mm⁻¹)")
        ax[i, 1].set_xlabel("distancia al extremo apical (mm)")
        ax[i, 1].grid(alpha=0.3)
        L = d.max()
        for k in (L / 3, 2 * L / 3):
            ax[i, 0].axvline(k, color="k", ls=":", lw=0.8)
            ax[i, 1].axvline(k, color="k", ls=":", lw=0.8)
        for nom, x in (("apical", L / 6), ("medio", L / 2), ("coronal", 5 * L / 6)):
            ax[i, 1].text(x, 1.01, nom, transform=ax[i, 1].get_xaxis_transform(), ha="center", va="bottom", fontsize=7)
    fig.suptitle(f"{nombre}. Izquierda: diámetro del conducto a lo largo de su longitud (sin línea = no hay medida válida en ese punto, "
                 "ver avisos). Derecha: curvatura (más alta = curva más cerrada).", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "control_perfil.png", dpi=90)
    plt.close(fig)
    # secciones a distancias fijas
    # Se dibuja EXACTAMENTE lo que usa el perfil: la ventana completa de medida (±mitad_mm, no un recorte), el trozo que
    # contiene la línea (diametros_seccion) y su contorno de barrido. Si no hay diámetro, el título (en rojo) dice por qué.
    nc = len(DIST_FIJAS_MM) - 1
    m = a.mitad_mm
    fig, ax = plt.subplots(nrow, nc, figsize=(3.6 * nc, 3.9 * nrow), squeeze=False)
    Qt = [Q + np.array(org) * v for (cid, Q, org, *_) in lineas]            # todas las líneas en coordenadas del diente
    bases = []
    for Q in Qt:
        T = cm.tangentes(Q)
        U, W = cm._base_perpendicular(T)
        bases.append((T, U, W))
    for i, (cid, Q, org, tabla, perf, vol, edt3) in enumerate(lineas):
        d = tabla["d_desde_apice_mm"]
        for j, d0 in enumerate(DIST_FIJAS_MM[:nc]):
            a_ = ax[i, j]
            a_.axis("off")
            if d0 > d.max():
                continue
            k = int(np.argmin(np.abs(d - d0)))
            sec, c, paso = perf["_sec"][k], perf["_c"], perf["_paso"]
            dd = cm.diametros_seccion(sec, c, paso)
            a_.imshow(sec, cmap="gray", origin="lower", extent=(-m, m, -m, m), vmin=0, vmax=1)
            a_.plot([-m + 0.1, -m + 0.6], [-m + 0.1, -m + 0.1], "w-", lw=2)
            a_.text(-m + 0.35, -m + 0.16, "0,5 mm", color="w", ha="center", fontsize=7)
            a_.plot(0, 0, "x", color="yellow", ms=6)                         # punto de la línea central (centro de la ventana)
            T_, U_, W_ = (arr[k] for arr in bases[i])                        # base del plano de esta sección
            cruces = [(0.0, 0.0)]                                            # puntos de línea en este plano (la propia y las otras)
            for jj, (cid2, *_r) in enumerate(lineas):                        # dónde cruzan este plano las OTRAS líneas
                if jj == i:
                    continue
                rel = Qt[jj] - Qt[i][k]
                s_ = rel @ T_
                cruce = np.flatnonzero(s_[:-1] * s_[1:] <= 0)
                if len(cruce) == 0:
                    continue
                best = None
                for q in cruce:
                    f = s_[q] / (s_[q] - s_[q + 1]) if s_[q] != s_[q + 1] else 0.0
                    pq = rel[q] + f * (rel[q + 1] - rel[q])
                    xy = (pq @ W_, pq @ U_)
                    if best is None or np.hypot(*xy) < np.hypot(*best):
                        best = xy
                if abs(best[0]) < m and abs(best[1]) < m:
                    a_.plot(best[0], best[1], "x", color="cyan", ms=5)
                    a_.text(best[0], best[1] + 0.08, cid2, color="cyan", fontsize=7, ha="center")
                    cruces.append((best[0], best[1]))
            # trozos de conducto de la ventana a los que no llega ninguna línea (a más de 0,1 mm de todas)
            lab_, n_ = ndi.label(ndi.binary_closing(np.pad(sec >= 0.5, 2), iterations=1)[2:-2, 2:-2] | (sec >= 0.5))
            for kk in range(1, n_ + 1):
                comp = lab_ == kk
                if comp.sum() * paso ** 2 < 0.005:
                    continue
                dk = ndi.distance_transform_edt(~comp) * paso
                if all(dk[int(np.clip(round(c + y / paso), 0, sec.shape[0] - 1)), int(np.clip(round(c + x / paso), 0, sec.shape[1] - 1))] > 0.1
                       for x, y in cruces):
                    yy_, xx_ = np.nonzero(comp)
                    a_.text((xx_.mean() - c) * paso, (yy_.mean() - c) * paso + 0.1, "sin línea", color="orange", fontsize=7, ha="center")
            if dd["centro_ij"] is not None:
                ci, cj = dd["centro_ij"]
                xc, yc = (cj - c) * paso, (ci - c) * paso
                if np.isfinite(dd["d_mayor"]):                               # barrido válido: contorno rojo = el que mide mayor/menor
                    ang = np.deg2rad(np.arange(360))
                    a_.plot(xc + dd["rad"] * np.sin(ang), yc + dd["rad"] * np.cos(ang), "r-", lw=1)
                else:                                                        # barrido no válido: se dibuja el borde real del trozo
                    xs = (np.arange(sec.shape[0]) - c) * paso
                    a_.contour(xs, xs, dd["region"].astype(float), levels=[0.5], colors="orange", linewidths=0.8)
                a_.plot(xc, yc, "r+")                                        # centro de la sección
            if np.isfinite(dd["d_eq"]) and dd["motivo"] is None:
                otros = f", {dd['n_otros']} trozo(s) vecino(s)" if dd["n_otros"] else ""
                alarg = f", alargada ×{dd['alargamiento']:.0f}" if dd["alargamiento"] >= 4 else ""
                a_.set_title(f"{cid} a {d0:g} mm: Deq {dd['d_eq']:.2f} mm{otros}{alarg}", fontsize=8)
            else:
                txt = dd["motivo"] or "sin diámetro"
                if np.isfinite(dd["d_eq"]):
                    txt = f"Deq {dd['d_eq']:.2f} mm; " + txt
                a_.set_title(textwrap.fill(f"{cid} a {d0:g} mm: {txt}", 58), fontsize=6.5, color="r")
    fig.suptitle(f"{nombre}: secciones perpendiculares a la línea (ventana {2 * m:g}×{2 * m:g} mm). × amarilla = punto de la línea; "
                 "+ roja = centro de la sección;\ncontorno rojo = el que mide mayor/menor; naranja = borde del trozo cuando el "
                 "barrido no es válido; título rojo = sin diámetro (motivo);\n× celeste = donde cruza este plano la línea de OTRO "
                 "conducto (ese conducto se mide en su propia fila)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "control_secciones.png", dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", type=Path, nargs="+", help="results/microct/Diente_XX")
    ap.add_argument("--out", type=Path, default=HERE.parents[2] / "results" / "medidas")
    ap.add_argument("--gap-mm", type=float, default=0.13, help="une trozos separados menos de esto en z (como split_canal.py)")
    ap.add_argument("--min-mm3", type=float, default=0.02, help="descarta grupos más pequeños")
    ap.add_argument("--metodo", choices=["ramas", "sistema", "teasar"], default="ramas",
                    help="ramas = un camino ápice→orificio por conducto (recomendado); sistema = un solo eje por grupo "
                         "(centroide de todo el grupo por corte: solo vale si el grupo es UN conducto); "
                         "teasar = primer método, descartado (ramas falsas en el embudo)")
    ap.add_argument("--umbral-fusion", type=float, default=0.6, help="fracción de cortes en la misma sección para dar dos caminos por el mismo conducto")
    ap.add_argument("--zona-apical-mm", type=float, default=3.0, help="solo se buscan ápices en los mm más bajos del grupo")
    ap.add_argument("--coarse", type=int, default=2, help="factor de la rejilla gruesa para el camino inicial")
    ap.add_argument("--min-rama-mm", type=float, default=2.0, help="una rama lateral con menos tramo nuevo se descarta")
    ap.add_argument("--min-descenso-mm", type=float, default=1.0, help="una rama lateral debe acabar al menos esto por debajo de donde se separa")
    ap.add_argument("--max-ramas", type=int, default=4, help="máximo de ramas por grupo")
    ap.add_argument("--ds-mm", type=float, default=0.04, help="espaciado de los puntos de la línea central")
    ap.add_argument("--suav-mm", type=float, default=0.25, help="sigma del suavizado de la línea central (curvatura): 0,25 mm minimiza el error de κ en los conductos artificiales")
    ap.add_argument("--mitad-mm", type=float, default=2.0, help="semilado de las secciones perpendiculares")
    ap.add_argument("--Dk-mm", type=float, default=1.0, help="cuerda de referencia del giro máximo (Česaitienė: ~1/10 del largo)")
    ap.add_argument("--tol-mm", type=float, default=0.1, help="desvío de la recta 1 que marca el inicio de la curva (Schneider 3D)")
    ap.add_argument("--ext-max-mm", type=float, default=2.0, help="máximo que se extiende el extremo apical")
    a = ap.parse_args()
    for f in a.folders:
        procesar(f.resolve(), a.out.resolve(), a)


if __name__ == "__main__":
    main()
