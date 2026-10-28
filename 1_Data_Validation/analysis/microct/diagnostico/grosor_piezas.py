"""Grosor (radio inscrito) de los grupos y de las piezas de una máscara de conducto, y motas sueltas.

Reproduce el análisis de RESUMEN_PROYECTO.md (apartado 3.3) que llevó al filtro de grosor `--min-thick-mm 0.05`:
las rayas de una caries tienen radio p90 ≤ 0,03 mm y fracción de vóxeles a < 40 µm de la pared ≈ 1; los conductos, ≥ 0,07 mm.

    python grosor_piezas.py --resultados results/microct [--mascara canal_completo.npy] [--gap-mm 0.13]

Por diente imprime (1) un histograma del tamaño de las piezas 3D (motas), (2) por GRUPO (piezas unidas por la holgura en
z): volumen, largo, radio p90 y fracción a < 40 µm de la pared, y (3) por pieza: lo mismo y su contacto con otras piezas.
OJO: con la máscara ya filtrada (`canal_completo.npy` actual) los grupos descartados por finos ya no aparecen; los números
del RESUMEN se calcularon sobre una máscara intermedia, antes del filtro, que se puede regenerar con
`preprocess_microct.py --min-thick-mm 0` (y `--out` distinto).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--resultados", type=Path, default=HERE.parents[2] / "results" / "microct")
ap.add_argument("--dientes", nargs="*", default=["Diente_35", "Diente_51", "Diente_52"])
ap.add_argument("--mascara", default="canal_completo.npy")
ap.add_argument("--gap-mm", type=float, default=0.13)
ap.add_argument("--min-pieza-mm3", type=float, default=0.005, help="no listar piezas más pequeñas")
a = ap.parse_args()

for d in a.dientes:
    V = json.loads((a.resultados / d / "info.json").read_text())["voxel_mm"]
    g = int(round(a.gap_mm / V))
    c = np.load(a.resultados / d / a.mascara).astype(bool)
    pl, pn = ndi.label(c)
    pbox = ndi.find_objects(pl)
    vox = np.bincount(pl.ravel())[1:]
    print(f"\n{d} ({a.mascara}): {pn} piezas 3D, volumen {c.sum() * V ** 3:.3f} mm3")
    for th in (10, 100, 1000, 10000):
        m = vox < th
        print(f"   piezas < {th:>5} vóxeles: {int(m.sum()):>6}, suman {vox[m].sum() * V ** 3:.4f} mm3")
    grp = ndi.label(ndi.binary_dilation(c, structure=np.ones((2 * g + 1, 1, 1), bool)))[0] * c
    for gi, b in enumerate(ndi.find_objects(grp), 1):
        if b is None:
            continue
        m = np.pad(grp[b] == gi, 1)
        e = ndi.distance_transform_edt(m) * V
        print(f"   GRUPO {gi}: vol {m.sum() * V ** 3:7.3f} mm3, largo z {(b[0].stop - b[0].start) * V:5.2f} mm, "
              f"radio p90 {np.percentile(e[m], 90):.3f} mm, fracción a <40µm de la pared {(e[m] < 0.04).mean():.2f}")
    for i, b in enumerate(pbox, 1):
        pm = pl[b] == i
        vol = pm.sum() * V ** 3
        if vol < a.min_pieza_mm3:
            continue
        m = np.pad(pm, 1)
        e = ndi.distance_transform_edt(m) * V
        z0, z1 = max(b[0].start - g, 0), min(b[0].stop + g, c.shape[0])
        other = (pl[z0:z1, b[1], b[2]] > 0) & (pl[z0:z1, b[1], b[2]] != i)
        od = ndi.binary_dilation(other, structure=np.ones((2 * g + 1, 1, 1), bool))
        cols = ((pl[z0:z1, b[1], b[2]] == i) & od).any(axis=0).sum() * V ** 2
        print(f"   pieza {i:>3}: vol {vol:7.3f} mm3, z {b[0].start}..{b[0].stop - 1}, radio p90 {np.percentile(e[m], 90):.3f} mm, "
              f"contacto con otras piezas {cols:.4f} mm2")
