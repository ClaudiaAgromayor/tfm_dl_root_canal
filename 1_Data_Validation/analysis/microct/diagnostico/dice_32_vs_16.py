"""Estabilidad del conducto entre resoluciones: volumen, componentes y Dice entre la máscara de 16 µm y la de 32 µm.

Reproduce results/microct/resumen_etapa1.json (RESUMEN_PROYECTO.md, apartado 3.3), salvo las columnas "antes de los filtros".

    python dice_32_vs_16.py --r16 results/microct --r32 results/microct_32um [--mascara canal.npy] [--out salida.json]

Dice (rejilla de 32 µm): la máscara de 16 µm se reduce por mayoría (>= 4 de 8 vóxeles de cada bloque 2x2x2) y se compara con
la de 32 µm. Dice (rejilla de 16 µm): la de 32 µm se repite 2x2x2 y se compara con la de 16 µm. Por tercios del conducto en z.
Es estabilidad entre resoluciones del MISMO método, no exactitud frente a un experto.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--r16", type=Path, default=HERE.parents[2] / "results" / "microct")
ap.add_argument("--r32", type=Path, default=HERE.parents[2] / "results" / "microct_32um")
ap.add_argument("--dientes", nargs="*", default=["Diente_35", "Diente_51", "Diente_52"])
ap.add_argument("--mascara", default="canal.npy")
ap.add_argument("--out", type=Path, default=None)
a = ap.parse_args()


def dice(x, y):
    s = x.sum() + y.sum()
    return 2 * float((x & y).sum()) / s if s else 1.0


rows = []
for d in a.dientes:
    c1 = np.load(a.r16 / d / a.mascara, mmap_mode="r")
    c2 = np.load(a.r32 / d / a.mascara).astype(bool)
    i1, i2 = (json.loads((r / d / "info.json").read_text()) for r in (a.r16, a.r32))
    nz, ny, nx = c2.shape
    c1 = np.asarray(c1[:nz * 2, :ny * 2, :nx * 2]).astype(bool)
    c1_on2 = c1.reshape(nz, 2, ny, 2, nx, 2).sum(axis=(1, 3, 5)) >= 4
    c2_on1 = c2.repeat(2, 0).repeat(2, 1).repeat(2, 2)
    zs = np.flatnonzero(c2.any(axis=(1, 2)))
    e = np.linspace(zs.min(), zs.max() + 1, 4).astype(int)
    rows.append({"diente": d, "vol32": i2["volumen_canal_mm3"], "comp32": i2["componentes_canal"],
                 "vol16": i1["volumen_canal_mm3"], "comp16": i1["componentes_canal"],
                 "dif_vol_%": 100 * abs(i1["volumen_canal_mm3"] - i2["volumen_canal_mm3"]) / i2["volumen_canal_mm3"],
                 "dice_rejilla32": dice(c1_on2, c2), "dice_rejilla16": dice(c1, c2_on1[:c1.shape[0], :c1.shape[1], :c1.shape[2]]),
                 "dice_tercios_z_bajo_a_alto": [round(dice(c1_on2[e[k]:e[k + 1]], c2[e[k]:e[k + 1]]), 3) for k in range(3)]})
    print(json.dumps(rows[-1], ensure_ascii=False, default=float))
if a.out:
    a.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=float))
