"""Sección más grande del conducto por corte, y sensibilidad del orificio al umbral de área de `split_canal.py`.

Reproduce lo que sustenta el criterio del orificio (RESUMEN_PROYECTO.md, apartado 3.5): en la raíz hay 3-4 secciones
pequeñas (la mayor ≤ ~1,15 mm2) y en la cámara se funden en una grande; y cuánto se mueve el orificio al cambiar
`--area-camara-mm2` (1,25 ... 3,0 mm2).

    python secciones_y_orificio.py --resultados results/microct [--areas 1.25 1.5 2 2.5 3] [--paso 40]

Usa canal_completo.npy y la orientación de info.json (ejecutar antes split_canal.py). Imprime, por diente, la tabla
z | mayor sección | nº de secciones | área total, y para cada umbral el z del orificio y su diferencia en mm con el de 1,5 mm2.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import split_canal as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--resultados", type=Path, default=HERE.parents[2] / "results" / "microct")
ap.add_argument("--dientes", nargs="*", default=["Diente_35", "Diente_51", "Diente_52"])
ap.add_argument("--areas", type=float, nargs="*", default=[1.25, 1.5, 2.0, 2.5, 3.0])
ap.add_argument("--sostenido-mm", type=float, default=0.1)
ap.add_argument("--paso", type=int, default=40, help="imprimir un corte de cada `paso`")
a = ap.parse_args()

for d in a.dientes:
    info = json.loads((a.resultados / d / "info.json").read_text())
    v = info["voxel_mm"]
    crown_high = info["orientacion"]["corona_en_z"] == "alto"
    canal = np.load(a.resultados / d / "canal_completo.npy").astype(bool)
    if not crown_high:
        canal = canal[::-1]                       # como split_canal.py: siempre con la corona hacia z alto
    big, nsec = S.section_stats(canal, v)
    tot = canal.sum(axis=(1, 2)) * v ** 2
    zs = np.flatnonzero(canal.any(axis=(1, 2)))
    run = max(int(round(a.sostenido_mm / v)), 1)
    print(f"\n{d} (corona en z {'alto' if crown_high else 'bajo (volteado)'}): z | mayor sección mm2 | nº secciones >= 0.02 | área total mm2")
    for z in range(int(zs.min()), int(zs.max()) + 1, a.paso):
        print(f"  {z:5d} | {big[z]:6.3f} | {nsec[z]:2d} | {tot[z]:6.3f}")
    ref = S.find_floor(big, int(zs.min()), int(zs.max()), 1.5, run)
    for ac in a.areas:
        zf = S.find_floor(big, int(zs.min()), int(zs.max()), ac, run)
        print(f"  umbral {ac:4.2f} mm2 -> orificio z={zf}" + (f"  ({(zf - ref) * v:+.2f} mm respecto a 1.5)" if zf is not None and ref is not None else ""))
