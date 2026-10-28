"""Cortes axiales ampliados desde el final de la máscara de un conducto hacia la punta de la raíz.

Sirve para comprobar a ojo lo que dice RESUMEN_PROYECTO.md (apartado 3.4): hasta el final de la máscara el conducto está
encerrado por dentina; ~0,06 mm más abajo el hueco se abre al exterior (foramen) y por debajo hay dentina maciza.

    python ver_apice.py --diente Diente_52 --raiz 1 --conducto 1 [--resultados results/microct] [--out salida.png]
    python ver_apice.py --diente Diente_52 --todos          # una imagen por cada conducto de info.json["apice"]

Los números de raíz y conducto son los de info.json["apice"]["raices"] (empiezan en 1). Cada panel es un corte axial de
0,64 mm de lado centrado en el final del conducto; verde = máscara de solo conducto; el título da la distancia en z
respecto al final de la máscara. Hace falta haber ejecutado split_canal.py (usa su info.json).
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
ap.add_argument("--diente", required=True)
ap.add_argument("--raiz", type=int, default=1)
ap.add_argument("--conducto", type=int, default=1)
ap.add_argument("--todos", action="store_true")
ap.add_argument("--paso", type=int, default=8, help="cortes entre paneles (8 = 0,128 mm a 16 µm)")
ap.add_argument("--out", type=Path, default=None)
a = ap.parse_args()

folder = a.resultados / a.diente
info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
V = info["voxel_mm"]
norm = np.load(folder / "norm.npy", mmap_mode="r")
solo = np.load(folder / "canal_solo_conducto.npy", mmap_mode="r")
crown_high = info["orientacion"]["corona_en_z"] == "alto"
sg = 1 if crown_high else -1                     # hacia la punta = z decreciente si la corona está arriba
out_dir = a.out.parent if a.out else Path("results/diagnostico")
out_dir.mkdir(parents=True, exist_ok=True)
pares = [(r["raiz"], k + 1) for r in info["apice"]["raices"] for k in range(len(r["conductos"]))] if a.todos else [(a.raiz, a.conducto)]
for ri, ci in pares:
    r = info["apice"]["raices"][ri - 1]
    c = r["conductos"][ci - 1]
    ze, (y, x) = c["z_fin_vox"], c["yx_fin_vox"]
    tip = ze - sg * int(round(c["mm_a_la_punta_local"] / V))
    dist = int(round(c["mm_a_la_punta_local"] / V))                 # cortes entre el final de la máscara y la punta local
    zs = [ze + sg * k for k in range(12, -(dist + 5), -a.paso)]       # desde 12 cortes hacia la corona hasta pasada la punta
    zs = [z for z in zs if 0 <= z < norm.shape[0]][:16]
    cols = 8
    fig, ax = plt.subplots((len(zs) + cols - 1) // cols, cols, figsize=(2.6 * cols, 2.8 * ((len(zs) + cols - 1) // cols)), squeeze=False)
    for q in ax.ravel():
        q.axis("off")
    h = 40
    for q, z in zip(ax.ravel(), zs):
        q.imshow(np.asarray(norm[z, y - h:y + h, x - h:x + h], np.float32), cmap="gray", vmin=0, vmax=1.3)
        m = np.asarray(solo[z, y - h:y + h, x - h:x + h])
        if m.any():
            q.contour(m, [0.5], colors="lime", linewidths=0.8)
        q.set_title(f"z={z} ({sg * (ze - z) * V:+.2f} mm hacia la punta)", fontsize=7)
    plt.suptitle(f"{a.diente} raíz {ri} conducto {ci}: fin de máscara z={ze}, punta local z≈{tip}; verde = máscara; paneles de 0.64 mm", fontsize=9)
    plt.tight_layout()
    f = (a.out if (a.out and not a.todos) else out_dir / f"apice_{a.diente}_r{ri}c{ci}.png")
    plt.savefig(f, dpi=60)
    plt.close(fig)
    print(f)
