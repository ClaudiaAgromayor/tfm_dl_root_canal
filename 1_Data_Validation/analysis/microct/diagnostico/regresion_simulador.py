"""Prueba de regresión del simulador: genera parches con recortes y semillas FIJOS y los guarda (o los compara).

Sirve para demostrar que un cambio en simulate_cbct.py que NO debería alterar la imagen (por ejemplo, generalizar el
tamaño de vóxel) no la altera, o para medir cuánto la altera un cambio que sí debe hacerlo.

    # línea base con el código actual
    python regresion_simulador.py --out base.npz
    # línea base con una copia antigua del simulador (carpeta que contiene simulate_cbct.py)
    python regresion_simulador.py --moddir <carpeta_antigua> --out base.npz
    # comparar el código actual con la línea base
    python regresion_simulador.py --compare base.npz [--legacy]

--legacy pone `crop.reduccion: adaptativa` y `crop.third_weights: null` (el comportamiento anterior) si el
simulador actual conoce esas opciones; con ellas debe salir idéntico a la línea base de la versión antigua.
Los recortes se fijan por diente con una semilla (no se sortean con `pick_start`, que puede cambiar).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--moddir", type=Path, default=HERE.parent, help="carpeta con simulate_cbct.py (y su yaml)")
ap.add_argument("--teeth", type=Path, nargs="*", default=None, help="carpetas results/microct/Diente_XX (por defecto las 3)")
ap.add_argument("--results", type=Path, default=HERE.parents[2] / "results" / "microct")
ap.add_argument("--out", type=Path, default=None)
ap.add_argument("--compare", type=Path, default=None)
ap.add_argument("--legacy", action="store_true")
ap.add_argument("--n", type=int, default=2, help="recortes por diente")
a = ap.parse_args()

sys.path.insert(0, str(a.moddir))
import simulate_cbct as sc  # noqa: E402

if sc.DEV.type == "cuda":
    torch.set_default_device("cuda")
cfg = yaml.safe_load((a.moddir / "simulate_cbct_config.yaml").read_text(encoding="utf-8"))
if a.legacy:
    cfg["crop"]["reduccion"] = "adaptativa"
    cfg["crop"]["third_weights"] = None
sim = sc.Simulator(cfg)
n = getattr(sim, "n", None)
teeth = a.teeth or [a.results / f"Diente_{d}" for d in (35, 51, 52)]
out = {}
for tp in teeth:
    t = sc.Tooth(tp)
    n_t = n if n is not None else sim.crop_size(t.voxel)
    rng0 = np.random.default_rng(2024)
    coords = t.coords[rng0.integers(len(t.coords), size=a.n)]
    for i, c in enumerate(coords):
        start = np.round(c - n_t / 2).astype(int)
        for mode in ("centro", "azar"):
            o, P = sim.patch(t, np.random.default_rng(100 + i), start=start, center=(mode == "centro"))
            key = f"{tp.name}_{i}_{mode}"
            out[key + "_imagen"] = o["image"].cpu().numpy()
            out[key + "_soft"] = o["label_soft"].cpu().numpy()
            out[key + "_peso"] = o["weight"].cpu().numpy()
if a.out:
    np.savez_compressed(a.out, **out)
    print("guardado", a.out, "|", len(out) // 3, "parches")
if a.compare:
    ref = np.load(a.compare)
    worst = {}
    for k in out:
        kind = k.rsplit("_", 1)[1]
        d = float(np.abs(out[k] - ref[k]).max())
        worst[kind] = max(worst.get(kind, 0.0), d)
    print("diferencia máxima respecto a la línea base (imagen en HU; etiqueta blanda y pesos en 0..1):")
    for kind, d in worst.items():
        print(f"  {kind:7s}: {d:.6g}")
