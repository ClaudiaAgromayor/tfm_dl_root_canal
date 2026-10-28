"""Cuánto tarda un parche del simulador, por etapas, y en CPU frente a GPU.

Reproduce los tiempos de RESUMEN_PROYECTO.md (apartado 3.6): `patch()` ~0,5 s en GPU (la primera llamada incluye el
arranque) frente a ~87 s en CPU (20 hilos).

    python tiempos_simulador.py --que etapas            # lectura del disco, conversión y patch() en GPU
    python tiempos_simulador.py --que cpu_gpu           # el mismo parche en GPU y en CPU (en CPU, minutos)

CPU/GPU: se cambia `simulate_cbct.DEV` a "cpu" dentro del proceso. (`CUDA_VISIBLE_DEVICES=-1` NO sirve en la máquina
del laboratorio: Python se cae al importar torch con el código 0xC0000005.) Las imágenes de CPU y GPU NO son comparables
número a número: el generador aleatorio de cada dispositivo produce secuencias distintas.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--results", type=Path, default=HERE.parents[2] / "results" / "microct")
ap.add_argument("--diente", default="Diente_51")
ap.add_argument("--que", choices=["etapas", "cpu_gpu"], default="etapas")
a = ap.parse_args()
sys.path.insert(0, str(HERE.parent))
import simulate_cbct as sc  # noqa: E402

cfg = yaml.safe_load((HERE.parent / "simulate_cbct_config.yaml").read_text(encoding="utf-8"))


def T(f):
    if sc.DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    r = f()
    if sc.DEV.type == "cuda":
        torch.cuda.synchronize()
    return r, time.time() - t0


if a.que == "etapas":
    if sc.DEV.type == "cuda":
        torch.set_default_device("cuda")
    sim = sc.Simulator(cfg)
    t = sc.Tooth(a.results / a.diente)
    n = sim.crop_size(t.voxel)
    rng = np.random.default_rng(0)
    for rep in range(3):
        start, d0 = T(lambda: sim.pick_start(t, rng))
        _, d1 = T(lambda: t.read(t.norm, start, n))
        _, d2 = T(lambda: t.read(t.canal, start, n))
        _, d3 = T(lambda: t.read(t.chamber, start, n))
        _, d4 = T(lambda: torch.from_numpy(t.read(t.norm, start, n).astype(np.float32)).to(sc.DEV))
        _, d5 = T(lambda: torch.from_numpy(t.read(t.norm, start, n)).to(sc.DEV).float())
        _, d6 = T(lambda: sim.patch(t, rng, start=start))
        print(f"repetición {rep}: pick_start {d0:.2f} s | leer norm {d1:.2f}, canal {d2:.2f}, cámara {d3:.2f} | norm a GPU convirtiendo en CPU "
              f"{d4:.2f} frente a convirtiendo en GPU {d5:.2f} | patch() completo {d6:.2f} s")
else:
    print("hilos de CPU de torch:", torch.get_num_threads())
    imgs = {}
    for dev in ("cuda", "cpu"):
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        sc.DEV = torch.device(dev)
        torch.set_default_device(dev)
        sim = sc.Simulator(cfg)
        t = sc.Tooth(a.results / a.diente)
        start = sim.pick_start(t, np.random.default_rng(0))
        for rep in range(2 if dev == "cuda" else 1):
            out, dt = T(lambda: sim.patch(t, np.random.default_rng(1), start=start, center=True))
            print(f"{dev}: patch() {dt:.2f} s (repetición {rep})", flush=True)
        imgs[dev] = out["image"].cpu().numpy()
    if len(imgs) == 2:
        print("(no se comparan las dos imágenes: los generadores aleatorios de CPU y GPU dan secuencias distintas)")
