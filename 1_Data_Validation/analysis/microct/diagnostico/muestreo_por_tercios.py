"""Cómo se reparten los parches del simulador entre los tercios apical, medio y coronal del conducto.

Para cada diente y cada configuración de `crop.third_weights` ("proporcional" = null, el muestreo antiguo según el volumen
de conducto; "equilibrado" = [1, 1, 1]) llama muchas veces a `Simulator.pick_start` y mide:
  - en qué tercio cae el CENTRO del bloque (o fuera del conducto / sin información);
  - cuánto conducto de cada tercio contiene de media el bloque (mm3), y en qué fracción de los bloques hay conducto apical;
  - cuántas veces ningún intento (de 20) cumplió lo pedido (sin_cumplir).

    python muestreo_por_tercios.py [--n-centros 4000] [--n-bloques 150] [--dientes Diente_35 Diente_51 Diente_52]
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--results", type=Path, default=HERE.parents[2] / "results" / "microct")
ap.add_argument("--dientes", nargs="*", default=["Diente_35", "Diente_51", "Diente_52"])
ap.add_argument("--n-centros", type=int, default=4000, help="llamadas a pick_start por diente y configuración")
ap.add_argument("--n-bloques", type=int, default=150, help="de ellas, cuántas se leen para contar conducto por tercio")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, str(HERE.parent))
import simulate_cbct as sc  # noqa: E402

if sc.DEV.type == "cuda":
    torch.set_default_device("cuda")
cfg0 = yaml.safe_load((HERE.parent / "simulate_cbct_config.yaml").read_text(encoding="utf-8"))
teeth = [sc.Tooth(a.results / d) for d in a.dientes]
NAMES = ("apical", "medio", "coronal")

for label, weights in (("proporcional (antiguo)", None), ("equilibrado [1,1,1]", [1, 1, 1])):
    cfg = copy.deepcopy(cfg0)
    cfg["crop"]["third_weights"] = weights
    sim = sc.Simulator(cfg)
    print(f"\n=== {label}")
    print(f"{'diente':10s} | centro en tercio: {'apical':>7s} {'medio':>7s} {'coronal':>8s} {'fuera':>6s} | conducto por bloque (mm3): "
          f"{'apical':>7s} {'medio':>7s} {'coronal':>8s} | bloques con >=0.05 mm3 apical | sin_cumplir")
    tot_c, tot_v, tot_b = np.zeros(4), np.zeros(3), 0
    for t in teeth:
        rng = np.random.default_rng(a.seed)
        n = sim.crop_size(t.voxel)
        cnt, vol, withap = np.zeros(4), np.zeros(3), 0
        for i in range(a.n_centros):
            st = sim.pick_start(t, rng)
            cz = st[0] + n / 2
            k = int(t.third_of_z(cz))
            cnt[k if k >= 0 else 3] += 1
            if i < a.n_bloques:
                can = t.read(t.canal, st, n)
                per_slice = can.reshape(n, -1).sum(1) * t.voxel ** 3
                th = t.third_of_z(st[0] + np.arange(n))
                v = [per_slice[th == j].sum() for j in range(3)]
                vol += v
                withap += v[0] >= 0.05
        nb = min(a.n_bloques, a.n_centros)
        p = 100 * cnt / cnt.sum()
        print(f"{t.name:10s} | {'':17s}{p[0]:6.1f}% {p[1]:6.1f}% {p[2]:7.1f}% {p[3]:5.1f}% | {'':27s}"
              f"{vol[0] / nb:7.2f} {vol[1] / nb:7.2f} {vol[2] / nb:8.2f} | {100 * withap / nb:5.0f} %{'':20s}| {sim.stats['sin_cumplir']}/{sim.stats['pick_start']}")
        tot_c += cnt; tot_v += vol; tot_b += nb; sim.stats.update(pick_start=0, sin_cumplir=0)
    p = 100 * tot_c / tot_c.sum()
    print(f"{'TOTAL':10s} | {'':17s}{p[0]:6.1f}% {p[1]:6.1f}% {p[2]:7.1f}% {p[3]:5.1f}% | {'':27s}"
          f"{tot_v[0] / tot_b:7.2f} {tot_v[1] / tot_b:7.2f} {tot_v[2] / tot_b:8.2f}")
