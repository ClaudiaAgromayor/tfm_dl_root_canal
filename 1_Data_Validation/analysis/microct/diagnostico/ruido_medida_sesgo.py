"""Por qué el "ruido en dentina" medido en el simulador sale más alto con rangos aleatorios que en el centro de los rangos.

Reproduce lo de RESUMEN_PROYECTO.md (apartado 5.1, hallazgo 14): la medida de ruido (calibrate_cbct.block_noise) usa una
ventana de gris FIJA para "dentina" (1450-1850); con rangos aleatorios, la dentina simulada cae a veces fuera de ella y la
ventana solo recoge bloques de borde, con más estructura. No es un fallo del ruido simulado, es un sesgo de la medida.

    python ruido_medida_sesgo.py [--n 36] [--parches 48]

Parte 1: sd medida en dentina sin ruido (suelo de estructura del µCT), con el ruido del centro del rango y con sd 20.
Parte 2: parche a parche con parámetros aleatorios: sd medida frente a sd sorteada, y su correlación con el gris de la
dentina sorteado, la PSF, etc.
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
ap.add_argument("--n", type=int, default=36, help="parches de la parte 1")
ap.add_argument("--parches", type=int, default=48, help="parches de la parte 2")
a = ap.parse_args()
sys.path.insert(0, str(HERE.parent))
import calibrate_cbct as cc  # noqa: E402
import simulate_cbct as sc  # noqa: E402

if sc.DEV.type == "cuda":
    torch.set_default_device("cuda")
cfg = yaml.safe_load((HERE.parent / "simulate_cbct_config.yaml").read_text(encoding="utf-8"))
teeth = [sc.Tooth(a.results / d) for d in a.dientes]


def run(c, label, n, seed=5, center=True):
    sim = sc.Simulator(c)
    rng = np.random.default_rng(seed)
    imgs = [sim.patch(teeth[i % len(teeth)], rng, center=center)[0]["image"].cpu().numpy().transpose(2, 1, 0).astype(np.float32) for i in range(n)]
    ns = cc.noise_stats(imgs)["dentina"]
    print(f"{label:52s} sd dentina {ns['sd_mediana']:6.1f} (cuartiles {ns['sd_p25']:.0f}-{ns['sd_p75']:.0f}, n={ns['n_bloques']}) | "
          f"autocorr {ns['autocorr_lag1_x_y_z'][0]:.3f} | curtosis {ns['curtosis_exceso_mediana']:.2f}")


c0 = copy.deepcopy(cfg); c0["crop"]["aug_xy"] = False
c1 = copy.deepcopy(c0); c1["noise"]["sd_hu"] = [0, 0]; c1["shading"].update(mult=[0, 0], add_hu=[0, 0])
c2 = copy.deepcopy(c0); c2["noise"]["sd_hu"] = [20, 20]
print("== Parte 1 (parámetros en el centro de los rangos)")
run(c1, "sin ruido ni brillo (solo estructura del µCT)", a.n)
run(c0, "ruido en el centro del rango", a.n)
run(c2, "ruido sd 20", a.n)

print("\n== Parte 2 (parámetros aleatorios, parche a parche)")
sim = sc.Simulator(cfg)
rng = np.random.default_rng(5)
rows = []
for i in range(a.parches):
    out, P = sim.patch(teeth[i % len(teeth)], rng)
    img = out["image"].cpu().numpy().transpose(2, 1, 0).astype(np.float32)
    m, s, ac, sk = cc.block_noise(img, 1450, 1850)
    if len(s) >= 5:
        rows.append((P["noise_sd_hu"], np.median(s), len(s), P["psf_sigma_mm"], P["dentin_hu"], P["poisson_counts"], P["noise_corr_vox"], P["shade_add_hu"], P["shade_mult"]))
r = np.array(rows)
print(f"parches con >=5 bloques de dentina: {len(r)} de {a.parches}")
print(f"sd sorteada (mediana) {np.median(r[:, 0]):.1f} | sd medida por parche (mediana) {np.median(r[:, 1]):.1f}")
print("correlación de la sd medida con: " + ", ".join(f"{n} {np.corrcoef(r[:, 1], r[:, k])[0, 1]:.2f}" for n, k in
      (("sd sorteada", 0), ("PSF", 3), ("gris de la dentina sorteado", 4), ("cuentas Poisson", 5), ("corr. vecinos", 6), ("brillo aditivo", 7), ("brillo mult.", 8))))
