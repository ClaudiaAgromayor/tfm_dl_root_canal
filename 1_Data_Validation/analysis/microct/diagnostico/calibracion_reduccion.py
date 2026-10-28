"""¿Cambia la comprobación del simulador contra el CBCT real (Isa) al cambiar la reducción a 75 µm o el muestreo?

Ejecuta la misma comprobación que `simulate_cbct.py --check` (mismas funciones de medida de calibrate_cbct.py, mismas
semillas, parámetros en el centro de cada rango) con varias configuraciones y las pone en una tabla junto a lo medido
en Isa:
  A  reducción adaptativa (la antigua, ventanas ~106 µm) + muestreo proporcional (la antigua)
  B  reducción exacta                                        + muestreo proporcional   -> solo cambia la reducción
  C  reducción exacta                                        + muestreo equilibrado por tercios (la nueva por defecto)
  D  como C pero con el centro de la PSF subido (--psf-centro), para ver si recupera lo medido en Isa

    python calibracion_reduccion.py --calibracion <calibration.json> [--n 36] [--seeds 1 2] [--psf-centro 0.14]
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--results", type=Path, default=HERE.parents[2] / "results" / "microct")
ap.add_argument("--calibracion", type=Path, default=HERE.parents[2] / "results" / "cbct_calibration" / "Isa_caso01" / "calibration.json")
ap.add_argument("--dientes", nargs="*", default=["Diente_35", "Diente_51", "Diente_52"])
ap.add_argument("--n", type=int, default=36)
ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2])
ap.add_argument("--psf-centro", type=float, default=None, help="si se da, añade la configuración D con el centro de psf.sigma_mm en este valor (mm)")
a = ap.parse_args()
sys.path.insert(0, str(HERE.parent))
import calibrate_cbct as cc  # noqa: E402
import simulate_cbct as sc  # noqa: E402

if sc.DEV.type == "cuda":
    torch.set_default_device("cuda")
cfg0 = yaml.safe_load((HERE.parent / "simulate_cbct_config.yaml").read_text(encoding="utf-8"))
teeth = [sc.Tooth(a.results / d) for d in a.dientes]
ref = json.loads(a.calibracion.read_text())

ROWS = [("gris dentina (mediana)", ("grises", "dentina", "mediana")), ("gris esmalte (mediana)", ("grises", "esmalte", "mediana")),
        ("gris tejido lejano (mediana)", ("grises", "tejido_lejano", "mediana")),
        ("ruido sd dentina", ("ruido", "dentina", "sd_mediana")), ("ruido sd tejido blando", ("ruido", "blando", "sd_mediana")),
        ("autocorr lag1 dentina (x)", ("ruido", "dentina", "autocorr_lag1_x_y_z")),
        ("asimetría ruido dentina", ("ruido", "dentina", "asimetria_mediana")), ("curtosis ruido dentina", ("ruido", "dentina", "curtosis_exceso_mediana")),
        ("borde dentina-exterior sigma (mm)", ("bordes", "dentina_exterior", "sigma_mm")),
        ("borde esmalte-exterior sigma (mm)", ("bordes", "esmalte_exterior", "sigma_mm")),
        ("borde dentina-conducto sigma (mm)", ("bordes", "dentina_conducto", "sigma_mm"))]


def get(d, path):
    for k in path:
        if d is None or k not in d:
            return None
        d = d[k]
    return d[0] if isinstance(d, list) else d


def run(cfg, seed):
    sim = sc.Simulator(cfg)
    rng = np.random.default_rng(seed)
    imgs, segs = [], []
    for i in range(a.n):
        out, _ = sim.patch(teeth[i % len(teeth)], rng, center=True)
        imgs.append(out["image"].cpu().numpy().transpose(2, 1, 0).astype(np.float32))
        segs.append(out["label"].cpu().numpy().transpose(2, 1, 0) > 0)
    return cc.measure(imgs, segs, cfg["crop"]["voxel_out_mm"])


configs = {}
for name, red, tw in (("A", "adaptativa", None), ("B", "exacta", None), ("C", "exacta", [1, 1, 1])):
    c = copy.deepcopy(cfg0); c["crop"]["reduccion"] = red; c["crop"]["third_weights"] = tw
    configs[name] = c
if a.psf_centro:
    c = copy.deepcopy(configs["C"]); w = 0.5 * (c["psf"]["sigma_mm"][1] - c["psf"]["sigma_mm"][0])
    c["psf"]["sigma_mm"] = [a.psf_centro - w, a.psf_centro + w]
    configs["D"] = c
res = {n: [run(c, s) for s in a.seeds] for n, c in configs.items()}
print(f"comprobación contra Isa; {a.n} parches por semilla, semillas {a.seeds}; media sobre semillas (entre paréntesis, rango)")
print(f"{'magnitud':36s} {'Isa':>9s} " + " ".join(f"{n:>17s}" for n in configs))
for label, path in ROWS:
    r = get(ref, path)
    cells = []
    for n in configs:
        v = [get(x, path) for x in res[n]]
        v = [q for q in v if q is not None]
        cells.append(f"{np.mean(v):9.3f} ({min(v):.3f}-{max(v):.3f})" if v else "-")
    print(f"{label:36s} {r if r is None else format(r, '9.3f')} " + " ".join(f"{c:>17s}" for c in cells))
