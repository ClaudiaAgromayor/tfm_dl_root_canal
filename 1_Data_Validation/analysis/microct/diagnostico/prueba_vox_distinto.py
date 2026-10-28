"""Prueba de que el simulador usa el vóxel de cada diente (no supone 16 µm).

Construye, en una carpeta temporal, un "diente" a 32 µm con los datos de results/microct_32um (norm.npy y
canal.npy a 32 µm) y la extensión del conducto del diente de 16 µm (sus z divididos por 2), y genera parches del mismo
recorte físico con la versión a 16 µm y la de 32 µm. Sin ruido ni brillo ni textura (para que la imagen sea determinista)
deben salir casi iguales (solo cambia la resolución de partida), con el mismo tamaño físico y el mismo volumen de conducto.

    python prueba_vox_distinto.py --diente Diente_51 [--tmp <carpeta>]

También imprime el tamaño de bloque que se usaría con otros vóxeles.
"""
import argparse
import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--diente", default="Diente_51")
ap.add_argument("--results", type=Path, default=HERE.parents[2] / "results")
ap.add_argument("--tmp", type=Path, default=None)
a = ap.parse_args()
sys.path.insert(0, str(HERE.parent))
import simulate_cbct as sc  # noqa: E402

if sc.DEV.type == "cuda":
    torch.set_default_device("cuda")
cfg = yaml.safe_load((HERE.parent / "simulate_cbct_config.yaml").read_text(encoding="utf-8"))
sim = sc.Simulator(cfg)
print("tamaño de bloque (vóxeles del µCT) según el vóxel del diente:",
      {v: sim.crop_size(v) for v in (0.0125, 0.016, 0.020, 0.025, 0.032, 0.0185)})

d16, d32 = a.results / "microct" / a.diente, a.results / "microct_32um" / a.diente
tmp = a.tmp or Path(tempfile.mkdtemp(prefix="diente32_"))
t32 = tmp / a.diente
t32.mkdir(parents=True, exist_ok=True)
shutil.copy2(d32 / "norm.npy", t32 / "norm.npy")
shutil.copy2(d32 / "canal.npy", t32 / "canal_solo_conducto.npy")
i16, i32 = json.loads((d16 / "info.json").read_text()), json.loads((d32 / "info.json").read_text())
z0, z1 = i16["separacion"]["canal_solo_z_vox"]
i32["separacion"] = {"canal_solo_z_vox": [z0 // 2, z1 // 2]}
i32["orientacion"] = i16["orientacion"]
(t32 / "info.json").write_text(json.dumps(i32))

# configuración determinista: sin ruido, brillo ni textura; fondo y pulpa fijos
c = copy.deepcopy(cfg)
c["noise"].update(sd_hu=[0, 0]); c["shading"].update(mult=[0, 0], add_hu=[0, 0])
c["background"].update(base_hu=[900, 900], texture_amp_hu=[0, 0]); c["background"]["pdl"]["prob"] = 0
c["intensity"]["canal_hu"] = [100, 100]; c["crop"]["aug_xy"] = False; c["labels"]["weight_mode"] = "none"
s = sc.Simulator(c)
# a 16 µm se usa también la máscara COMPLETA como etiqueta, igual que la copia de 32 µm, para que los volúmenes sean comparables
T16, T32 = sc.Tooth(d16, canal_name="canal_completo.npy", chamber_name=None), sc.Tooth(t32)
print(f"vóxel: {T16.voxel * 1000:.3f} µm y {T32.voxel * 1000:.3f} µm | bloque: {s.crop_size(T16.voxel)} y {s.crop_size(T32.voxel)} vóxeles")
rng = np.random.default_rng(0)
print("recorte | tamaño físico (mm) 16 / 32 µm | volumen conducto en el bloque (mm3) 16 / 32 | soft×0.075³ 16 / 32 | RMS dif. imagen (HU) | dif. máx. | contraste dentina-pulpa")
for k in range(4):
    c16 = T16.coords[rng.integers(len(T16.coords))]
    st16 = np.round(c16 - s.crop_size(T16.voxel) / 2).astype(int)
    st32 = np.round(c16 / 2 - s.crop_size(T32.voxel) / 2).astype(int)         # mismo punto físico en la rejilla de 32 µm
    o16, P16 = s.patch(T16, np.random.default_rng(1), start=st16, center=True)
    o32, P32 = s.patch(T32, np.random.default_rng(1), start=st32, center=True)
    n16, n32 = s.crop_size(T16.voxel), s.crop_size(T32.voxel)
    v16 = float(T16.read(T16.canal, st16, n16).sum()) * T16.voxel ** 3
    v32 = float(T32.read(T32.canal, st32, n32).sum()) * T32.voxel ** 3
    dif = (o16["image"] - o32["image"]).float()
    print(f"  {k}     | {n16 * T16.voxel:.3f} / {n32 * T32.voxel:.3f} | {v16:.3f} / {v32:.3f} | "
          f"{float(o16['label_soft'].sum()) * 0.075 ** 3:.3f} / {float(o32['label_soft'].sum()) * 0.075 ** 3:.3f} | "
          f"{float(dif.pow(2).mean().sqrt()):.1f} | {float(dif.abs().max()):.0f} | {float(o16['image'].max() - o16['image'].min()):.0f}")
if a.tmp is None:
    shutil.rmtree(tmp, ignore_errors=True)
