"""Simulador con etiqueta solo-conducto: relleno de la cámara, franja de "ignorar" y alineación imagen-etiqueta.

Reproduce lo de RESUMEN_PROYECTO.md (apartados 3.6 y 3.5 de CLAUDE.md "Simulador con la etiqueta nueva"):
  (1) con pulpa fija a 100 HU y hueso a 900 HU, sin ruido: gris medio del conducto, de la cámara y del fondo lejos del
      diente, y fracción del parche con peso 0 según `labels.weight_mode` (band / chamber / none);
  (2) alineación: con desenfoque y ruido apagados y el conducto como aire, la correlación entre la imagen y la etiqueta
      blanda desplazada 0, ±1, ±2 vóxeles debe ser la más negativa con desplazamiento 0 en los tres ejes.

    python prueba_alineacion_y_pesos.py [--dientes Diente_35 Diente_51 Diente_52]
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
a = ap.parse_args()
sys.path.insert(0, str(HERE.parent))
import simulate_cbct as sc  # noqa: E402

if sc.DEV.type == "cuda":
    torch.set_default_device("cuda")
cfg = yaml.safe_load((HERE.parent / "simulate_cbct_config.yaml").read_text(encoding="utf-8"))
c = copy.deepcopy(cfg)
c["psf"].update(sigma_mm=[0.03, 0.03], z_factor=[1, 1]); c["noise"].update(sd_hu=[0, 0]); c["shading"].update(mult=[0, 0], add_hu=[0, 0])
c["intensity"]["canal_hu"] = [100, 100]; c["labels"]["chamber_offset_hu"] = [0, 0]; c["crop"]["aug_xy"] = False
c["background"].update(base_hu=[900, 900], texture_amp_hu=[0, 0]); c["background"]["pdl"]["prob"] = 0
for mode in ("band", "chamber", "none"):
    cc = copy.deepcopy(c); cc["labels"]["weight_mode"] = mode
    sim = sc.Simulator(cc)
    print(f"\n--- weight_mode = {mode}")
    for d in a.dientes:
        t = sc.Tooth(a.results / d)
        z_or = t.info["separacion"]["z_orificio_vox"]
        near = t.coords[np.abs(t.coords[:, 0] - z_or) < 40]            # parche que cruza el orificio
        ctr = near[len(near) // 2]
        out, _ = sim.patch(t, np.random.default_rng(5), start=np.round(ctr - sim.crop_size(t.voxel) / 2).astype(int), center=True)
        img, lab, ch, w = out["image"], out["label_soft"], out["chamber"], out["weight"]
        mean = lambda m: float(img[m].mean()) if m.any() else float("nan")
        can, cam = lab > 0.95, ch > 0.95
        far = (lab < 0.01) & (ch < 0.01) & (out["tooth"] < 0.01)
        print(f"  {d}: gris medio conducto {mean(can):6.0f} | cámara {mean(cam):6.0f} | fondo lejos {mean(far):6.0f} | peso 0 en "
              f"{100 * float((w == 0).float().mean()):.2f} % del parche (dentro del conducto {int(((w == 0) & (lab > 0.5)).sum())} vox, "
              f"de la cámara {int(((w == 0) & (ch > 0.5)).sum())} vox)")

c2 = copy.deepcopy(c); c2["intensity"]["canal_hu"] = [-1000, -1000]; c2["labels"]["chamber_fill"] = "bg"
sim = sc.Simulator(c2)
print("\nalineación imagen vs etiqueta solo-conducto: correlación con la etiqueta desplazada -2,-1,0,+1,+2 (mínimo en 0 = alineada)")
for d in a.dientes:
    t = sc.Tooth(a.results / d)
    out, _ = sim.patch(t, np.random.default_rng(3))
    img, soft = out["image"], out["label_soft"]
    row = []
    for ax, an in enumerate("zyx"):
        cs = [float(torch.corrcoef(torch.stack([img.flatten(), torch.roll(soft, s, dims=ax).flatten()]))[0, 1]) for s in (-2, -1, 0, 1, 2)]
        row.append(f"{an}: " + " ".join(f"{q:+.3f}" for q in cs) + f" (mín en {(-2, -1, 0, 1, 2)[int(np.argmin(cs))]:+d})")
    print(" ", d, "|", " | ".join(row))
