"""Barrido de la holgura en z (`--gap-mm` de preprocess_microct.py): volumen y nº de componentes de conducto.

Reproduce el barrido de RESUMEN_PROYECTO.md (apartado 3.3): con gap = 0 el conducto del diente 51 a 16 µm se parte en
trozos y el filtro de largo mínimo tira los cortos; con 0,13 mm sale 1 componente de ~23 mm3, igual que a 32 µm.

    python barrido_gap.py --downsample 2 1 --gaps 0 0.065 0.13 0.2 0.3 0.5 [--dientes 35 51 52] [--min-piece-mm3 0.005]

Con `--min-piece-mm3 0` no se quitan las motas antes de unir (el barrido original, anterior a esa corrección): con
huecos pequeños que hacen de puente salen más piezas. El filtro de grosor (`--min-thick-mm`) NO se aplica aquí;
para eso, ejecutar preprocess_microct.py completo.
Cuesta unos minutos por diente a 16 µm (lee el .raw de 2-3 GB y etiqueta volúmenes de 1.400 millones de vóxeles).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import preprocess_microct as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", type=Path, default=HERE.parents[2] / "datasets" / "MicroCT")
ap.add_argument("--dientes", nargs="*", default=["35", "51", "52"])
ap.add_argument("--downsample", type=int, nargs="*", default=[2, 1])
ap.add_argument("--gaps", type=float, nargs="*", default=[0.0, 0.065, 0.13, 0.2, 0.3, 0.5])
ap.add_argument("--min-piece-mm3", type=float, default=0.005)
ap.add_argument("--min-mm3", type=float, default=0.05)
ap.add_argument("--min-len-mm", type=float, default=1.5)
a = ap.parse_args()

for d in a.dientes:
    vol, voxel = P.open_volume(a.datasets / f"Diente_{d}")
    for ds in a.downsample:
        v = P.downsample_mean(vol, ds) if ds > 1 else np.asarray(vol, np.float32)
        vx = voxel * ds
        norm = P.normalize(v, P.normalization_anchors(v))
        tooth = P.largest_component(norm > 0.5)
        cav = np.stack([ndi.binary_fill_holes(s) for s in tooth]) & ~tooth
        if a.min_piece_mm3 > 0:
            cav = P.remove_specks(cav, vx, a.min_piece_mm3)
        del v, norm
        for gm in a.gaps:
            g = int(round(gm / vx))
            j = ndi.binary_dilation(cav, structure=np.ones((2 * g + 1, 1, 1), bool)) if g else cav
            lab, n = ndi.label(j)
            vm = ndi.sum(cav, lab, range(1, n + 1)) * vx ** 3
            ln = np.array([(b[0].stop - b[0].start - 2 * g) * vx for b in ndi.find_objects(lab)])
            ok = (vm >= a.min_mm3) & (ln >= a.min_len_mm)
            print(f"diente {d} ds={ds} gap={gm:.3f} mm: volumen {vm[ok].sum():.2f} mm3, {int(ok.sum())} componentes "
                  f"{[round(float(x), 2) for x in vm[ok]]} largos {[round(float(x), 1) for x in ln[ok]]}", flush=True)
