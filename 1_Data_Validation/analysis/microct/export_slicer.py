"""Exporta un diente a NRRD para abrirlo en 3D Slicer: recortado a la caja del diente y a 32 µm.

    python export_slicer.py results/microct/Diente_35 results/microct/Diente_51 --out results/para_slicer

Por diente escribe (comprimido):
    Diente_XX_imagen_uint8.nrrd            gris = round(norm * 127.5), recortado a 0-255  (0 = aire, ~124 = dentina, ~204 = esmalte;
                                           lo que pasa de norm 2.0 se satura)
    Diente_XX_mascara_solo_conducto.nrrd   0/1 (la etiqueta)
    Diente_XX_mascara_camara_y_cavidades.nrrd
    Diente_XX_mascara_completa.nrrd        solo conducto + cámara y cavidades (canal_completo.npy)
Las máscaras de 32 µm se obtienen de las de 16 µm por mayoría (>= 4 de 8 vóxeles); la cámara se deriva como completa - solo conducto. La orientación (izquierda/derecha,
anterior/posterior) del escaneo NO se conoce: los ejes son los del volumen (x, y, z), con z el eje largo del diente.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_microct import largest_component  # noqa: E402


def block_mean2(arr, box):
    """Media por bloques de 2x2x2 de arr[box] (box = ((z0,z1),(y0,y1),(x0,x1)), tamaños pares), por losas."""
    (z0, z1), (y0, y1), (x0, x1) = box
    out = np.empty(((z1 - z0) // 2, (y1 - y0) // 2, (x1 - x0) // 2), np.float32)
    for k in range(out.shape[0]):
        slab = np.asarray(arr[z0 + 2 * k:z0 + 2 * k + 2, y0:y1, x0:x1], np.float32)
        out[k] = slab.reshape(2, (y1 - y0) // 2, 2, (x1 - x0) // 2, 2).mean(axis=(0, 2, 4))
    return out


def block_major2(arr, box):
    """Igual, pero para máscaras: un bloque vale 1 si >= 4 de sus 8 vóxeles valen 1."""
    (z0, z1), (y0, y1), (x0, x1) = box
    out = np.empty(((z1 - z0) // 2, (y1 - y0) // 2, (x1 - x0) // 2), np.uint8)
    for k in range(out.shape[0]):
        slab = np.asarray(arr[z0 + 2 * k:z0 + 2 * k + 2, y0:y1, x0:x1], np.uint8)
        out[k] = slab.reshape(2, (y1 - y0) // 2, 2, (x1 - x0) // 2, 2).sum(axis=(0, 2, 4)) >= 4
    return out


def write_nrrd(path: Path, vol_zyx, voxel_mm, desc):
    import nrrd
    hdr = {"space": "left-posterior-superior", "space directions": np.diag([voxel_mm] * 3),
           "space origin": np.zeros(3), "kinds": ["domain"] * 3, "encoding": "gzip", "description": desc}
    nrrd.write(str(path), np.ascontiguousarray(vol_zyx.transpose(2, 1, 0)), hdr, compression_level=9)


def export(folder: Path, out: Path, margin_vox=6):
    info = json.loads((folder / "info.json").read_text())
    v16 = float(info["voxel_mm"])
    norm = np.load(folder / "norm.npy", mmap_mode="r")
    masks = {"solo_conducto": "canal_solo_conducto.npy", "camara_y_cavidades": "camara_y_cavidades.npy",
             "completa": "canal_completo.npy"}
    arrs = {k: np.load(folder / f, mmap_mode="r") for k, f in masks.items()}
    # caja del diente: componente mayor de norm > 0.5, calculada a 32 µm
    full = (0, norm.shape[0] // 2 * 2), (0, norm.shape[1] // 2 * 2), (0, norm.shape[2] // 2 * 2)
    n32 = block_mean2(norm, full)
    box = [(int(a.min()), int(a.max()) + 1) for a in np.nonzero(largest_component(n32 > 0.5))]
    box = [(max(a - margin_vox, 0), min(b + margin_vox, s)) for (a, b), s in zip(box, n32.shape)]
    lo16 = tuple((a * 2, b * 2) for a, b in box)           # la misma caja en índices de 16 µm
    img = np.clip(np.rint(n32[tuple(slice(a, b) for a, b in box)] * 127.5), 0, 255).astype(np.uint8)
    v32 = v16 * 2
    out.mkdir(parents=True, exist_ok=True)
    tag = folder.name
    desc_img = "grey = round(norm*127.5) clipped to 0-255; norm: air=0, dentin~0.975, enamel~1.6"
    write_nrrd(out / f"{tag}_imagen_uint8.nrrd", img, v32, desc_img)
    res = {"diente": tag, "forma_zyx": list(img.shape), "voxel_mm": v32,
           "caja_16um_zyx": [list(map(int, b)) for b in lo16], "archivos": {}}
    # solo conducto y completa por mayoría; la cámara se deriva (completa menos solo conducto) para que no se solapen
    # en el plano del orificio (un bloque a caballo de ese plano tiene 4 vóxeles de cada lado y ganaría en los dos)
    m32 = {k: block_major2(arrs[k], lo16) for k in ("solo_conducto", "completa")}
    m32["camara_y_cavidades"] = (m32["completa"] & (1 - m32["solo_conducto"])).astype(np.uint8)
    for k in ("solo_conducto", "camara_y_cavidades", "completa"):
        m = m32[k]
        write_nrrd(out / f"{tag}_mascara_{k}.nrrd", m, v32, f"mask {k}, 0/1, 32 um (majority vote of 2x2x2 voxels of 16 um)")
        res["archivos"][k] = {"vox_32um": int(m.sum()), "mm3": round(float(m.sum() * v32 ** 3), 3)}
    res["archivos"]["imagen_uint8"] = {}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("results/para_slicer"))
    a = ap.parse_args()
    allres = []
    for f in a.folders:
        r = export(f, a.out)
        allres.append(r)
        print(json.dumps(r))
    (a.out / "resumen_exportacion.json").write_text(json.dumps(allres, indent=1))
