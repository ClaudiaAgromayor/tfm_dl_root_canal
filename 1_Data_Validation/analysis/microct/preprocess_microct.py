"""Preprocesado de µCT (Nikon XT, .raw uint16): normalización y umbralización del canal.

Uso (un diente o todos los de una carpeta):
    python analysis/microct/preprocess_microct.py datasets/MicroCT/Diente_35
    python analysis/microct/preprocess_microct.py datasets/MicroCT --all --downsample 4

Por cada diente guarda en results/microct/<diente>/:
    norm.npy         volumen normalizado (aire=0, tejido del diente=1), float16
    canal.npy        máscara binaria del canal (uint8)
    info.json        anclas de normalización, umbral y estadísticas
    qc.png           cortes con la máscara superpuesta, para revisar a ojo

Normalización: en cada volumen se calculan dos anclas robustas (aire y mediana del tejido)
y se reescala I' = (I - aire) / (tejido - aire). Así el umbral se expresa igual en todos los
dientes: 0.5 es el punto medio entre aire y tejido, que es donde cae el borde real de una
estructura aunque la imagen esté desenfocada (efecto de volumen parcial).

Umbralización del canal: el canal es aire dentro del diente, así que tiene el mismo gris que
el fondo. Se distingue por estar ENCERRADO por el diente: se rellenan los huecos de cada corte
axial (el eje z es el eje largo del diente) y se queda lo que era hueco y ahora es tejido.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu


# --------------------------------------------------------------------------- lectura
def read_xtek(path: Path) -> dict:
    """Lee pares clave=valor de un .xtekVolume / .xtekct (ignora la parte binaria)."""
    txt = path.read_bytes()[:20000].decode("latin-1")
    return dict(re.findall(r"^([A-Za-z0-9_.]+)=(.*?)\r?$", txt, flags=re.M))


def open_volume(folder: Path):
    """Devuelve (memmap [Z,Y,X] uint16, tamaño de vóxel en mm) de una carpeta Diente_XX."""
    raw = next(folder.glob("*.raw"))
    meta_file = next(folder.glob("*.xtekVolume"), None) or next(folder.glob("*.xtekct"))
    m = read_xtek(meta_file)
    nx, ny, nz = int(m["VoxelsX"]), int(m["VoxelsY"]), int(m["VoxelsZ"])
    voxel = float(m["VoxelSizeX"])
    expected = nx * ny * nz * 2
    if raw.stat().st_size != expected:
        raise ValueError(f"{raw.name}: pesa {raw.stat().st_size} bytes y las dimensiones "
                         f"{nx}x{ny}x{nz} uint16 esperan {expected}. ¿Fichero incompleto?")
    return np.memmap(raw, dtype=np.uint16, mode="r", shape=(nz, ny, nx)), voxel


def downsample_mean(vol, f: int) -> np.ndarray:
    """Reduce f veces por lado promediando bloques, leyendo por losas para no llenar la RAM."""
    nz, ny, nx = vol.shape
    nz, ny, nx = nz // f * f, ny // f * f, nx // f * f
    out = np.empty((nz // f, ny // f, nx // f), np.float32)
    for k in range(nz // f):
        slab = np.asarray(vol[k * f:(k + 1) * f, :ny, :nx], dtype=np.float32)
        out[k] = slab.reshape(f, ny // f, f, nx // f, f).mean(axis=(0, 2, 4))
    return out


# --------------------------------------------------------------------- normalización
def normalization_anchors(vol: np.ndarray) -> dict:
    """Anclas robustas: aire = mediana del fondo, tejido = mediana de lo que está sobre Otsu."""
    sub = vol[::3, ::3, ::3]
    t = float(threshold_otsu(sub))
    return {"otsu": t,
            "aire": float(np.median(sub[sub < t])),
            "tejido": float(np.median(sub[sub >= t]))}


def normalize(vol: np.ndarray, anchors: dict) -> np.ndarray:
    return (vol - anchors["aire"]) / (anchors["tejido"] - anchors["aire"])


# ------------------------------------------------------------------- umbralización
def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def canal_mask(norm: np.ndarray, voxel_mm: float, thr: float = 0.5, min_mm3: float = 0.05,
               min_len_mm: float = 1.5):
    """Canal = hueco encerrado por el diente en los cortes axiales.

    Se descartan los huecos pequeños (< min_mm3) y los cortos a lo largo de z (< min_len_mm):
    un conducto es un tubo largo, mientras que los poros y la caries de la corona son bolsas
    cortas. Devuelve (máscara uint8, diente bool, lista de componentes conservados).
    """
    tooth = largest_component(norm > thr)
    filled = np.stack([ndi.binary_fill_holes(s) for s in tooth])      # corte a corte (eje z)
    cavity = filled & ~tooth
    lab, n = ndi.label(cavity)
    if n == 0:
        return np.zeros_like(cavity, np.uint8), tooth, []
    vol_mm3 = ndi.sum(cavity, lab, range(1, n + 1)) * voxel_mm ** 3
    boxes = ndi.find_objects(lab)
    len_mm = np.array([(b[0].stop - b[0].start) * voxel_mm for b in boxes])
    ok = (vol_mm3 >= min_mm3) & (len_mm >= min_len_mm)
    keep = np.isin(lab, 1 + np.flatnonzero(ok))
    comps = [{"volumen_mm3": round(float(vol_mm3[i]), 3), "largo_z_mm": round(float(len_mm[i]), 2)}
             for i in np.flatnonzero(ok)]
    return keep.astype(np.uint8), tooth, comps


# ------------------------------------------------------------------------------- QC
def qc_figure(norm, canal, out: Path, n=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zs = np.flatnonzero(canal.any(axis=(1, 2)))
    zs = np.linspace(zs.min(), zs.max(), n).astype(int) if len(zs) else np.linspace(0, len(norm) - 1, n).astype(int)
    fig, ax = plt.subplots(2, n // 2, figsize=(4 * n // 2, 8))
    for a, z in zip(ax.ravel(), zs):
        a.imshow(norm[z], cmap="gray", vmin=0, vmax=1.6)
        a.contour(canal[z], levels=[0.5], colors="r", linewidths=0.8)
        a.set_title(f"z={z}")
        a.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=70)
    plt.close(fig)


def qc_sections(norm, canal, out: Path):
    """Corte sagital y coronal por el centro del canal: se ve hasta dónde llega hacia el ápice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not canal.any():
        return
    cz, cy, cx = [int(round(c)) for c in ndi.center_of_mass(canal)]
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    for a, img, msk, t in ((ax[0], norm[:, :, cx], canal[:, :, cx], f"sagital x={cx}"),
                           (ax[1], norm[:, cy, :], canal[:, cy, :], f"coronal y={cy}")):
        a.imshow(img, cmap="gray", vmin=0, vmax=1.6, aspect="auto")
        a.contour(msk, levels=[0.5], colors="r", linewidths=0.8)
        a.set_title(t)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=70)
    plt.close(fig)


# ------------------------------------------------------------------------------ main
def process(folder: Path, out_root: Path, downsample: int, thr: float, min_mm3: float,
            min_len_mm: float):
    vol, voxel = open_volume(folder)
    vol_ds = downsample_mean(vol, downsample) if downsample > 1 else np.asarray(vol, np.float32)
    vox = voxel * downsample
    anchors = normalization_anchors(vol_ds)
    norm = normalize(vol_ds, anchors).astype(np.float32)
    canal, tooth, comps = canal_mask(norm, vox, thr, min_mm3, min_len_mm)

    n = len(comps)
    out = out_root / folder.name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "norm.npy", norm.astype(np.float16))
    np.save(out / "canal.npy", canal)
    info = {"diente": folder.name, "forma_zyx": list(norm.shape), "voxel_mm": vox,
            "anclas": anchors, "umbral_normalizado": thr,
            "volumen_diente_mm3": float(tooth.sum() * vox ** 3),
            "volumen_canal_mm3": float(canal.sum() * vox ** 3),
            "componentes_canal": n, "componentes": comps}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    qc_figure(norm, canal, out / "qc.png")
    qc_sections(norm, canal, out / "qc_sagital_coronal.png")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="carpeta Diente_XX, o carpeta con varias si --all")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--downsample", type=int, default=4, help="1 = resolución completa (16 µm)")
    ap.add_argument("--thr", type=float, default=0.5, help="umbral normalizado del canal")
    ap.add_argument("--min-mm3", type=float, default=0.05, help="volumen mínimo de un hueco")
    ap.add_argument("--min-len-mm", type=float, default=1.5,
                    help="largo mínimo en z: descarta poros y caries de la corona")
    ap.add_argument("--out", type=Path, default=Path("results/microct"))
    a = ap.parse_args()
    for f in (sorted(p for p in a.path.iterdir() if p.is_dir()) if a.all else [a.path]):
        process(f, a.out, a.downsample, a.thr, a.min_mm3, a.min_len_mm)
