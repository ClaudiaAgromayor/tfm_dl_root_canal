"""Preprocesado de µCT (Nikon XT, .raw uint16): normalización y umbralización del canal.

Uso (un diente o todos los de una carpeta; desde cualquier directorio):
    python analysis/microct/preprocess_microct.py datasets/MicroCT/Diente_35
    python analysis/microct/preprocess_microct.py datasets/MicroCT --all --downsample 2

Por defecto (--downsample 1 = resolución completa, 16 µm) las etiquetas FINALES van a
1_Data_Validation/results/microct/<diente>/. Con otra resolución la salida va a una carpeta distinta
(results/microct_32um, results/microct_64um...), así que una prueba rápida no puede pisar las finales; y si la
carpeta de destino ya tiene etiquetas separadas por split_canal.py (canal_solo_conducto.npy...) o se calculó con
otra resolución, el script se niega a escribir salvo que se pase --force. --out cambia la raíz de salida.

Por cada diente guarda en <salida>/<diente>/:
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
    """Normaliza en el sitio (vol debe ser float32) para no duplicar la memoria a 16 µm."""
    vol -= np.float32(anchors["aire"])
    vol /= np.float32(anchors["tejido"] - anchors["aire"])
    return vol


# ------------------------------------------------------------------- umbralización
def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def remove_specks(cavity: np.ndarray, voxel_mm: float, min_piece_mm3: float) -> np.ndarray:
    """Quita las motas de ruido: piezas 3D de menos de min_piece_mm3 (huecos de 1-10 vóxeles).

    Hay que hacerlo ANTES de unir en z: si no, las motas hacen de puente y pegan al conducto
    cosas que no lo son (rayas de caries).
    """
    pl, pn = ndi.label(cavity)
    if pn == 0:
        return cavity
    ok = np.bincount(pl.ravel(), minlength=pn + 1) * voxel_mm ** 3 >= min_piece_mm3
    ok[0] = False
    return ok[pl]


def canal_mask(norm: np.ndarray, voxel_mm: float, thr: float = 0.5, min_mm3: float = 0.05,
               min_len_mm: float = 1.5, gap_mm: float = 0.13, min_piece_mm3: float = 0.005,
               min_thick_mm: float = 0.05):
    """Canal = hueco encerrado por el diente en los cortes axiales.

    1. Se quitan las motas de ruido (remove_specks).
    2. Se unen los trozos separados por menos de gap_mm en z. A resolución fina el conducto puede
       quedar sin encerrar en un corte suelto y partirse en trozos; la unión se usa solo para
       decidir qué se conserva (la máscara no se modifica).
    3. Se descarta cada grupo unido que sea pequeño (< min_mm3), corto en z (< min_len_mm: poros
       y bolsas de la corona) o delgado: radio inscrito (distancia a la pared) en el percentil
       90 menor que min_thick_mm. Un conducto es un tubo de radio >= ~0.07 mm y se afina hacia el
       ápice, pero eso lo mide el grupo entero, no cada trozo; el borde de una caries son rayas
       de 1-2 vóxeles. El grosor está pensado para vóxel <= 32 µm (a 64 µm una raya de 1 vóxel
       ya mide 0.064 mm).
    Devuelve (máscara uint8, diente bool, componentes conservados, grupos descartados).
    """
    tooth = largest_component(norm > thr)
    filled = np.stack([ndi.binary_fill_holes(s) for s in tooth])      # corte a corte (eje z)
    cavity = remove_specks(filled & ~tooth, voxel_mm, min_piece_mm3)
    g = int(round(gap_mm / voxel_mm))
    # holgura en z: dilatación solo para unir trozos; el largo se mide sin la holgura añadida
    joined = ndi.binary_dilation(cavity, structure=np.ones((2 * g + 1, 1, 1), bool)) if g > 0 else cavity
    lab, n = ndi.label(joined)
    if n == 0:
        return np.zeros_like(cavity, np.uint8), tooth, [], []
    vol_mm3 = ndi.sum(cavity, lab, range(1, n + 1)) * voxel_mm ** 3
    boxes = ndi.find_objects(lab)
    len_mm = np.array([(b[0].stop - b[0].start - 2 * g) * voxel_mm for b in boxes])
    comps, dropped, ok = [], [], np.zeros(n, bool)
    for i in range(n):
        if vol_mm3[i] < min_mm3 or len_mm[i] < min_len_mm:
            continue                      # poros, motas sueltas y bolsas cortas: no se listan
        m = np.pad((lab[boxes[i]] == i + 1) & cavity[boxes[i]], 1)
        r90 = float(np.percentile(ndi.distance_transform_edt(m)[m] * voxel_mm, 90))
        row = {"volumen_mm3": round(float(vol_mm3[i]), 3), "largo_z_mm": round(float(len_mm[i]), 2),
               "radio_p90_mm": round(r90, 3), "z_vox": [boxes[i][0].start, boxes[i][0].stop - 1]}
        if r90 < min_thick_mm:
            dropped.append(row)
        else:
            ok[i] = True
            comps.append(row)
    keep = np.isin(lab, 1 + np.flatnonzero(ok)) & cavity
    return keep.astype(np.uint8), tooth, comps, dropped


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
RESULTS = Path(__file__).resolve().parents[2] / "results"      # 1_Data_Validation/results, no depende del cwd
SPLIT_FILES = ("canal_solo_conducto.npy", "camara_y_cavidades.npy", "canal_completo.npy")   # los crea split_canal.py


def default_out_root(voxel_mm: float, downsample: int) -> Path:
    """results/microct para la resolución completa (las etiquetas finales); results/microct_<µm>um para el resto."""
    return RESULTS / ("microct" if downsample == 1 else f"microct_{round(voxel_mm * downsample * 1000)}um")


def guard_overwrite(out: Path, downsample: int, force: bool):
    """Se niega a pisar una carpeta con etiquetas ya separadas o hecha con otra resolución (salvo --force)."""
    if force or not out.exists():
        return
    prev = None
    if (out / "info.json").exists():
        prev = json.loads((out / "info.json").read_text()).get("parametros", {}).get("downsample")
    separadas = [f for f in SPLIT_FILES if (out / f).exists()]
    if separadas or (prev is not None and prev != downsample):
        raise SystemExit(
            f"{out} ya existe y se sobrescribiría ({'tiene ' + ', '.join(separadas) if separadas else ''}"
            f"{'; ' if separadas and prev not in (None, downsample) else ''}"
            f"{'hecha con --downsample ' + str(prev) if prev not in (None, downsample) else ''}). "
            "Usa --out para otra carpeta o --force si de verdad quieres pisarla (tendrás que volver a ejecutar split_canal.py).")


def process(folder: Path, out_root, downsample: int, thr: float, min_mm3: float,
            min_len_mm: float, gap_mm: float = 0.13, min_piece_mm3: float = 0.005,
            min_thick_mm: float = 0.05, force: bool = False):
    vol, voxel = open_volume(folder)
    out_root = default_out_root(voxel, downsample) if out_root is None else out_root
    guard_overwrite(out_root / folder.name, downsample, force)      # antes de gastar minutos de cálculo
    vol_ds = downsample_mean(vol, downsample) if downsample > 1 else np.asarray(vol, np.float32)
    vox = voxel * downsample
    anchors = normalization_anchors(vol_ds)
    norm = normalize(vol_ds, anchors)
    canal, tooth, comps, dropped = canal_mask(norm, vox, thr, min_mm3, min_len_mm, gap_mm,
                                              min_piece_mm3, min_thick_mm)

    n = len(comps)
    out = out_root / folder.name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "norm.npy", norm.astype(np.float16))
    np.save(out / "canal.npy", canal)
    info = {"diente": folder.name, "forma_zyx": list(norm.shape), "voxel_mm": vox,
            "anclas": anchors, "umbral_normalizado": thr,
            "volumen_diente_mm3": float(tooth.sum() * vox ** 3),
            "volumen_canal_mm3": float(canal.sum() * vox ** 3),
            "componentes_canal": n, "componentes": comps,
            "descartados_por_delgados": dropped,
            "parametros": {"downsample": downsample, "thr": thr, "min_mm3": min_mm3,
                           "min_len_mm": min_len_mm, "gap_mm": gap_mm,
                           "min_piece_mm3": min_piece_mm3, "min_thick_mm": min_thick_mm}}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    qc_figure(norm, canal, out / "qc.png")
    qc_sections(norm, canal, out / "qc_sagital_coronal.png")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="carpeta Diente_XX, o carpeta con varias si --all")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--downsample", type=int, default=1,
                    help="1 = resolución completa (16 µm, las etiquetas finales); 2 = 32 µm; 4 = 64 µm (solo para pruebas "
                         "rápidas: el filtro de grosor no vale a 64 µm)")
    ap.add_argument("--thr", type=float, default=0.5, help="umbral normalizado del canal")
    ap.add_argument("--min-mm3", type=float, default=0.05, help="volumen mínimo de un hueco")
    ap.add_argument("--min-len-mm", type=float, default=1.5,
                    help="largo mínimo en z: descarta poros y caries de la corona")
    ap.add_argument("--gap-mm", type=float, default=0.13,
                    help="trozos de canal separados por menos de esto en z se miden como uno solo")
    ap.add_argument("--min-piece-mm3", type=float, default=0.005,
                    help="quita piezas sueltas de canal más pequeñas que esto (motas de ruido)")
    ap.add_argument("--min-thick-mm", type=float, default=0.05,
                    help="quita piezas finas (rayas de caries): radio inscrito p90 menor que esto")
    ap.add_argument("--out", type=Path, default=None,
                    help="raíz de salida; por defecto 1_Data_Validation/results/microct (16 µm) o microct_<µm>um")
    ap.add_argument("--force", action="store_true", help="permite sobrescribir una carpeta con etiquetas separadas")
    a = ap.parse_args()
    for f in (sorted(p for p in a.path.iterdir() if p.is_dir()) if a.all else [a.path]):
        process(f, a.out, a.downsample, a.thr, a.min_mm3, a.min_len_mm, a.gap_mm, a.min_piece_mm3,
                a.min_thick_mm, a.force)
