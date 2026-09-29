"""
plot_ian_hd95.py
Genera una figura de diagnóstico HD95 para un paciente, igual al estilo de la imagen de referencia:
  - Arriba izq:  Vista 3D superficie GT vs Manual IAN
  - Arriba der:  Zoom al peor par de puntos
  - Abajo izq:   Corte 2D axial en el slice del peor punto
  - Abajo der:   Histograma de distancias superficie-a-superficie

Uso:
    python analysis/ian/plot_ian_hd95.py              # usa PATIENT_ID configurado abajo
    python analysis/ian/plot_ian_hd95.py P48          # paciente por argumento

Ajusta las constantes de la sección CONFIGURACIÓN.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import nrrd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy import ndimage

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
PATIENT_ID = sys.argv[1] if len(sys.argv) > 1 else "P35"
IAN_LABEL  = 2      # label del IAN en el .seg.nrrd  (ver columna labels_ian del Excel)

BASE_DIR   = Path(
    r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation"
)
SLICER_DIR  = BASE_DIR.parent / "2_UNet" / "datasets" / "Segmentaciones 3DSlicer"
OUTPUT_DIR  = BASE_DIR / "results" / "ian" / "hd95_explicado"
RESUMEN_XLSX = BASE_DIR / "results" / "canales" / "canales_por_capa" / "resumen_canales_pacientes.xlsx"
# ─────────────────────────────────────────────────────────────────────────────

# importar funciones del proyecto (ian_metrics.py está en esta misma carpeta)
from ian_metrics import load_gt_ian, dice_score


# ── helpers ──────────────────────────────────────────────────────────────────

def surface_voxels(mask: np.ndarray) -> np.ndarray:
    st = ndimage.generate_binary_structure(mask.ndim, 1)
    return np.logical_xor(mask, ndimage.binary_erosion(mask, structure=st, border_value=0))


def all_surface_distances(mask_a: np.ndarray, mask_b: np.ndarray):
    """Devuelve (distances, source_flags) donde source_flags indica de qué máscara viene cada dist."""
    sa, sb = surface_voxels(mask_a), surface_voxels(mask_b)
    dm_b = ndimage.distance_transform_edt(~sb)
    dm_a = ndimage.distance_transform_edt(~sa)
    d_ab = dm_b[sa]   # desde a hacia b
    d_ba = dm_a[sb]   # desde b hacia a
    # coordenadas de superficie
    coords_sa = np.argwhere(sa)
    coords_sb = np.argwhere(sb)
    return d_ab, d_ba, coords_sa, coords_sb, dm_b, dm_a


def hd95(d_ab, d_ba):
    d = np.concatenate([d_ab, d_ba])
    return float(np.percentile(d, 95)) if d.size else float("nan")


def load_ian_label_from_excel(patient_id: str) -> int:
    """Lee labels_ian del Excel; devuelve IAN_LABEL global si no encuentra nada."""
    try:
        import pandas as pd
        df = pd.read_excel(str(RESUMEN_XLSX), dtype={"labels_ian": str})
        rows = df[df["paciente"] == patient_id]
        if not rows.empty:
            raw = rows["labels_ian"].dropna().iloc[0]
            return int(float(str(raw).split(",")[0].strip()))
    except Exception:
        pass
    return IAN_LABEL


def filter_gt_to_side(gt: np.ndarray, manual: np.ndarray) -> np.ndarray:
    if not manual.any() or not gt.any():
        return gt
    cx = np.argwhere(manual)[:, 0].mean()
    mid = gt.shape[0] / 2.0
    filtered = gt.copy()
    if cx < mid:
        filtered[int(mid):, :, :] = False
    else:
        filtered[:int(mid), :, :] = False
    return filtered if filtered.any() else gt


def subsample_surface(coords: np.ndarray, max_pts: int = 4000) -> np.ndarray:
    if len(coords) <= max_pts:
        return coords
    idx = np.random.choice(len(coords), max_pts, replace=False)
    return coords[idx]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[{PATIENT_ID}] Cargando datos...", flush=True)

    # 1. Manual IAN desde NRRD
    nrrd_path = SLICER_DIR / PATIENT_ID / "Segmentation.seg.nrrd"
    if not nrrd_path.exists():
        candidates = list((SLICER_DIR / PATIENT_ID).glob("*.seg.nrrd")) if (SLICER_DIR / PATIENT_ID).exists() else []
        if not candidates:
            raise FileNotFoundError(f"No se encontró NRRD en {SLICER_DIR / PATIENT_ID}")
        nrrd_path = candidates[0]

    data, header = nrrd.read(str(nrrd_path))
    ian_lbl = load_ian_label_from_excel(PATIENT_ID)
    manual_ian = (data == ian_lbl)
    print(f"[{PATIENT_ID}] Manual IAN: {manual_ian.sum()} voxels (label={ian_lbl})", flush=True)

    # 2. GT IAN
    gt_full = load_gt_ian(PATIENT_ID)
    if gt_full is None:
        raise ValueError(f"load_gt_ian devolvió None para {PATIENT_ID}")
    gt_side = filter_gt_to_side(gt_full.astype(bool), manual_ian)
    print(f"[{PATIENT_ID}] GT IAN (filtrado): {gt_side.sum()} voxels", flush=True)

    # 3. Distancias superficie
    print(f"[{PATIENT_ID}] Calculando distancias...", flush=True)
    d_ab, d_ba, coords_sa, coords_sb, dm_b, dm_a = all_surface_distances(manual_ian, gt_side)
    all_dists = np.concatenate([d_ab, d_ba])
    hd95_val  = hd95(d_ab, d_ba)
    dice_val  = dice_score(manual_ian, gt_side)
    max_dist  = float(all_dists.max()) if all_dists.size else 0.0
    print(f"[{PATIENT_ID}] Dice={dice_val:.3f}  HD95={hd95_val:.1f}  Max={max_dist:.1f}", flush=True)

    # ── Punto representativo del HD95 ──────────────────────────────────────────
    # HD95 = percentil 95 de TODAS las distancias superficie-a-superficie (d_ab + d_ba).
    # Buscamos el punto de superficie cuya distancia al opuesto es la más cercana
    # al valor del percentil 95, en la dirección que más contribuye a ese valor.
    #
    # Estrategia:
    #   1. Concatenar d_ab y d_ba con etiquetas de origen.
    #   2. Calcular el percentil 95 del conjunto completo.
    #   3. Encontrar el punto cuya distancia es más cercana a hd95_val.
    #   4. Calcular el nearest neighbor en la superficie opuesta mediante el
    #      distance transform ya computado (sin recalcular).

    all_dists_ab = d_ab   # distancias de cada voxel de superficie manual → GT
    all_dists_ba = d_ba   # distancias de cada voxel de superficie GT → manual

    # Índice del punto más cercano al p95 en cada dirección
    idx_p95_ab = int(np.argmin(np.abs(all_dists_ab - hd95_val)))
    idx_p95_ba = int(np.argmin(np.abs(all_dists_ba - hd95_val)))
    dist_p95_ab = float(all_dists_ab[idx_p95_ab])
    dist_p95_ba = float(all_dists_ba[idx_p95_ba])

    # Elegimos la dirección cuyo punto está más cerca del valor exacto del HD95
    if abs(dist_p95_ab - hd95_val) <= abs(dist_p95_ba - hd95_val):
        p95_src  = coords_sa[idx_p95_ab]   # punto en superficie manual
        # Nearest neighbor en superficie GT (usando coordenadas de superficie)
        surf_gt_coords = np.argwhere(surface_voxels(gt_side))
        diffs = surf_gt_coords - p95_src
        nn_idx = int(np.argmin((diffs**2).sum(axis=1)))
        p95_nn = surf_gt_coords[nn_idx]
        source_label = "Manual → GT"
    else:
        p95_src  = coords_sb[idx_p95_ba]   # punto en superficie GT
        surf_man_coords = np.argwhere(surface_voxels(manual_ian))
        diffs = surf_man_coords - p95_src
        nn_idx = int(np.argmin((diffs**2).sum(axis=1)))
        p95_nn = surf_man_coords[nn_idx]
        source_label = "GT → Manual"

    # También guardamos el punto con distancia máxima (para referencia en histograma)
    max_dist = float(all_dists.max()) if all_dists.size else 0.0

    # ── FIGURA ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
    fig.suptitle(
        f"{PATIENT_ID} | IAN vs IAN only: HD95 is the 95th percentile of all 3D surface distances",
        fontsize=13, color="white", fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28,
                           left=0.05, right=0.97, top=0.93, bottom=0.07)

    # ── Colores ──
    C_GT     = "#4ea8de"   # azul GT
    C_MANUAL = "#f4a261"   # naranja manual
    C_WORST  = "#e63946"   # rojo peor punto
    C_NN     = "#06d6a0"   # verde punto más cercano

    # Submuestrear superficies para 3D
    np.random.seed(0)
    pts_gt  = subsample_surface(coords_sb)
    pts_man = subsample_surface(coords_sa)

    # ── Plot 1: Vista 3D completa ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0], projection="3d", facecolor="#161b22")
    ax1.scatter(pts_gt[:, 0],  pts_gt[:, 1],  pts_gt[:, 2],
                c=C_GT,     s=0.4, alpha=0.5, rasterized=True)
    ax1.scatter(pts_man[:, 0], pts_man[:, 1], pts_man[:, 2],
                c=C_MANUAL, s=0.4, alpha=0.5, rasterized=True)
    ax1.scatter(*p95_src, c=C_WORST, s=60, zorder=10)
    ax1.scatter(*p95_nn,  c=C_NN,    s=60, zorder=10, label="Nearest point on opposite surface")
    ax1.plot([p95_src[0], p95_nn[0]],
             [p95_src[1], p95_nn[1]],
             [p95_src[2], p95_nn[2]], color=C_WORST, linewidth=1.5)
    _style_3d(ax1, f"{PATIENT_ID} | 3D surface distance example")

    # leyenda manual con proxies
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_GT,     markersize=6, label="GT IAN surface"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_MANUAL, markersize=6, label="Manual IAN surface"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_WORST,  markersize=7, label=f"HD95 representative point = {hd95_val:.1f} vox"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_NN,     markersize=7, label="Nearest point on opposite surface"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=7,
               facecolor="#21262d", edgecolor="#30363d", labelcolor="white")

    # ── Plot 2: Zoom al peor par ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1], projection="3d", facecolor="#161b22")
    margin = 60
    x0, x1 = max(0, min(p95_src[0], p95_nn[0]) - margin), min(data.shape[0], max(p95_src[0], p95_nn[0]) + margin)
    y0, y1 = max(0, min(p95_src[1], p95_nn[1]) - margin), min(data.shape[1], max(p95_src[1], p95_nn[1]) + margin)
    z0, z1 = max(0, min(p95_src[2], p95_nn[2]) - margin), min(data.shape[2], max(p95_src[2], p95_nn[2]) + margin)

    def in_box(pts):
        mask = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
                (pts[:, 1] >= y0) & (pts[:, 1] <= y1) &
                (pts[:, 2] >= z0) & (pts[:, 2] <= z1))
        return pts[mask]

    pts_gt_z  = in_box(coords_sb)
    pts_man_z = in_box(coords_sa)
    if len(pts_gt_z):
        ax2.scatter(pts_gt_z[:, 0],  pts_gt_z[:, 1],  pts_gt_z[:, 2],
                    c=C_GT,     s=1.0, alpha=0.6, rasterized=True)
    if len(pts_man_z):
        ax2.scatter(pts_man_z[:, 0], pts_man_z[:, 1], pts_man_z[:, 2],
                    c=C_MANUAL, s=1.0, alpha=0.6, rasterized=True)
    ax2.scatter(*p95_src, c=C_WORST, s=80, zorder=10)
    ax2.scatter(*p95_nn,  c=C_NN,    s=80, zorder=10)
    ax2.plot([p95_src[0], p95_nn[0]],
             [p95_src[1], p95_nn[1]],
             [p95_src[2], p95_nn[2]], color=C_WORST, linewidth=2)
    _style_3d(ax2, "Zoom around the HD95 representative pair")

    # ── Plot 3: Corte 2D axial en z del peor punto ───────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    z_slice = int(p95_src[2])
    ax3.set_facecolor("black")

    # Contornos de ambas máscaras
    gt_sl  = gt_side[:, :, z_slice]
    man_sl = manual_ian[:, :, z_slice]

    if gt_sl.any():
        ax3.contour(gt_sl.T,  levels=[0.5], colors=[C_GT],     linewidths=1.2)
    if man_sl.any():
        ax3.contour(man_sl.T, levels=[0.5], colors=[C_MANUAL], linewidths=1.2)

    ax3.scatter(p95_src[0], p95_src[1], c=C_WORST, s=60, zorder=5, label="Max source point")
    ax3.scatter(p95_nn[0],  p95_nn[1],  c=C_NN,    s=60, zorder=5, label="Nearest target point")
    ax3.set_title(f"2D IAN contours at z={z_slice}  (HD95 point)", color="white", fontsize=10)
    ax3.set_xlim(0, data.shape[0])
    ax3.set_ylim(0, data.shape[1])
    ax3.invert_yaxis()
    ax3.legend(fontsize=7, facecolor="#21262d", edgecolor="#30363d", labelcolor="white")
    ax3.tick_params(colors="gray")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Plot 4: Histograma ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#161b22")
    ax4.hist(all_dists, bins=80, color=C_GT, alpha=0.75, edgecolor="none")
    ax4.axvline(hd95_val,  color="#2ecc71", linestyle="--", linewidth=1.5, label=f"HD95 = {hd95_val:.1f} vox")
    ax4.axvline(max_dist,  color=C_WORST,   linestyle="-",  linewidth=1.5, label=f"Max = {max_dist:.1f} vox")

    textstr = (f"DICE IAN = {dice_val:.3f}\n"
               f"HD95 = {hd95_val:.1f} vox\n"
               f"Max distance = {max_dist:.1f} vox\n"
               f"Source: {source_label} surface")
    ax4.text(0.97, 0.97, textstr, transform=ax4.transAxes, fontsize=8,
             verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d",
                       edgecolor="#30363d", alpha=0.9),
             color="white")
    ax4.set_title("Distribution of surface-to-surface distances", color="white", fontsize=10)
    ax4.set_xlabel("Distance in voxels", color="gray", fontsize=9)
    ax4.set_ylabel("Count", color="gray", fontsize=9)
    ax4.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="white")
    ax4.tick_params(colors="gray")
    for spine in ax4.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Guardar ──────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{PATIENT_ID}_hd95_explained.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[{PATIENT_ID}] Guardado: {out_path}", flush=True)


def _style_3d(ax, title: str):
    ax.set_title(title, color="white", fontsize=9, pad=4)
    ax.set_xlabel("X", color="gray", fontsize=7)
    ax.set_ylabel("Y", color="gray", fontsize=7)
    ax.set_zlabel("Z", color="gray", fontsize=7)
    ax.tick_params(colors="gray", labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#30363d")
    ax.yaxis.pane.set_edgecolor("#30363d")
    ax.zaxis.pane.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", linewidth=0.5)


if __name__ == "__main__":
    main()
    