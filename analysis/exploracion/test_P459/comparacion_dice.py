import argparse
import csv
import itertools
from collections import deque
from pathlib import Path

import nibabel as nib
import nrrd
import numpy as np


DEFAULT_SEG_PATH = (
    r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas"
    r"\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\P459_3DSlicer_prueba\Segmentation_1.seg.nrrd"
)

DEFAULT_GT_PATH = (
    r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas"
    r"\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\Pulpy3D\P459\gt_pulp.nii.gz"
)

DEFAULT_OUT_CSV = "results/exploracion/test_P459/dice_por_capa.csv"
DEFAULT_OUT_DIR = "results/exploracion/test_P459/salida_comparacion_dice"


def to_3d_label_volume(seg_raw: np.ndarray) -> np.ndarray:
    if seg_raw.ndim == 3:
        return seg_raw
    if seg_raw.ndim == 4:
        label_volume = np.zeros(seg_raw.shape[1:], dtype=np.uint8)
        for i in range(seg_raw.shape[0]):
            label_volume[seg_raw[i] > 0] = i + 1
        return label_volume
    raise ValueError(f"NRRD de segmentación con dimensión no soportada: {seg_raw.ndim}")


def load_segmentation_nrrd(path: str) -> np.ndarray:
    seg_raw, _ = nrrd.read(path)
    seg_3d = to_3d_label_volume(seg_raw)
    return seg_3d > 0


def load_gt_nii(path: str, threshold: float = 0.5) -> np.ndarray:
    img = nib.load(path)
    data = img.get_fdata()
    return data > threshold


def load_gt_nii_with_ref(path: str, threshold: float = 0.5) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(path)
    data = img.get_fdata()
    return data > threshold, img


def dice_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    a_sum = int(np.count_nonzero(a))
    b_sum = int(np.count_nonzero(b))
    if a_sum == 0 and b_sum == 0:
        return 1.0
    inter = int(np.count_nonzero(a & b))
    return 2.0 * inter / (a_sum + b_sum)


def jaccard_index(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 1.0
    inter = int(np.count_nonzero(a & b))
    return inter / union


def precision_recall(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    tp = int(np.count_nonzero(a & b))
    fp = int(np.count_nonzero(a & ~b))
    fn = int(np.count_nonzero(~a & b))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return precision, recall


def volume_similarity(a: np.ndarray, b: np.ndarray) -> float:
    va = int(np.count_nonzero(a))
    vb = int(np.count_nonzero(b))
    denom = va + vb
    if denom == 0:
        return 1.0
    return 1.0 - abs(va - vb) / denom


def transformed_candidates(seg: np.ndarray, target_shape: tuple[int, int, int]):
    axes = (0, 1, 2)
    for perm in itertools.permutations(axes):
        seg_perm = np.transpose(seg, perm)
        if seg_perm.shape != target_shape:
            continue
        for flip_mask in range(8):
            candidate = seg_perm
            applied_flips = []
            for ax in axes:
                if flip_mask & (1 << ax):
                    candidate = np.flip(candidate, axis=ax)
                    applied_flips.append(ax)
            yield candidate, perm, tuple(applied_flips)


def find_best_alignment(seg: np.ndarray, gt: np.ndarray):
    if seg.ndim != 3 or gt.ndim != 3:
        raise ValueError("Solo se admiten máscaras 3D")

    best = None
    total_checked = 0

    for cand, perm, flips in transformed_candidates(seg, gt.shape):
        total_checked += 1
        d = dice_coefficient(cand, gt)
        j = jaccard_index(cand, gt)

        if best is None or d > best["dice"] or (np.isclose(d, best["dice"]) and j > best["jaccard"]):
            best = {
                "mask": cand,
                "perm": perm,
                "flips": flips,
                "dice": d,
                "jaccard": j,
            }

    if best is None:
        raise ValueError(
            "No se pudo alinear segmentación y GT por permutación+flip. "
            f"shape_seg={seg.shape}, shape_gt={gt.shape}"
        )

    best["checked"] = total_checked
    return best


def metrics_globales(seg: np.ndarray, gt: np.ndarray) -> dict:
    dice = dice_coefficient(seg, gt)
    jac = jaccard_index(seg, gt)
    precision, recall = precision_recall(seg, gt)
    vs = volume_similarity(seg, gt)

    tp = int(np.count_nonzero(seg & gt))
    fp = int(np.count_nonzero(seg & ~gt))
    fn = int(np.count_nonzero(~seg & gt))

    seg_vox = int(np.count_nonzero(seg))
    gt_vox = int(np.count_nonzero(gt))

    return {
        "dice": dice,
        "jaccard": jac,
        "precision": precision,
        "recall": recall,
        "volume_similarity": vs,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "vox_seg": seg_vox,
        "vox_gt": gt_vox,
    }


def dice_por_capa(seg: np.ndarray, gt: np.ndarray, axis: int = 2) -> list[dict]:
    if seg.shape != gt.shape:
        raise ValueError("seg y gt deben tener la misma forma para cálculo por capa")

    n_layers = seg.shape[axis]
    rows = []

    for idx in range(n_layers):
        seg_slice = np.take(seg, idx, axis=axis)
        gt_slice = np.take(gt, idx, axis=axis)

        d = dice_coefficient(seg_slice, gt_slice)
        j = jaccard_index(seg_slice, gt_slice)

        seg_vox = int(np.count_nonzero(seg_slice))
        gt_vox = int(np.count_nonzero(gt_slice))
        inter = int(np.count_nonzero(seg_slice & gt_slice))
        union = int(np.count_nonzero(seg_slice | gt_slice))

        rows.append(
            {
                "axis": axis,
                "layer": idx,
                "dice": d,
                "jaccard": j,
                "vox_seg": seg_vox,
                "vox_gt": gt_vox,
                "intersection": inter,
                "union": union,
                "both_empty": int(seg_vox == 0 and gt_vox == 0),
            }
        )

    return rows


def write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No hay filas para guardar en CSV")

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_mask_nifti(mask: np.ndarray, reference_img: nib.Nifti1Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(mask.astype(np.uint8), reference_img.affine, reference_img.header)
    nib.save(nii, str(out_path))


def build_overlap_map(seg: np.ndarray, gt: np.ndarray) -> np.ndarray:
    overlap = np.zeros(seg.shape, dtype=np.uint8)
    tp = seg & gt
    fp = seg & ~gt
    fn = ~seg & gt

    overlap[tp] = 1
    overlap[fp] = 2
    overlap[fn] = 3
    return overlap


def connected_components_26(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    if mask.ndim != 3:
        raise ValueError("connected_components_26 requiere máscara 3D")

    labels = np.zeros(mask.shape, dtype=np.int32)
    component_sizes = [0]
    comp_id = 0

    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    coords = np.argwhere(mask)
    sx, sy, sz = mask.shape

    for x, y, z in coords:
        if labels[x, y, z] != 0:
            continue

        comp_id += 1
        q = deque([(int(x), int(y), int(z))])
        labels[x, y, z] = comp_id
        size = 0

        while q:
            cx, cy, cz = q.popleft()
            size += 1

            for dx, dy, dz in neighbor_offsets:
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                if nx < 0 or ny < 0 or nz < 0 or nx >= sx or ny >= sy or nz >= sz:
                    continue
                if not mask[nx, ny, nz]:
                    continue
                if labels[nx, ny, nz] != 0:
                    continue

                labels[nx, ny, nz] = comp_id
                q.append((nx, ny, nz))

        component_sizes.append(size)

    return labels, component_sizes


def component_pair_analysis(seg: np.ndarray, gt: np.ndarray) -> dict:
    seg_labels, seg_sizes = connected_components_26(seg)
    gt_labels, gt_sizes = connected_components_26(gt)

    n_seg = len(seg_sizes) - 1
    n_gt = len(gt_sizes) - 1

    overlap_mask = (seg_labels > 0) & (gt_labels > 0)
    if np.any(overlap_mask):
        seg_ids = seg_labels[overlap_mask].astype(np.int64)
        gt_ids = gt_labels[overlap_mask].astype(np.int64)
        packed = seg_ids * (n_gt + 1) + gt_ids
        uniq, counts = np.unique(packed, return_counts=True)
    else:
        uniq = np.array([], dtype=np.int64)
        counts = np.array([], dtype=np.int64)

    best = None
    for packed_id, inter in zip(uniq, counts):
        seg_id = int(packed_id // (n_gt + 1))
        gt_id = int(packed_id % (n_gt + 1))

        s = seg_sizes[seg_id]
        g = gt_sizes[gt_id]
        d = 2.0 * int(inter) / (s + g)
        j = int(inter) / (s + g - int(inter))

        if best is None or d > best["dice"]:
            best = {
                "seg_component_id": seg_id,
                "gt_component_id": gt_id,
                "seg_voxels": int(s),
                "gt_voxels": int(g),
                "intersection": int(inter),
                "dice": float(d),
                "jaccard": float(j),
            }

    return {
        "n_components_seg": int(n_seg),
        "n_components_gt": int(n_gt),
        "best_pair": best,
    }


def resumen_capas(rows: list[dict]) -> dict:
    dice_vals = np.array([r["dice"] for r in rows], dtype=float)
    empty_flags = np.array([r["both_empty"] for r in rows], dtype=int)

    non_empty = dice_vals[empty_flags == 0]

    summary = {
        "n_capas": int(len(rows)),
        "n_capas_ambas_vacias": int(empty_flags.sum()),
        "dice_mean_all": float(np.mean(dice_vals)) if len(dice_vals) else np.nan,
        "dice_median_all": float(np.median(dice_vals)) if len(dice_vals) else np.nan,
        "dice_min_all": float(np.min(dice_vals)) if len(dice_vals) else np.nan,
        "dice_max_all": float(np.max(dice_vals)) if len(dice_vals) else np.nan,
    }

    if len(non_empty):
        summary.update(
            {
                "dice_mean_non_empty": float(np.mean(non_empty)),
                "dice_median_non_empty": float(np.median(non_empty)),
                "dice_min_non_empty": float(np.min(non_empty)),
                "dice_max_non_empty": float(np.max(non_empty)),
            }
        )
    else:
        summary.update(
            {
                "dice_mean_non_empty": np.nan,
                "dice_median_non_empty": np.nan,
                "dice_min_non_empty": np.nan,
                "dice_max_non_empty": np.nan,
            }
        )

    return summary


def print_metrics(
    global_m: dict,
    align_info: dict,
    sum_z: dict,
    sum_x: dict,
    sum_y: dict,
    comp_info: dict,
    csv_path: Path,
) -> None:
    print("=== Comparación 3D completa (todas las capas) ===")
    print(f"Transformaciones probadas: {align_info['checked']}")
    print(f"Mejor permutación de ejes aplicada a SEG: {align_info['perm']}")
    print(f"Flips aplicados (ejes): {align_info['flips'] if align_info['flips'] else 'ninguno'}")
    print()

    print("Métricas globales de volumen:")
    print(f"  Dice               : {global_m['dice']:.6f}")
    print(f"  Jaccard (IoU)      : {global_m['jaccard']:.6f}")
    print(f"  Precision          : {global_m['precision']:.6f}")
    print(f"  Recall             : {global_m['recall']:.6f}")
    print(f"  Volume Similarity  : {global_m['volume_similarity']:.6f}")
    print(f"  Voxeles SEG        : {global_m['vox_seg']}")
    print(f"  Voxeles GT         : {global_m['vox_gt']}")
    print(f"  TP / FP / FN       : {global_m['tp']} / {global_m['fp']} / {global_m['fn']}")
    print()

    print("Resumen Dice por capas (eje Z axial):")
    print(f"  Capas totales      : {sum_z['n_capas']}")
    print(f"  Capas ambas vacías : {sum_z['n_capas_ambas_vacias']}")
    print(f"  Dice mean (all)    : {sum_z['dice_mean_all']:.6f}")
    print(f"  Dice mean (no vac) : {sum_z['dice_mean_non_empty']:.6f}")
    print()

    print("Resumen Dice por capas (eje X sagital):")
    print(f"  Dice mean (all)    : {sum_x['dice_mean_all']:.6f}")
    print(f"  Dice mean (no vac) : {sum_x['dice_mean_non_empty']:.6f}")
    print()

    print("Resumen Dice por capas (eje Y coronal):")
    print(f"  Dice mean (all)    : {sum_y['dice_mean_all']:.6f}")
    print(f"  Dice mean (no vac) : {sum_y['dice_mean_non_empty']:.6f}")
    print()

    print("Análisis por componentes 3D (diente/región):")
    print(f"  Componentes SEG    : {comp_info['n_components_seg']}")
    print(f"  Componentes GT     : {comp_info['n_components_gt']}")
    best_pair = comp_info["best_pair"]
    if best_pair is None:
        print("  Mejor par          : no hay intersección entre componentes")
    else:
        print(
            "  Mejor par          : "
            f"SEG#{best_pair['seg_component_id']} vs GT#{best_pair['gt_component_id']}"
        )
        print(f"  Dice mejor par     : {best_pair['dice']:.6f}")
        print(f"  Jaccard mejor par  : {best_pair['jaccard']:.6f}")
        print(
            "  Vox (SEG/GT/INT)   : "
            f"{best_pair['seg_voxels']} / {best_pair['gt_voxels']} / {best_pair['intersection']}"
        )
    print()

    print(f"CSV por capas guardado en: {csv_path.resolve()}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compara SEG NRRD de 3D Slicer contra GT NIfTI usando Dice y métricas 3D/por-capa"
    )
    parser.add_argument("--seg", type=str, default=DEFAULT_SEG_PATH, help="Ruta a Segmentation_1.seg.nrrd")
    parser.add_argument("--gt", type=str, default=DEFAULT_GT_PATH, help="Ruta a gt_pulp.nii.gz")
    parser.add_argument("--out-csv", type=str, default=DEFAULT_OUT_CSV, help="Salida CSV por capas")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Carpeta de salida para volúmenes NIfTI de comparación 3D",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    seg = load_segmentation_nrrd(args.seg)
    gt, gt_img = load_gt_nii_with_ref(args.gt)

    align = find_best_alignment(seg, gt)
    seg_aligned = align["mask"]

    global_m = metrics_globales(seg_aligned, gt)

    rows_z = dice_por_capa(seg_aligned, gt, axis=2)
    rows_x = dice_por_capa(seg_aligned, gt, axis=0)
    rows_y = dice_por_capa(seg_aligned, gt, axis=1)

    # Un único CSV con las tres orientaciones
    rows_all = rows_z + rows_x + rows_y
    out_csv = Path(args.out_csv)
    write_csv(rows_all, out_csv)

    out_dir = Path(args.out_dir)
    seg_out = out_dir / "seg_alineada_mask.nii.gz"
    gt_out = out_dir / "gt_mask.nii.gz"
    overlap_out = out_dir / "overlap_tp_fp_fn.nii.gz"

    overlap = build_overlap_map(seg_aligned, gt)
    save_mask_nifti(seg_aligned, gt_img, seg_out)
    save_mask_nifti(gt, gt_img, gt_out)
    save_mask_nifti(overlap, gt_img, overlap_out)

    sum_z = resumen_capas(rows_z)
    sum_x = resumen_capas(rows_x)
    sum_y = resumen_capas(rows_y)
    comp_info = component_pair_analysis(seg_aligned, gt)

    print_metrics(global_m, align, sum_z, sum_x, sum_y, comp_info, out_csv)
    print("\nVolúmenes NIfTI para ver en 3D Slicer:")
    print(f"  SEG alineada : {seg_out.resolve()}")
    print(f"  GT binaria   : {gt_out.resolve()}")
    print(f"  Overlap      : {overlap_out.resolve()}")
    print("\nLeyenda overlap_tp_fp_fn.nii.gz:")
    print("  0 = fondo")
    print("  1 = TP (coinciden SEG y GT)")
    print("  2 = FP (solo SEG)")
    print("  3 = FN (solo GT)")


if __name__ == "__main__":
    main()
