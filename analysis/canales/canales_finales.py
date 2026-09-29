from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import nrrd
import pandas as pd
from scipy import ndimage


BASE_DIR = Path(
    r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\Pulpy3D"
)
MANUAL_DIR = Path(
    r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\2_UNet\datasets\Segmentaciones 3DSlicer"
)
OUTPUT_EXCEL = Path("results") / "canales" / "canales_finales" / "dice_canales_finales_primera_hoja.xlsx"
OUTPUT_ROOT = Path("results") / "canales" / "canales_finales"

GT_RIGHT_LABEL = 36
GT_LEFT_LABEL = 46
IAN_TOKENS = ("ian", "nerv")
SIDE_TOKENS = {
    "derecha": ("der", "right", "36"),
    "izquierda": ("izq", "left", "46"),
}


TARGETS = {
    "P35": {"mode": "manual_left_to_gt46", "description": "Solo izq (equiv a 46 en GT)"},
    "P48": {"mode": "manual_left_to_gt46", "description": "Solo izq (equiv a 46 en GT)"},
    "P59": {"mode": "solo_ian", "description": "Solo ian"},
    "P61": {"mode": "manual_left_to_gt46", "description": "Solo izq (equiv a 46 en GT)"},
    "P81": {"mode": "manual_right_to_gt36", "description": "Solo der (equiv 36 en GT)"},
    "P100": {"mode": "manual_right_to_gt36", "description": "Solo der (equiv 36 en GT)"},
    "P111": {"mode": "manual_left_to_gt46", "description": "Solo izq (equiv a 46 en GT)"},
    "P133": {"mode": "manual_right_to_gt36", "description": "Solo der (equiv 36 en GT)"},
    "P137": {"mode": "manual_left_to_gt36", "description": "Solo izq (equiv 36 en GT)"},
    "P146": {"mode": "solo_ian", "description": "Solo ian"},
    "P191": {"mode": "manual_right_to_gt46", "description": "Solo der (equiv 46 en GT)"},
    "P195": {"mode": "manual_right_to_gt36", "description": "Solo der (equiv 36 en GT)"},
    "P208": {"mode": "manual_right_to_gt36", "description": "Solo der (equiv 36 en GT)"},
    "P217": {"mode": "comun", "description": "36 y 46 diferentes pixeles"},
    "P222": {"mode": "manual_left_to_gt46", "description": "Solo izq (equiv a 46 en GT)"},
    "P249": {"mode": "manual_right_to_gt46", "description": "Solo der (equiv 46 en GT)"},
    "P254": {"mode": "manual_right_to_gt46", "description": "Solo der (equiv 46 en GT)"},
    "P261": {"mode": "manual_right_to_gt36", "description": "Solo der (equiv 36 en GT)"},
    "P343": {"mode": "comun", "description": "36 y 46 diferentes pixeles"},
    "P380": {"mode": "manual_right_to_gt36", "description": "Solo der (equiv 36 en GT)"},
    "P381": {"mode": "manual_right_to_gt36", "description": "Solo der (equiv 36 en GT)"},
    "P394": {"mode": "manual_left_to_gt46", "description": "Solo izq (equiv a 46 en GT)"},
    "P402": {"mode": "manual_left_to_gt46", "description": "Solo izq (equiv a 46 en GT)"},
    "P422": {"mode": "manual_left_to_gt46", "description": "Solo izq (equiv a 46 en GT)"},
    "P447": {"mode": "solo_ian", "description": "Solo ian"},
    "P448": {"mode": "manual_right_to_gt36", "description": "Solo der (equiv 36 en GT)"},
    "P466": {"mode": "manual_right_to_gt36", "description": "Solo der (equiv 36 en GT)"},
    "P476": {"mode": "solo_ian", "description": "Solo ian"},
    "P511": {"mode": "manual_left_to_gt46", "description": "Solo izq (equiv a 46 en GT)"},
    "P534": {"mode": "manual_left_to_gt46", "description": "Solo izq (equiv a 46 en GT)"},
    "P546": {"mode": "manual_right_to_gt36", "description": "Solo der (equiv 36 en GT)"},
}


def dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    intersection = np.count_nonzero(mask_a & mask_b)
    total = np.count_nonzero(mask_a) + np.count_nonzero(mask_b)
    if total == 0:
        return float("nan")
    return 2.0 * intersection / total


def align_shape_with_permutations(seg_array: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray | None:
    if seg_array.shape == target_shape:
        return seg_array

    for perm in itertools.permutations(range(3)):
        candidate = np.transpose(seg_array, perm)
        if candidate.shape == target_shape:
            return candidate

    return None


def extract_label_map(header: dict) -> dict[int, str]:
    label_map: dict[int, str] = {}
    for key, value in header.items():
        if not key.endswith("_Name"):
            continue
        segment_idx = key[:-5]
        label_key = f"{segment_idx}_LabelValue"
        if label_key not in header:
            continue
        try:
            label_value = int(header[label_key])
        except Exception:
            continue
        label_map[label_value] = str(value)
    return label_map


def infer_side_from_name(name: str) -> str | None:
    name_low = name.lower()
    for side, tokens in SIDE_TOKENS.items():
        if any(token in name_low for token in tokens):
            return side
    return None


def classify_manual_labels(seg: np.ndarray, label_map: dict[int, str]) -> tuple[dict[str, list[int]], list[int]]:
    labels = [int(v) for v in np.unique(seg) if v > 0]
    roles = {"derecha": [], "izquierda": [], "ian": [], "otros": []}

    for label_value in labels:
        name = label_map.get(label_value, "")
        name_low = name.lower()
        if any(token in name_low for token in IAN_TOKENS):
            roles["ian"].append(label_value)
            continue
        side = infer_side_from_name(name)
        if side is not None:
            roles[side].append(label_value)
        else:
            roles["otros"].append(label_value)

    if not roles["derecha"] and not roles["izquierda"]:
        non_ian = [label for label in labels if label not in roles["ian"]]
        if len(non_ian) >= 2:
            roles["derecha"] = [non_ian[0]]
            roles["izquierda"] = [non_ian[1]]
        elif len(non_ian) == 1:
            roles["derecha"] = [non_ian[0]]

    return roles, labels


def make_mask(seg: np.ndarray, labels: list[int]) -> np.ndarray:
    mask = np.zeros_like(seg, dtype=np.uint8)
    for label_value in labels:
        mask |= (seg == label_value).astype(np.uint8)
    return mask


def count_clusters_2d(slice_mask: np.ndarray, min_area: int = 8) -> int:
    if np.count_nonzero(slice_mask) == 0:
        return 0
    structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    cc, n_comp = ndimage.label(slice_mask.astype(np.uint8), structure=structure)
    if n_comp == 0:
        return 0
    sizes = ndimage.sum(slice_mask, labels=cc, index=np.arange(1, n_comp + 1))
    return int(np.count_nonzero(np.asarray(sizes) >= min_area))


def detect_root_final_region(gt_mask: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    valid_slices = np.flatnonzero(np.count_nonzero(gt_mask, axis=(0, 1)) > 0)
    if valid_slices.size == 0:
        return np.zeros_like(gt_mask, dtype=np.uint8), {
            "z_start": -1,
            "z_end": -1,
            "split_z": -1,
            "root_side": "none",
            "valid_slices": 0,
            "region_slices": 0,
            "clusters_max": 0,
            "keep_slices": [],
            "clusters_per_z": [],
        }

    clusters_per_z = np.zeros(gt_mask.shape[2], dtype=np.int32)
    for z in valid_slices:
        clusters_per_z[z] = count_clusters_2d(gt_mask[:, :, z])

    valid_list = valid_slices.tolist()
    q = max(1, len(valid_list) // 4)
    low_mean = float(np.mean([clusters_per_z[z] for z in valid_list[:q]]))
    high_mean = float(np.mean([clusters_per_z[z] for z in valid_list[-q:]]))
    root_on_high = high_mean >= low_mean

    ordered = valid_list if root_on_high else list(reversed(valid_list))
    split_z = None
    for idx, z in enumerate(ordered):
        window = ordered[idx : idx + 3]
        if sum(clusters_per_z[w] >= 3 for w in window) >= 2:
            split_z = z
            break

    if split_z is None:
        split_z = max(ordered, key=lambda zi: clusters_per_z[zi])

    if root_on_high:
        keep_slices = [z for z in valid_list if z >= split_z]
        root_side = "high_z"
    else:
        keep_slices = [z for z in valid_list if z <= split_z]
        root_side = "low_z"

    region = np.zeros_like(gt_mask, dtype=np.uint8)
    region[:, :, keep_slices] = gt_mask[:, :, keep_slices]

    info: dict[str, object] = {
        "z_start": int(valid_list[0]),
        "z_end": int(valid_list[-1]),
        "split_z": int(split_z),
        "root_side": root_side,
        "valid_slices": int(len(valid_list)),
        "region_slices": int(len(keep_slices)),
        "clusters_max": int(np.max(clusters_per_z[valid_slices])),
        "keep_slices": keep_slices,
        "clusters_per_z": clusters_per_z,
    }
    return region, info


def load_patient_data(patient_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str], dict[str, list[int]], list[int]]:
    instance_path = BASE_DIR / patient_id / "gt_instance.nii.gz"
    data_path = BASE_DIR / patient_id / "data.nii.gz"
    manual_dir = MANUAL_DIR / patient_id
    seg_files = sorted(manual_dir.glob("*.seg.nrrd"))
    if not instance_path.exists():
        raise FileNotFoundError(f"Missing gt_instance for {patient_id}")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data.nii.gz for {patient_id}")
    if not seg_files:
        raise FileNotFoundError(f"Missing .seg.nrrd for {patient_id}")

    data = nib.load(str(data_path)).get_fdata()
    instance = np.rint(nib.load(str(instance_path)).get_fdata()).astype(np.int32)
    seg_data, header = nrrd.read(str(seg_files[0]))
    seg_rotated = np.flip(np.flip(np.flip(seg_data, axis=1), axis=0), axis=0)
    seg_aligned = align_shape_with_permutations(seg_rotated, instance.shape)
    if seg_aligned is None:
        raise ValueError(f"Incompatible shapes for {patient_id}: {seg_rotated.shape} vs {instance.shape}")

    seg = np.rint(seg_aligned).astype(np.int32)
    label_map = extract_label_map(header)
    roles, all_labels = classify_manual_labels(seg, label_map)
    return data, instance, seg, label_map, roles, all_labels


def compute_slice_dice_table(gt_target: np.ndarray, manual_target: np.ndarray, keep_slices: list[int], clusters_per_z: np.ndarray) -> pd.DataFrame:
    gt_counts = np.count_nonzero(gt_target, axis=(0, 1))
    manual_counts = np.count_nonzero(manual_target, axis=(0, 1))
    valid_slices = np.flatnonzero((gt_counts > 0) | (manual_counts > 0))
    keep_set = set(keep_slices)

    if valid_slices.size == 0:
        return pd.DataFrame(columns=["z", "dice_slice", "gt_voxels", "manual_voxels", "intersection_px", "clusters_gt", "en_region_canales_finales"])

    start_z = int(valid_slices[0])
    end_z = int(valid_slices[-1])
    for z in range(end_z + 1, gt_target.shape[2]):
        if gt_counts[z] == 0 and manual_counts[z] == 0:
            end_z = int(z)
            break

    rows: list[dict[str, object]] = []
    for z in range(start_z, end_z + 1):
        gt_slice = gt_target[:, :, z]
        manual_slice = manual_target[:, :, z]
        intersection_px = int(np.count_nonzero(gt_slice & manual_slice))
        total_px = int(gt_counts[z] + manual_counts[z])
        rows.append(
            {
                "z": int(z),
                "dice_slice": float(0.0 if total_px == 0 else (2.0 * intersection_px / total_px)),
                "gt_voxels": int(gt_counts[z]),
                "manual_voxels": int(manual_counts[z]),
                "intersection_px": intersection_px,
                "clusters_gt": int(clusters_per_z[z]),
                "en_region_canales_finales": int(z in keep_set),
            }
        )
    return pd.DataFrame(rows)


def create_overlap_graph(df: pd.DataFrame, patient_id: str, results_subdir: Path) -> Path:
    """Create a per-slice overlap graph matching the IAN analysis style."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"{patient_id} - Analisis por capas (canales)", fontsize=14, fontweight="bold")

    axes[0].plot(df["z"], df["gt_voxels"], label="Pulpy3D", color="red", linewidth=2)
    axes[0].plot(df["z"], df["manual_voxels"], label="Manual", color="blue", linewidth=2)
    axes[0].plot(df["z"], df["intersection_px"], label="Coincidencia", color="magenta", linewidth=2.5, linestyle="--")
    axes[0].set_xlabel("Corte Z (pixeles)", fontweight="bold")
    axes[0].set_ylabel("Cantidad de pixeles", fontweight="bold")
    axes[0].set_title("Cantidad de pixeles por capa")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(df["z"], df["dice_slice"], color="green", alpha=0.7, label="DICE")
    dice_mean = float(df["dice_slice"].mean()) if not df.empty else 0.0
    axes[1].axhline(y=dice_mean, color="darkgreen", linestyle="--", label=f"DICE promedio: {dice_mean:.3f}")
    axes[1].set_xlabel("Corte Z (pixeles)", fontweight="bold")
    axes[1].set_ylabel("Coeficiente DICE", fontweight="bold")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("DICE por capa")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    filename = results_subdir / f"{patient_id}_overlap_graph.png"
    fig.savefig(filename, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return filename


def choose_count_boundaries(info: dict[str, object]) -> tuple[int, int]:
    keep_slices = [int(z) for z in info.get("keep_slices", [])]
    split_z = int(info.get("split_z", -1))
    root_side = str(info.get("root_side", "none"))
    if not keep_slices:
        return -1, -1

    if root_side == "low_z":
        start_count = split_z
        end_count = min(keep_slices)
    elif root_side == "high_z":
        start_count = split_z
        end_count = max(keep_slices)
    else:
        start_count = keep_slices[0]
        end_count = keep_slices[-1]
    return int(start_count), int(end_count)


def save_overlay_png(data: np.ndarray, gt_mask: np.ndarray, manual_mask: np.ndarray, z: int, out_path: Path, title: str) -> None:
    if z < 0 or z >= data.shape[2]:
        return

    base = data[:, :, z]
    gt_slice = gt_mask[:, :, z]
    manual_slice = manual_mask[:, :, z]

    vmin, vmax = np.percentile(base, [1, 99])
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(base.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    ax.imshow(np.ma.masked_where(gt_slice.T == 0, gt_slice.T), cmap="Blues", alpha=0.55, origin="lower")
    ax.imshow(np.ma.masked_where(manual_slice.T == 0, manual_slice.T), cmap="autumn", alpha=0.55, origin="lower")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_patient_outputs(
    patient_id: str,
    data: np.ndarray,
    gt_target: np.ndarray,
    manual_target: np.ndarray,
    info: dict[str, object],
    summary: dict[str, object],
) -> None:
    patient_out = OUTPUT_ROOT / patient_id
    patient_out.mkdir(parents=True, exist_ok=True)

    keep_slices = [int(z) for z in info.get("keep_slices", [])]
    clusters_per_z = info.get("clusters_per_z")
    if isinstance(clusters_per_z, list):
        clusters_array = np.asarray(clusters_per_z, dtype=np.int32)
    else:
        clusters_array = np.asarray(clusters_per_z, dtype=np.int32)

    per_slice_df = compute_slice_dice_table(gt_target, manual_target, keep_slices, clusters_array)
    per_slice_df.to_csv(patient_out / "dice_por_capa.csv", index=False)
    create_overlap_graph(per_slice_df, patient_id, patient_out)

    pd.DataFrame([summary]).to_csv(patient_out / "resumen_metricas.csv", index=False)

    start_count, end_count = choose_count_boundaries(info)
    split_z = int(info.get("split_z", -1))

    save_overlay_png(
        data=data,
        gt_mask=gt_target,
        manual_mask=manual_target,
        z=split_z,
        out_path=patient_out / "inicio_conteo_canales.png",
        title=f"{patient_id} | inicio conteo canales | z={split_z}",
    )
    save_overlay_png(
        data=data,
        gt_mask=gt_target,
        manual_mask=manual_target,
        z=end_count,
        out_path=patient_out / "fin_conteo_canales.png",
        title=f"{patient_id} | fin conteo canales | z={end_count}",
    )

    report_lines = [
        f"paciente: {patient_id}",
        f"dice_total_objetivo: {summary['dice_objetivo_completo']}",
        f"dice_canales_finales: {summary['dice_canales_finales']}",
        f"z_inicio_objetivo: {summary['z_start_objetivo']}",
        f"z_fin_objetivo: {summary['z_end_objetivo']}",
        f"z_inicio_conteo(split_3_clusters): {split_z}",
        f"z_fin_conteo: {end_count}",
        f"root_side: {summary['root_side']}",
        f"region_slices: {summary['region_slices']}",
    ]
    (patient_out / "README_resumen.txt").write_text("\n".join(report_lines), encoding="utf-8")


def build_target_masks(patient_id: str, instance: np.ndarray, seg: np.ndarray, roles: dict[str, list[int]]) -> tuple[np.ndarray, np.ndarray, str]:
    gt_right = (instance == GT_RIGHT_LABEL).astype(np.uint8)
    gt_left = (instance == GT_LEFT_LABEL).astype(np.uint8)

    manual_ian = make_mask(seg, roles["ian"])
    manual_right = make_mask(seg, roles["derecha"])
    manual_left = make_mask(seg, roles["izquierda"])

    target_mode = TARGETS[patient_id]["mode"]
    if target_mode == "comun":
        gt_target = (gt_right | gt_left).astype(np.uint8)
        manual_target = (make_mask(seg, roles["derecha"] + roles["izquierda"] + roles["otros"]) & (manual_ian == 0)).astype(np.uint8)
        mode = "comun"
    elif target_mode == "manual_left_to_gt46":
        gt_target = gt_left.astype(np.uint8)
        if np.count_nonzero(manual_left) == 0:
            fallback = roles["otros"][:1] or roles["derecha"][:1]
            manual_left = make_mask(seg, fallback)
        manual_target = (manual_left & (manual_ian == 0)).astype(np.uint8)
        mode = "manual_left_to_gt46"
    elif target_mode == "manual_right_to_gt36":
        gt_target = gt_right.astype(np.uint8)
        if np.count_nonzero(manual_right) == 0:
            fallback = roles["otros"][:1] or roles["izquierda"][:1]
            manual_right = make_mask(seg, fallback)
        manual_target = (manual_right & (manual_ian == 0)).astype(np.uint8)
        mode = "manual_right_to_gt36"
    elif target_mode == "manual_left_to_gt36":
        gt_target = gt_right.astype(np.uint8)
        if np.count_nonzero(manual_left) == 0:
            fallback = roles["otros"][:1] or roles["derecha"][:1]
            manual_left = make_mask(seg, fallback)
        manual_target = (manual_left & (manual_ian == 0)).astype(np.uint8)
        mode = "manual_left_to_gt36"
    elif target_mode == "manual_right_to_gt46":
        gt_target = gt_left.astype(np.uint8)
        if np.count_nonzero(manual_right) == 0:
            fallback = roles["otros"][:1] or roles["izquierda"][:1]
            manual_right = make_mask(seg, fallback)
        manual_target = (manual_right & (manual_ian == 0)).astype(np.uint8)
        mode = "manual_right_to_gt46"
    elif target_mode == "solo_ian":
        gt_target = np.zeros_like(gt_right, dtype=np.uint8)
        manual_target = np.zeros_like(gt_right, dtype=np.uint8)
        mode = "solo_ian"
    else:
        raise ValueError(f"Unsupported mode for {patient_id}")

    return gt_target, manual_target, mode


def process_patient(patient_id: str) -> dict[str, object]:
    data, instance, seg, label_map, roles, all_labels = load_patient_data(patient_id)
    gt_target, manual_target, mode = build_target_masks(patient_id, instance, seg, roles)

    if mode == "solo_ian":
        summary = {
            "paciente": patient_id,
            "estado": "solo_ian_sin_diente",
            "descripcion_objetivo": TARGETS[patient_id]["description"],
            "modo": mode,
            "manual_labels": ",".join(map(str, all_labels)),
            "label_names": " | ".join(f"{k}:{v}" for k, v in sorted(label_map.items())),
            "labels_derecha": ",".join(map(str, roles["derecha"])),
            "labels_izquierda": ",".join(map(str, roles["izquierda"])),
            "labels_ian": ",".join(map(str, roles["ian"])),
            "labels_otros": ",".join(map(str, roles["otros"])),
            "gt_voxels_objetivo": 0,
            "manual_voxels_objetivo": 0,
            "dice_objetivo_completo": float("nan"),
            "gt_voxels_canales_finales": 0,
            "manual_voxels_canales_finales": 0,
            "dice_canales_finales": float("nan"),
            "z_start_objetivo": -1,
            "z_end_objetivo": -1,
            "split_z_3_clusters": -1,
            "root_side": "none",
            "valid_slices": 0,
            "region_slices": 0,
            "clusters_max": 0,
        }
        export_patient_outputs(
            patient_id=patient_id,
            data=data,
            gt_target=gt_target,
            manual_target=manual_target,
            info={"keep_slices": [], "clusters_per_z": np.zeros(gt_target.shape[2], dtype=np.int32), "split_z": -1, "root_side": "none"},
            summary=summary,
        )
        return summary

    gt_root_region, info = detect_root_final_region(gt_target)
    keep_slices = np.asarray(info.get("keep_slices", []), dtype=np.int32)
    manual_root_region = np.zeros_like(manual_target, dtype=np.uint8)
    if keep_slices.size > 0:
        manual_root_region[:, :, keep_slices] = manual_target[:, :, keep_slices]

    dice_global_target = dice_score(manual_target, gt_target)
    dice_root_final = dice_score(manual_root_region, gt_root_region)

    summary = {
        "paciente": patient_id,
        "estado": "ok",
        "descripcion_objetivo": TARGETS[patient_id]["description"],
        "modo": mode,
        "manual_labels": ",".join(map(str, all_labels)),
        "label_names": " | ".join(f"{k}:{v}" for k, v in sorted(label_map.items())),
        "labels_derecha": ",".join(map(str, roles["derecha"])),
        "labels_izquierda": ",".join(map(str, roles["izquierda"])),
        "labels_ian": ",".join(map(str, roles["ian"])),
        "labels_otros": ",".join(map(str, roles["otros"])),
        "gt_voxels_objetivo": int(np.count_nonzero(gt_target)),
        "manual_voxels_objetivo": int(np.count_nonzero(manual_target)),
        "dice_objetivo_completo": dice_global_target,
        "gt_voxels_canales_finales": int(np.count_nonzero(gt_root_region)),
        "manual_voxels_canales_finales": int(np.count_nonzero(manual_root_region)),
        "dice_canales_finales": dice_root_final,
        "z_start_objetivo": info["z_start"],
        "z_end_objetivo": info["z_end"],
        "split_z_3_clusters": info["split_z"],
        "root_side": info["root_side"],
        "valid_slices": info["valid_slices"],
        "region_slices": info["region_slices"],
        "clusters_max": info["clusters_max"],
    }
    export_patient_outputs(
        patient_id=patient_id,
        data=data,
        gt_target=gt_target,
        manual_target=manual_target,
        info=info,
        summary=summary,
    )
    return summary


def main() -> None:
    rows: list[dict[str, object]] = []
    for patient_id in TARGETS:
        try:
            rows.append(process_patient(patient_id))
        except Exception as error:
            rows.append({"paciente": patient_id, "estado": f"error: {error}"})

    df = pd.DataFrame(rows)
    OUTPUT_EXCEL.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_EXCEL, index=False)

    print(f"Saved: {OUTPUT_EXCEL}")
    show_cols = [
        "paciente",
        "estado",
        "descripcion_objetivo",
        "modo",
        "dice_objetivo_completo",
        "dice_canales_finales",
        "split_z_3_clusters",
        "root_side",
        "region_slices",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
