from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from canales_por_capa import detect_canal_region
from canales_finales import TARGETS, build_target_masks, load_patient_data, make_mask


OUTPUT_ROOT = Path("results") / "canales" / "hd95_canales_finales"
OUTPUT_EXCEL = OUTPUT_ROOT / "hd95_canales_global_summary.xlsx"
PATIENTS = ["P5"] + list(TARGETS.keys())


def dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    intersection = np.count_nonzero(mask_a & mask_b)
    total = np.count_nonzero(mask_a) + np.count_nonzero(mask_b)
    if total == 0:
        return float("nan")
    return 2.0 * intersection / total


def hd95_3d(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    if np.count_nonzero(mask_a) == 0 or np.count_nonzero(mask_b) == 0:
        return float("nan")

    def surface_voxels(mask: np.ndarray) -> np.ndarray:
        structure = ndimage.generate_binary_structure(mask.ndim, 1)
        return np.logical_xor(mask, ndimage.binary_erosion(mask, structure=structure, border_value=0))

    def directed_surface_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source_surface = surface_voxels(source)
        target_surface = surface_voxels(target)
        if not np.any(source_surface) or not np.any(target_surface):
            return np.array([], dtype=np.float32)
        distance_map = ndimage.distance_transform_edt(~target_surface)
        return distance_map[source_surface]

    distances = np.concatenate(
        [
            directed_surface_distances(mask_a, mask_b),
            directed_surface_distances(mask_b, mask_a),
        ]
    )
    if distances.size == 0:
        return float("nan")
    return float(np.percentile(distances, 95))


def build_side_mask(manual: np.ndarray, roles: dict[str, list[int]], side: str) -> np.ndarray:
    mask = make_mask(manual, roles[side])
    if roles.get("ian"):
        mask &= (make_mask(manual, roles["ian"]) == 0).astype(np.uint8)
    return mask


def build_patient_rows(patient_id: str) -> list[dict[str, object]]:
    data, instance, manual, label_map, roles, all_labels = load_patient_data(patient_id)
    rows: list[dict[str, object]] = []

    if patient_id == "P5":
        gt_mask = ((instance == 36) | (instance == 46)).astype(np.uint8)
        manual_mask = build_side_mask(manual, roles, "derecha") | build_side_mask(manual, roles, "izquierda")
        gt_root_region, info = detect_canal_region(gt_mask)
        keep_slices = np.asarray(info.get("keep_slices", []), dtype=np.int32)
        manual_root_region = np.zeros_like(manual_mask, dtype=np.uint8)
        if keep_slices.size > 0:
            manual_root_region[:, :, keep_slices] = manual_mask[:, :, keep_slices]

        rows.append(
            {
                "paciente": patient_id,
                "lado": "ambos",
                "diente_gt": "36+46",
                "modo": "dual_same",
                "manual_labels": ",".join(map(str, all_labels)),
                "manual_labels_usadas": ",".join(map(str, [int(v) for v in np.unique(manual_mask) if v > 0])),
                "gt_voxels_objetivo": int(np.count_nonzero(gt_mask)),
                "manual_voxels_objetivo": int(np.count_nonzero(manual_mask)),
                "gt_voxels_canales_finales": int(np.count_nonzero(gt_root_region)),
                "manual_voxels_canales_finales": int(np.count_nonzero(manual_root_region)),
                "dice_objetivo_completo": float(dice_score(gt_mask, manual_mask)),
                "dice_canales_finales": float(dice_score(gt_root_region, manual_root_region)),
                "hd95_canales_finales": float(hd95_3d(gt_root_region, manual_root_region)),
                "z_start_objetivo": int(info.get("z_start", -1)),
                "z_end_objetivo": int(info.get("z_end", -1)),
                "valid_slices": int(info.get("valid_slices", 0)),
                "region_slices": int(info.get("region_slices", 0)),
                "clusters_max": int(info.get("clusters_max", 0)),
            }
        )
        return rows

    spec = TARGETS.get(patient_id)
    if spec is None:
        return rows

    mode = spec["mode"]

    if mode == "solo_ian":
        rows.append(
            {
                "paciente": patient_id,
                "lado": "ian",
                "diente_gt": "-",
                "modo": mode,
                "manual_labels": ",".join(map(str, all_labels)),
                "manual_labels_usadas": "-",
                "gt_voxels_objetivo": 0,
                "manual_voxels_objetivo": int(np.count_nonzero(make_mask(manual, roles["ian"]))),
                "gt_voxels_canales_finales": 0,
                "manual_voxels_canales_finales": 0,
                "dice_objetivo_completo": float("nan"),
                "dice_canales_finales": float("nan"),
                "hd95_canales_finales": float("nan"),
                "z_start_objetivo": -1,
                "z_end_objetivo": -1,
                "valid_slices": 0,
                "region_slices": 0,
                "clusters_max": 0,
            }
        )
        return rows

    gt_target, manual_target, _ = build_target_masks(patient_id, instance, manual, roles)
    gt_root_region, info = detect_canal_region(gt_target)
    keep_slices = np.asarray(info.get("keep_slices", []), dtype=np.int32)
    manual_root_region = np.zeros_like(manual_target, dtype=np.uint8)
    if keep_slices.size > 0:
        manual_root_region[:, :, keep_slices] = manual_target[:, :, keep_slices]

    rows.append(
        {
            "paciente": patient_id,
            "lado": "derecha" if mode in {"manual_right_to_gt36", "manual_right_to_gt46"} else "izquierda",
            "diente_gt": 36 if mode in {"manual_right_to_gt36", "manual_left_to_gt36"} else 46,
            "modo": mode,
            "manual_labels": ",".join(map(str, all_labels)),
            "manual_labels_usadas": ",".join(map(str, [int(v) for v in np.unique(manual_target) if v > 0])),
            "gt_voxels_objetivo": int(np.count_nonzero(gt_target)),
            "manual_voxels_objetivo": int(np.count_nonzero(manual_target)),
            "gt_voxels_canales_finales": int(np.count_nonzero(gt_root_region)),
            "manual_voxels_canales_finales": int(np.count_nonzero(manual_root_region)),
            "dice_objetivo_completo": float(dice_score(gt_target, manual_target)),
            "dice_canales_finales": float(dice_score(gt_root_region, manual_root_region)),
            "hd95_canales_finales": float(hd95_3d(gt_root_region, manual_root_region)),
            "z_start_objetivo": int(info.get("z_start", -1)),
            "z_end_objetivo": int(info.get("z_end", -1)),
            "valid_slices": int(info.get("valid_slices", 0)),
            "region_slices": int(info.get("region_slices", 0)),
            "clusters_max": int(info.get("clusters_max", 0)),
        }
    )
    return rows


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []

    print("HD95 3D solo para canales finales")
    for patient_id in PATIENTS:
        try:
            rows = build_patient_rows(patient_id)
            if not rows:
                print(f"{patient_id} | sin dientes aplicables")
                continue
            all_rows.extend(rows)
            for row in rows:
                dice_text = "nan" if pd.isna(row["dice_canales_finales"]) else f"{row['dice_canales_finales']:.6f}"
                hd95_text = "nan" if pd.isna(row["hd95_canales_finales"]) else f"{row['hd95_canales_finales']:.6f}"
                print(f"{row['paciente']} | {row['lado']} | GT={row['diente_gt']} | DICE_canales={dice_text} | HD95_canales={hd95_text}")
        except Exception as error:
            all_rows.append(
                {
                    "paciente": patient_id,
                    "lado": "error",
                    "diente_gt": "-",
                    "modo": "error",
                    "manual_labels": "",
                    "manual_labels_usadas": "",
                    "gt_voxels_objetivo": 0,
                    "manual_voxels_objetivo": 0,
                    "gt_voxels_canales_finales": 0,
                    "manual_voxels_canales_finales": 0,
                    "dice_objetivo_completo": float("nan"),
                    "dice_canales_finales": float("nan"),
                    "hd95_canales_finales": float("nan"),
                    "z_start_objetivo": -1,
                    "z_end_objetivo": -1,
                    "valid_slices": 0,
                    "region_slices": 0,
                    "clusters_max": 0,
                    "error": str(error),
                }
            )
            print(f"{patient_id} | error | {error}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_ROOT / "hd95_canales_global_summary.csv", index=False, sep=";", decimal=",")
    df.to_excel(OUTPUT_EXCEL, index=False)
    float_formatter = lambda value: "NaN" if pd.isna(value) else f"{value:.6f}".replace(".", ",")
    (OUTPUT_ROOT / "hd95_canales_global_summary.txt").write_text(
        df.to_string(
            index=False,
            formatters={
                "dice_objetivo_completo": float_formatter,
                "dice_canales_finales": float_formatter,
                "hd95_canales_finales": float_formatter,
            },
        ),
        encoding="utf-8",
    )
    print(f"Saved: {OUTPUT_ROOT / 'hd95_canales_global_summary.csv'}")


if __name__ == "__main__":
    main()