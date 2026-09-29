from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import nrrd
from scipy import ndimage

from canales_por_capa import TARGETS, PATIENT_SIDE_OVERRIDES, SIDE_CONFIGS, load_patient_data, make_mask


OUTPUT_DIR = Path("results") / "canales" / "hd95_diente_completo"
PATIENTS = ["P5"] + list(TARGETS.keys())
MANUAL_BASE_DIR = Path(
    r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\2_UNet\datasets\Segmentaciones 3DSlicer"
)


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
        rows.append(
            {
                "paciente": patient_id,
                "lado": "ambos",
                "diente_gt": "36+46",
                "modo": "dual_same",
                "manual_labels": ",".join(map(str, all_labels)),
                "manual_labels_usadas": ",".join(map(str, [int(v) for v in np.unique(manual_mask) if v > 0])),
                "gt_voxels": int(np.count_nonzero(gt_mask)),
                "manual_voxels": int(np.count_nonzero(manual_mask)),
                "dice_3d": float(dice_score(gt_mask, manual_mask)),
                "hd95_3d": float(hd95_3d(gt_mask, manual_mask)),
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
                "gt_voxels": 0,
                "manual_voxels": int(np.count_nonzero(make_mask(manual, roles["ian"]))),
                "dice_3d": float("nan"),
                "hd95_3d": float("nan"),
            }
        )
        return rows

    if mode == "comun":
        for side in ("derecha", "izquierda"):
            gt_label = PATIENT_SIDE_OVERRIDES.get(patient_id, {}).get(side, SIDE_CONFIGS[side]["gt_label"])
            gt_mask = (instance == gt_label).astype(np.uint8)
            manual_mask = build_side_mask(manual, roles, side)
            rows.append(
                {
                    "paciente": patient_id,
                    "lado": side,
                    "diente_gt": gt_label,
                    "modo": mode,
                    "manual_labels": ",".join(map(str, all_labels)),
                    "manual_labels_usadas": ",".join(map(str, [int(v) for v in np.unique(manual_mask) if v > 0])),
                    "gt_voxels": int(np.count_nonzero(gt_mask)),
                    "manual_voxels": int(np.count_nonzero(manual_mask)),
                    "dice_3d": float(dice_score(gt_mask, manual_mask)),
                    "hd95_3d": float(hd95_3d(gt_mask, manual_mask)),
                }
            )
        return rows

    if mode == "manual_right_to_gt46":
        gt_label = 46
        side = "derecha"
    elif mode == "manual_right_to_gt36":
        gt_label = 36
        side = "derecha"
    elif mode == "manual_left_to_gt36":
        gt_label = 36
        side = "izquierda"
    elif mode == "manual_left_to_gt46":
        gt_label = 46
        side = "izquierda"
    else:
        return rows

    gt_mask = (instance == gt_label).astype(np.uint8)
    manual_mask = build_side_mask(manual, roles, side)
    rows.append(
        {
            "paciente": patient_id,
            "lado": side,
            "diente_gt": gt_label,
            "modo": mode,
            "manual_labels": ",".join(map(str, all_labels)),
            "manual_labels_usadas": ",".join(map(str, [int(v) for v in np.unique(manual_mask) if v > 0])),
            "gt_voxels": int(np.count_nonzero(gt_mask)),
            "manual_voxels": int(np.count_nonzero(manual_mask)),
            "dice_3d": float(dice_score(gt_mask, manual_mask)),
            "hd95_3d": float(hd95_3d(gt_mask, manual_mask)),
        }
    )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []

    print("HD95 3D global por paciente")
    for patient_id in PATIENTS:
        try:
            rows = build_patient_rows(patient_id)
            if not rows:
                print(f"{patient_id} | sin dientes aplicables")
                continue
            all_rows.extend(rows)
            for row in rows:
                dice_value = row["dice_3d"]
                hd95_value = row["hd95_3d"]
                dice_text = "nan" if np.isnan(dice_value) else f"{dice_value:.6f}"
                hd95_text = "nan" if np.isnan(hd95_value) else f"{hd95_value:.6f}"
                print(f"{row['paciente']} | {row['lado']} | GT={row['diente_gt']} | DICE={dice_text} | HD95_3D={hd95_text}")
        except Exception as error:
            all_rows.append(
                {
                    "paciente": patient_id,
                    "lado": "error",
                    "diente_gt": "-",
                    "modo": "error",
                    "manual_labels": "",
                    "manual_labels_usadas": "",
                    "gt_voxels": 0,
                    "manual_voxels": 0,
                    "dice_3d": float("nan"),
                    "hd95_3d": float("nan"),
                    "error": str(error),
                }
            )
            print(f"{patient_id} | error | {error}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_DIR / "hd95_global_summary.csv", index=False, sep=";", decimal=",")
    df.to_excel(OUTPUT_DIR / "hd95_global_summary.xlsx", index=False)
    float_formatter = lambda value: "NaN" if pd.isna(value) else f"{value:.6f}".replace(".", ",")
    (OUTPUT_DIR / "hd95_global_summary.txt").write_text(
        df.to_string(
            index=False,
            formatters={
                "dice_3d": float_formatter,
                "hd95_3d": float_formatter,
            },
        ),
        encoding="utf-8",
    )
    print(f"Saved: {OUTPUT_DIR / 'hd95_global_summary.csv'}")
if __name__ == "__main__":
    main()
