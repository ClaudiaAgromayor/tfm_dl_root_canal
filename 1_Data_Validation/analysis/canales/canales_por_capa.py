from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from canales_finales import (
    build_target_masks,
    create_overlap_graph,
    count_clusters_2d,
    dice_score,
    make_mask,
    load_patient_data,
    TARGETS,
)


OUTPUT_ROOT = Path("results") / "canales" / "canales_por_capa"
OUTPUT_EXCEL = OUTPUT_ROOT / "resumen_canales_pacientes.xlsx"
TARGET_PATIENTS = list(TARGETS.keys())
GENERATE_PRESTART_IMAGES = False
GENERATE_CONTEO_IMAGES = False


SIDE_CONFIGS = {
    "derecha": {"gt_label": 36, "manual_role": "derecha", "descripcion_suffix": "(derecha)"},
    "izquierda": {"gt_label": 46, "manual_role": "izquierda", "descripcion_suffix": "(izquierda)"},
}

PATIENT_SIDE_OVERRIDES = {
    "P217": {
        "derecha": 46,
        "izquierda": 36,
    }
}


def detect_canal_region(gt_mask: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    valid_slices = np.flatnonzero(np.count_nonzero(gt_mask, axis=(0, 1)) > 0)
    if valid_slices.size == 0:
        return np.zeros_like(gt_mask, dtype=np.uint8), {
            "z_start": -1,
            "z_end": -1,
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
    start_z = None
    for idx in range(len(valid_list) - 2):
        window = valid_list[idx : idx + 3]
        if sum(clusters_per_z[z] >= 3 for z in window) >= 2 and (idx == 0 or clusters_per_z[valid_list[idx - 1]] < 3):
            start_z = int(next(z for z in window if clusters_per_z[z] >= 3))
            break

    if start_z is None:
        start_z = int(max(valid_list, key=lambda z: clusters_per_z[z]))

    end_z = int(valid_list[-1])
    keep_slices = [z for z in valid_list if z >= start_z]

    region = np.zeros_like(gt_mask, dtype=np.uint8)
    region[:, :, keep_slices] = gt_mask[:, :, keep_slices]

    info: dict[str, object] = {
        "z_start": start_z,
        "z_end": end_z,
        "valid_slices": int(len(valid_list)),
        "region_slices": int(len(keep_slices)),
        "clusters_max": int(np.max(clusters_per_z[valid_slices])),
        "keep_slices": keep_slices,
        "clusters_per_z": clusters_per_z,
    }
    return region, info


def compute_slice_dice_table(
    gt_target: np.ndarray,
    manual_target: np.ndarray,
    keep_slices: list[int],
    clusters_per_z: np.ndarray,
) -> pd.DataFrame:
    gt_counts = np.count_nonzero(gt_target, axis=(0, 1))
    manual_counts = np.count_nonzero(manual_target, axis=(0, 1))
    valid_slices = np.flatnonzero((gt_counts > 0) | (manual_counts > 0))
    keep_set = set(keep_slices)

    if valid_slices.size == 0:
        return pd.DataFrame(columns=["z", "dice_slice", "gt_voxels", "manual_voxels", "intersection_px", "clusters_gt", "en_region_canales"])

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
                "en_region_canales": int(z in keep_set),
            }
        )
    return pd.DataFrame(rows)


def build_side_target_masks(
    instance: np.ndarray,
    seg: np.ndarray,
    roles: dict[str, list[int]],
    side: str,
    gt_label: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    config = SIDE_CONFIGS[side]
    label_value = config["gt_label"] if gt_label is None else gt_label
    gt_target = (instance == label_value).astype(np.uint8)
    manual_side = make_mask(seg, roles[config["manual_role"]])
    manual_ian = make_mask(seg, roles["ian"])
    manual_target = (manual_side & (manual_ian == 0)).astype(np.uint8)
    return gt_target, manual_target, side


def save_overlay_png(
    data: np.ndarray,
    gt_mask: np.ndarray,
    manual_mask: np.ndarray,
    z: int,
    out_path: Path,
    title: str,
) -> None:
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
    clusters_per_z = np.asarray(info.get("clusters_per_z", []), dtype=np.int32)

    per_slice_df = compute_slice_dice_table(gt_target, manual_target, keep_slices, clusters_per_z)
    per_slice_df.to_csv(patient_out / "dice_por_capa.csv", index=False)
    create_overlap_graph(per_slice_df, patient_id, patient_out)

    start_z = int(info.get("z_start", -1))
    end_z = int(info.get("z_end", -1))
    fin_z = int(per_slice_df["z"].iloc[-1]) if not per_slice_df.empty else end_z

    if GENERATE_CONTEO_IMAGES:
        save_overlay_png(
            data=data,
            gt_mask=gt_target,
            manual_mask=manual_target,
            z=start_z,
            out_path=patient_out / "inicio_conteo_canales.png",
            title=f"{patient_id} | inicio canales | z={start_z}",
        )

    save_overlay_png(
        data=data,
        gt_mask=gt_target,
        manual_mask=manual_target,
        z=fin_z,
        out_path=patient_out / "fin_conteo_canales.png",
        title=f"{patient_id} | fin canales | z={fin_z}",
    )

    if GENERATE_PRESTART_IMAGES:
        prev_dir = patient_out / "capas_antes_inicio"
        prev_dir.mkdir(parents=True, exist_ok=True)
        for z in range(max(0, start_z - 10), start_z):
            save_overlay_png(
                data=data,
                gt_mask=gt_target,
                manual_mask=manual_target,
                z=z,
                out_path=prev_dir / f"capa_{z:03d}.png",
                title=f"{patient_id} | capa previa al inicio | z={z}",
            )

    report_lines = [
        f"paciente: {patient_id}",
        f"dice_total_objetivo: {summary['dice_objetivo_completo']}",
        f"dice_canales_finales: {summary['dice_canales_finales']}",
        f"z_inicio_canales: {start_z}",
        f"z_fin_canales: {end_z}",
        f"capas_previas_inicio: {max(0, start_z - 10)}-{start_z - 1}",
        f"valid_slices: {summary['valid_slices']}",
        f"region_slices: {summary['region_slices']}",
        f"clusters_max: {summary['clusters_max']}",
    ]
    (patient_out / "README_resumen.txt").write_text("\n".join(report_lines), encoding="utf-8")


def process_patient(patient_id: str) -> list[dict[str, object]]:
    data, instance, seg, label_map, roles, all_labels = load_patient_data(patient_id)
    mode = TARGETS[patient_id]["mode"]
    description = TARGETS[patient_id]["description"]

    summaries: list[dict[str, object]] = []
    if mode == "comun":
        for side in ["derecha", "izquierda"]:
            gt_label = PATIENT_SIDE_OVERRIDES.get(patient_id, {}).get(side, SIDE_CONFIGS[side]["gt_label"])
            gt_target, manual_target, _ = build_side_target_masks(instance, seg, roles, side, gt_label=gt_label)
            gt_root_region, info = detect_canal_region(gt_target)
            keep_slices = np.asarray(info.get("keep_slices", []), dtype=np.int32)
            manual_root_region = np.zeros_like(manual_target, dtype=np.uint8)
            if keep_slices.size > 0:
                manual_root_region[:, :, keep_slices] = manual_target[:, :, keep_slices]

            summary = {
                "paciente": patient_id,
                "lado": side,
                "diente_gt": gt_label,
                "estado": "ok",
                "descripcion_objetivo": f"{description} {SIDE_CONFIGS[side]['descripcion_suffix']}",
                "modo": mode,
                "manual_labels": ",".join(map(str, all_labels)),
                "label_names": " | ".join(f"{k}:{v}" for k, v in sorted(label_map.items())),
                "labels_derecha": ",".join(map(str, roles["derecha"])),
                "labels_izquierda": ",".join(map(str, roles["izquierda"])),
                "labels_ian": ",".join(map(str, roles["ian"])),
                "labels_otros": ",".join(map(str, roles["otros"])),
                "gt_voxels_objetivo": int(np.count_nonzero(gt_target)),
                "manual_voxels_objetivo": int(np.count_nonzero(manual_target)),
                "dice_objetivo_completo": float(dice_score(manual_target, gt_target)),
                "gt_voxels_canales_finales": int(np.count_nonzero(gt_root_region)),
                "manual_voxels_canales_finales": int(np.count_nonzero(manual_root_region)),
                "dice_canales_finales": float(dice_score(manual_root_region, gt_root_region)),
                "z_start_objetivo": int(info["z_start"]),
                "z_end_objetivo": int(info["z_end"]),
                "valid_slices": int(info["valid_slices"]),
                "region_slices": int(info["region_slices"]),
                "clusters_max": int(info["clusters_max"]),
            }

            export_patient_outputs(
                patient_id=f"{patient_id}_{side}",
                data=data,
                gt_target=gt_target,
                manual_target=manual_target,
                info=info,
                summary=summary,
            )
            summaries.append(summary)
        return summaries

    gt_target, manual_target, mode = build_target_masks(patient_id, instance, seg, roles)
    if mode == "solo_ian":
        manual_target = make_mask(seg, roles["ian"])
        gt_target = np.zeros_like(manual_target, dtype=np.uint8)

    gt_root_region, info = detect_canal_region(gt_target)
    if mode == "solo_ian":
        gt_root_region, info = detect_canal_region(manual_target)
        gt_target = np.zeros_like(manual_target, dtype=np.uint8)
    keep_slices = np.asarray(info.get("keep_slices", []), dtype=np.int32)
    manual_root_region = np.zeros_like(manual_target, dtype=np.uint8)
    if keep_slices.size > 0:
        manual_root_region[:, :, keep_slices] = manual_target[:, :, keep_slices]

    side = "derecha" if mode in {"manual_right_to_gt36", "manual_right_to_gt46"} else "izquierda" if mode == "manual_left_to_gt46" else "ian"
    summary = {
        "paciente": patient_id,
        "lado": side,
        "diente_gt": SIDE_CONFIGS.get(side, {}).get("gt_label", 0),
        "estado": "ok",
        "descripcion_objetivo": description,
        "modo": mode,
        "manual_labels": ",".join(map(str, all_labels)),
        "label_names": " | ".join(f"{k}:{v}" for k, v in sorted(label_map.items())),
        "labels_derecha": ",".join(map(str, roles["derecha"])),
        "labels_izquierda": ",".join(map(str, roles["izquierda"])),
        "labels_ian": ",".join(map(str, roles["ian"])),
        "labels_otros": ",".join(map(str, roles["otros"])),
        "gt_voxels_objetivo": int(np.count_nonzero(gt_target)),
        "manual_voxels_objetivo": int(np.count_nonzero(manual_target)),
        "dice_objetivo_completo": float(dice_score(manual_target, gt_target)),
        "gt_voxels_canales_finales": int(np.count_nonzero(gt_root_region)),
        "manual_voxels_canales_finales": int(np.count_nonzero(manual_root_region)),
        "dice_canales_finales": float(dice_score(manual_root_region, gt_root_region)),
        "z_start_objetivo": int(info["z_start"]),
        "z_end_objetivo": int(info["z_end"]),
        "valid_slices": int(info["valid_slices"]),
        "region_slices": int(info["region_slices"]),
        "clusters_max": int(info["clusters_max"]),
    }

    export_patient_outputs(
        patient_id=patient_id,
        data=data,
        gt_target=gt_target,
        manual_target=manual_target,
        info=info,
        summary=summary,
    )
    summaries.append(summary)

    return summaries


def main() -> None:
    rows: list[dict[str, object]] = []
    for patient_id in TARGET_PATIENTS:
        try:
            rows.extend(process_patient(patient_id))
        except Exception as error:
            rows.append({"paciente": patient_id, "estado": f"error: {error}"})

    df = pd.DataFrame(rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_EXCEL, index=False)
    df.to_csv(OUTPUT_ROOT / "resumen_canales_pacientes.csv", index=False)

    print(f"Saved: {OUTPUT_EXCEL}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()