from __future__ import annotations

from pathlib import Path
import itertools

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import nibabel as nib
import numpy as np
import nrrd
import pandas as pd


BASE_DIR = Path(
	r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\Pulpy3D"
)
MANUAL_DIR = Path(
	r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\2_UNet\datasets\Segmentaciones 3DSlicer"
)
OUTPUT_DIR = Path("results") / "manual_vs_gt"
FIG_DIR = Path("results") / "manual_vs_gt" / "figuras_dice"
OUTPUT_EXCEL = OUTPUT_DIR / "dice_manual_vs_gt_reglas_final.xlsx"

GT_RIGHT_LABEL = 36
GT_LEFT_LABEL = 46
GT_IAN_FILENAME = "gt_ian.nii.gz"
GT_INSTANCE_FILENAME = "gt_instance.nii.gz"
IAN_TOKENS = ("ian", "nerv", "canal")
SIDE_TOKENS = {
	"derecha": ("der", "right", "36"),
	"izquierda": ("izq", "left", "46"),
}
SOLO_36_PATIENTS = {
	"P35",
	"P48",
	"P59",
	"P61",
	"P81",
	"P111",
	"P137",
	"P146",
	"P191",
	"P254",
	"P380",
	"P394",
	"P422",
	"P466",
	"P511",
	"P534",
	"P546",
}
SOLO_46_PATIENTS = {
	"P100",
	"P133",
	"P208",
	"P222",
	"P249",
	"P261",
	"P381",
	"P448",
	"P476",
}
AMBOS_PATIENTS = {
	"P217",
	"P195",
	"P343",
	"P402",
	"P447",
}
EXCLUIDOS = {"P1", "P459"}
MANUAL_PALETTE = [
	"#fdae61",
	"#abdda4",
	"#984ea3",
	"#ff7f00",
	"#4daf4a",
	"#377eb8",
	"#e41a1c",
	"#ffff33",
	"#a65628",
	"#999999",
]


def dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
	mask_a = mask_a.astype(bool)
	mask_b = mask_b.astype(bool)
	intersection = np.count_nonzero(mask_a & mask_b)
	total = np.count_nonzero(mask_a) + np.count_nonzero(mask_b)
	if total == 0:
		return np.nan
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
	label_map = {}
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


def load_manual_segmentation(patient_id: str, target_shape: tuple[int, int, int]) -> tuple[np.ndarray, dict[int, str]]:
	manual_patient_dir = MANUAL_DIR / patient_id
	if not manual_patient_dir.exists():
		return np.zeros(target_shape, dtype=np.int32), {}

	seg_files = sorted(manual_patient_dir.glob("*.seg.nrrd"))
	if not seg_files:
		return np.zeros(target_shape, dtype=np.int32), {}

	seg_data, header = nrrd.read(str(seg_files[0]))
	seg_rotated = np.flip(np.flip(np.flip(seg_data, axis=1), axis=0), axis=0)
	seg_aligned = align_shape_with_permutations(seg_rotated, target_shape)
	if seg_aligned is None:
		raise ValueError(f"Shape incompatible: {seg_rotated.shape} vs {target_shape}")

	seg_aligned = np.rint(seg_aligned).astype(np.int32)
	return seg_aligned, extract_label_map(header)


def load_gt_instance(patient_id: str) -> np.ndarray:
	instance_path = BASE_DIR / patient_id / GT_INSTANCE_FILENAME
	if not instance_path.exists():
		raise FileNotFoundError(f"No existe {GT_INSTANCE_FILENAME} en {patient_id}")
	return np.rint(nib.load(str(instance_path)).get_fdata()).astype(np.int32)


def load_gt_ian(patient_id: str) -> np.ndarray | None:
	ian_path = BASE_DIR / patient_id / GT_IAN_FILENAME
	if not ian_path.exists():
		return None
	return (nib.load(str(ian_path)).get_fdata() > 0).astype(np.uint8)


def get_patient_rule(patient_id: str) -> str:
	if patient_id in EXCLUIDOS:
		return "excluir"
	if patient_id in SOLO_36_PATIENTS:
		return "solo_36"
	if patient_id in SOLO_46_PATIENTS:
		return "solo_46"
	if patient_id in AMBOS_PATIENTS:
		return "ambos"
	return "sin_regla"


def infer_side_from_name(name: str) -> str | None:
	name_low = name.lower()
	for side, tokens in SIDE_TOKENS.items():
		if any(token in name_low for token in tokens):
			return side
	return None


def classify_manual_labels(seg: np.ndarray, label_map: dict[int, str]) -> tuple[dict[str, list[int]], list[int], list[int], list[int]]:
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

	if not roles["ian"] and len(labels) >= 3:
		# Si el header no nombra el IAN pero hay varias etiquetas, intenta detectarlo por el valor más pequeño que no sea diente.
		roles["ian"] = [label for label in labels if label not in roles["derecha"] + roles["izquierda"]][:-1]

	if not roles["derecha"] and not roles["izquierda"]:
		# Fallback conservador: usar las dos primeras etiquetas no IAN como lados.
		tooth_labels = [label for label in labels if label not in roles["ian"]]
		if len(tooth_labels) >= 2:
			roles["derecha"] = [tooth_labels[0]]
			roles["izquierda"] = [tooth_labels[1]]
		elif len(tooth_labels) == 1:
			roles["derecha"] = [tooth_labels[0]]

	return roles, labels, roles["ian"], roles["otros"]


def make_mask(seg: np.ndarray, labels: list[int]) -> np.ndarray:
	mask = np.zeros_like(seg, dtype=np.uint8)
	for label_value in labels:
		mask |= (seg == label_value).astype(np.uint8)
	return mask


def choose_slice(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
	combined = (mask_a > 0) | (mask_b > 0)
	voxels = combined.sum(axis=(0, 1))
	if voxels.max() == 0:
		return mask_a.shape[2] // 2
	return int(np.argmax(voxels))


def colorize_labels(slice_labels: np.ndarray, label_colors: dict[int, str], alpha: float) -> np.ndarray:
	color_image = np.zeros(slice_labels.shape + (4,), dtype=np.float32)
	for label_value, color in label_colors.items():
		mask = slice_labels == label_value
		if not np.any(mask):
			continue
		color_image[mask, :3] = to_rgb(color)
		color_image[mask, 3] = alpha
	return color_image


def save_review_figure(patient_id: str, data: np.ndarray, gt_tooth: np.ndarray, manual_tooth: np.ndarray, gt_ian: np.ndarray | None, manual_ian: np.ndarray | None, out_path: Path) -> None:
	gt_tooth_labels = [GT_RIGHT_LABEL, GT_LEFT_LABEL]
	gt_colors = {GT_RIGHT_LABEL: "#2c7bb6", GT_LEFT_LABEL: "#d7191c"}
	manual_labels = [int(v) for v in np.unique(manual_tooth) if v > 0]
	manual_colors = {label_value: MANUAL_PALETTE[idx % len(MANUAL_PALETTE)] for idx, label_value in enumerate(manual_labels)}
	z = choose_slice(gt_tooth, manual_tooth)
	base = data[:, :, z]
	gt_tooth_slice = gt_tooth[:, :, z]
	manual_tooth_slice = manual_tooth[:, :, z]

	if patient_id == "P217":
		base = np.rot90(base, 2)
		gt_tooth_slice = np.rot90(gt_tooth_slice, 2)
		manual_tooth_slice = np.rot90(manual_tooth_slice, 2)

	fig, axes = plt.subplots(1, 3, figsize=(21, 7))
	vmin, vmax = np.percentile(base, [1, 99])
	for ax in axes:
		ax.imshow(base.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
		ax.axis("off")

	axes[0].imshow(colorize_labels(gt_tooth_slice.T, gt_colors, alpha=0.8), origin="lower")
	axes[0].set_title("GT dientes (36/46)")

	axes[1].imshow(colorize_labels(manual_tooth_slice.T, manual_colors, alpha=0.8), origin="lower")
	axes[1].set_title("Manual dientes sin IAN")

	axes[2].imshow(colorize_labels(gt_tooth_slice.T, gt_colors, alpha=0.55), origin="lower")
	axes[2].imshow(colorize_labels(manual_tooth_slice.T, manual_colors, alpha=0.45), origin="lower")
	axes[2].set_title("Superposición dientes")

	if gt_ian is not None and manual_ian is not None:
		gt_ian_slice = gt_ian[:, :, z]
		manual_ian_slice = manual_ian[:, :, z]
		if patient_id == "P217":
			gt_ian_slice = np.rot90(gt_ian_slice, 2)
			manual_ian_slice = np.rot90(manual_ian_slice, 2)
		ian_fig, ian_axes = plt.subplots(1, 3, figsize=(21, 7))
		for ax in ian_axes:
			ax.imshow(base.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
			ax.axis("off")
		ian_axes[0].imshow(np.ma.masked_where(gt_ian_slice.T == 0, gt_ian_slice.T), cmap="Blues", alpha=0.8, origin="lower")
		ian_axes[0].set_title("GT IAN")
		ian_axes[1].imshow(np.ma.masked_where(manual_ian_slice.T == 0, manual_ian_slice.T), cmap="Oranges", alpha=0.8, origin="lower")
		ian_axes[1].set_title("Manual IAN")
		ian_axes[2].imshow(np.ma.masked_where(gt_ian_slice.T == 0, gt_ian_slice.T), cmap="Blues", alpha=0.55, origin="lower")
		ian_axes[2].imshow(np.ma.masked_where(manual_ian_slice.T == 0, manual_ian_slice.T), cmap="Oranges", alpha=0.45, origin="lower")
		ian_axes[2].set_title("Superposición IAN")
		ian_fig.suptitle(f"{patient_id} | corte z={z} | IAN")
		plt.tight_layout()
		ian_out = out_path.parent / f"{patient_id}_ian.png"
		ian_fig.savefig(ian_out, dpi=200, bbox_inches="tight")
		plt.close(ian_fig)

	fig.suptitle(f"{patient_id} | corte z={z} | dientes sin IAN")
	plt.tight_layout()
	fig.savefig(out_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def summarize_patient(patient_id: str) -> dict[str, object]:
	rule = get_patient_rule(patient_id)
	if rule == "excluir":
		return {
			"paciente": patient_id,
			"estado": "excluido",
			"regla": rule,
		}
	if rule == "sin_regla":
		return {
			"paciente": patient_id,
			"estado": "sin_regla",
			"regla": rule,
		}

	patient_dir = BASE_DIR / patient_id
	manual_dir = MANUAL_DIR / patient_id
	instance = load_gt_instance(patient_id)
	gt_ian = load_gt_ian(patient_id)
	data_path = patient_dir / "data.nii.gz"
	if not data_path.exists():
		raise FileNotFoundError(f"No existe data.nii.gz en {patient_id}")
	data = nib.load(str(data_path)).get_fdata()

	seg_files = sorted(manual_dir.glob("*.seg.nrrd"))
	if not seg_files:
		return {"paciente": patient_id, "estado": "sin_segmentation_seg_nrrd"}

	seg, label_map = load_manual_segmentation(patient_id, data.shape)
	roles, all_labels, ian_labels, unknown_labels = classify_manual_labels(seg, label_map)

	gt_right = (instance == GT_RIGHT_LABEL).astype(np.uint8)
	gt_left = (instance == GT_LEFT_LABEL).astype(np.uint8)
	gt_tooth = (gt_right | gt_left).astype(np.uint8)
	manual_ian = make_mask(seg, ian_labels)
	manual_right = make_mask(seg, roles["derecha"])
	manual_left = make_mask(seg, roles["izquierda"])
	manual_tooth = (manual_right | manual_left).astype(np.uint8)
	gt_tooth_without_ian = gt_tooth.copy()
	manual_tooth_without_ian = manual_tooth.copy()

	if gt_ian is not None:
		gt_tooth_without_ian = gt_tooth_without_ian & (gt_ian == 0)
		manual_tooth_without_ian = manual_tooth_without_ian & (manual_ian == 0)
		gt_right_without_ian = gt_right & (gt_ian == 0)
		gt_left_without_ian = gt_left & (gt_ian == 0)
	else:
		gt_right_without_ian = gt_right
		gt_left_without_ian = gt_left

	if rule == "solo_36":
		manual_right = manual_right & (manual_ian == 0)
		manual_left = np.zeros_like(manual_right, dtype=np.uint8)
		dice_right = dice_score(manual_right, gt_right_without_ian)
		dice_left = 0.0
		dice_common = 0.0
		modo_dientes = "solo_36"
		diente_objetivo = 36
		gt_objetivo = gt_right_without_ian
		manual_objetivo = manual_right
	elif rule == "solo_46":
		manual_right = np.zeros_like(manual_left, dtype=np.uint8)
		manual_left = manual_left & (manual_ian == 0)
		dice_right = 0.0
		dice_left = dice_score(manual_left, gt_left_without_ian)
		dice_common = 0.0
		modo_dientes = "solo_46"
		diente_objetivo = 46
		gt_objetivo = gt_left_without_ian
		manual_objetivo = manual_left
	else:
		# ambos
		# Segmentación conjunta: usar un DICE común y dejar lados a 0.
		dice_common = dice_score(manual_tooth_without_ian, gt_tooth_without_ian)
		dice_right = 0.0
		dice_left = 0.0
		modo_dientes = "ambos"
		diente_objetivo = 0
		gt_objetivo = gt_tooth_without_ian
		manual_objetivo = manual_tooth_without_ian

	if gt_ian is None:
		dice_ian = np.nan
		estado_ian = "sin_gt_ian"
		manual_ian_mask = None
	else:
		manual_ian_mask = manual_ian
		dice_ian = dice_score(manual_ian_mask, gt_ian)
		estado_ian = "ok"

	review_dir = FIG_DIR / patient_id
	review_dir.mkdir(parents=True, exist_ok=True)
	save_review_figure(
		patient_id=patient_id,
		data=data,
		gt_tooth=gt_tooth_without_ian.astype(np.uint8),
		manual_tooth=manual_tooth_without_ian.astype(np.uint8),
		gt_ian=gt_ian,
		manual_ian=manual_ian_mask,
		out_path=review_dir / f"{patient_id}_dientes.png",
	)

	gt_target_voxels = int(np.count_nonzero(gt_objetivo))
	manual_target_voxels = int(np.count_nonzero(manual_objetivo))

	return {
		"paciente": patient_id,
		"estado": "ok",
		"regla": rule,
		"diente_objetivo": diente_objetivo,
		"archivo_manual": seg_files[0].name,
		"manual_labels": ",".join(map(str, all_labels)) if all_labels else "",
		"ian_labels": ",".join(map(str, ian_labels)) if ian_labels else "",
		"unknown_labels": ",".join(map(str, unknown_labels)) if unknown_labels else "",
		"modo_dientes": modo_dientes,
		"gt_voxels_objetivo": gt_target_voxels,
		"manual_voxels_objetivo": manual_target_voxels,
		"dice_derecha": dice_right,
		"dice_izquierda": dice_left,
		"dice_comun": dice_common,
		"dice_ian": dice_ian,
		"estado_ian": estado_ian,
		"gt_right_label": GT_RIGHT_LABEL,
		"gt_left_label": GT_LEFT_LABEL,
		"gt_ian_file": GT_IAN_FILENAME if gt_ian is not None else "",
		"nota": "dientes sin IAN; IAN solo con gt_ian.nii.gz",
	}


def iter_patient_ids() -> list[str]:
	selected = sorted((SOLO_36_PATIENTS | SOLO_46_PATIENTS | AMBOS_PATIENTS) - EXCLUIDOS)
	return [p for p in selected if (MANUAL_DIR / p).is_dir()]


def main() -> None:
	rows = []
	for patient_id in iter_patient_ids():
		try:
			rows.append(summarize_patient(patient_id))
		except Exception as error:
			rows.append({"paciente": patient_id, "estado": f"error: {error}"})

	df = pd.DataFrame(rows)
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	df.to_excel(OUTPUT_EXCEL, index=False)
	print(f"Excel guardado en: {OUTPUT_EXCEL}")
	print(f"Figuras guardadas en: {FIG_DIR}")
	cols = [c for c in ["paciente", "estado", "regla", "diente_objetivo", "modo_dientes", "gt_voxels_objetivo", "manual_voxels_objetivo", "dice_derecha", "dice_izquierda", "dice_comun", "dice_ian", "estado_ian"] if c in df.columns]
	print(df[cols].to_string(index=False))


if __name__ == "__main__":
	main()