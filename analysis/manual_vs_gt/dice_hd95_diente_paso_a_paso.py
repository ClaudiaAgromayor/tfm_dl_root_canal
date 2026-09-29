from pathlib import Path
import csv
import itertools

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import nibabel as nib
import numpy as np
import nrrd
from scipy import ndimage


BASE_DIR = Path(
	r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\Pulpy3D"
)
MANUAL_DIR = Path(
	r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\2_UNet\datasets\Segmentaciones 3DSlicer"
)
OUTPUT_DIR = Path("results") / "manual_vs_gt" / "dice_paso_a_paso"

CASES = {
	"P222": {"gt_label": 46, "manual_side": "left"},
	"P249": {"gt_label": 46, "manual_side": "right"},
	"P254": {"gt_label": 46, "manual_side": "right"},
	"P261": {"gt_label": 36, "manual_side": "right"},
	"P343": {"mode": "dual"},
	"P380": {"gt_label": 36, "manual_side": "right"},
	"P381": {"gt_label": 36, "manual_side": "right"},
	"P394": {"gt_label": 46, "manual_side": "left"},
	"P402": {"gt_label": 46, "manual_side": "left"},
	"P422": {"gt_label": 46, "manual_side": "left"},
	"P448": {"gt_label": 36, "manual_side": "right"},
	"P466": {"gt_label": 36, "manual_side": "right"},
	"P511": {"gt_label": 46, "manual_side": "left"},
	"P534": {"gt_label": 46, "manual_side": "left"},
	"P546": {"gt_label": 36, "manual_side": "right"},
	"P259": {"gt_label": 46, "manual_side": "right"},
}

IAN_TOKENS = ("ian", "nerv", "canal")
SIDE_TOKENS = {
	"right": ("der", "right", "36"),
	"left": ("izq", "left", "46"),
}


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


def align_shape_with_permutations(seg_array: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray | None:
	if seg_array.shape == target_shape:
		return seg_array
	for perm in itertools.permutations(range(3)):
		candidate = np.transpose(seg_array, perm)
		if candidate.shape == target_shape:
			return candidate
	return None


def load_manual_segmentation(patient_id: str, target_shape: tuple[int, int, int]) -> tuple[np.ndarray, dict[int, str]]:
	manual_dir = MANUAL_DIR / patient_id
	seg_files = sorted(manual_dir.glob("*.seg.nrrd"))
	if not seg_files:
		raise FileNotFoundError(f"No hay .seg.nrrd en {manual_dir}")

	seg_data, header = nrrd.read(str(seg_files[0]))
	seg_rot = np.flip(np.flip(np.flip(seg_data, axis=1), axis=0), axis=0)
	seg_aligned = align_shape_with_permutations(seg_rot, target_shape)
	if seg_aligned is None:
		raise ValueError(f"Shape incompatible: {seg_rot.shape} vs {target_shape}")

	seg_aligned = np.rint(seg_aligned).astype(np.int32)
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
	return seg_aligned, label_map


def load_patient(patient_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
	patient_dir = BASE_DIR / patient_id
	data = nib.load(str(patient_dir / "data.nii.gz")).get_fdata()
	instance = np.rint(nib.load(str(patient_dir / "gt_instance.nii.gz")).get_fdata()).astype(np.int32)
	manual, label_map = load_manual_segmentation(patient_id, data.shape)
	if data.shape != instance.shape:
		raise ValueError(f"Shape distinta: data={data.shape} vs instance={instance.shape}")
	return data, instance, manual, label_map


def build_manual_mask(manual: np.ndarray, label_map: dict[int, str], side: str) -> tuple[np.ndarray, list[int], list[int]]:
	labels = [int(v) for v in np.unique(manual) if v > 0]
	keep_labels = []
	removed_ian_labels = []
	for label_value in labels:
		name = str(label_map.get(label_value, "")).lower()
		if any(token in name for token in IAN_TOKENS):
			removed_ian_labels.append(label_value)
		elif any(token in name for token in SIDE_TOKENS[side]):
			keep_labels.append(label_value)

	if not keep_labels:
		# Fallback conservador si el header no nombra el lado.
		candidate_labels = [label for label in labels if label not in removed_ian_labels]
		if candidate_labels:
			keep_labels = [candidate_labels[0]]

	mask = np.zeros_like(manual, dtype=np.uint8)
	for label_value in keep_labels:
		mask |= (manual == label_value).astype(np.uint8)
	return mask, keep_labels, removed_ian_labels


def analyze_slices(gt_mask: np.ndarray, manual_mask: np.ndarray) -> list[dict[str, object]]:
	rows = []
	for z in range(gt_mask.shape[2]):
		gt_slice = gt_mask[:, :, z].astype(np.uint8)
		manual_slice = manual_mask[:, :, z].astype(np.uint8)
		gt_voxels = int(gt_slice.sum())
		manual_voxels = int(manual_slice.sum())
		both_voxels = int(np.count_nonzero(gt_slice & manual_slice))
		rows.append(
			{
				"z": z,
				"dice": dice_score(gt_slice, manual_slice),
				"gt_voxels": gt_voxels,
				"manual_voxels": manual_voxels,
				"both_voxels": both_voxels,
				"only_gt": gt_voxels > 0 and manual_voxels == 0,
				"only_manual": manual_voxels > 0 and gt_voxels == 0,
				"both_present": gt_voxels > 0 and manual_voxels > 0,
			}
		)
	return rows


def contiguous_ranges(indices: list[int]) -> list[tuple[int, int]]:
	if not indices:
		return []
	ranges = []
	start = prev = indices[0]
	for value in indices[1:]:
		if value == prev + 1:
			prev = value
			continue
		ranges.append((start, prev))
		start = prev = value
	ranges.append((start, prev))
	return ranges


def colorize_mask(slice_mask: np.ndarray, color: str, alpha: float) -> np.ndarray:
	colored = np.zeros(slice_mask.shape + (4,), dtype=np.float32)
	colored[slice_mask > 0, :3] = to_rgb(color)
	colored[slice_mask > 0, 3] = alpha
	return colored


def save_slice_analysis(patient: str, slice_rows: list[dict[str, object]]) -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	csv_path = OUTPUT_DIR / f"{patient}_slice_analysis.csv"
	with csv_path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=["z", "dice", "hd95_3d", "gt_voxels", "manual_voxels", "both_voxels", "only_gt", "only_manual", "both_present"])
		writer.writeheader()
		writer.writerows(slice_rows)

	valid = [row for row in slice_rows if row["gt_voxels"] > 0 or row["manual_voxels"] > 0]
	if valid:
		highest = max(valid, key=lambda row: row["dice"] if not np.isnan(row["dice"]) else -1)
		lowest = min(valid, key=lambda row: row["dice"] if not np.isnan(row["dice"]) else np.inf)
	else:
		highest = lowest = None

	only_gt = [row["z"] for row in slice_rows if row["only_gt"]]
	only_manual = [row["z"] for row in slice_rows if row["only_manual"]]
	both = [row["z"] for row in slice_rows if row["both_present"]]

	print(f"\nAnálisis por cortes de {patient}")
	if highest is not None:
		print(f"  DICE más alto: z={highest['z']} | dice={highest['dice']:.6f} | gt={highest['gt_voxels']} | manual={highest['manual_voxels']}")
		print(f"  DICE más bajo: z={lowest['z']} | dice={lowest['dice']:.6f} | gt={lowest['gt_voxels']} | manual={lowest['manual_voxels']}")
	print(f"  Cortes solo GT: {contiguous_ranges(only_gt)}")
	print(f"  Cortes solo manual: {contiguous_ranges(only_manual)}")
	print(f"  Cortes con ambos: {contiguous_ranges(both)}")
	if slice_rows:
		print(f"  HD95 3D global: {slice_rows[0]['hd95_3d']}")
	print(f"  CSV guardado en: {csv_path}")

	plot_path = OUTPUT_DIR / f"{patient}_dice_por_corte.png"
	zs = [row["z"] for row in slice_rows]
	dices = [np.nan_to_num(row["dice"], nan=-0.1) for row in slice_rows]
	gt_counts = [row["gt_voxels"] for row in slice_rows]
	manual_counts = [row["manual_voxels"] for row in slice_rows]

	fig, ax1 = plt.subplots(figsize=(14, 5))
	ax1.plot(zs, dices, color="#1f77b4", linewidth=1.5, label="DICE por corte")
	ax1.set_xlabel("Corte z")
	ax1.set_ylabel("DICE")
	ax1.set_ylim(-0.05, 1.05)
	ax1.grid(True, alpha=0.25)
	ax1.legend(loc="upper left")

	ax2 = ax1.twinx()
	ax2.plot(zs, gt_counts, color="#2c7bb6", alpha=0.35, label="GT voxels")
	ax2.plot(zs, manual_counts, color="#fdae61", alpha=0.35, label="Manual voxels")
	ax2.set_ylabel("Voxels")
	ax2.legend(loc="upper right")

	fig.suptitle(f"{patient} | DICE y presencia por corte")
	plt.tight_layout()
	plt.savefig(plot_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"  Gráfico guardado en: {plot_path}")


def save_dual_slice_analysis(patient: str, slice_rows: list[dict[str, object]]) -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	csv_path = OUTPUT_DIR / f"{patient}_slice_analysis.csv"
	with csv_path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=[
				"z",
				"gt_right_voxels",
				"manual_right_voxels",
				"dice_right",
				"hd95_right_3d",
				"gt_left_voxels",
				"manual_left_voxels",
				"dice_left",
				"hd95_left_3d",
				"only_gt",
				"only_manual",
				"both_present",
			],
		)
		writer.writeheader()
		writer.writerows(slice_rows)

	valid_right = [row for row in slice_rows if row["gt_right_voxels"] > 0 or row["manual_right_voxels"] > 0]
	valid_left = [row for row in slice_rows if row["gt_left_voxels"] > 0 or row["manual_left_voxels"] > 0]
	highest_right = max(valid_right, key=lambda row: row["dice_right"] if not np.isnan(row["dice_right"]) else -1) if valid_right else None
	highest_left = max(valid_left, key=lambda row: row["dice_left"] if not np.isnan(row["dice_left"]) else -1) if valid_left else None

	only_gt = [row["z"] for row in slice_rows if row["only_gt"]]
	only_manual = [row["z"] for row in slice_rows if row["only_manual"]]
	both = [row["z"] for row in slice_rows if row["both_present"]]

	print(f"\nAnálisis por cortes de {patient}")
	if highest_right is not None:
		print(f"  DICE derecho más alto: z={highest_right['z']} | dice={highest_right['dice_right']:.6f} | gt={highest_right['gt_right_voxels']} | manual={highest_right['manual_right_voxels']}")
	if highest_left is not None:
		print(f"  DICE izquierdo más alto: z={highest_left['z']} | dice={highest_left['dice_left']:.6f} | gt={highest_left['gt_left_voxels']} | manual={highest_left['manual_left_voxels']}")
	print(f"  Cortes solo GT: {contiguous_ranges(only_gt)}")
	print(f"  Cortes solo manual: {contiguous_ranges(only_manual)}")
	print(f"  Cortes con ambos: {contiguous_ranges(both)}")
	if slice_rows:
		print(f"  HD95 3D derecho global: {slice_rows[0]['hd95_right_3d']}")
		print(f"  HD95 3D izquierdo global: {slice_rows[0]['hd95_left_3d']}")
	print(f"  CSV guardado en: {csv_path}")

	plot_path = OUTPUT_DIR / f"{patient}_dice_por_corte.png"
	zs = [row["z"] for row in slice_rows]
	dices_right = [np.nan_to_num(row["dice_right"], nan=-0.1) for row in slice_rows]
	dices_left = [np.nan_to_num(row["dice_left"], nan=-0.1) for row in slice_rows]
	gt_right_counts = [row["gt_right_voxels"] for row in slice_rows]
	manual_right_counts = [row["manual_right_voxels"] for row in slice_rows]
	gt_left_counts = [row["gt_left_voxels"] for row in slice_rows]
	manual_left_counts = [row["manual_left_voxels"] for row in slice_rows]

	fig, ax1 = plt.subplots(figsize=(14, 5))
	ax1.plot(zs, dices_right, color="#2c7bb6", linewidth=1.5, label="DICE derecho")
	ax1.plot(zs, dices_left, color="#d7191c", linewidth=1.5, label="DICE izquierdo")
	ax1.set_xlabel("Corte z")
	ax1.set_ylabel("DICE")
	ax1.set_ylim(-0.05, 1.05)
	ax1.grid(True, alpha=0.25)
	ax1.legend(loc="upper left")

	ax2 = ax1.twinx()
	ax2.plot(zs, gt_right_counts, color="#2c7bb6", alpha=0.25, linestyle="--", label="GT derecho")
	ax2.plot(zs, manual_right_counts, color="#fdae61", alpha=0.25, linestyle="--", label="Manual derecho")
	ax2.plot(zs, gt_left_counts, color="#d7191c", alpha=0.25, linestyle=":", label="GT izquierdo")
	ax2.plot(zs, manual_left_counts, color="#abdda4", alpha=0.25, linestyle=":", label="Manual izquierdo")
	ax2.set_ylabel("Voxels")
	ax2.legend(loc="upper right")

	fig.suptitle(f"{patient} | DICE y presencia por corte")
	plt.tight_layout()
	plt.savefig(plot_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"  Gráfico guardado en: {plot_path}")


def save_dual_summary_figure(
	patient: str,
	base: np.ndarray,
	right_gt: np.ndarray,
	right_manual: np.ndarray,
	left_gt: np.ndarray,
	left_manual: np.ndarray,
	right_dice: float,
	left_dice: float,
) -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	out_path = OUTPUT_DIR / f"{patient}_dice.png"

	def best_slice(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
		combined = (mask_a > 0) | (mask_b > 0)
		voxels = combined.sum(axis=(0, 1))
		if voxels.max() == 0:
			return mask_a.shape[2] // 2
		return int(np.argmax(voxels))

	z_right = best_slice(right_gt, right_manual)
	z_left = best_slice(left_gt, left_manual)

	fig, axes = plt.subplots(2, 3, figsize=(21, 14))
	for row_axes, z, gt_slice, manual_slice, title, dice_value in [
		(axes[0], z_right, right_gt[:, :, z_right], right_manual[:, :, z_right], "Derecha / GT 36", right_dice),
		(axes[1], z_left, left_gt[:, :, z_left], left_manual[:, :, z_left], "Izquierda / GT 46", left_dice),
	]:
		base_slice = base[:, :, z]
		vmin, vmax = np.percentile(base_slice, [1, 99])
		for ax in row_axes:
			ax.imshow(base_slice.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
			ax.axis("off")
		row_axes[0].imshow(colorize_mask(gt_slice.T, "#2c7bb6", 0.85), origin="lower")
		row_axes[0].set_title(f"GT | z={z}")
		row_axes[1].imshow(colorize_mask(manual_slice.T, "#fdae61", 0.85), origin="lower")
		row_axes[1].set_title("Manual")
		row_axes[2].imshow(colorize_mask(gt_slice.T, "#2c7bb6", 0.55), origin="lower")
		row_axes[2].imshow(colorize_mask(manual_slice.T, "#fdae61", 0.40), origin="lower")
		row_axes[2].set_title(f"Superposición | DICE={dice_value:.6f}")

	fig.suptitle(f"{patient} | resumen de ambas muelas")
	plt.tight_layout()
	plt.savefig(out_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"Imagen guardada en: {out_path}")


def save_summary_figure(patient: str, base: np.ndarray, z: int, gt_slice: np.ndarray, manual_slice: np.ndarray, dice_value: float) -> None:
	if patient == "P217":
		base = np.rot90(base, 2)
		gt_slice = np.rot90(gt_slice, 2)
		manual_slice = np.rot90(manual_slice, 2)

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	out_path = OUTPUT_DIR / f"{patient}_dice.png"

	fig, axes = plt.subplots(1, 3, figsize=(21, 7))
	vmin, vmax = np.percentile(base, [1, 99])
	for ax in axes:
		ax.imshow(base.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
		ax.axis("off")

	axes[0].imshow(colorize_mask(gt_slice.T, "#2c7bb6", 0.85), origin="lower")
	axes[0].set_title("GT")
	axes[1].imshow(colorize_mask(manual_slice.T, "#fdae61", 0.85), origin="lower")
	axes[1].set_title("Manual")
	axes[2].imshow(colorize_mask(gt_slice.T, "#2c7bb6", 0.55), origin="lower")
	axes[2].imshow(colorize_mask(manual_slice.T, "#fdae61", 0.40), origin="lower")
	axes[2].set_title("Superposición")

	fig.suptitle(f"{patient} | DICE entero={dice_value:.6f} | z={z}")
	plt.tight_layout()
	plt.savefig(out_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"Imagen guardada en: {out_path}")


def run_case(patient_id: str, gt_label: int, manual_side: str) -> None:
	try:
		data, instance, manual, label_map = load_patient(patient_id)
	except FileNotFoundError as error:
		print(f"[SKIP] {patient_id}: {error}")
		return

	gt_mask = (instance == gt_label).astype(np.uint8)
	manual_mask, manual_labels_used, manual_labels_removed_ian = build_manual_mask(manual, label_map, manual_side)
	dice_value = dice_score(gt_mask, manual_mask)
	hd95_value = hd95_3d(gt_mask, manual_mask)
	slice_rows = analyze_slices(gt_mask, manual_mask)
	z = int(np.argmax((gt_mask > 0).sum(axis=(0, 1)) + (manual_mask > 0).sum(axis=(0, 1))))
	for row in slice_rows:
		row["hd95_3d"] = hd95_value

	save_summary_figure(patient_id, data[:, :, z], z, gt_mask[:, :, z], manual_mask[:, :, z], dice_value)
	save_slice_analysis(patient_id, slice_rows)

	print(f"Paciente: {patient_id}")
	print(f"GT label: {gt_label}")
	print(f"Manual side: {manual_side}")
	print(f"Manual labels (total): {sorted([int(v) for v in np.unique(manual) if v > 0])}")
	print(f"Manual labels usadas: {manual_labels_used}")
	print(f"Manual labels IAN quitadas: {manual_labels_removed_ian}")
	print(f"DICE entero: {dice_value:.6f}")
	print(f"HD95 3D entero: {hd95_value if not np.isnan(hd95_value) else 'nan'}")


def run_dual_case(patient_id: str) -> None:
	try:
		data, instance, manual, label_map = load_patient(patient_id)
	except FileNotFoundError as error:
		print(f"[SKIP] {patient_id}: {error}")
		return

	right_gt = (instance == 36).astype(np.uint8)
	left_gt = (instance == 46).astype(np.uint8)
	right_manual, right_used, right_removed = build_manual_mask(manual, label_map, "right")
	left_manual, left_used, left_removed = build_manual_mask(manual, label_map, "left")
	right_dice = dice_score(right_gt, right_manual)
	left_dice = dice_score(left_gt, left_manual)
	right_hd95 = hd95_3d(right_gt, right_manual)
	left_hd95 = hd95_3d(left_gt, left_manual)
	slice_rows = []
	for z in range(instance.shape[2]):
		right_gt_slice = right_gt[:, :, z].astype(np.uint8)
		right_manual_slice = right_manual[:, :, z].astype(np.uint8)
		left_gt_slice = left_gt[:, :, z].astype(np.uint8)
		left_manual_slice = left_manual[:, :, z].astype(np.uint8)
		right_gt_voxels = int(right_gt_slice.sum())
		right_manual_voxels = int(right_manual_slice.sum())
		left_gt_voxels = int(left_gt_slice.sum())
		left_manual_voxels = int(left_manual_slice.sum())
		slice_rows.append(
			{
				"z": z,
				"gt_right_voxels": right_gt_voxels,
				"manual_right_voxels": right_manual_voxels,
				"dice_right": dice_score(right_gt_slice, right_manual_slice),
					"hd95_right_3d": right_hd95,
				"gt_left_voxels": left_gt_voxels,
				"manual_left_voxels": left_manual_voxels,
				"dice_left": dice_score(left_gt_slice, left_manual_slice),
					"hd95_left_3d": left_hd95,
				"only_gt": (right_gt_voxels + left_gt_voxels) > 0 and (right_manual_voxels + left_manual_voxels) == 0,
				"only_manual": (right_manual_voxels + left_manual_voxels) > 0 and (right_gt_voxels + left_gt_voxels) == 0,
				"both_present": (right_gt_voxels + left_gt_voxels) > 0 and (right_manual_voxels + left_manual_voxels) > 0,
			}
		)

	save_dual_summary_figure(patient_id, data, right_gt, right_manual, left_gt, left_manual, right_dice, left_dice)
	save_dual_slice_analysis(patient_id, slice_rows)

	print(f"Paciente: {patient_id}")
	print("GT labels: 36 y 46")
	print(f"Manual labels derecha usadas: {right_used}")
	print(f"Manual labels izquierda usadas: {left_used}")
	print(f"Manual labels IAN quitadas derecha: {right_removed}")
	print(f"Manual labels IAN quitadas izquierda: {left_removed}")
	print(f"DICE derecho global: {right_dice:.6f}")
	print(f"DICE izquierdo global: {left_dice:.6f}")
	print(f"HD95 3D derecho global: {right_hd95 if not np.isnan(right_hd95) else 'nan'}")
	print(f"HD95 3D izquierdo global: {left_hd95 if not np.isnan(left_hd95) else 'nan'}")


def main() -> None:
	for patient_id, config in CASES.items():
		if config.get("mode") == "dual":
			run_dual_case(patient_id)
		else:
			run_case(patient_id, config["gt_label"], config["manual_side"])


if __name__ == "__main__":
	main()