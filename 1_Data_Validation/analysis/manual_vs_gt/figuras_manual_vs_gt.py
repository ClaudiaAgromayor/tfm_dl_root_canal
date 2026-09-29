from pathlib import Path
import itertools

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import nibabel as nib
import numpy as np
import nrrd


BASE_DIR = Path(
	r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\Pulpy3D"
)
FER_BASE_DIR = Path(
	r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\2_UNet\datasets\Segmentaciones 3DSlicer"
)
OUTPUT_DIR = Path("results") / "manual_vs_gt" / "figuras"

GT_LABELS = (36, 46)
GT_COLORS = {
	36: "#2c7bb6",
	46: "#d7191c",
}
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
	interseccion = np.count_nonzero(mask_a & mask_b)
	denominador = np.count_nonzero(mask_a) + np.count_nonzero(mask_b)
	if denominador == 0:
		return 1.0
	return 2.0 * interseccion / denominador


def align_shape_with_permutations(seg_array: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray | None:
	if seg_array.shape == target_shape:
		return seg_array

	for perm in itertools.permutations(range(3)):
		candidate = np.transpose(seg_array, perm)
		if candidate.shape == target_shape:
			return candidate

	return None


def escoger_corte_con_mascara(mask: np.ndarray) -> int:
	mask_bin = mask > 0
	voxeles_por_corte = mask_bin.sum(axis=(0, 1))
	if voxeles_por_corte.max() == 0:
		return mask.shape[2] // 2
	return int(np.argmax(voxeles_por_corte))


def extraer_mapa_etiquetas(header: dict) -> dict[int, str]:
	mapa = {}
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
		mapa[label_value] = str(value)
	return mapa


def cargar_seg_manual(patient_id: str, target_shape: tuple[int, int, int]) -> tuple[np.ndarray, dict[int, str]]:
	fer_patient_dir = FER_BASE_DIR / patient_id
	if not fer_patient_dir.exists():
		return np.zeros(target_shape, dtype=np.int32), {}

	seg_files = sorted(fer_patient_dir.glob("*.seg.nrrd"))
	if not seg_files:
		return np.zeros(target_shape, dtype=np.int32), {}

	seg_data, header = nrrd.read(str(seg_files[0]))
	seg_rot = np.flip(np.flip(np.flip(seg_data, axis=1), axis=0), axis=0)
	seg_aligned = align_shape_with_permutations(seg_rot, target_shape)
	if seg_aligned is None:
		return np.zeros(target_shape, dtype=np.int32), {}

	seg_aligned = np.rint(seg_aligned).astype(np.int32)
	return seg_aligned, extraer_mapa_etiquetas(header)


def cargar_volumenes(patient_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
	data_path = patient_dir / "data.nii.gz"
	instance_path = patient_dir / "gt_instance.nii.gz"

	if not data_path.exists() or not instance_path.exists():
		raise FileNotFoundError(
			f"Faltan archivos en {patient_dir.name}: data.nii.gz o gt_instance.nii.gz"
		)

	data = nib.load(str(data_path)).get_fdata()
	instance = nib.load(str(instance_path)).get_fdata()
	instance_labels = np.rint(instance).astype(np.int32)
	manual_labels, manual_map = cargar_seg_manual(patient_dir.name, data.shape)

	if data.shape != instance_labels.shape:
		raise ValueError(
			f"Shape distinta en {patient_dir.name}: data={data.shape} vs instance={instance_labels.shape}"
		)

	return data, instance_labels, manual_labels, manual_map


def _colorize_labels(slice_labels: np.ndarray, label_colors: dict[int, str], alpha: float) -> np.ndarray:
	color_image = np.zeros(slice_labels.shape + (4,), dtype=np.float32)
	for label_value, color in label_colors.items():
		mask = slice_labels == label_value
		if not np.any(mask):
			continue
		color_image[mask, :3] = to_rgb(color)
		color_image[mask, 3] = alpha
	return color_image


def _draw_colored_mask(ax, base_slice: np.ndarray, label_slice: np.ndarray, label_colors: dict[int, str], title: str) -> None:
	vmin, vmax = np.percentile(base_slice, [1, 99])
	ax.imshow(base_slice.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
	ax.imshow(_colorize_labels(label_slice.T, label_colors, alpha=0.8), origin="lower")
	ax.set_title(title)
	ax.axis("off")


def _label_title(labels: list[int], label_map: dict[int, str]) -> str:
	if not labels:
		return "sin etiquetas"
	parts = []
	for label_value in labels:
		name = label_map.get(label_value)
		if name:
			parts.append(f"{label_value}:{name}")
		else:
			parts.append(str(label_value))
	return ", ".join(parts)


def _manual_label_colors(labels: list[int]) -> dict[int, str]:
	return {label_value: MANUAL_PALETTE[idx % len(MANUAL_PALETTE)] for idx, label_value in enumerate(labels)}


def mostrar_comparacion_paciente(base_dir: Path, paciente: str, output_png: Path) -> None:
	patient_dir = base_dir / paciente
	data, instance_labels, manual_labels, manual_map = cargar_volumenes(patient_dir)

	manual_unique = [int(v) for v in np.unique(manual_labels) if v > 0]
	gt_unique = [int(v) for v in np.unique(instance_labels) if v in GT_LABELS]
	z = escoger_corte_con_mascara((instance_labels > 0) | (manual_labels > 0))

	corte_data = data[:, :, z]
	corte_gt = instance_labels[:, :, z]
	corte_manual = manual_labels[:, :, z]

	if paciente == "P217":
		corte_data = np.rot90(corte_data, 2)
		corte_gt = np.rot90(corte_gt, 2)
		corte_manual = np.rot90(corte_manual, 2)

	manual_colors = _manual_label_colors(manual_unique)
	gt_colors = {label_value: GT_COLORS[label_value] for label_value in gt_unique}

	fig, axes = plt.subplots(1, 3, figsize=(21, 7))
	_draw_colored_mask(axes[0], corte_data, corte_gt, gt_colors, f"Ground truth | {_label_title(gt_unique, {})}")
	_draw_colored_mask(axes[1], corte_data, corte_manual, manual_colors, f"Manual | {_label_title(manual_unique, manual_map)}")

	vmin, vmax = np.percentile(corte_data, [1, 99])
	axes[2].imshow(corte_data.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
	axes[2].imshow(_colorize_labels(corte_gt.T, gt_colors, alpha=0.65), origin="lower")
	axes[2].imshow(_colorize_labels(corte_manual.T, manual_colors, alpha=0.45), origin="lower")
	axes[2].set_title("Superposición GT + manual")
	axes[2].axis("off")

	dice_global = dice_score(instance_labels > 0, manual_labels > 0)
	fig.suptitle(
		f"{paciente} | DICE global={dice_global:.4f} | corte z={z}",
		fontsize=14,
	)
	plt.tight_layout()
	output_png.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_png, dpi=200, bbox_inches="tight")
	plt.close(fig)

	print(f"Paciente: {paciente}")
	print(f"DICE global (GT vs manual): {dice_global:.4f}")
	print(f"Etiquetas manuales: {manual_unique if manual_unique else 'sin etiquetas'}")
	print(f"Imagen guardada en: {output_png}")


def iterar_pacientes_manual() -> list[str]:
	return sorted([p.name for p in FER_BASE_DIR.iterdir() if p.is_dir()])


def main() -> None:
	for paciente in iterar_pacientes_manual():
		try:
			output_png = OUTPUT_DIR / paciente / f"{paciente}_comparacion.png"
			mostrar_comparacion_paciente(BASE_DIR, paciente, output_png)
		except Exception as error:
			print(f"[SKIP] {paciente}: {error}")


if __name__ == "__main__":
	main()
