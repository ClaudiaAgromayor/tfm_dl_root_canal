from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


PREFERRED_MASK_PATTERNS = [
    "* gt_pulp.nrrd",
    "* gt_instance.nrrd",
    "* gt_ian.nrrd",
    "Segmentation.seg.nrrd",
    "*.seg.nrrd",
    "*.nrrd",
]


def find_mask_file(patient_dir: Path) -> Path | None:
    for pattern in PREFERRED_MASK_PATTERNS:
        candidates = sorted(patient_dir.glob(pattern))
        if candidates:
            return candidates[0]
    return None


def load_mask_as_zxy(mask_path: Path) -> np.ndarray:
    image = sitk.ReadImage(str(mask_path))
    array = sitk.GetArrayFromImage(image)

    if array.ndim == 4:
        return np.any(array != 0, axis=-1)
    return array != 0


def analyze_patient(patient_dir: Path) -> tuple[str, str, str, str, str]:
    mask_path = find_mask_file(patient_dir)
    patient = patient_dir.name

    if mask_path is None:
        return patient, "SIN_ARCHIVO", "SIN_ARCHIVO", "SIN_ARCHIVO", "SIN_ARCHIVO"

    mask = load_mask_as_zxy(mask_path)
    capas_z = int(mask.shape[0])

    with_pixels_per_slice = np.any(mask, axis=(1, 2))
    annotated_indices = np.flatnonzero(with_pixels_per_slice)
    capas_anotadas = int(annotated_indices.size)

    if capas_anotadas == 0:
        capa_inicio = "SIN_PIXELES"
        capa_fin = "SIN_PIXELES"
    else:
        capa_inicio = str(int(annotated_indices[0]) + 1)
        capa_fin = str(int(annotated_indices[-1]) + 1)

    return patient, str(capas_z), str(capas_anotadas), capa_inicio, capa_fin


def build_report(input_root: Path, output_path: Path) -> None:
    patient_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()], key=lambda p: p.name)

    lines = ["paciente\tcapas_z\tcapas_anotadas\tcapa_inicio\tcapa_fin"]
    for patient_dir in patient_dirs:
        row = analyze_patient(patient_dir)
        lines.append("\t".join(row))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcula capas Z y rango anotado por paciente en segmentaciones 3DSlicer."
    )
    parser.add_argument(
        "--input",
        default="../2_UNet/datasets/Segmentaciones 3DSlicer",
        help="Carpeta con subcarpetas de pacientes.",
    )
    parser.add_argument(
        "--output",
        default="results/dataset_checks/pacientes_cortes_profundidad.txt",
        help="Ruta del fichero de salida.",
    )
    args = parser.parse_args()

    input_root = Path(args.input)
    output_path = Path(args.output)

    build_report(input_root, output_path)
    print(f"Archivo generado: {output_path}")


if __name__ == "__main__":
    main()