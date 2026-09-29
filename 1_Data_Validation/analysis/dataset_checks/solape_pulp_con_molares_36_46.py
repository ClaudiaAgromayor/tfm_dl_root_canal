"""
Analiza pacientes con DICE <= 0.9999 y comprueba, de forma directa, si
gt_pulp_mandible toca las instancias 36 y/o 46 en gt_instance.

Método (como pediste):
- overlap_36 = voxels donde (gt_instance == 36) AND (gt_pulp_mandible > 0)
- overlap_46 = voxels donde (gt_instance == 46) AND (gt_pulp_mandible > 0)

Salida:
- Print por paciente.
- CSV resumen: pacientes_malos_1molar_resumen.csv
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


BASE_DIR = Path(
    r"C:/Users/clauu/OneDrive - Universidad Pontificia Comillas/Documentos/ICAI/"
    r"Beca IIT/1_Data_Validation/datasets/Pulpy3D"
)

OUTPUT_CSV = Path(
    r"C:/Users/clauu/OneDrive - Universidad Pontificia Comillas/Documentos/ICAI/"
    r"Beca IIT/1_Data_Validation/results/dataset_checks/pacientes_malos_1molar_resumen.csv"
)

PACIENTES_MALOS = [
    "P100",
    "P137",
    "P191",
    "P217",
    "P254",
    "P380",
    "P459",
    "P466",
    "P48",
    "P5",
    "P511",
    "P534",
    "P546",
    "P61",
    "P81",
]


def clasificar_paciente(patient_id: str) -> dict:
    pulp_path = BASE_DIR / patient_id / "gt_pulp_mandible.nii.gz"
    inst_path = BASE_DIR / patient_id / "gt_instance.nii.gz"

    if not pulp_path.exists() or not inst_path.exists():
        return {
            "paciente": patient_id,
            "estado": "faltan_archivos",
            "pulp_total": np.nan,
            "instance_36_total": np.nan,
            "instance_46_total": np.nan,
            "overlap_36": np.nan,
            "overlap_46": np.nan,
            "pct_pulp_36": np.nan,
            "pct_pulp_46": np.nan,
            "categoria": "faltan_archivos",
        }

    pulp = nib.load(str(pulp_path)).get_fdata()
    instance = nib.load(str(inst_path)).get_fdata()

    if pulp.shape != instance.shape:
        return {
            "paciente": patient_id,
            "estado": "shape_distinta",
            "pulp_total": np.nan,
            "instance_36_total": np.nan,
            "instance_46_total": np.nan,
            "overlap_36": np.nan,
            "overlap_46": np.nan,
            "pct_pulp_36": np.nan,
            "pct_pulp_46": np.nan,
            "categoria": "shape_distinta",
        }

    pulp_mask = pulp > 0
    inst_labels = np.rint(instance).astype(np.int32)
    inst_36 = inst_labels == 36
    inst_46 = inst_labels == 46

    pulp_total = int(np.count_nonzero(pulp_mask))
    instance_36_total = int(np.count_nonzero(inst_36))
    instance_46_total = int(np.count_nonzero(inst_46))

    overlap_36 = int(np.count_nonzero(inst_36 & pulp_mask))
    overlap_46 = int(np.count_nonzero(inst_46 & pulp_mask))

    pct_pulp_36 = 100.0 * overlap_36 / pulp_total if pulp_total > 0 else 0.0
    pct_pulp_46 = 100.0 * overlap_46 / pulp_total if pulp_total > 0 else 0.0

    toca_36 = overlap_36 > 0
    toca_46 = overlap_46 > 0

    if toca_36 and not toca_46:
        categoria = "solo_36"
    elif toca_46 and not toca_36:
        categoria = "solo_46"
    elif toca_36 and toca_46:
        categoria = "36_y_46"
    else:
        categoria = "ninguno_36_46"

    return {
        "paciente": patient_id,
        "estado": "ok",
        "pulp_total": pulp_total,
        "instance_36_total": instance_36_total,
        "instance_46_total": instance_46_total,
        "overlap_36": overlap_36,
        "overlap_46": overlap_46,
        "pct_pulp_36": pct_pulp_36,
        "pct_pulp_46": pct_pulp_46,
        "categoria": categoria,
    }


def main() -> None:
    results = []
    for patient in PACIENTES_MALOS:
        try:
            row = clasificar_paciente(patient)
        except Exception as error:
            row = {
                "paciente": patient,
                "estado": f"error: {error}",
                "pulp_total": np.nan,
                "instance_36_total": np.nan,
                "instance_46_total": np.nan,
                "overlap_36": np.nan,
                "overlap_46": np.nan,
                "pct_pulp_36": np.nan,
                "pct_pulp_46": np.nan,
                "categoria": "error",
            }
        results.append(row)

    df = pd.DataFrame(results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("Resumen solape directo con instance==36 y instance==46:")
    for _, row in df.iterrows():
        if row["estado"] == "ok":
            print(
                f"{row['paciente']}: pulp_total={int(row['pulp_total'])}, "
                f"overlap36={int(row['overlap_36'])} ({row['pct_pulp_36']:.1f}% del pulp), "
                f"overlap46={int(row['overlap_46'])} ({row['pct_pulp_46']:.1f}% del pulp), "
                f"categoria={row['categoria']}"
            )
        else:
            print(f"{row['paciente']}: {row['estado']}")

    print(f"\nCSV guardado en: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
