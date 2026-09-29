"""
Recorre todos los pacientes de datasets/Pulpy3D y comprueba si la máscara
gt_pulp_mandible.nii.gz contiene información.

Criterio:
- Si np.max(mascara_pulpa) == 0, se considera que NO tiene información.

Salida:
- Imprime la lista de pacientes con máximo 0.
- Guarda esa lista en un CSV: pacientes_canal_sin_info.csv
"""

import os

import nibabel as nib
import numpy as np
import pandas as pd


BASE_DIR = (
    r"C:/Users/clauu/OneDrive - Universidad Pontificia Comillas/Documentos/ICAI/"
    r"Beca IIT/1_Data_Validation/datasets/Pulpy3D"
)
OUTPUT_CSV = (
    r"C:/Users/clauu/OneDrive - Universidad Pontificia Comillas/Documentos/ICAI/"
    r"Beca IIT/1_Data_Validation/results/dataset_checks/pacientes_canal_sin_info_2.csv"
)


def main() -> None:
    pacientes = sorted(
        d
        for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d)) and d.startswith("P")
    )

    pacientes_max_0 = []
    pacientes_faltan_archivo = []
    pacientes_error = []

    for paciente in pacientes:
        ruta_pulpa = os.path.join(BASE_DIR, paciente, "gt_pulp_mandible.nii.gz")

        if not os.path.exists(ruta_pulpa):
            pacientes_faltan_archivo.append(paciente)
            continue

        try:
            mascara_pulpa = nib.load(ruta_pulpa).get_fdata()
            valor_maximo = np.max(mascara_pulpa)

            if valor_maximo == 0:
                pacientes_max_0.append(paciente)
        except Exception:
            pacientes_error.append(paciente)

    print("Pacientes con np.max(gt_pulp_mandible) == 0:")
    if pacientes_max_0:
        for paciente in pacientes_max_0:
            print(f"- {paciente}")
    else:
        print("Ninguno")

    print(f"\nTotal con max == 0: {len(pacientes_max_0)}")
    print(f"Total sin archivo gt_pulp_mandible.nii.gz: {len(pacientes_faltan_archivo)}")
    print(f"Total con error de lectura: {len(pacientes_error)}")

    df = pd.DataFrame({"paciente": pacientes_max_0})
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCSV guardado en: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
