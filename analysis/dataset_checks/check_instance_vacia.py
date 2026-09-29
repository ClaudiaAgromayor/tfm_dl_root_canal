"""
Recorre todos los pacientes de datasets/Pulpy3D y comprueba si la máscara
gt_instance.nii.gz contiene información.

Criterio:
- Si np.max(mascara_instance) == 0, se considera que la instance está vacía.

Salida:
- Imprime la lista de pacientes con instance vacía.
- Guarda esa lista en un CSV: pacientes_instance_vacios.csv
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
    r"Beca IIT/1_Data_Validation/results/dataset_checks/pacientes_instance_vacios.csv"
)


def main() -> None:
    pacientes = sorted(
        d
        for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d)) and d.startswith("P")
    )

    pacientes_instance_vacia = []
    pacientes_faltan_archivo = []
    pacientes_error = []

    for paciente in pacientes:
        ruta_instance = os.path.join(BASE_DIR, paciente, "gt_instance.nii.gz")

        if not os.path.exists(ruta_instance):
            pacientes_faltan_archivo.append(paciente)
            continue

        try:
            mascara_instance = nib.load(ruta_instance).get_fdata()
            valor_maximo = np.max(mascara_instance)

            if valor_maximo == 0:
                pacientes_instance_vacia.append(paciente)
        except Exception:
            pacientes_error.append(paciente)

    print("Pacientes con np.max(gt_instance) == 0:")
    if pacientes_instance_vacia:
        for paciente in pacientes_instance_vacia:
            print(f"- {paciente}")
    else:
        print("Ninguno")

    print(f"\nTotal con instance vacía (max == 0): {len(pacientes_instance_vacia)}")
    print(f"Total sin archivo gt_instance.nii.gz: {len(pacientes_faltan_archivo)}")
    print(f"Total con error de lectura: {len(pacientes_error)}")

    df = pd.DataFrame({"paciente": pacientes_instance_vacia})
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCSV guardado en: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
