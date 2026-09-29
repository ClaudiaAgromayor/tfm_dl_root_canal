"""
Script para comparar el coeficiente DICE entre las máscaras pulp_mandible e instance
de TODOS los pacientes de la carpeta Pulpy3D.

Para cada paciente, compara:
- gt_pulp_mandible.nii.gz (máscara binaria de la pulpa mandibular)
- gt_instance.nii.gz (máscara de instancias, binarizada)

Output:
- CSV con el DICE de todos los pacientes procesados.
- Conteo de pacientes con DICE > 0.9999 y DICE <= 0.9999.
"""
import os
import nibabel as nib
import numpy as np
import pandas as pd

# Ruta base de los pacientes
base_dir = r'C:/Users/clauu/OneDrive - Universidad Pontificia Comillas/Documentos/ICAI/Beca IIT/1_Data_Validation/datasets/Pulpy3D'
output_csv = r'C:/Users/clauu/OneDrive - Universidad Pontificia Comillas/Documentos/ICAI/Beca IIT/1_Data_Validation/results/dataset_checks/dice_pulp_vs_instance.csv'

# Listar carpetas de pacientes
all_patients = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('P')]
all_patients = sorted(all_patients)

results = []
count_gt_09999 = 0
count_le_09999 = 0

def dice_score(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    return 2. * intersection / (np.sum(mask1) + np.sum(mask2) + 1e-8)

for patient in all_patients:
    pulp_path = os.path.join(base_dir, patient, 'gt_pulp_mandible.nii.gz')
    inst_path = os.path.join(base_dir, patient, 'gt_instance.nii.gz')
    if not (os.path.exists(pulp_path) and os.path.exists(inst_path)):
        print(f"Saltando {patient}: falta algún archivo.")
        results.append({'paciente': patient, 'dice': np.nan, 'estado': 'faltan_archivos'})
        continue
    try:
        pulp_img = nib.load(pulp_path)
        pulp_data = pulp_img.get_fdata()
        inst_img = nib.load(inst_path)
        inst_data = inst_img.get_fdata()
        # Binarizar
        pulp_mask = (pulp_data > 0).astype(np.uint8)
        inst_mask = (inst_data > 0).astype(np.uint8)
        if pulp_mask.shape != inst_mask.shape:
            print(f"Saltando {patient}: shapes distintas {pulp_mask.shape} vs {inst_mask.shape}.")
            results.append({'paciente': patient, 'dice': np.nan, 'estado': 'shape_distinta'})
            continue
        # DICE global
        dice = dice_score(pulp_mask, inst_mask)
        results.append({'paciente': patient, 'dice': dice, 'estado': 'ok'})

        if dice > 0.9999:
            count_gt_09999 += 1
        else:
            count_le_09999 += 1

        print(f"{patient}: DICE = {dice:.4f}")
    except Exception as e:
        print(f"Error en {patient}: {e}")
        results.append({'paciente': patient, 'dice': np.nan, 'estado': f'error: {e}'})

# Guardar resultados
results_df = pd.DataFrame(results)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
results_df.to_csv(output_csv, index=False)
print(f"Resultados guardados en {output_csv}")
print(f"Pacientes con DICE > 0.9999: {count_gt_09999}")
print(f"Pacientes con DICE <= 0.9999: {count_le_09999}")
