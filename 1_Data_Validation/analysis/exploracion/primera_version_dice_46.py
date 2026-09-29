import nibabel as nib
import numpy as np
import nrrd
import matplotlib.pyplot as plt
import os
import csv
import itertools


def dice_score(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    return 2. * intersection / (np.sum(mask1) + np.sum(mask2) + 1e-8)


def align_shape_with_permutations(seg_array, target_shape):
    if seg_array.shape == target_shape:
        return seg_array, "identity"

    for perm in itertools.permutations(range(3)):
        candidate = np.transpose(seg_array, perm)
        if candidate.shape == target_shape:
            return candidate, f"transpose{perm}"

    return None, "no_shape_match"


def load_good_patients(csv_path):
    patients = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get("paciente", "").strip()
            if p:
                patients.append(p)
    return patients


base_seg = r'C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\2_UNet\datasets\Segmentaciones 3DSlicer'
base_pulpy = r'C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\Pulpy3D'
good_csv = r'C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation\results\dataset_checks\pacientes_dice_bueno.csv'

if os.path.exists(good_csv):
    good_patients = load_good_patients(good_csv)
    print(f"\n[INFO] Pacientes buenos en CSV: {len(good_patients)}")

    for patient in good_patients:
        seg_patient_dir = os.path.join(base_seg, patient)
        instance_path_patient = os.path.join(base_pulpy, patient, 'gt_instance.nii.gz')

        if not os.path.isdir(seg_patient_dir):
            continue
        if not os.path.exists(instance_path_patient):
            print(f"[SKIP] {patient}: no existe gt_instance.nii.gz")
            continue

        seg_files = sorted([f for f in os.listdir(seg_patient_dir) if f.endswith('.seg.nrrd')])
        if len(seg_files) == 0:
            print(f"[SKIP] {patient}: no hay .seg.nrrd")
            continue

        seg_path_patient = os.path.join(seg_patient_dir, seg_files[0])

        instance_img_patient = nib.load(instance_path_patient)
        instance_data_patient = np.rint(instance_img_patient.get_fdata()).astype(np.int32)

        seg_data_patient, _ = nrrd.read(seg_path_patient)
        fer_data_rotated =np.flip(np.flip(np.flip(seg_data_patient, axis=1), axis=0), axis=0)

        seg_aligned, align_mode = align_shape_with_permutations(fer_data_rotated, instance_data_patient.shape)
        if seg_aligned is None:
            print(f"[SKIP] {patient}: shape incompatible ({fer_data_rotated.shape} vs {instance_data_patient.shape})")
            continue

        seg_unique = np.unique(seg_aligned)
        seg_unique_nonzero = [int(v) for v in seg_unique if v > 0]

        if len(seg_unique_nonzero) >= 3:
            seg_mask_patient = ((seg_aligned == 1) | (seg_aligned == 2)).astype(np.uint8)
            labels_rule = "seg=[0,1,2,3...] -> usar 1 y 2"
        elif len(seg_unique_nonzero) == 2:
            seg_mask_patient = (seg_aligned == 1).astype(np.uint8)
            labels_rule = "seg=[0,1,2] -> usar solo 1"
        elif len(seg_unique_nonzero) == 1:
            seg_mask_patient = (seg_aligned == seg_unique_nonzero[0]).astype(np.uint8)
            labels_rule = f"seg=[0,{seg_unique_nonzero[0]}] -> usar {seg_unique_nonzero[0]}"
        else:
            seg_mask_patient = np.zeros_like(seg_aligned, dtype=np.uint8)
            labels_rule = "seg vacía"

        instancia_mask_patient = (instance_data_patient == 46).astype(np.uint8)
        dice_patient = dice_score(seg_mask_patient, instancia_mask_patient)

        scores_z = (
            np.count_nonzero(seg_mask_patient, axis=(0, 1))
            + np.count_nonzero(instancia_mask_patient, axis=(0, 1))
        )
        slice_idx_patient = int(np.argmax(scores_z)) if scores_z.size > 0 else 0

        print(
            f"{patient}: DICE={dice_patient:.4f} | align={align_mode} | "
            f"seg_file={seg_files[0]} | regla={labels_rule}"
        )

        plt.figure(figsize=(8, 8))
        plt.imshow(instancia_mask_patient[:, :, slice_idx_patient], cmap='Blues', alpha=0.55)
        plt.imshow(seg_mask_patient[:, :, slice_idx_patient], cmap='autumn', alpha=0.55)
        plt.title(
            f"{patient} | DICE={dice_patient:.4f} | z={slice_idx_patient}\n"
            f"Instance azul | Segmentación rojo"
        )
        plt.axis('off')
        plt.show()

else:
    print(f"\n[WARN] No existe el CSV de pacientes buenos: {good_csv}")
################################################################################

