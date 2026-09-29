"""
Script para comparar la segmentación de pulpy_mandible e instancia en un paciente.

- Carga gt_pulp.nii.gz y gt_instance.nii.gz
- Binariza ambas máscaras
- Calcula el DICE por cada slice en el eje Z, considerando solo los cortes donde alguna máscara tiene más de 5 píxeles
- Muestra el corte con menor DICE en matplotlib
- Imprime el índice del peor slice, el DICE y el número de píxeles diferentes en ese corte
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Rutas de los archivos (ajusta a tu paciente)
pulpy_path = r'C:/Users/clauu/OneDrive - Universidad Pontificia Comillas/Documentos/ICAI/Beca IIT/1_Data_Validation/datasets/Pulpy3D/P459/gt_pulp_mandible.nii.gz'
instancia_path = r'C:/Users/clauu/OneDrive - Universidad Pontificia Comillas/Documentos/ICAI/Beca IIT/1_Data_Validation/datasets/Pulpy3D/P459/gt_instance.nii.gz'

# Cargar datos
pulpy_img = nib.load(pulpy_path)
pulpy_data = pulpy_img.get_fdata()
instancia_img = nib.load(instancia_path)
instancia_data = instancia_img.get_fdata()


# Binarizar
pulpy_mask = (pulpy_data > 0).astype(np.uint8)
instancia_mask = (instancia_data > 0).astype(np.uint8)


# DICE por slice
num_slices_pulpy = pulpy_mask.shape[2]
num_slices_instancia = instancia_mask.shape[2]
print(f"Slices Pulpy: {num_slices_pulpy} | Slices Instancia: {num_slices_instancia}")

if num_slices_pulpy != num_slices_instancia:
    raise ValueError("Pulpy e Instancia no tienen el mismo número de slices en Z.")

num_slices = num_slices_pulpy
dice_slices = []
valid_slices = []
common_slices = []
min_pixels = 5

def dice_score(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    return 2. * intersection / (np.sum(mask1) + np.sum(mask2) + 1e-8)

pulpy_pixels_per_slice = np.sum(pulpy_mask, axis=(0, 1))
instancia_pixels_per_slice = np.sum(instancia_mask, axis=(0, 1))

for i in range(num_slices):
    m1 = pulpy_mask[:, :, i]
    m2 = instancia_mask[:, :, i]
    if (pulpy_pixels_per_slice[i] > min_pixels) or (instancia_pixels_per_slice[i] > min_pixels):
        d = dice_score(m1, m2)
        dice_slices.append(d)
        valid_slices.append(i)
    if (pulpy_pixels_per_slice[i] > min_pixels) and (instancia_pixels_per_slice[i] > min_pixels):
        common_slices.append(i)

if len(dice_slices) == 0:
    print("No hay slices con suficiente información para comparar.")
    exit()

if len(common_slices) > 0:
    dice_common = [dice_score(pulpy_mask[:, :, i], instancia_mask[:, :, i]) for i in common_slices]
    worst_idx = common_slices[int(np.argmin(dice_common))]
    worst_dice = dice_common[int(np.argmin(dice_common))]
    criterio = "(ambas máscaras > min_pixels)"
else:
    worst_idx = valid_slices[int(np.argmin(dice_slices))]
    worst_dice = dice_slices[int(np.argmin(dice_slices))]
    criterio = "(al menos una máscara > min_pixels)"

num_diff_pixels = np.sum(pulpy_mask[:, :, worst_idx] != instancia_mask[:, :, worst_idx])
print(f"Peor slice: {worst_idx} (DICE={worst_dice:.4f}) {criterio} | Diferencias de píxeles: {num_diff_pixels}")
print(f"Pixeles Pulpy en slice {worst_idx}: {int(pulpy_pixels_per_slice[worst_idx])}")
print(f"Pixeles Instancia en slice {worst_idx}: {int(instancia_pixels_per_slice[worst_idx])}")

nonzero_pulpy = np.where(pulpy_pixels_per_slice > 0)[0]
nonzero_instancia = np.where(instancia_pixels_per_slice > 0)[0]
if len(nonzero_pulpy) > 0 and len(nonzero_instancia) > 0:
    print(f"Rango slices con contenido Pulpy: {nonzero_pulpy[0]} - {nonzero_pulpy[-1]}")
    print(f"Rango slices con contenido Instancia: {nonzero_instancia[0]} - {nonzero_instancia[-1]}")

#Valores de affine y únicos
print("¿Arrays idénticos?:", np.array_equal(pulpy_mask, instancia_mask))
print("Máxima diferencia absoluta:", np.max(np.abs(pulpy_mask - instancia_mask)))

print("Pulpy suma:", np.sum(pulpy_mask))
print("Instancia suma:", np.sum(instancia_mask))

diff_vol = pulpy_mask - instancia_mask
print("Total de píxeles diferentes en todo el volumen:", np.sum(diff_vol != 0))

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(pulpy_mask[:, :, worst_idx], cmap='gray', vmin=0, vmax=1)
plt.title('Pulpy Mandible')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(instancia_mask[:, :, worst_idx], cmap='gray', vmin=0, vmax=1)
plt.title('Instancia (binaria)')
plt.axis('off')

plt.subplot(1, 3, 3)
diff = pulpy_mask[:, :, worst_idx] - instancia_mask[:, :, worst_idx]
plt.imshow(diff, cmap='bwr', vmin=-1, vmax=1)
plt.title('Diferencia')
plt.axis('off')

plt.suptitle(
    f'Slice {worst_idx} | DICE={worst_dice:.4f} | Diff={num_diff_pixels} | '
    f'Pulpy={int(pulpy_pixels_per_slice[worst_idx])} | Instancia={int(instancia_pixels_per_slice[worst_idx])}'
)
plt.tight_layout()
plt.show()