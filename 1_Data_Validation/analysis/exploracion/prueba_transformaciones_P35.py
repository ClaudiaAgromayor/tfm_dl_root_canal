import nibabel as nib
import nrrd
import numpy as np
import matplotlib.pyplot as plt
import os

# Rutas
p35_pulpy_path = r"c:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\Pulpy3D\P35"
p35_slicer_path = r"c:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\2_UNet\datasets\Segmentaciones 3DSlicer\P35"

# ========== CARGAR DATOS ==========
print("Cargando datos...")

# Data original
data_img = nib.load(os.path.join(p35_pulpy_path, "data.nii.gz"))
data = data_img.get_fdata()

# IAN Pulpy3D
gt_ian_img = nib.load(os.path.join(p35_pulpy_path, "gt_ian.nii.gz"))
gt_ian = gt_ian_img.get_fdata()

# IAN Manual (3DSlicer)
seg_data, _ = nrrd.read(os.path.join(p35_slicer_path, "Segmentation.seg.nrrd"))
ian_manual = (seg_data == 2).astype(float)

print(f"Data shape: {data.shape}")
print(f"Pulpy3D shape: {gt_ian.shape}")
print(f"Manual shape: {ian_manual.shape}")

# Probar diferentes transformaciones (solo en cada corte 2D, sin cambiar dimensión Z)
transformations = {
    'Original': ian_manual,
    'Flip X': np.fliplr(ian_manual),
    'Flip Y': np.flipud(ian_manual),
    'Flip X+Y': np.flipud(np.fliplr(ian_manual)),
}

print("\n" + "="*70)
print("PROBANDO TRANSFORMACIONES")
print("="*70)

z_slice = gt_ian.shape[2] // 2

best_overlap = 0
best_transform = None

for name, transformed in transformations.items():
    pulpy_slice = gt_ian[:, :, z_slice]
    manual_slice = transformed[:, :, z_slice]
    
    pulpy_mask = pulpy_slice > 0
    manual_mask = manual_slice > 0
    
    overlap = np.logical_and(pulpy_mask, manual_mask).sum()
    
    print(f"{name:<15} - Solapamiento: {overlap} vóxeles")
    
    if overlap > best_overlap:
        best_overlap = overlap
        best_transform = (name, transformed)

print(f"\n✓ Mejor transformación: {best_transform[0]} con {best_overlap} vóxeles de solapamiento")

# Visualizar la mejor
print("\nCreando visualización con mejor transformación...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Normalizar data
data_slice = data[:, :, z_slice]
data_normalized = np.clip((data_slice + 1000) / 2000, 0, 1)

pulpy_slice = gt_ian[:, :, z_slice]
manual_slice = best_transform[1][:, :, z_slice]

# 1. Imagen original
ax = axes[0, 0]
ax.imshow(data_normalized, cmap='gray')
ax.set_title(f'Imagen Original (Z={z_slice})', fontsize=12, fontweight='bold')

# 2. Pulpy3D overlay
ax = axes[0, 1]
ax.imshow(data_normalized, cmap='gray')
pulpy_mask = pulpy_slice > 0
ax.imshow(np.ma.masked_where(~pulpy_mask, pulpy_mask), cmap='Reds', alpha=0.6)
ax.set_title(f'Pulpy3D IAN (Rojo)', fontsize=12, fontweight='bold')

# 3. Manual overlay (transformado)
ax = axes[1, 0]
ax.imshow(data_normalized, cmap='gray')
manual_mask = manual_slice > 0
ax.imshow(np.ma.masked_where(~manual_mask, manual_mask), cmap='Blues', alpha=0.6)
ax.set_title(f'Manual IAN - {best_transform[0]} (Azul)', fontsize=12, fontweight='bold')

# 4. Ambos combinados
ax = axes[1, 1]
ax.imshow(data_normalized, cmap='gray')

overlay = np.zeros((*pulpy_slice.shape, 3))
pulpy_only = pulpy_mask & ~manual_mask
manual_only = manual_mask & ~pulpy_mask
both = pulpy_mask & manual_mask

overlay[pulpy_only] = [1, 0, 0]
overlay[manual_only] = [0, 0, 1]
overlay[both] = [1, 0, 1]

ax.imshow(overlay, alpha=0.7)
ax.set_title(f'Comparación (Rojo=Pulpy3D, Azul=Manual, Morado=Ambos)', fontsize=11, fontweight='bold')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.6, label=f'Solo Pulpy3D ({pulpy_only.sum()} vóx)'),
    Patch(facecolor='blue', alpha=0.6, label=f'Solo Manual ({manual_only.sum()} vóx)'),
    Patch(facecolor='magenta', alpha=0.6, label=f'Solapamiento ({both.sum()} vóx)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
output_dir = os.path.join(p35_pulpy_path, "..", "..", "..", "results", "exploracion")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "P35_comparacion_transformaciones.png")
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ Guardado: {output_file}")
plt.show()

print("\n" + "="*70)
print(f"MEJOR TRANSFORMACIÓN: {best_transform[0]}")
print("="*70)
print(f"Solapamiento: {best_overlap} vóxeles")
print(f"Pulpy3D vóxeles en corte: {pulpy_mask.sum()}")
print(f"Manual vóxeles en corte: {manual_mask.sum()}")
if best_overlap > 0:
    pct_pulpy = (best_overlap / pulpy_mask.sum() * 100) if pulpy_mask.sum() > 0 else 0
    pct_manual = (best_overlap / manual_mask.sum() * 100) if manual_mask.sum() > 0 else 0
    print(f"% Solapamiento en Pulpy3D: {pct_pulpy:.1f}%")
    print(f"% Solapamiento en Manual: {pct_manual:.1f}%")
