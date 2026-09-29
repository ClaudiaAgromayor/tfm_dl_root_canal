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

# Rotar 180 grados
ian_manual_rotated = np.rot90(ian_manual, k=2, axes=(0, 1))

print(f"Data shape: {data.shape}")
print(f"Data min/max: {data.min():.1f} / {data.max():.1f}")
print(f"Pulpy3D IAN vóxeles: {(gt_ian > 0).sum()}")
print(f"Manual IAN vóxeles (original): {ian_manual.sum()}")
print(f"Manual IAN vóxeles (rotado 180°): {ian_manual_rotated.sum()}")

# Usar la versión rotada
ian_manual = ian_manual_rotated

# ========== CREAR VISUALIZACIÓN ==========
print("\nCreando visualización...")

# Seleccionar un corte en el medio
z_slice = data.shape[2] // 2

# Normalizar la imagen para mejor visualización
data_slice = data[:, :, z_slice]
# Aplicar windowing para DICOM
data_normalized = np.clip((data_slice + 1000) / 2000, 0, 1)

# Segmentaciones en este corte
pulpy_slice = gt_ian[:, :, z_slice]
manual_slice = ian_manual[:, :, z_slice]

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== 1. Imagen original ==========
ax = axes[0, 0]
ax.imshow(data_normalized, cmap='gray')
ax.set_title(f'Imagen Original (Corte Z={z_slice})', fontsize=12, fontweight='bold')
ax.axis('on')

# ========== 2. Overlay Pulpy3D ==========
ax = axes[0, 1]
ax.imshow(data_normalized, cmap='gray')
# Overlay en rojo
pulpy_mask = pulpy_slice > 0
ax.imshow(np.ma.masked_where(~pulpy_mask, pulpy_mask), cmap='Reds', alpha=0.6)
ax.set_title(f'Pulpy3D IAN Overlay (Rojo)', fontsize=12, fontweight='bold')
ax.axis('on')

# ========== 3. Overlay Manual ==========
ax = axes[1, 0]
ax.imshow(data_normalized, cmap='gray')
# Overlay en azul
manual_mask = manual_slice > 0
ax.imshow(np.ma.masked_where(~manual_mask, manual_mask), cmap='Blues', alpha=0.6)
ax.set_title(f'Manual IAN Overlay (Azul)', fontsize=12, fontweight='bold')
ax.axis('on')

# ========== 4. Ambos overlays combinados ==========
ax = axes[1, 1]
ax.imshow(data_normalized, cmap='gray')

# Crear imagen RGB para mostrar ambas segmentaciones
overlay = np.zeros((*pulpy_slice.shape, 3))

# Rojo: solo Pulpy3D
pulpy_only = pulpy_mask & ~manual_mask
overlay[pulpy_only] = [1, 0, 0]  # Rojo

# Azul: solo Manual
manual_only = manual_mask & ~pulpy_mask
overlay[manual_only] = [0, 0, 1]  # Azul

# Morado: ambos (solapamiento)
both = pulpy_mask & manual_mask
overlay[both] = [1, 0, 1]  # Morado

# Mostrar overlay
ax.imshow(overlay, alpha=0.7)
ax.set_title(f'Comparación (Rojo=Pulpy3D, Azul=Manual, Morado=Ambos)', fontsize=12, fontweight='bold')
ax.axis('on')

# Añadir leyenda
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.6, label=f'Pulpy3D solo ({pulpy_only.sum()} vóx)'),
    Patch(facecolor='blue', alpha=0.6, label=f'Manual solo ({manual_only.sum()} vóx)'),
    Patch(facecolor='magenta', alpha=0.6, label=f'Solapamiento ({both.sum()} vóx)')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
output_dir = os.path.join(p35_pulpy_path, "..", "..", "..", "results", "exploracion")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "P35_comparacion_visual_pulpy_vs_manual.png")
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Imagen guardada: {output_file}")
plt.show()

# ========== ESTADÍSTICAS ==========
print("\n" + "="*70)
print("ESTADÍSTICAS DEL CORTE Z=" + str(z_slice))
print("="*70)
print(f"Vóxeles Pulpy3D: {pulpy_mask.sum()}")
print(f"Vóxeles Manual: {manual_mask.sum()}")
print(f"Solapamiento: {both.sum()}")
print(f"Solo Pulpy3D: {pulpy_only.sum()}")
print(f"Solo Manual: {manual_only.sum()}")

if both.sum() > 0:
    pct_pulpy = (both.sum() / pulpy_mask.sum() * 100) if pulpy_mask.sum() > 0 else 0
    pct_manual = (both.sum() / manual_mask.sum() * 100) if manual_mask.sum() > 0 else 0
    print(f"\nSolapamiento % Pulpy3D: {pct_pulpy:.1f}%")
    print(f"Solapamiento % Manual: {pct_manual:.1f}%")
