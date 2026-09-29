import nibabel as nib # para trabajar con archivos nii
import matplotlib.pyplot as plt
import os
import glob # para buscar archivos con patrones de nombres 
from matplotlib.widgets import Slider #para el control interactivo
import numpy as np 

BASE_DIR = r'c:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation'
folder = os.path.join(BASE_DIR, 'datasets', 'Pulpy3D', 'P35')
nii_files = glob.glob(os.path.join(folder, '*.nii.gz')) #busca todos los archivos con extension .nii.gz

if not os.path.isdir(folder):
    print(f'La carpeta no existe: {folder}')
    raise SystemExit(0)

if not nii_files:
    print(f'No hay archivos .nii.gz en: {folder}')
    raise SystemExit(0)

num_files = len(nii_files)


fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 filas, 4 columnas, tamaño 20x10 pulgadas
axes = axes.flatten() # matriz 2D de ejes en una lista 1D

fig2_main = None  # placeholder para la segunda figura
last_index = -1

for i, file_path in enumerate(nii_files):
    if i >= 8:  # si hay más de 8, solo mostramos los primeros 8
        break
    last_index = i
    img = nib.load(file_path) # carga el archivo nii
    data = img.get_fdata() # extrae los datos numericos

    #coronal_slice = data[data.shape[0]//2, :, :]
    #sagital_slice = data[:, data.shape[1]//2, :]
    middle_slice = data[:, :, data.shape[2]//2] # visualizar un corte medio de la imagen original
    axes[i].imshow(middle_slice, cmap='gray') #muestra  la imagen en escala de grises
    axes[i].set_title(os.path.basename(file_path)) #titlo
    axes[i].axis('off') #quita ejes

gt_instance_path = os.path.join(folder, 'gt_instance.nii.gz') # cargamos el archivo 3D completo
tooth_number = 36  # numero de diente a visualizar (formato IPE)


if os.path.exists(gt_instance_path): #verifica que el archivo existe y lo carga
    img_gt = nib.load(gt_instance_path)
    data_gt = img_gt.get_fdata()
    
    print(f"Shape del archivo: {data_gt.shape}")
    print(f"Valores únicos en data_gt: {np.unique(data_gt)}")
    
    primer_molar_mask = (data_gt == tooth_number) #crear mascara booleana donde solo el diente 36 es True
    print(f"Pixeles con valor {tooth_number}: {primer_molar_mask.sum()} píxeles")
    
    fig2, ax = plt.subplots(figsize=(10, 10))
    fig2.patch.set_facecolor('white') 
    ax.set_facecolor('white') 
    plt.subplots_adjust(bottom=0.3)
    
    max_slices = data_gt.shape[2] #calcula cuantas capas  tiene
    slice_idx = max_slices // 2  # empieza en el medio
    
    data_slice = data_gt[:, :, slice_idx]
    im = ax.imshow(data_slice, cmap='gray')
    
    mask_slice = primer_molar_mask[:, :, slice_idx].astype(float) # superponer la máscara del diente seleccionado en rojo
    im_mask = ax.imshow(mask_slice, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
    
    ax.set_title(f'gt_instance_jan.nii.gz', fontsize=14, color='black')
    ax.tick_params(colors='black')
    
    ax_slider = plt.axes([0.15, 0.15, 0.7, 0.03], facecolor='lightgray') #crea slider 
    slider = Slider(ax_slider, 'Capa', 0, max_slices - 1, valinit=slice_idx, valstep=1)
    
    def update(val): # cada vez que meuves la mascara del slider se actualiza la imagen
        idx = int(slider.val)
        
        data_slice_new = data_gt[:, :, idx]
        im.set_data(data_slice_new)
        
        mask_data = primer_molar_mask[:, :, idx].astype(float)
        im_mask.set_data(mask_data)
        
        pixel_count = mask_data.sum()
        ax.set_title(f'gt_instance_jan.nii.gz - Capa {idx}/{max_slices-1} ({int(pixel_count)} píxeles)', fontsize=14, color='black')
        fig2.canvas.draw_idle()
    
    slider.on_changed(update)
    fig2_main = fig2
else:
    print(f"Archivo gt_instance.nii.gz no encontrado en {folder}")

for j in range(last_index + 1, 8):
    axes[j].axis('off')

plt.show()

if fig2_main is not None:
    fig2_main.show()
