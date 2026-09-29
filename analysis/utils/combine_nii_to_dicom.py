"""
Script para combinar data.nii.gz con gt_pulp.nii.gz y convertir a DICOM
La pulpa se resalta con un valor alto para que sea visible en la reconstrucción 3D
"""

import nibabel as nib
import numpy as np
import os
import glob
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import datetime


def create_combined_dicom(patient_folder, output_folder, pulp_intensity=3000):
    """
    Combina data.nii.gz con gt_pulp.nii.gz y genera una serie DICOM única.
    La pulpa se superpone con un valor de intensidad alto para destacar en 3D.
    
    Args:
        patient_folder: Carpeta del paciente (ej: Pulpy3D/P1)
        output_folder: Carpeta donde guardar los DICOM combinados
        pulp_intensity: Valor de intensidad para la pulpa (default: 3000 HU)
    """
    patient_name = os.path.basename(patient_folder)
    
    # Buscar archivos necesarios
    data_path = os.path.join(patient_folder, 'data.nii.gz')
    pulp_path = os.path.join(patient_folder, 'gt_pulp.nii.gz')
    
    # Verificar que existen los archivos
    if not os.path.exists(data_path):
        print(f"  ERROR: No se encontró data.nii.gz en {patient_folder}")
        return 0
    
    if not os.path.exists(pulp_path):
        print(f"  AVISO: No se encontró gt_pulp.nii.gz, usando solo data.nii.gz")
        pulp_path = None
    
    # Cargar data (imagen CT)
    print(f"  Cargando data.nii.gz...")
    data_img = nib.load(data_path)
    data = data_img.get_fdata()
    header = data_img.header
    pixdim = header.get_zooms()
    
    # Si existe pulp, combinarlo
    if pulp_path:
        print(f"  Cargando gt_pulp.nii.gz...")
        pulp_img = nib.load(pulp_path)
        pulp_data = pulp_img.get_fdata()
        
        # Crear máscara booleana de la pulpa
        pulp_mask = pulp_data > 0
        
        # Combinar: donde hay pulpa, poner valor alto
        combined_data = data.copy()
        combined_data[pulp_mask] = pulp_intensity
        
        print(f"  Pulpa detectada: {pulp_mask.sum()} voxels")
    else:
        combined_data = data
    
    # Crear carpeta de salida
    patient_output = os.path.join(output_folder, patient_name)
    os.makedirs(patient_output, exist_ok=True)
    
    # Generar UIDs
    study_instance_uid = generate_uid()
    series_instance_uid = generate_uid()
    frame_of_reference_uid = generate_uid()
    
    # Fecha y hora
    dt = datetime.datetime.now()
    date_str = dt.strftime('%Y%m%d')
    time_str = dt.strftime('%H%M%S.%f')
    
    # Normalizar datos para DICOM
    data_min = combined_data.min()
    data_max = combined_data.max()
    
    if data_max != data_min:
        scaled_data = ((combined_data - data_min) / (data_max - data_min) * 65535).astype(np.uint16)
    else:
        scaled_data = np.zeros_like(combined_data, dtype=np.uint16)
    
    rescale_slope = (data_max - data_min) / 65535 if data_max != data_min else 1.0
    rescale_intercept = data_min
    
    num_slices = scaled_data.shape[2]
    print(f"  Generando {num_slices} slices DICOM combinados...")
    
    for slice_idx in range(num_slices):
        slice_data = scaled_data[:, :, slice_idx]
        slice_data = np.rot90(slice_data)
        
        filename = f"slice_{slice_idx:04d}.dcm"
        filepath = os.path.join(patient_output, filename)
        
        # Crear FileDataset
        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()
        
        ds = FileDataset(filepath, {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # Información del paciente
        ds.PatientName = patient_name
        ds.PatientID = patient_name
        ds.PatientBirthDate = ""
        ds.PatientSex = ""
        
        # Información del estudio
        ds.StudyInstanceUID = study_instance_uid
        ds.StudyDate = date_str
        ds.StudyTime = time_str
        ds.StudyDescription = "Teeth + Pulp Combined"
        ds.AccessionNumber = ""
        ds.ReferringPhysicianName = ""
        ds.StudyID = "1"
        
        # Información de la serie
        ds.SeriesInstanceUID = series_instance_uid
        ds.SeriesNumber = 1
        ds.SeriesDescription = "data_with_pulp"
        ds.Modality = "CT"
        ds.SeriesDate = date_str
        ds.SeriesTime = time_str
        
        # Frame of Reference
        ds.FrameOfReferenceUID = frame_of_reference_uid
        ds.PositionReferenceIndicator = ""
        
        # Información de la imagen
        ds.SOPClassUID = pydicom.uid.CTImageStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.InstanceNumber = slice_idx + 1
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        
        # Dimensiones
        ds.Rows = slice_data.shape[0]
        ds.Columns = slice_data.shape[1]
        ds.PixelSpacing = [float(pixdim[0]), float(pixdim[1])]
        ds.SliceThickness = float(pixdim[2]) if len(pixdim) > 2 else 1.0
        ds.SpacingBetweenSlices = float(pixdim[2]) if len(pixdim) > 2 else 1.0
        
        # Posición del slice
        ds.ImagePositionPatient = [0.0, 0.0, float(slice_idx * pixdim[2]) if len(pixdim) > 2 else float(slice_idx)]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.SliceLocation = float(slice_idx * pixdim[2]) if len(pixdim) > 2 else float(slice_idx)
        
        # Información de píxeles
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        
        # Rescale
        ds.RescaleIntercept = float(rescale_intercept)
        ds.RescaleSlope = float(rescale_slope)
        ds.RescaleType = "HU"
        
        # Window/Level para visualización (ajustado para ver la pulpa)
        ds.WindowCenter = float(pulp_intensity / 2)
        ds.WindowWidth = float(pulp_intensity)
        
        # Datos de píxeles
        ds.PixelData = slice_data.tobytes()
        
        ds.ContentDate = date_str
        ds.ContentTime = time_str
        
        ds.save_as(filepath)
    
    print(f"  ¡Completado! {num_slices} slices guardados en {patient_output}")
    return num_slices


def main():
    # Carpetas
    pulpy3d_folder = r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\Pulpy3D"
    output_folder = r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation\results\dicom\Pulpy3D_DICOM_Combined"
    
    os.makedirs(output_folder, exist_ok=True)
    
    # patient_folders = [os.path.join(pulpy3d_folder, 'P1')]
    patient_folders = sorted(glob.glob(os.path.join(pulpy3d_folder, 'P*')))
    
    print(f"Procesando {len(patient_folders)} pacientes")
    print(f"Salida: {output_folder}")
    print("=" * 60)
    
    for patient_folder in patient_folders:
        if os.path.isdir(patient_folder):
            patient_name = os.path.basename(patient_folder)
            print(f"\nProcesando {patient_name}...")
            try:
                create_combined_dicom(patient_folder, output_folder)
            except Exception as e:
                print(f"  ERROR: {patient_name} saltado - {e}")
    
    print("\n" + "=" * 60)
    print("¡Conversión completada!")
    print(f"Archivos DICOM combinados en: {output_folder}")
    print("\nAbre la carpeta del paciente (ej: P1/) directamente en Carestream")


if __name__ == "__main__":
    main()
