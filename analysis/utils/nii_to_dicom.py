"""
Script para convertir archivos NIfTI (.nii.gz) a DICOM
Recorre todas las carpetas P1, P2, ... en Pulpy3D y convierte cada archivo .nii.gz a una serie DICOM
"""

import nibabel as nib
import numpy as np
import os
import glob
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from pydicom.sequence import Sequence
import datetime
import tempfile

def create_dicom_from_nifti(nifti_path, output_folder, patient_id="Patient", series_description=None, study_instance_uid=None, frame_of_reference_uid=None):
    """
    Convierte un archivo NIfTI a una serie de archivos DICOM (uno por slice).
    
    Args:
        nifti_path: Ruta al archivo .nii o .nii.gz
        output_folder: Carpeta donde guardar los archivos DICOM
        patient_id: ID del paciente para los metadatos DICOM
        series_description: Descripción de la serie (si es None, usa el nombre del archivo)
        study_instance_uid: UID del estudio (compartido por paciente)
        frame_of_reference_uid: UID del frame of reference (compartido por paciente)
    """
    # Cargar el archivo NIfTI
    nii_img = nib.load(nifti_path)
    nii_data = nii_img.get_fdata()
    header = nii_img.header
    affine = nii_img.affine
    
    # Obtener información del header NIfTI
    pixdim = header.get_zooms()  # Dimensiones del voxel (spacing)
    
    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # Generar UIDs únicos para la serie
    if study_instance_uid is None:
        study_instance_uid = generate_uid()
    if frame_of_reference_uid is None:
        frame_of_reference_uid = generate_uid()
    series_instance_uid = generate_uid()
    
    # Nombre del archivo como descripción de la serie
    if series_description is None:
        series_description = os.path.basename(nifti_path).replace('.nii.gz', '').replace('.nii', '')
    
    # Fecha y hora actuales
    dt = datetime.datetime.now()
    date_str = dt.strftime('%Y%m%d')
    time_str = dt.strftime('%H%M%S.%f')
    
    # Normalizar los datos para DICOM (convertir a enteros)
    # DICOM típicamente usa valores enteros de 16 bits
    data_min = nii_data.min()
    data_max = nii_data.max()
    
    if data_max != data_min:
        # Escalar a rango 0-65535 para uint16
        scaled_data = ((nii_data - data_min) / (data_max - data_min) * 65535).astype(np.uint16)
    else:
        scaled_data = np.zeros_like(nii_data, dtype=np.uint16)
    
    # Calcular slope e intercept para recuperar valores originales
    rescale_slope = (data_max - data_min) / 65535 if data_max != data_min else 1.0
    rescale_intercept = data_min
    
    num_slices = scaled_data.shape[2]
    
    print(f"  Convirtiendo {os.path.basename(nifti_path)}: {scaled_data.shape} -> {num_slices} slices DICOM")
    
    # Crear un archivo DICOM por cada slice
    for slice_idx in range(num_slices):
        # Extraer el slice
        slice_data = scaled_data[:, :, slice_idx]
        
        # Transponer para que la orientación sea correcta
        slice_data = np.rot90(slice_data)
        
        # Crear un archivo temporal (requerido por FileDataset)
        suffix = '.dcm'
        filename = f"slice_{slice_idx:04d}{suffix}"
        filepath = os.path.join(output_folder, filename)
        
        # Crear el FileDataset
        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()
        
        ds = FileDataset(filepath, {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # Información del paciente
        ds.PatientName = patient_id
        ds.PatientID = patient_id
        ds.PatientBirthDate = ""
        ds.PatientSex = ""
        
        # Información del estudio
        ds.StudyInstanceUID = study_instance_uid
        ds.StudyDate = date_str
        ds.StudyTime = time_str
        ds.StudyDescription = "NIfTI to DICOM Conversion"
        ds.AccessionNumber = ""
        ds.ReferringPhysicianName = ""
        ds.StudyID = "1"
        
        # Información de la serie
        ds.SeriesInstanceUID = series_instance_uid
        ds.SeriesNumber = 1
        ds.SeriesDescription = series_description
        ds.Modality = "CT"  # O "MR" dependiendo del tipo de imagen
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
        
        # Dimensiones de la imagen
        ds.Rows = slice_data.shape[0]
        ds.Columns = slice_data.shape[1]
        ds.PixelSpacing = [float(pixdim[0]), float(pixdim[1])]
        ds.SliceThickness = float(pixdim[2]) if len(pixdim) > 2 else 1.0
        ds.SpacingBetweenSlices = float(pixdim[2]) if len(pixdim) > 2 else 1.0
        
        # Posición del slice
        ds.ImagePositionPatient = [0.0, 0.0, float(slice_idx * pixdim[2]) if len(pixdim) > 2 else float(slice_idx)]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.SliceLocation = float(slice_idx * pixdim[2]) if len(pixdim) > 2 else float(slice_idx)
        
        # Información de los píxeles
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0  # unsigned
        
        # Rescale para recuperar valores originales
        ds.RescaleIntercept = float(rescale_intercept)
        ds.RescaleSlope = float(rescale_slope)
        ds.RescaleType = "HU"
        
        # Window/Level para visualización
        ds.WindowCenter = float((data_max + data_min) / 2)
        ds.WindowWidth = float(data_max - data_min) if data_max != data_min else 1.0
        
        # Datos de píxeles
        ds.PixelData = slice_data.tobytes()
        
        # Establecer la fecha de creación
        ds.ContentDate = date_str
        ds.ContentTime = time_str
        
        # Guardar el archivo DICOM
        ds.save_as(filepath)
    
    return num_slices


def convert_folder(folder_path, output_base_folder):
    """
    Convierte todos los archivos NIfTI en una carpeta a DICOM.
    
    Args:
        folder_path: Ruta a la carpeta con archivos .nii.gz (ej: Pulpy3D/P1)
        output_base_folder: Carpeta base donde crear las subcarpetas DICOM
    """
    folder_name = os.path.basename(folder_path)
    print(f"\nProcesando carpeta: {folder_name}")
    
    # Buscar todos los archivos NIfTI
    nii_files = glob.glob(os.path.join(folder_path, '*.nii.gz'))
    nii_files += glob.glob(os.path.join(folder_path, '*.nii'))
    
    if not nii_files:
        print(f"  No se encontraron archivos NIfTI en {folder_path}")
        return
    
    # Crear carpeta de salida para este paciente
    patient_output_folder = os.path.join(output_base_folder, folder_name)
    os.makedirs(patient_output_folder, exist_ok=True)
    
    # Generar UIDs compartidos por paciente
    study_instance_uid = generate_uid()
    frame_of_reference_uid = generate_uid()
    
    total_slices = 0
    for nii_file in nii_files:
        # Nombre del archivo sin extensión para la subcarpeta
        file_name = os.path.basename(nii_file).replace('.nii.gz', '').replace('.nii', '')
        series_output_folder = os.path.join(patient_output_folder, file_name)
        
        try:
            num_slices = create_dicom_from_nifti(
                nii_file, 
                series_output_folder, 
                patient_id=folder_name,
                series_description=file_name,
                study_instance_uid=study_instance_uid,
                frame_of_reference_uid=frame_of_reference_uid
            )
            total_slices += num_slices
        except Exception as e:
            print(f"  Error convirtiendo {nii_file}: {str(e)}")
    
    print(f"  Total: {len(nii_files)} archivos NIfTI -> {total_slices} slices DICOM")


def main():
    # Carpeta base con los datos NIfTI
    pulpy3d_folder = r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\Pulpy3D"
    
    # Carpeta de salida para los DICOM
    output_folder = r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation\results\dicom\Pulpy3D_DICOM"
    
    # Crear carpeta de salida
    os.makedirs(output_folder, exist_ok=True)
    
    # Buscar todas las carpetas de pacientes (P1, P2, P3, ...)
    # MODO TEST: solo procesar la primera carpeta (P1)
    # patient_folders = [os.path.join(pulpy3d_folder, 'P1')]
    # Para procesar todas, comenta las siguientes 2 líneas y descomenta la línea original
    patient_folders = sorted(glob.glob(os.path.join(pulpy3d_folder, 'P*')))
    
    print(f"Encontradas {len(patient_folders)} carpetas de pacientes")
    print(f"Carpeta de salida: {output_folder}")
    print("=" * 60)
    
    for patient_folder in patient_folders:
        if os.path.isdir(patient_folder):
            convert_folder(patient_folder, output_folder)
    
    print("\n" + "=" * 60)
    print("¡Conversión completada!")
    print(f"Los archivos DICOM se han guardado en: {output_folder}")


if __name__ == "__main__":
    main()
