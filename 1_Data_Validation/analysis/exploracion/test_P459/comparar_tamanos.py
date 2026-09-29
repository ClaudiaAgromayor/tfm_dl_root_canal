import numpy as np
import nrrd
import nibabel as nib


DATA_NRRD_PATH = (
    r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas"
    r"\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\P459_3DSlicer_prueba\1 data.nrrd"
)

SEG_NRRD_PATH = (
    r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas"
    r"\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\P459_3DSlicer_prueba\Segmentation_1.seg.nrrd"
)

DATA_NII_PATH = (
    r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas"
    r"\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\Pulpy3D\P459\data.nii.gz"
)

GT_PULP_NII_PATH = (
    r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas"
    r"\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\Pulpy3D\P459\gt_pulp.nii.gz"
)


def extract_spacing_from_nrrd(header):
    if "spacings" in header and header["spacings"] is not None:
        return header["spacings"]

    if "space directions" in header and header["space directions"] is not None:
        dirs = np.array(header["space directions"], dtype=object)
        spacing = []
        for d in dirs:
            if d is None:
                spacing.append(None)
            else:
                arr = np.array(d, dtype=float)
                spacing.append(float(np.linalg.norm(arr)))
        return spacing

    return None


def to_3d_label_volume(seg):
    if seg.ndim == 3:
        return seg
    if seg.ndim == 4:
        label_volume = np.zeros(seg.shape[1:], dtype=np.uint8)
        for i in range(seg.shape[0]):
            label_volume[seg[i] > 0] = i + 1
        return label_volume
    raise ValueError(f"Dimensión no soportada para segmentación: {seg.ndim}")


def main():
    data_nrrd_raw, data_nrrd_header = nrrd.read(DATA_NRRD_PATH)
    seg_raw, seg_header = nrrd.read(SEG_NRRD_PATH)
    data_img = nib.load(DATA_NII_PATH)
    gt_img = nib.load(GT_PULP_NII_PATH)
    data = data_img.get_fdata()
    gt = gt_img.get_fdata()

    data_nrrd = to_3d_label_volume(data_nrrd_raw)
    seg = to_3d_label_volume(seg_raw)

    print("\n[1 data.nrrd]")
    print(f"  Shape original: {data_nrrd_raw.shape}")
    print(f"  Shape usado (3D): {data_nrrd.shape}")
    print(f"  Dtype: {data_nrrd_raw.dtype}")
    print(f"  Voxeles totales (3D): {data_nrrd.size:,}")
    print(f"  Spacing (mm): {extract_spacing_from_nrrd(data_nrrd_header)}")

    print("\n[Segmentation_1.seg.nrrd]")
    print(f"  Shape original: {seg_raw.shape}")
    print(f"  Shape usado (3D): {seg.shape}")
    print(f"  Dtype: {seg_raw.dtype}")
    print(f"  Voxeles totales (3D): {seg.size:,}")
    print(f"  Spacing (mm): {extract_spacing_from_nrrd(seg_header)}")

    print("\n[data.nii]")
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    print(f"  Voxeles totales: {data.size:,}")
    print(f"  Spacing (mm): {data_img.header.get_zooms()}")

    print("\n[gt_pulp.nii]")
    print(f"  Shape: {gt.shape}")
    print(f"  Dtype: {gt.dtype}")
    print(f"  Voxeles totales: {gt.size:,}")
    print(f"  Spacing (mm): {gt_img.header.get_zooms()}")

if __name__ == "__main__":
    main()
