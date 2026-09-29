import numpy as np
import nrrd
import nibabel as nib
import matplotlib.pyplot as plt


DATA_NII_PATH = (
    r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas"
    r"\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\Pulpy3D\P459\data.nii.gz"
)

DATA_NRRD_PATH = (
    r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas"
    r"\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\P459_3DSlicer_prueba\1 data.nrrd"
)

SEG1_NRRD_PATH = (
    r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas"
    r"\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\P459_3DSlicer_prueba\Segmentation_1.seg.nrrd"
)

GT_PULP_PATH = (
    r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas"
    r"\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\Pulpy3D\P459\gt_pulp.nii.gz"
)

OUTPUT_PANEL_DATA = "results/exploracion/test_P459/comparacion_side.png"
OUTPUT_BLACK_SEG1_GT = "results/exploracion/test_P459/superposicion.png"
MANUAL_SLICE_Z = 69  # Ejemplo: 64. Si es None, usa la mejor capa automática.
OVERLAY_BACKGROUND_MODE = "nii"  # "black", "nrrd" o "nii"
DIFF_COLOR = (1.0, 0.35, 0.0)  # naranja/rojo fosforito para zonas de más
SEG_LABEL_VALUES = (1,)  # p.ej. (1,) para solo Segment0; (2,) para Segment1; (1, 2) para ambos
SEG_CROP_BBOX = None  # (x_min, x_max, y_min, y_max, z_min, z_max), límites inclusivos


def to_3d_label_volume(seg): #convierte segmentacion .nrrd a 3D
    if seg.ndim == 3:
        return seg
    if seg.ndim == 4:
        label_volume = np.zeros(seg.shape[1:], dtype=np.uint8)
        for i in range(seg.shape[0]):
            label_volume[seg[i] > 0] = i + 1
        return label_volume
    raise ValueError(f"Segmentación NRRD con dimensión no soportada: {seg.ndim}")


def apply_crop_bbox(mask, crop_bbox):
    if crop_bbox is None:
        return mask

    if len(crop_bbox) != 6:
        raise ValueError("SEG_CROP_BBOX debe tener 6 valores: (x_min, x_max, y_min, y_max, z_min, z_max)")

    x_min, x_max, y_min, y_max, z_min, z_max = [int(v) for v in crop_bbox]
    sx, sy, sz = mask.shape

    if not (0 <= x_min <= x_max < sx and 0 <= y_min <= y_max < sy and 0 <= z_min <= z_max < sz):
        raise ValueError(
            f"SEG_CROP_BBOX fuera de rango para shape {mask.shape}: {crop_bbox}"
        )

    out = np.zeros_like(mask, dtype=bool)
    out[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1] = mask[
        x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1
    ]
    return out


def load_slicer_seg(seg_path, labels_to_keep=None, crop_bbox=None):
    seg_raw, _ = nrrd.read(seg_path)
    seg_3d = to_3d_label_volume(seg_raw)

    if labels_to_keep is None:
        mask = seg_3d > 0
    else:
        labels = tuple(int(v) for v in labels_to_keep)
        if len(labels) == 0:
            raise ValueError("SEG_LABEL_VALUES no puede estar vacío")
        mask = np.isin(seg_3d, labels)

    return apply_crop_bbox(mask, crop_bbox)


def mirror_y(mask):
    return np.flip(mask, axis=1)


def load_gt(gt_path):
    gt_img = nib.load(gt_path)
    return gt_img.get_fdata() > 0.5


def load_data_nrrd_3d(data_path):
    data_raw, _ = nrrd.read(data_path)
    if data_raw.ndim == 3:
        return data_raw
    if data_raw.ndim == 4:
        return data_raw[0]
    raise ValueError(f"NRRD de imagen con dimensión no soportada: {data_raw.ndim}")


def best_slice(mask_a, mask_b): #coge mejor corte axial para visualizar el que tiene mas voxeles 
    union = np.logical_or(mask_a, mask_b)
    s = union.sum(axis=(0, 1))
    if np.all(s == 0):
        return union.shape[2] // 2
    return int(np.argmax(s))


def choose_slice(mask_a, mask_b, manual_z=None):
    if manual_z is None:
        return best_slice(mask_a, mask_b)
    if not (0 <= manual_z < mask_a.shape[2]):
        raise ValueError(f"MANUAL_SLICE_Z fuera de rango: {manual_z}. Rango válido: [0, {mask_a.shape[2]-1}]")
    return int(manual_z)


def format_slice(slice_2d):
    return slice_2d.T


def normalize_data_for_display(data):
    d = data.astype(np.float32)
    valid = d[d > 0]
    if valid.size > 0:
        low, high = np.percentile(valid, [1, 99])
    else:
        low, high = float(d.min()), float(d.max())
    d = np.clip(d, low, high)
    return (d - low) / (high - low + 1e-8)


def save_overlay_black_pair(mask_a, mask_b, label_a, label_b, out_path, bg_data=None, bg_label="Fondo negro"):
    z = choose_slice(mask_a, mask_b, MANUAL_SLICE_Z)
    a = format_slice(mask_a[:, :, z])
    b = format_slice(mask_b[:, :, z])

    ov = np.zeros((*a.shape, 4), dtype=float)
    ov[a, 0] = 1.0
    ov[b, 1] = 1.0
    ov[np.logical_or(a, b), 3] = 0.95

    plt.figure(figsize=(8, 8))
    if bg_data is not None:
        bg = format_slice(bg_data[:, :, z])
        plt.imshow(bg, cmap="gray", origin="lower")
    plt.imshow(ov, origin="lower")
    plt.title(f"{bg_label} | Rojo={label_a} | Verde={label_b} (z={z})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_panel_on_data(data, seg1, gt, out_path):
    z = choose_slice(seg1, gt, MANUAL_SLICE_Z)
    d = format_slice(data[:, :, z])
    s1 = format_slice(seg1[:, :, z])
    g = format_slice(gt[:, :, z])
    seg_only = np.logical_and(s1, np.logical_not(g))
    gt_only = np.logical_and(g, np.logical_not(s1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(d, cmap="gray", origin="lower")
    ov_seg1 = np.zeros((*d.shape, 4), dtype=float)
    ov_seg1[s1, 2] = 1.0
    ov_seg1[s1, 3] = 0.6
    ov_seg1[seg_only, 0] = DIFF_COLOR[0]
    ov_seg1[seg_only, 1] = DIFF_COLOR[1]
    ov_seg1[seg_only, 2] = DIFF_COLOR[2]
    ov_seg1[seg_only, 3] = 0.95
    axes[0].imshow(ov_seg1, origin="lower")
    axes[0].set_title(f"Segmentation_1.seg sobre data.nii (z={z}) | exceso=rojo/naranja")
    axes[0].axis("off")

    axes[1].imshow(d, cmap="gray", origin="lower")
    ov_gt = np.zeros((*d.shape, 4), dtype=float)
    ov_gt[g, 1] = 1.0
    ov_gt[g, 3] = 0.6
    ov_gt[gt_only, 0] = DIFF_COLOR[0]
    ov_gt[gt_only, 1] = DIFF_COLOR[1]
    ov_gt[gt_only, 2] = DIFF_COLOR[2]
    ov_gt[gt_only, 3] = 0.95
    axes[1].imshow(ov_gt, origin="lower")
    axes[1].set_title(f"GT sobre data.nii (z={z}) | exceso=rojo/naranja")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    data_img = nib.load(DATA_NII_PATH)
    data_norm = normalize_data_for_display(data_img.get_fdata())
    data_nrrd_norm = normalize_data_for_display(load_data_nrrd_3d(DATA_NRRD_PATH))

    gt_bin = load_gt(GT_PULP_PATH)

    seg1 = load_slicer_seg(
        SEG1_NRRD_PATH,
        labels_to_keep=SEG_LABEL_VALUES,
        crop_bbox=SEG_CROP_BBOX,
    )
    seg1 = mirror_y(seg1)

    expected_shape = data_img.shape[:3]
    if seg1.shape != expected_shape or gt_bin.shape != expected_shape:
        raise ValueError(
            f"Shapes distintos. data={expected_shape}, seg1={seg1.shape}, gt={gt_bin.shape}"
        )

    if data_nrrd_norm.shape != expected_shape:
        raise ValueError(
            f"Shape de data.nrrd distinto. esperado={expected_shape}, data_nrrd={data_nrrd_norm.shape}"
        )

    bg_mode = OVERLAY_BACKGROUND_MODE.lower().strip()
    if bg_mode == "black":
        bg_data = None
        bg_label = "Fondo negro"
    elif bg_mode == "nrrd":
        bg_data = data_nrrd_norm
        bg_label = "Fondo 1 data.nrrd"
    elif bg_mode == "nii":
        bg_data = data_norm
        bg_label = "Fondo data.nii"
    else:
        raise ValueError("OVERLAY_BACKGROUND_MODE debe ser 'black', 'nrrd' o 'nii'")

    save_panel_on_data(data_norm, seg1, gt_bin, OUTPUT_PANEL_DATA)
    save_overlay_black_pair(
        seg1,
        gt_bin,
        "Segmentation_1.seg",
        "GT",
        OUTPUT_BLACK_SEG1_GT,
        bg_data=bg_data,
        bg_label=bg_label,
    )

    print(f"PNG panel sobre data (seg1 vs gt): {OUTPUT_PANEL_DATA}")
    print(f"PNG fondo negro seg1 vs gt: {OUTPUT_BLACK_SEG1_GT}")
    print(f"SEG_LABEL_VALUES usados: {SEG_LABEL_VALUES}")
    print(f"SEG_CROP_BBOX usada: {SEG_CROP_BBOX}")


if __name__ == "__main__":
    main()
