import nibabel as nib
import nrrd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import os

from skimage import measure
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist

print(">>> EJECUTANDO VERSION NUEVA <<<")

# --------------------------------------------------
# RENDER SETUP (IMPORTANT FOR WINDOWS)
# --------------------------------------------------
pio.renderers.default = "browser"

# --------------------------------------------------
# PATHS
# --------------------------------------------------
gt_path = r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\Pulpy3D\P448\gt_ian.nii.gz"

seg_path = r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\2_UNet\datasets\Segmentaciones 3DSlicer\P448\Segmentation.seg.nrrd"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
gt = nib.load(gt_path).get_fdata()
seg, _ = nrrd.read(seg_path)

# --------------------------------------------------
# BINARY MASKS
# --------------------------------------------------
gt = (gt > 0).astype(np.uint8)
seg = (seg == 2).astype(np.uint8)   # IAN label

# --------------------------------------------------
# SIDE SELECTION (igual que plot3d, ANTES del flip)
# --------------------------------------------------
keep_side = "right"   # <-- cambia aquí: "right", "left", o "both"
x_mid = seg.shape[0] // 2

if keep_side == "right":
    gt = gt[x_mid:, :, :]
    seg = seg[x_mid:, :, :]
elif keep_side == "left":
    gt = gt[:x_mid, :, :]
    seg = seg[:x_mid, :, :]
# si keep_side == "both", no se hace nada

# --------------------------------------------------
# ALIGNMENT (igual que plot3d, DESPUÉS del side selection)
# --------------------------------------------------
gt = np.flip(gt, axis=2)
seg = np.flip(seg, axis=2)
gt = np.flip(gt, axis=1)

# --------------------------------------------------
# CROP Y (tu recorte manual)
# --------------------------------------------------
gt = gt[:, 78:234, :]
seg = seg[:, 78:234, :]

# --------------------------------------------------
# OVERLAP (Dice logic)
# --------------------------------------------------
tp = (gt == 1) & (seg == 1)
fp = (seg == 1) & (gt == 0)
fn = (gt == 1) & (seg == 0)

# --------------------------------------------------
# SURFACE EXTRACTION
# --------------------------------------------------
def surface(mask):
    mask = mask.astype(bool)
    return mask ^ binary_erosion(mask)

tp_s = surface(tp)
fp_s = surface(fp)
fn_s = surface(fn)
gt_s = surface(gt)
seg_s = surface(seg)

# --------------------------------------------------
# HD95 FUNCTION (slicing along Y axis)
# --------------------------------------------------
def hd95_2d_slicewise(gt, seg):
    hd_slices = []
    y_list = []

    for y in range(gt.shape[1]):
        gt_slice = gt[:, y, :]
        seg_slice = seg[:, y, :]

        if gt_slice.sum() == 0 and seg_slice.sum() == 0:
            continue

        if gt_slice.sum() == 0 or seg_slice.sum() == 0:
            continue

        gt_pts = np.array(np.where(gt_slice)).T
        seg_pts = np.array(np.where(seg_slice)).T

        if len(gt_pts) == 0 or len(seg_pts) == 0:
            continue

        d1 = cdist(gt_pts, seg_pts)
        d2 = cdist(seg_pts, gt_pts)

        hd1 = np.percentile(np.min(d1, axis=1), 95)
        hd2 = np.percentile(np.min(d2, axis=1), 95)

        hd_slices.append(max(hd1, hd2))
        y_list.append(y)

    if len(hd_slices) == 0:
        return np.nan, None

    idx = int(np.argmax(hd_slices))

    worst_hd = hd_slices[idx]
    worst_y = y_list[idx]

    return worst_hd, worst_y


hd95_val, worst_y = hd95_2d_slicewise(gt, seg)
# Y del corte en la segmentación de 3D Slicer = inicio del recorte + índice dentro del recorte
Y_CROP_START = 78
worst_y_original = worst_y + Y_CROP_START if worst_y is not None else None
print("DEBUG -> hd95_val:", hd95_val, "worst_y (recortado):", worst_y, "worst_y (original):", worst_y_original)

if worst_y is None:
    print("⚠️ No se pudo calcular HD95: no hay slices con GT y SEG simultáneamente.")
    print("GT total voxels:", gt.sum(), "| SEG total voxels:", seg.sum())
else:
    save_dir = r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation\results\por_paciente\p448"
    os.makedirs(save_dir, exist_ok=True)

    data_path = r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\Pulpy3D\P448\data.nii.gz"
    data = nib.load(data_path).get_fdata()

    # mismo side selection que gt y seg
    if keep_side == "right":
        data = data[x_mid:, :, :]
    elif keep_side == "left":
        data = data[:x_mid, :, :]

    data = np.flip(data, axis=2)
    data = np.flip(data, axis=1)
    data = data[:, 78:234, :]

    data_slice = data[:, int(worst_y), :]
    gt_slice = gt[:, int(worst_y), :]
    seg_slice = seg[:, int(worst_y), :]

    gt_pts = np.array(np.where(gt_slice)).T
    seg_pts = np.array(np.where(seg_slice)).T

    d1 = cdist(gt_pts, seg_pts)
    d2 = cdist(seg_pts, gt_pts)

    min_d1 = np.min(d1, axis=1)
    min_d2 = np.min(d2, axis=1)

    p95_1 = np.percentile(min_d1, 95)
    p95_2 = np.percentile(min_d2, 95)

    if p95_1 >= p95_2:
        idx_a = (np.abs(min_d1 - p95_1)).argmin()
        pt_a = gt_pts[idx_a]
        idx_b = d1[idx_a].argmin()
        pt_b = seg_pts[idx_b]
    else:
        idx_a = (np.abs(min_d2 - p95_2)).argmin()
        pt_a = seg_pts[idx_a]
        idx_b = d2[idx_a].argmin()
        pt_b = gt_pts[idx_b]

    save_path = os.path.join(save_dir, f"p448_worst_hd95_y{worst_y_original}.png")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(data_slice.T, cmap='gray', origin='lower')

    ax.imshow(np.ma.masked_where(gt_slice == 0, gt_slice).T,
              cmap='Blues', alpha=0.5, origin='lower')
    ax.imshow(np.ma.masked_where(seg_slice == 0, seg_slice).T,
              cmap='Reds', alpha=0.5, origin='lower')

    ax.plot([pt_a[0], pt_b[0]], [pt_a[1], pt_b[1]],
            color='yellow', linewidth=2, marker='o', markersize=4)

    ax.set_title(f"Worst HD95 slice Y={worst_y_original} | HD95={hd95_val:.2f}")

    ax.axis('off')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print("✅ Saved image at:", save_path)

# --------------------------------------------------
# MARCHING CUBES (SAFE)
# --------------------------------------------------
def mesh(mask):
    mask = mask.astype(np.float32)
    if mask.sum() == 0:
        return None, None
    return measure.marching_cubes(mask, 0.5)[:2]

v_gt, f_gt = mesh(gt)
v_seg, f_seg = mesh(seg)

if v_gt is None or v_seg is None:
    raise ValueError("Empty mesh after marching cubes")

def dice_3d(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    return 2 * inter / (a.sum() + b.sum() + 1e-8)

dice = dice_3d(gt, seg)
print("DICE 3D:", dice)

# --------------------------------------------------
# PLOT
# --------------------------------------------------
fig = go.Figure()

# GT
fig.add_trace(go.Mesh3d(
    x=v_gt[:, 0], y=v_gt[:, 1], z=v_gt[:, 2],
    i=f_gt[:, 0], j=f_gt[:, 1], k=f_gt[:, 2],
    color='blue', opacity=0.3, name='GT'
))

# PRED
fig.add_trace(go.Mesh3d(
    x=v_seg[:, 0], y=v_seg[:, 1], z=v_seg[:, 2],
    i=f_seg[:, 0], j=f_seg[:, 1], k=f_seg[:, 2],
    color='red', opacity=0.3, name='Pred'
))

# HD95 visualization
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode='markers',
    marker=dict(size=0.1, color='yellow'),
    name='HD95 (computed per slice)'
))

# --------------------------------------------------
# LAYOUT
# --------------------------------------------------
fig.update_layout(
    title="IAN GT vs Prediction (Dice + HD95 visualization)",
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data',
        xaxis=dict(autorange=True),
        yaxis=dict(autorange=True),
        zaxis=dict(autorange=True),
    ),
    width=900,
    height=800
)

fig.show()