import nibabel as nib
import nrrd
import numpy as np
from skimage import measure
import plotly.graph_objects as go

from scipy.spatial.distance import cdist
import os
import matplotlib.pyplot as plt

# --------------------------------------------------
# PATHS
# --------------------------------------------------

gt_path = r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\Pulpy3D\P476\gt_ian.nii.gz"

seg_path = r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\2_UNet\datasets\Segmentaciones 3DSlicer\P476\Segmentation.seg.nrrd"

save_dir = r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation\results\por_paciente\p476"
os.makedirs(save_dir, exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

gt = nib.load(gt_path).get_fdata()
seg, _ = nrrd.read(seg_path)

# --------------------------------------------------
# KEEP IAN ONLY
# --------------------------------------------------

seg_ian = (seg == 2).astype(np.uint8)

# --------------------------------------------------
# OPTIONAL: keep one side
# --------------------------------------------------

keep_side = "right"
x_mid = seg_ian.shape[0] // 2

if keep_side == "right":
    gt = gt[x_mid:, :, :]
    seg_ian = seg_ian[x_mid:, :, :]
elif keep_side == "left":
    gt = gt[:x_mid, :, :]
    seg_ian = seg_ian[:x_mid, :, :]

# --------------------------------------------------
# FLIP (alignment fix)
# --------------------------------------------------

gt = np.flip(gt, axis=2)
seg_ian = np.flip(seg_ian, axis=2)
gt = np.flip(gt, axis=1)

# --------------------------------------------------
# HD95 2D SLICE-WISE (ADDED)
# --------------------------------------------------

def hd95_2d_slicewise(gt, seg):
    hd_slices = []
    z_list = []

    for z in range(gt.shape[2]):
        gt_slice = gt[:, :, z]
        seg_slice = seg[:, :, z]

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
        z_list.append(z)

    if len(hd_slices) == 0:
        return np.nan, None

    idx = int(np.argmax(hd_slices))
    return hd_slices[idx], z_list[idx]



# --------------------------------------------------
# MARCHING CUBES
# --------------------------------------------------

verts_gt, faces_gt, _, _ = measure.marching_cubes(gt, level=0.5)
verts_seg, faces_seg, _, _ = measure.marching_cubes(seg_ian, level=0.5)

# --------------------------------------------------
# PLOTLY MESHES
# --------------------------------------------------

fig = go.Figure()

# GT (blue)
fig.add_trace(go.Mesh3d(
    x=verts_gt[:, 0],
    y=verts_gt[:, 1],
    z=verts_gt[:, 2],
    i=faces_gt[:, 0],
    j=faces_gt[:, 1],
    k=faces_gt[:, 2],
    color='blue',
    opacity=0.4,
    name='GT'
))

# SEG (red)
fig.add_trace(go.Mesh3d(
    x=verts_seg[:, 0],
    y=verts_seg[:, 1],
    z=verts_seg[:, 2],
    i=faces_seg[:, 0],
    j=faces_seg[:, 1],
    k=faces_seg[:, 2],
    color='red',
    opacity=0.4,
    name='IAN Seg'
))

# --------------------------------------------------
# LAYOUT
# --------------------------------------------------

fig.update_layout(
    title="3D Visualization - GT vs IAN (Interactive)",
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    width=900,
    height=800
)

# --------------------------------------------------
# SHOW IN BROWSER
# --------------------------------------------------

fig.show()

fig.write_html(os.path.join(save_dir, "ian_3d_visualization.html"), include_plotlyjs="cdn")