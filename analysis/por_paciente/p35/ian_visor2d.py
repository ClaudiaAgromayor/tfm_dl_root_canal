import nibabel as nib
import nrrd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --------------------------------------------------
# Paths
# --------------------------------------------------

gt_path = r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation\datasets\Pulpy3D\P35\gt_ian.nii.gz"

seg_path = r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\2_UNet\datasets\Segmentaciones 3DSlicer\P35\Segmentation.seg.nrrd"

# --------------------------------------------------
# Load data
# --------------------------------------------------

gt = nib.load(gt_path).get_fdata()
seg, header = nrrd.read(seg_path)

print("GT shape:", gt.shape)
print("SEG shape:", seg.shape)
print("GT unique values:", np.unique(gt))
print("SEG unique values:", np.unique(seg))


# --------------------------------------------------
# Keep only IAN label
# --------------------------------------------------

seg_ian = (seg == 2).astype(np.uint8)

print("IAN voxels:", seg_ian.sum())


from scipy.ndimage import label, center_of_mass

# --------------------------------------------------
# KEEP ONLY ONE IAN (LEFT or RIGHT)
# --------------------------------------------------

keep_side = "left"   # cambia a "left" si quieres el otro

x_mid = seg_ian.shape[0] // 2  # eje izquierda/derecha

if keep_side == "right":
    seg_ian = seg_ian[x_mid:, :, :]
    gt = gt[x_mid:, :, :]
elif keep_side == "left":
    seg_ian = seg_ian[:x_mid, :, :]
    gt = gt[:x_mid, :, :]

# --------------------------------------------------
# Check shape consistency
# --------------------------------------------------

if gt.shape != seg_ian.shape:
    raise ValueError(
        f"Shapes differ:\nGT: {gt.shape}\nSEG: {seg_ian.shape}"
    )

# --------------------------------------------------
# OPTION 1: ONLY FLIP (mirror correction)
# --------------------------------------------------

gt = np.flip(gt, axis=2)
seg_ian = np.flip(seg_ian, axis=2)

gt = np.flip(gt, axis=1)


# --------------------------------------------------
# Viewer
# --------------------------------------------------

slice_idx = gt.shape[2] // 2

fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.15)

img_gt = ax.imshow(
    gt[:, :, slice_idx],
    cmap="Blues",
    alpha=0.5
)

img_seg = ax.imshow(
    seg_ian[:, :, slice_idx],
    cmap="Reds",
    alpha=0.5
)

ax.set_title(f"Slice {slice_idx}")

slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])

slider = Slider(
    slider_ax,
    "Slice",
    0,
    gt.shape[2] - 1,
    valinit=slice_idx,
    valstep=1
)

def update(val):
    s = int(slider.val)

    img_gt.set_data(gt[:, :, s])
    img_seg.set_data(seg_ian[:, :, s])

    ax.set_title(f"Slice {s}")
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()