"""
Compare IAN segmentations (manual vs Pulpy3D) on vertical (Y-axis) slices.
Generates per-slice pixel overlap analysis and representative visualizations.
"""

import os
import nibabel as nib
import nrrd
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Configuration
BASE_PATH = r"c:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos\ICAI\Beca IIT\1_Data_Validation"
RESULTS_DIR = os.path.join(BASE_PATH, "results", "ian", "cortes_verticales")
os.makedirs(RESULTS_DIR, exist_ok=True)
HD95_RESULTS_DIR = os.path.join(BASE_PATH, "results", "ian", "hd95_cortes_verticales")
os.makedirs(HD95_RESULTS_DIR, exist_ok=True)

# Patients to analyze
PATIENTS = ["P195"]


def dice_3d(a, b):
    """Compute 3D DICE coefficient."""
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum()
    return 2.0 * inter / (a.sum() + b.sum() + 1e-8)


def hd95_3d(mask_a, mask_b):
    """Compute 3D Hausdorff 95 distance using surface voxels."""
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    if np.count_nonzero(mask_a) == 0 or np.count_nonzero(mask_b) == 0:
        return float("nan")

    def surface_voxels(mask):
        structure = ndimage.generate_binary_structure(mask.ndim, 1)
        return np.logical_xor(mask, ndimage.binary_erosion(mask, structure=structure, border_value=0))

    def directed_surface_distances(source, target):
        source_surface = surface_voxels(source)
        target_surface = surface_voxels(target)
        if not np.any(source_surface) or not np.any(target_surface):
            return np.array([], dtype=np.float32)
        distance_map = ndimage.distance_transform_edt(~target_surface)
        return distance_map[source_surface]

    distances = np.concatenate(
        [
            directed_surface_distances(mask_a, mask_b),
            directed_surface_distances(mask_b, mask_a),
        ]
    )
    if distances.size == 0:
        return float("nan")
    return float(np.percentile(distances, 95))


def get_overlap_metrics(seg1, seg2):
    """
    Compute overlap metrics between two segmentations.
    Returns: intersection, union, dice, jaccard
    """
    seg1_bin = seg1 > 0
    seg2_bin = seg2 > 0
    inter = np.logical_and(seg1_bin, seg2_bin).sum()
    union = np.logical_or(seg1_bin, seg2_bin).sum()
    dice = 2.0 * inter / (seg1_bin.sum() + seg2_bin.sum() + 1e-8)
    jaccard = inter / (union + 1e-8) if union > 0 else 0
    return inter, union, dice, jaccard


def load_patient_data(patient_id):
    """Load manual (Slicer) and Pulpy3D segmentations for a patient."""
    pulpy_path = os.path.join(BASE_PATH, "datasets", "Pulpy3D", patient_id)
    slicer_path = os.path.join(os.path.dirname(BASE_PATH), "2_UNet", "datasets", "Segmentaciones 3DSlicer", patient_id)
    
    try:
        # Load Pulpy3D GT
        gt_ian_img = nib.load(os.path.join(pulpy_path, "gt_ian.nii.gz"))
        gt_ian = gt_ian_img.get_fdata().astype(np.uint8)
        
        # Load manual (Slicer)
        seg_file = os.path.join(slicer_path,  "Segmentation-label.nrrd")
        seg_data, _ = nrrd.read(seg_file)
        manual = (seg_data == 2).astype(np.uint8)
        
        # Load original data for visualization
        data_img = nib.load(os.path.join(pulpy_path, "data.nii.gz"))
        data = data_img.get_fdata()
        
        return gt_ian, manual, data, gt_ian_img.affine
    except Exception as e:
        print(f"  ERROR loading {patient_id}: {e}")
        return None, None, None, None


def find_best_manual_variant(manual, gt_ian):
    """Try original and flipped, choose best overlap."""
    variants = {
        'original': manual,
        'flip_x': np.fliplr(manual)
    }
    
    best_name = 'original'
    best_overlap = 0
    
    for name, m in variants.items():
        ov = np.logical_and(m > 0, gt_ian > 0).sum()
        if ov > best_overlap:
            best_overlap = ov
            best_name = name
    
    return variants[best_name], best_name


def select_same_side_components(gt_ian, manual):
    """Select only Pulpy3D components on same side as manual."""
    # Detect manual side
    coords = np.array(np.where(manual > 0))
    if coords.size == 0:
        return np.zeros_like(gt_ian, dtype=np.uint8)
    
    center_x = coords[0].mean()
    mid_x = gt_ian.shape[0] / 2.0
    manual_side = 'Right' if center_x < mid_x else 'Left'
    
    # Label and select matching side
    labeled, ncomp = ndimage.label(gt_ian > 0)
    gt_matched = np.zeros_like(gt_ian, dtype=np.uint8)
    
    for comp in range(1, ncomp + 1):
        comp_mask = (labeled == comp)
        coords = np.array(np.where(comp_mask))
        cx = coords[0].mean()
        side = 'Right' if cx < mid_x else 'Left'
        if side == manual_side:
            gt_matched[comp_mask] = 1
    
    return gt_matched


def analyze_vertical_slices(pulpy, manual, patient_id):
    """Analyze Y-axis slices (vertical/coronal)."""
    y_max = pulpy.shape[1]
    slice_data = []
    
    for y in range(y_max):
        pulpy_slice = pulpy[:, y, :]
        manual_slice = manual[:, y, :]
        
        inter, union, dice, jaccard = get_overlap_metrics(pulpy_slice, manual_slice)
        
        pulpy_voxels = int((pulpy_slice > 0).sum())
        manual_voxels = int((manual_slice > 0).sum())
        
        slice_data.append({
            'Y': y,
            'Intersection_px': int(inter),
            'Union_px': int(union),
            'Pulpy_voxels': pulpy_voxels,
            'Manual_voxels': manual_voxels,
            'DICE': dice,
            'Jaccard': jaccard
        })
    
    return pd.DataFrame(slice_data)


def select_representative_slices(df, total_slices):
    """Select 5 representative slices."""
    selected = []
    
    # Minimum Y with content
    has_content = (df['Pulpy_voxels'] > 0) | (df['Manual_voxels'] > 0)
    if has_content.any():
        selected.append(int(df[has_content]['Y'].min()))
    
    # Maximum Y with content
    if has_content.any():
        selected.append(int(df[has_content]['Y'].max()))
    
    # Middle
    selected.append(total_slices // 2)
    
    # Highest DICE
    selected.append(int(df['DICE'].idxmax()))
    
    # Highest intersection
    if df['Intersection_px'].max() > 0:
        selected.append(int(df['Intersection_px'].idxmax()))
    
    # Return unique, sorted slices
    return sorted(list(set([s for s in selected if 0 <= s < total_slices])))[:5]


def create_slice_visualization(pulpy, manual, data, y_slice, patient_id, variant, results_subdir):
    """Create 2x2 comparison for one Y slice."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'{patient_id} - Corte Y={y_slice}', fontsize=14, fontweight='bold')
    
    # Background from original data
    bg = data[:, y_slice, :]
    bg_norm = np.clip((bg + 1000) / 2000.0, 0, 1)
    
    pulpy_slice = (pulpy[:, y_slice, :] > 0).astype(float)
    manual_slice = (manual[:, y_slice, :] > 0).astype(float)
    
    # Plot 1: Original
    axes[0, 0].imshow(bg_norm, cmap='gray')
    axes[0, 0].set_title('Original', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Plot 2: Pulpy3D
    axes[0, 1].imshow(bg_norm, cmap='gray')
    pulpy_masked = np.ma.masked_where(pulpy_slice == 0, pulpy_slice)
    axes[0, 1].imshow(pulpy_masked, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
    axes[0, 1].set_title('Pulpy3D (Rojo)', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Plot 3: Manual
    axes[1, 0].imshow(bg_norm, cmap='gray')
    manual_masked = np.ma.masked_where(manual_slice == 0, manual_slice)
    axes[1, 0].imshow(manual_masked, cmap='Blues', alpha=0.7, vmin=0, vmax=1)
    axes[1, 0].set_title('Manual (Azul)', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Plot 4: Overlay
    axes[1, 1].imshow(bg_norm, cmap='gray')
    overlay = np.zeros((*pulpy_slice.shape, 3))
    overlay[(pulpy_slice > 0.5) & (manual_slice <= 0.5)] = [1, 0, 0]      # Only Pulpy: Red
    overlay[(pulpy_slice <= 0.5) & (manual_slice > 0.5)] = [0, 0, 1]      # Only Manual: Blue
    overlay[(pulpy_slice > 0.5) & (manual_slice > 0.5)] = [1, 0, 1]       # Both: Magenta
    axes[1, 1].imshow(overlay, alpha=0.7)
    axes[1, 1].set_title('Overlay (R=Pulpy, B=Manual, M=Ambos)', fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    filename = os.path.join(results_subdir, f'{patient_id}_Y{y_slice:03d}.png')
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filename


def create_overlap_graph(df, patient_id, results_subdir):
    """Create graph of pixel overlaps across Y slices."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f'{patient_id} - Análisis en Cortes Verticales (Eje Y)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Voxel counts
    axes[0].plot(df['Y'], df['Pulpy_voxels'], label='Pulpy3D', color='red', linewidth=2)
    axes[0].plot(df['Y'], df['Manual_voxels'], label='Manual', color='blue', linewidth=2)
    axes[0].plot(df['Y'], df['Intersection_px'], label='Coincidencia', 
                 color='magenta', linewidth=2.5, linestyle='--')
    axes[0].set_xlabel('Corte Y (píxeles)', fontweight='bold')
    axes[0].set_ylabel('Cantidad de píxeles', fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Cantidad de píxeles por corte')
    
    # Plot 2: DICE per slice
    axes[1].bar(df['Y'], df['DICE'], color='green', alpha=0.7, label='DICE')
    axes[1].axhline(y=df['DICE'].mean(), color='darkgreen', linestyle='--', 
                    label=f'DICE promedio: {df["DICE"].mean():.3f}')
    axes[1].set_xlabel('Corte Y (píxeles)', fontweight='bold')
    axes[1].set_ylabel('Coeficiente DICE', fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('DICE por corte')
    
    plt.tight_layout()
    filename = os.path.join(results_subdir, f'{patient_id}_overlap_graph.png')
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filename


def process_patient(patient_id):
    """Full analysis pipeline for one patient."""
    print(f"\n{'='*60}")
    print(f"Processing {patient_id}...")
    print('='*60)
    
    # Load data
    pulpy, manual, data, affine = load_patient_data(patient_id)
    if pulpy is None:
        return False
    
    print(f"  Data loaded | Shapes: Pulpy={pulpy.shape}, Manual={manual.shape}")
    
    # Validate shapes match
    if pulpy.shape != manual.shape:
        print(f"  Incompatible shapes: Pulpy{pulpy.shape} vs Manual{manual.shape}. Skipping...")
        return False
    
    # Find best manual variant
    manual, variant = find_best_manual_variant(manual, pulpy)
    print(f"  Manual variant: {variant}")
    
    # ---- LADO FORZADO (como en ian_hd95_plot3d.py) ----
    KEEP_SIDE = "right"  # Cambia a "left" si necesitas el otro lado
    x_mid = pulpy.shape[0] // 2

    if KEEP_SIDE == "right":
        pulpy = pulpy[x_mid:, :, :]
        manual = manual[x_mid:, :, :]
        data   = data[x_mid:, :, :]
    elif KEEP_SIDE == "left":
        pulpy = pulpy[:x_mid, :, :]
        manual = manual[:x_mid, :, :]
        data   = data[:x_mid, :, :]
    # ---- FIN LADO FORZADO ----

    # Select same side
    pulpy_matched = select_same_side_components(pulpy, manual)
    print(f"  Same-side components selected")

    dice_global = dice_3d(pulpy_matched, manual)
    hd95_global = hd95_3d(pulpy_matched, manual)
    hd95_text = f"{hd95_global:.4f}" if not np.isnan(hd95_global) else "nan"
    
    # Create patient results directory
    patient_dir = os.path.join(RESULTS_DIR, patient_id)
    os.makedirs(patient_dir, exist_ok=True)
    
    # Analyze Y slices
    df = analyze_vertical_slices(pulpy_matched, manual, patient_id)
    
    # Save CSV
    csv_file = os.path.join(patient_dir, f"{patient_id}_vertical_slices_analysis.csv")
    df.to_csv(csv_file, index=False)
    print(f"  CSV saved: {csv_file}")
    
    # Global statistics
    dice_global = dice_3d(pulpy_matched, manual)
    inter_total = (pulpy_matched > 0) & (manual > 0)
    inter_total_px = inter_total.sum()
    
    print(f"  DICE 3D global: {dice_global:.4f}")
    print(f"  HD95 3D global: {hd95_text}")
    print(f"  Total intersection: {int(inter_total_px)} pixels")
    print(f"  Mean intersection per slice: {df['Intersection_px'].mean():.1f} pixels")
    
    # Create overlap graph
    graph_file = create_overlap_graph(df, patient_id, patient_dir)
    print(f"  Graph saved: {graph_file}")
    
    # Select representative slices
    rep_slices = select_representative_slices(df, pulpy.shape[1])
    print(f"  Representative slices selected: {rep_slices}")
    
    # Create visualizations for representative slices
    for y_slice in rep_slices:
        img_file = create_slice_visualization(pulpy_matched, manual, data, y_slice, 
                                             patient_id, variant, patient_dir)
    print(f"  {len(rep_slices)} images saved in {patient_dir}")
    
    # Save summary
    summary_file = os.path.join(patient_dir, f"{patient_id}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"ANALYSIS VERTICAL (Y AXIS) - {patient_id}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Manual variant used: {variant}\n")
        f.write(f"Global DICE 3D: {dice_global:.4f}\n")
        f.write(f"Global HD95 3D: {hd95_text}\n")
        f.write(f"Total intersection: {int(inter_total_px)} pixels\n")
        f.write(f"Mean intersection per Y slice: {df['Intersection_px'].mean():.1f} pixels\n")
        f.write(f"Mean DICE per slice: {df['DICE'].mean():.4f}\n")
        f.write(f"Best DICE at Y={int(df['DICE'].idxmax())}: {df['DICE'].max():.4f}\n")
        f.write(f"Worst DICE at Y={int(df['DICE'].idxmin())}: {df['DICE'].min():.4f}\n")
        f.write(f"\nRepresentative slices analyzed: {rep_slices}\n")
    
    print(f"  Summary saved: {summary_file}")

    return {
        'paciente': patient_id,
        'variant': variant,
        'dice_3d': float(dice_global),
        'hd95_3d': float(hd95_global),
        'intersection_total_px': int(inter_total_px),
        'mean_intersection_px_per_slice': float(df['Intersection_px'].mean()),
        'mean_dice_per_slice': float(df['DICE'].mean()),
        'best_dice_slice': int(df['DICE'].idxmax()),
        'best_dice': float(df['DICE'].max()),
        'worst_dice_slice': int(df['DICE'].idxmin()),
        'worst_dice': float(df['DICE'].min()),
    }


def main():
    """Process all patients."""
    print("\n" + "="*60)
    print("IAN COMPARISON: VERTICAL (Y AXIS)")
    print("="*60)
    
    successful = 0
    hd95_rows = []
    for patient_id in PATIENTS:
        result = process_patient(patient_id)
        if result:
            successful += 1
            hd95_rows.append(result)

    if hd95_rows:
        hd95_df = pd.DataFrame(hd95_rows)
        csv_path = os.path.join(HD95_RESULTS_DIR, "hd95_ian_global_summary.csv")
        xlsx_path = os.path.join(HD95_RESULTS_DIR, "hd95_ian_global_summary.xlsx")
        txt_path = os.path.join(HD95_RESULTS_DIR, "hd95_ian_global_summary.txt")
        hd95_df.to_csv(csv_path, index=False)
        hd95_df.to_excel(xlsx_path, index=False)
        hd95_table = hd95_df.to_string(index=False)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(hd95_table)
        print(f"\nHD95 IAN saved in: {csv_path}")
    
    print(f"\n{'='*60}")
    print(f"Completed: {successful}/{len(PATIENTS)} patients processed")
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"HD95 saved in: {HD95_RESULTS_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
