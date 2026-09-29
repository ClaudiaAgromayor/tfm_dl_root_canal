"""
ian_metrics.py
--------------
Pure functions for mask I/O and segmentation metrics.

Key design: GT IAN side selection uses TARGETS[patient_id]["gt_ian_label"]
(derived from analysis/canales/canales_finales.py mode strings) instead of a centroid
heuristic.  This eliminates the cross-side comparison bug that produced
HD95 ≈ 105 vox for P35 instead of the correct ≈ 247 vox.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import nibabel as nib
import numpy as np
import nrrd
from scipy import ndimage

from ian_targets import (
    BASE_PULPY,
    GT_IAN_FILENAME,
    GT_INSTANCE_FILENAME,
    SLICER_DIR,
    TARGETS,
)

IAN_TOKENS = ("ian", "nerv", "canal")


# ---------------------------------------------------------------------------
# Mask I/O
# ---------------------------------------------------------------------------

def load_gt_ian(patient_id: str) -> np.ndarray | None:
    """Load binary IAN GT mask (both sides merged). Returns None if missing."""
    path = BASE_PULPY / patient_id / GT_IAN_FILENAME
    if not path.exists():
        return None
    return (nib.load(str(path)).get_fdata() > 0).astype(bool)


def load_gt_instance(patient_id: str) -> np.ndarray | None:
    """Load labelled instance volume (labels 36 / 46). Returns None if missing."""
    path = BASE_PULPY / patient_id / GT_INSTANCE_FILENAME
    if not path.exists():
        return None
    return np.rint(nib.load(str(path)).get_fdata()).astype(np.int32)


def select_gt_ian_side(
    gt_ian: np.ndarray,
    gt_instance: np.ndarray | None,
    gt_ian_label: int,
) -> np.ndarray:
    """
    Return the subset of gt_ian that belongs to gt_ian_label side.

    Strategy (in order of reliability):
      1. Intersect gt_ian with the instance label mask — most reliable.
      2. If gt_instance is unavailable, return gt_ian unchanged (best effort).
      3. If gt_ian_label == 0 (comun / solo_ian), return gt_ian unchanged.
    """
    if gt_ian_label == 0 or gt_instance is None:
        return gt_ian

    side_mask = (gt_instance == gt_ian_label)
    filtered = gt_ian & side_mask

    # Safety: if the intersection is empty (rare geometry edge case) fall back
    # to the full mask rather than returning an empty volume.
    if np.count_nonzero(filtered) == 0:
        return gt_ian

    return filtered


def _align_shape(arr: np.ndarray, target: tuple[int, ...]) -> np.ndarray | None:
    """Try all axis permutations to match target shape."""
    if arr.shape == target:
        return arr
    for perm in itertools.permutations(range(arr.ndim)):
        candidate = np.transpose(arr, perm)
        if candidate.shape == target:
            return candidate
    return None


def _extract_label_map(header: dict) -> dict[int, str]:
    label_map: dict[int, str] = {}
    for key, value in header.items():
        if not key.endswith("_Name"):
            continue
        prefix = key[: -len("_Name")]
        lv_key = f"{prefix}_LabelValue"
        if lv_key not in header:
            continue
        try:
            label_map[int(header[lv_key])] = str(value)
        except (ValueError, TypeError):
            pass
    return label_map


def _ian_label_from_header(header: dict) -> int | None:
    """Guess the IAN label value from segment names in the NRRD header."""
    i = 0
    while f"Segment{i}_Name" in header:
        name = header[f"Segment{i}_Name"].strip().upper()
        if any(tok in name for tok in ("IAN", "CANAL", "NERV")):
            try:
                return int(header[f"Segment{i}_LabelValue"])
            except (KeyError, ValueError):
                pass
        i += 1
    return None


def load_manual_ian(patient_id: str, reference_shape: tuple[int, ...]) -> np.ndarray | None:
    """
    Load the manual IAN segmentation from the 3DSlicer NRRD for patient_id.

    Returns a boolean mask aligned to reference_shape, or None if the file
    cannot be found or the IAN segment cannot be identified.
    """
    patient_dir = SLICER_DIR / patient_id
    if not patient_dir.exists():
        return None

    candidates = sorted(patient_dir.glob("*.seg.nrrd"))
    if not candidates:
        return None

    data, header = nrrd.read(str(candidates[0]))
    label_map = _extract_label_map(header)

    # Identify IAN label values from segment names
    ian_labels = [
        lv for lv, name in label_map.items()
        if any(tok in name.lower() for tok in IAN_TOKENS)
    ]

    # Fallback: search raw header
    if not ian_labels:
        guess = _ian_label_from_header(header)
        if guess is not None:
            ian_labels = [guess]

    if not ian_labels:
        return None

    # Build binary IAN mask
    mask = np.zeros(data.shape, dtype=bool)
    for lv in ian_labels:
        mask |= data == lv

    # Align to reference volume shape (flip + permute, matching manual_vs_gt/dice_manual_vs_gt_reglas.py)
    rotated = np.flip(np.flip(np.flip(data, axis=1), axis=0), axis=0)
    # Apply same flips to mask
    mask_rotated = np.flip(np.flip(np.flip(mask, axis=1), axis=0), axis=0)
    aligned = _align_shape(mask_rotated, reference_shape)
    if aligned is None:
        return None

    return aligned.astype(bool)


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

def dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.count_nonzero(a & b)
    total = np.count_nonzero(a) + np.count_nonzero(b)
    if total == 0:
        return float("nan")
    return 2.0 * intersection / total


def hd95(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Symmetric 95th-percentile Hausdorff distance (in voxels).

    Computed from surface voxels only via EDT, matching the standard
    medical-image-segmentation convention.
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    if not np.any(a) or not np.any(b):
        return float("nan")

    def surface(m: np.ndarray) -> np.ndarray:
        struct = ndimage.generate_binary_structure(m.ndim, 1)
        return m ^ ndimage.binary_erosion(m, structure=struct, border_value=0)

    def directed(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        src_surf = surface(src)
        tgt_surf = surface(tgt)
        if not np.any(src_surf) or not np.any(tgt_surf):
            return np.array([], dtype=np.float32)
        dist_map = ndimage.distance_transform_edt(~tgt_surf)
        return dist_map[src_surf]

    distances = np.concatenate([directed(a, b), directed(b, a)])
    if distances.size == 0:
        return float("nan")
    return float(np.percentile(distances, 95))
