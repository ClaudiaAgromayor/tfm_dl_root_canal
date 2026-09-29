"""
ian_targets.py
--------------
Single source of truth for patient targets, GT labels, and directory paths.
All other modules import from here.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
BASE_PULPY = Path(
    r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos"
    r"\ICAI\Beca IIT\1_Data_Validation\datasets\Pulpy3D"
)
SLICER_DIR = Path(
    r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos"
    r"\ICAI\Beca IIT\2_UNet\datasets\Segmentaciones 3DSlicer"
)
OUTPUT_DIR = Path(
    r"C:\Users\clauu\OneDrive - Universidad Pontificia Comillas\Documentos"
    r"\ICAI\Beca IIT\1_Data_Validation\results\ian"
)

OUTPUT_XLSX = OUTPUT_DIR / "ian_hd95_summary.xlsx"
OUTPUT_CSV  = OUTPUT_DIR / "ian_hd95_summary.csv"

# ---------------------------------------------------------------------------
# GT label constants
# ---------------------------------------------------------------------------
GT_RIGHT_LABEL = 36   # label 36 → right molar / right IAN in gt_instance.nii.gz
GT_LEFT_LABEL  = 46   # label 46 → left  molar / left  IAN in gt_instance.nii.gz

GT_IAN_FILENAME      = "gt_ian.nii.gz"       # binary IAN mask (both sides merged)
GT_INSTANCE_FILENAME = "gt_instance.nii.gz"  # labelled instance (36 / 46)

# ---------------------------------------------------------------------------
# Per-patient targets
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
# "mode" encodes which manual label maps to which GT label:
#   manual_left_to_gt46  → manual left canal  compared against GT label 46
#   manual_right_to_gt36 → manual right canal compared against GT label 36
#   manual_left_to_gt36  → manual left canal  compared against GT label 36 (crossed anatomy)
#   manual_right_to_gt46 → manual right canal compared against GT label 46 (crossed anatomy)
#   comun                → both sides segmented together
#   solo_ian             → IAN not available for tooth comparison (skip)
# ---------------------------------------------------------------------------
TARGETS: dict[str, dict[str, str]] = {
    "P35":  {"mode": "manual_left_to_gt46",  "gt_ian_label": GT_LEFT_LABEL},
    "P48":  {"mode": "manual_left_to_gt46",  "gt_ian_label": GT_LEFT_LABEL},
    "P59":  {"mode": "solo_ian",             "gt_ian_label": 0},
    "P61":  {"mode": "manual_left_to_gt46",  "gt_ian_label": GT_LEFT_LABEL},
    "P81":  {"mode": "manual_right_to_gt36", "gt_ian_label": GT_RIGHT_LABEL},
    "P100": {"mode": "manual_right_to_gt36", "gt_ian_label": GT_RIGHT_LABEL},
    "P111": {"mode": "manual_left_to_gt46",  "gt_ian_label": GT_LEFT_LABEL},
    "P133": {"mode": "manual_right_to_gt36", "gt_ian_label": GT_RIGHT_LABEL},
    "P137": {"mode": "manual_left_to_gt36",  "gt_ian_label": GT_RIGHT_LABEL},
    "P146": {"mode": "solo_ian",             "gt_ian_label": 0},
    "P191": {"mode": "manual_right_to_gt46", "gt_ian_label": GT_LEFT_LABEL},
    "P195": {"mode": "manual_right_to_gt36", "gt_ian_label": GT_RIGHT_LABEL},
    "P208": {"mode": "manual_right_to_gt36", "gt_ian_label": GT_RIGHT_LABEL},
    "P217": {"mode": "comun",               "gt_ian_label": 0},
    "P222": {"mode": "manual_left_to_gt46",  "gt_ian_label": GT_LEFT_LABEL},
    "P249": {"mode": "manual_right_to_gt46", "gt_ian_label": GT_LEFT_LABEL},
    "P254": {"mode": "manual_right_to_gt46", "gt_ian_label": GT_LEFT_LABEL},
    "P261": {"mode": "manual_right_to_gt36", "gt_ian_label": GT_RIGHT_LABEL},
    "P343": {"mode": "comun",               "gt_ian_label": 0},
    "P380": {"mode": "manual_right_to_gt36", "gt_ian_label": GT_RIGHT_LABEL},
    "P381": {"mode": "manual_right_to_gt36", "gt_ian_label": GT_RIGHT_LABEL},
    "P394": {"mode": "manual_left_to_gt46",  "gt_ian_label": GT_LEFT_LABEL},
    "P402": {"mode": "manual_left_to_gt46",  "gt_ian_label": GT_LEFT_LABEL},
    "P422": {"mode": "manual_left_to_gt46",  "gt_ian_label": GT_LEFT_LABEL},
    "P447": {"mode": "solo_ian",             "gt_ian_label": 0},
    "P448": {"mode": "manual_right_to_gt36", "gt_ian_label": GT_RIGHT_LABEL},
    "P466": {"mode": "manual_right_to_gt36", "gt_ian_label": GT_RIGHT_LABEL},
    "P476": {"mode": "solo_ian",             "gt_ian_label": 0},
    "P511": {"mode": "manual_left_to_gt46",  "gt_ian_label": GT_LEFT_LABEL},
    "P534": {"mode": "manual_left_to_gt46",  "gt_ian_label": GT_LEFT_LABEL},
    "P546": {"mode": "manual_right_to_gt36", "gt_ian_label": GT_RIGHT_LABEL},
}

# Ordered patient list for iteration
PATIENTS: list[str] = list(TARGETS.keys())
