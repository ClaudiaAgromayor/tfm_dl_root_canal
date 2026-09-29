from __future__ import annotations

from pathlib import Path
import sys

import canales_finales as cc


PATIENTS = ["P5", "P35"]
PRE_SLICES = 5


def main():
    out_root = Path("results") / "canales" / "canales_finales"
    for pid in PATIENTS:
        try:
            data, instance, seg, label_map, roles, all_labels = cc.load_patient_data(pid)
            # If patient not present in TARGETS mapping (e.g. P5), use conservative default 'comun'
            if pid not in getattr(cc, 'TARGETS', {}):
                cc.TARGETS[pid] = {"mode": "comun", "description": "auto_comun"}
            gt_target, manual_target, mode = cc.build_target_masks(pid, instance, seg, roles)
            gt_root_region, info = cc.detect_root_final_region(gt_target)

            split_z = int(info.get("split_z", -1))
            if split_z < 0:
                print(f"{pid}: no split_z found, skipping")
                continue

            z_start = max(0, split_z - PRE_SLICES)
            z_list = list(range(z_start, split_z))

            out_dir = out_root / pid / "prestart_slices"
            out_dir.mkdir(parents=True, exist_ok=True)

            for z in z_list:
                out_path = out_dir / f"preview_z{z}.png"
                title = f"{pid} | pre-start preview | z={z} | split={split_z}"
                cc.save_overlay_png(data=data, gt_mask=gt_target, manual_mask=manual_target, z=z, out_path=out_path, title=title)
                print(f"Saved {out_path}")

        except Exception as e:
            print(f"Error processing {pid}: {e}")


if __name__ == "__main__":
    main()
