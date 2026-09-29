from pathlib import Path
import canales_finales as cc

PID = "P5"
ZS = list(range(60, 68))  # 60..67 inclusive

def main():
    out_dir = Path("results") / "canales" / "canales_finales" / PID / "prestart_more"
    out_dir.mkdir(parents=True, exist_ok=True)

    data, instance, seg, label_map, roles, all_labels = cc.load_patient_data(PID)
    if PID not in getattr(cc, 'TARGETS', {}):
        cc.TARGETS[PID] = {"mode": "comun", "description": "auto_comun"}
    gt_target, manual_target, mode = cc.build_target_masks(PID, instance, seg, roles)

    for z in ZS:
        out_path = out_dir / f"preview_z{z}.png"
        title = f"{PID} | extra pre-start | z={z}"
        cc.save_overlay_png(data=data, gt_mask=gt_target, manual_mask=manual_target, z=z, out_path=out_path, title=title)
        print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
