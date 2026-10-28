"""Bucle de entrenamiento con CBCT simulado (etapa 3: solo infraestructura).

    python train_sim.py -c configs/sim_canal.yml [--holdout Diente_35] [--iterations 200] [--tag fold35]

- Los dientes se descubren por carpeta (dataloader/SimCBCT.py). Se entrena con todos menos los de `holdout`;
  la validación usa parches FIJOS (semilla fija) del diente dejado fuera, nunca recortes del mismo diente.
- Pérdida con etiqueta blanda y mapa de pesos (losses/SoftDiceBCE.py); Adam; bf16.
- Los parches de entrenamiento se centran según `crop.third_weights` del yaml del simulador (por defecto a partes
  iguales entre tercios); `--third-weights proporcional` recupera el muestreo antiguo (según el volumen de conducto).
  La validación usa SIEMPRE parches repartidos a partes iguales entre tercios, para que el ápice cuente igual.
- Métricas por tercios (apical, medio, coronal) en cada validación (metrics_canal.py).
- Guarda en project_dir/<title>_<tag>/: log.txt, metrics.json, last.pt, config.yml.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from dataloader.SimCBCT import SimCBCT, discover_teeth
from losses.SoftDiceBCE import SoftDiceBCE
from metrics_canal import metrics_by_thirds
from models.CanalUNet3D import CanalUNet3D


def build_model(cfg):
    m = cfg["model"]
    assert m["name"] == "CanalUNet3D", "de momento solo CanalUNet3D (los UNet de Pulpy3D necesitan embedding de posición)"
    return CanalUNet3D(1, 1, m["init_features"], m["depth"], m["dropout"])


@torch.no_grad()
def validate(model, bank, loss_fn, amp, voxel_mm=0.075):
    model.eval()
    agg = {k: {"dice": [], "hd95_mm": []} for k in ("apical", "medio", "coronal", "total")}
    losses = []
    for item in bank:
        x, y, w = item["image"][None], item["target"][None], item["weight"][None]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logit = model(x)
        losses.append(float(loss_fn(logit, y, w)))
        pred = (torch.sigmoid(logit.float())[0, 0] > 0.5).cpu().numpy()
        gt = item["label"][0].cpu().numpy() > 0
        keep = item["weight"][0].cpu().numpy() > 0           # lo ignorado no cuenta en las métricas
        r = metrics_by_thirds(pred & keep, gt & keep, item["meta"]["thirds"], voxel_mm)
        for k in agg:
            for q in ("dice", "hd95_mm"):
                agg[k][q].append(r[k][q])
    model.train()
    out = {k: {q: (float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")) for q, v in d.items()} for k, d in agg.items()}
    out["loss"] = float(np.mean(losses))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", type=Path, required=True)
    ap.add_argument("--holdout", nargs="*", default=None, help="sustituye a data.holdout")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=None, help="sustituye a seed del yml (muestreo y pesos iniciales)")
    ap.add_argument("--third-weights", choices=["config", "equilibrado", "proporcional"], default="config",
                    help="muestreo de entrenamiento por tercios: el del yaml del simulador, a partes iguales o proporcional al volumen")
    ap.add_argument("--val-patches", type=int, default=None, help="sustituye a data.val_patches")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    base = a.config.resolve().parent.parent
    d = cfg["data"]
    holdout = a.holdout if a.holdout is not None else d["holdout"]
    iters = a.iterations or cfg["trainer"]["iterations"]
    root = (base / d["root"]).resolve()
    sim_cfg = (base / d["simulator_config"]).resolve()
    seed = cfg["seed"] if a.seed is None else a.seed
    over = {"config": {}, "equilibrado": {"third_weights": [1, 1, 1]}, "proporcional": {"third_weights": None}}[a.third_weights]
    torch.manual_seed(seed)
    np.random.seed(seed)
    assert torch.cuda.is_available(), "hace falta GPU (la máquina del laboratorio)"
    torch.set_default_device("cuda")

    names = [p.name for p in discover_teeth(root)]
    train_names = [n for n in names if n not in holdout]
    assert train_names, "no queda ningún diente para entrenar"
    print(f"dientes: {names} | entrenamiento: {train_names} | fuera (validación): {holdout}")
    out_dir = (base / cfg["project_dir"] / f"{cfg['title']}{('_' + a.tag) if a.tag else ''}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yml").write_text(yaml.safe_dump({**cfg, "holdout_usado": holdout, "dientes_entrenamiento": train_names,
                                                  "semilla_usada": seed, "muestreo_tercios_entrenamiento": a.third_weights}), encoding="utf-8")

    train = SimCBCT(root, train_names, sim_cfg, seed=seed, prefetch=d["prefetch"], crop_overrides=over)
    n_val = a.val_patches or d["val_patches"]
    bank = (SimCBCT(root, holdout, sim_cfg, seed=0, prefetch=0, crop_overrides={"third_weights": [1, 1, 1]})
            .fixed_bank(n_val, seed=12345) if holdout else [])

    model = build_model(cfg).cuda()
    o = cfg["optimizer"]
    assert o["name"] == "Adam"
    opt = torch.optim.Adam(model.parameters(), lr=o["learning_rate"], weight_decay=o["weight_decay"])
    loss_fn = SoftDiceBCE(cfg["loss"]["bce_weight"], cfg["loss"]["dice_weight"])
    amp = cfg["trainer"]["amp"] == "bf16"
    n_par = sum(p.numel() for p in model.parameters())
    print(f"modelo {cfg['model']['name']}: {n_par / 1e6:.2f} M parámetros | lote {d['batch_size']} | {iters} iteraciones")

    log = open(out_dir / "log.txt", "w", encoding="utf-8")
    history, t_data, t_step = [], 0.0, 0.0
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    model.train()
    for it in range(1, iters + 1):
        t0 = time.time()
        b = train.next_batch(d["batch_size"])
        torch.cuda.synchronize()
        t1 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logit = model(b["image"])
        loss = loss_fn(logit, b["target"], b["weight"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        t2 = time.time()
        t_data += t1 - t0
        t_step += t2 - t1
        row = {"it": it, "loss": float(loss)}
        if it % cfg["trainer"]["val_every"] == 0 or it == iters:
            if bank:
                row["val"] = validate(model, bank, loss_fn, amp)
            row["s_por_it_datos"] = t_data / it
            row["s_por_it_red"] = t_step / it
            row["gpu_pico_GB"] = torch.cuda.max_memory_allocated() / 1e9
            print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        history.append(row)
    total = time.time() - t_start
    train.close()
    torch.save({"model": model.state_dict(), "cfg": cfg, "holdout": holdout}, out_dir / "last.pt")
    summary = {"iteraciones": iters, "segundos_total": total, "s_por_iteracion": total / iters,
               "s_esperando_datos_por_it": t_data / iters, "s_en_la_red_por_it": t_step / iters,
               "gpu_pico_GB": torch.cuda.max_memory_allocated() / 1e9, "parametros_M": n_par / 1e6,
               "dientes_entrenamiento": train_names, "dientes_fuera": holdout,
               "validacion_final": next((r["val"] for r in reversed(history) if "val" in r), None)}
    (out_dir / "metrics.json").write_text(json.dumps({"resumen": summary, "historia": history}, indent=1))
    print("RESUMEN", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
