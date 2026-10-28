"""Parches de CBCT simulado para entrenar (etapa 3). No usa splits.json: descubre los dientes por carpeta.

Cada subcarpeta de `root` (p. ej. 1_Data_Validation/results/microct) que tenga info.json, norm.npy y
canal_solo_conducto.npy es un diente (los crea preprocess_microct.py + split_canal.py). Añadir un µCT = añadir su
carpeta; no hay lista que editar.

El simulador (1_Data_Validation/analysis/microct/simulate_cbct.py) genera cada parche en la GPU a partir del µCT:
imagen de 75 µm, etiqueta blanda (fracción de conducto por vóxel) y un mapa de pesos (0 = ignorar en la pérdida).
Hace falta generar en el proceso principal (la GPU no se comparte bien con procesos hijos), así que un hilo de
prefetch va preparando lotes mientras la red entrena.

    src = SimCBCT(root, train_teeth, sim_config_path, seed=0)
    batch = src.next_batch(2)               # image, target, weight: tensores Bx1xMxMxM en la GPU
    bank = SimCBCT(root, [held_out], sim_config_path).fixed_bank(6, seed=123)   # validación reproducible

`crop_overrides` cambia claves de `crop:` del yaml del simulador (p. ej. {"third_weights": None} para el muestreo
proporcional al volumen, {"third_weights": [1, 1, 1]} para repartir los parches a partes iguales entre los tercios).
"""
import queue
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import yaml

_MICROCT = Path(__file__).resolve().parents[2] / "1_Data_Validation" / "analysis" / "microct"
sys.path.insert(0, str(_MICROCT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import simulate_cbct as sc  # noqa: E402
from metrics_canal import third_map  # noqa: E402

REQUIRED = ("info.json", "norm.npy", "canal_solo_conducto.npy")


def discover_teeth(root) -> list:
    """Carpetas-diente dentro de root, por orden alfabético."""
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_dir() and all((p / f).exists() for f in REQUIRED))


def normalize_patch(img: torch.Tensor) -> torch.Tensor:
    """Gris robusto por parche (percentiles 1 y 99.5): así la red no depende de la escala de gris de un escáner."""
    lo, hi = torch.quantile(img.flatten(), torch.tensor([0.01, 0.995], device=img.device))
    return ((img - lo) / (hi - lo).clamp(min=1.0)).clamp(-0.5, 1.5)


class SimCBCT:
    def __init__(self, root, teeth=None, sim_config=None, seed=0, prefetch=2, crop_overrides=None):
        folders = discover_teeth(root)
        if teeth is not None:
            names = {p.name: p for p in folders}
            missing = [t for t in teeth if t not in names]
            if missing:
                raise FileNotFoundError(f"dientes sin carpeta válida en {root}: {missing}; hay {sorted(names)}")
            folders = [names[t] for t in teeth]
        if not folders:
            raise FileNotFoundError(f"ningún diente en {root} (¿falta split_canal.py?)")
        cfg = yaml.safe_load(Path(sim_config).read_text(encoding="utf-8"))
        cfg["crop"].update(crop_overrides or {})
        self.sim = sc.Simulator(cfg)
        self.teeth = [sc.Tooth(f) for f in folders]
        self.m = self.sim.m
        self.rng = np.random.default_rng(seed)
        self.prefetch = prefetch
        self._q, self._thread, self._stop = None, None, threading.Event()

    # ---- un parche
    def _one(self, tooth, rng):
        out, P = self.sim.patch(tooth, rng)
        thirds = third_map(tooth.info, P["start_zyx_vox"][0], self.sim.crop_size(tooth.voxel), self.m)
        return {"image": normalize_patch(out["image"])[None], "target": out["label_soft"][None],
                "weight": out["weight"][None], "label": out["label"][None],
                "meta": {"tooth": tooth.name, "params": P, "thirds": thirds}}

    def _batch(self, bs, rng):
        items = [self._one(self.teeth[rng.integers(len(self.teeth))], rng) for _ in range(bs)]
        return {"image": torch.stack([i["image"] for i in items]), "target": torch.stack([i["target"] for i in items]),
                "weight": torch.stack([i["weight"] for i in items]), "label": torch.stack([i["label"] for i in items]),
                "meta": [i["meta"] for i in items]}

    # ---- lotes para entrenar (con hilo de prefetch)
    def _worker(self, bs):
        with torch.device(sc.DEV):
            while not self._stop.is_set():
                b = self._batch(bs, self.rng)
                while not self._stop.is_set():
                    try:
                        self._q.put(b, timeout=0.5)
                        break
                    except queue.Full:
                        pass

    def next_batch(self, bs):
        if self.prefetch <= 0:
            return self._batch(bs, self.rng)
        if self._thread is None:
            self._q = queue.Queue(maxsize=self.prefetch)
            self._thread = threading.Thread(target=self._worker, args=(bs,), daemon=True)
            self._thread.start()
        return self._q.get()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # ---- validación reproducible: los mismos parches cada vez (misma semilla)
    def fixed_bank(self, n, seed):
        rng = np.random.default_rng(seed)
        return [self._one(self.teeth[i % len(self.teeth)], rng) for i in range(n)]
