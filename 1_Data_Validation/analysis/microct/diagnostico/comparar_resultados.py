"""Compara resultados regenerados con los guardados (para comprobar que un paso se reproduce).

    python comparar_resultados.py pre <carpeta_nueva/Diente_XX> <carpeta_guardada/Diente_XX>   # norm.npy, canal.npy, info.json
    python comparar_resultados.py hash <fichero> [<fichero> ...]                              # SHA-256 (16 primeros caracteres)
    python comparar_resultados.py json <a.json> <b.json>                                      # igualdad de dos json
    python comparar_resultados.py calib <nuevo/calibration.json> <guardado/calibration.json>

Uso típico (RESUMEN_PROYECTO.md, apéndice A):
    python preprocess_microct.py <datasets/MicroCT/Diente_52> --downsample 1 --out <tmp>
    python comparar_resultados.py pre <tmp>/Diente_52 results/microct/Diente_52
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

cmd = sys.argv[1]
if cmd == "pre":
    a, b = Path(sys.argv[2]), Path(sys.argv[3])
    ia, ib = json.loads((a / "info.json").read_text()), json.loads((b / "info.json").read_text())
    print("volumen del canal: nuevo %.6f | guardado %.6f | idéntico: %s" % (ia["volumen_canal_mm3"], ib["volumen_canal_mm3"], ia["volumen_canal_mm3"] == ib["volumen_canal_mm3"]))
    print("anclas iguales:", ia["anclas"] == ib["anclas"])
    for nombre in ("canal.npy", "norm.npy"):
        x, y = np.load(a / nombre, mmap_mode="r"), np.load(b / nombre, mmap_mode="r")
        print(f"{nombre} idéntica byte a byte:", x.shape == y.shape and all(np.array_equal(x[i:i + 100], y[i:i + 100]) for i in range(0, x.shape[0], 100)))
elif cmd == "hash":
    for f in sys.argv[2:]:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 24), b""):
                h.update(blk)
        print(Path(f).name, h.hexdigest()[:16])
elif cmd == "json":
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[2:4])
    print("json idénticos:", a == b)
    if a != b:
        for k in a:
            if a[k] != b.get(k):
                print("  difiere:", k)
elif cmd == "calib":
    a, b = (json.loads(Path(p).read_text()) for p in sys.argv[2:4])
    print("calibración idéntica:", a == b)
    print("  gris dentina:", a["grises"]["dentina"]["mediana"], "frente a", b["grises"]["dentina"]["mediana"])
    print("  ruido dentina:", a["ruido"]["dentina"]["sd_mediana"], "frente a", b["ruido"]["dentina"]["sd_mediana"])
    print("  sigma borde dentina-exterior:", a["bordes"]["dentina_exterior"]["sigma_mm"], "frente a", b["bordes"]["dentina_exterior"]["sigma_mm"])
else:
    sys.exit(__doc__)
