"""Comprobaciones pequeñas y rápidas del simulador, las métricas y los datos reales (las "E-x" del RESUMEN_PROYECTO.md).

    python comprobaciones.py [--que isa muestreo ventanas ganancia_ruido conservacion dilatacion semilla tercios]
    (sin --que ejecuta todas)

isa            segmentación de Isa: etiquetas, vóxeles, volumen, componentes conexas, Status de la cabecera (necesita el NRRD)
muestreo       reparto de los centros de parche por tercio con el muestreo ANTIGUO (proporcional) y el nuevo (ver también
               muestreo_por_tercios.py, que mide además el conducto por bloque)
ventanas       anchos de las ventanas de la reducción antigua (adaptive_avg_pool3d) y de la exacta (box_weights)
ganancia_ruido sd de ruido blanco unitario tras el filtro gaussiano frente a simulate_cbct.noise_gain
conservacion   suma de la etiqueta blanda × (0,075 mm)^3 frente al volumen de conducto del bloque, con las dos reducciones
dilatacion     dilate_box (separable) igual que un máximo 3D
semilla        la misma semilla da el mismo parche; otra, uno distinto
tercios        third_map y metrics_by_thirds con un caso conocido, y coincidencia con Tooth.third_of_z
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--que", nargs="*", default=["isa", "muestreo", "ventanas", "ganancia_ruido", "conservacion", "dilatacion", "semilla", "tercios"])
ap.add_argument("--results", type=Path, default=HERE.parents[2] / "results" / "microct")
ap.add_argument("--isa", type=Path, default=HERE.parents[2] / "datasets" / "CBCT_75um" / "Isa_caso01" / "Segmentation.seg.nrrd")
a = ap.parse_args()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2].parent / "2_UNet"))
import simulate_cbct as sc  # noqa: E402

if sc.DEV.type == "cuda":
    torch.set_default_device("cuda")
cfg = yaml.safe_load((HERE.parent / "simulate_cbct_config.yaml").read_text(encoding="utf-8"))
NOMBRES = ("Diente_35", "Diente_51", "Diente_52")


def titulo(t):
    print(f"\n== {t}")


if "isa" in a.que:
    titulo("segmentación de Isa")
    import nrrd
    from scipy import ndimage as ndi
    seg, h = nrrd.read(str(a.isa))
    m = seg > 0
    lab, n = ndi.label(m)
    sizes = np.sort(np.bincount(lab.ravel())[1:])[::-1]
    print("etiquetas distintas:", np.unique(seg).tolist(), "| vóxeles:", int(m.sum()), "| volumen mm3: %.2f" % (m.sum() * 0.075 ** 3))
    print("componentes conexas (6-conectividad):", n, "| 5 mayores (vóx):", sizes[:5].tolist(), "| con >= 100 vóx:", int((sizes >= 100).sum()))
    print("Segment0_Tags:", h.get("Segment0_Tags", "")[:60])

if "muestreo" in a.que:
    titulo("centros de parche por tercio (600 recortes por diente; solo el centro, sin leer conducto)")
    import copy
    for label, w in (("proporcional (antiguo)", None), ("equilibrado [1,1,1]", [1, 1, 1])):
        c = copy.deepcopy(cfg); c["crop"]["third_weights"] = w
        sim = sc.Simulator(c)
        tot = np.zeros(4)
        for d in NOMBRES:
            t = sc.Tooth(a.results / d); rng = np.random.default_rng(0); n = sim.crop_size(t.voxel)
            for _ in range(600):
                k = int(t.third_of_z(sim.pick_start(t, rng)[0] + n / 2))
                tot[k if k >= 0 else 3] += 1
        p = 100 * tot / tot.sum()
        print(f"  {label:24s}: apical {p[0]:.1f} % | medio {p[1]:.1f} % | coronal {p[2]:.1f} % | fuera del conducto {p[3]:.1f} %")

if "ventanas" in a.que:
    titulo("ventanas de la reducción 300 -> 128 (vóxeles de 32 µm)")
    L, M = 300, 128
    w = [int(np.ceil((i + 1) * L / M) - np.floor(i * L / M)) for i in range(M)]
    print("antigua (adaptive_avg_pool3d): anchos", {k: w.count(k) for k in sorted(set(w))}, "| media %.4f vóxeles = %.1f µm (paso %.3f vóxeles = 75 µm)" % (np.mean(w), np.mean(w) * 32, L / M))
    x = torch.arange(300.0, device="cpu").view(1, 1, 300, 1, 1)
    exp = torch.tensor([np.mean(np.arange(int(np.floor(i * L / M)), int(np.ceil((i + 1) * L / M)))) for i in range(M)], dtype=torch.float32, device="cpu")
    print("  PyTorch coincide con la fórmula floor/ceil:", bool(torch.allclose(F.adaptive_avg_pool3d(x, (128, 1, 1))[0, 0, :, 0, 0], exp)))
    W = sc.box_weights(300, 128).cpu()
    anchos = (W > 0).sum(1).numpy() - 0
    print("exacta (box_weights): suma de cada fila = 1:", bool(torch.allclose(W.sum(1), torch.ones(128, device="cpu"), atol=1e-6)),
          "| cada salida cubre exactamente %.4f vóxeles de entrada (= %.1f µm); vóxeles tocados por salida: %s" % (L / M, L / M * 32, {int(k): int((anchos == k).sum()) for k in sorted(set(anchos))}))

if "ganancia_ruido" in a.que:
    titulo("noise_gain frente a la medida")
    for s in (1.0, 1.3, 1.7):
        g = torch.Generator(device=sc.DEV); g.manual_seed(0)
        zf = sc.blur(torch.randn((96, 96, 96), generator=g, device=sc.DEV), (s, s, s))[20:-20, 20:-20, 20:-20]
        print(f"  sigma {s}: sd medida {float(zf.std()):.4f} | noise_gain {sc.noise_gain((s, s, s)):.4f}")

if "conservacion" in a.que:
    titulo("conservación de la etiqueta blanda (un parche por diente y reducción)")
    import copy
    for red in ("adaptativa", "exacta"):
        c = copy.deepcopy(cfg); c["crop"]["reduccion"] = red
        sim = sc.Simulator(c)
        diffs = []
        for d in NOMBRES:
            t = sc.Tooth(a.results / d)
            for k in range(2):
                rng = np.random.default_rng(11 + k)
                st = sim.pick_start(t, rng)
                o, _ = sim.patch(t, rng, start=st, center=True)
                v16 = float(t.read(t.canal, st, sim.crop_size(t.voxel)).sum()) * t.voxel ** 3
                diffs.append(100 * (float(o["label_soft"].sum()) * 0.075 ** 3 - v16) / v16)
        print(f"  reducción {red:10s}: diferencia de volumen {min(diffs):+.3f} % ... {max(diffs):+.3f} % en {len(diffs)} parches")

if "dilatacion" in a.que:
    titulo("dilate_box frente al máximo 3D")
    x = (torch.rand(60, 70, 50) > 0.995).float()
    for r in (3, 9):
        print(f"  r={r}: iguales: {bool((sc.dilate_box(x, r) == (F.max_pool3d(x[None, None], 2 * r + 1, 1, r)[0, 0] > 0)).all())}")

if "semilla" in a.que:
    titulo("misma semilla -> mismo parche")
    sim = sc.Simulator(cfg); t = sc.Tooth(a.results / "Diente_51")
    st = np.array([300, 500, 300])
    o1, P1 = sim.patch(t, np.random.default_rng(7), start=st); o2, P2 = sim.patch(t, np.random.default_rng(7), start=st); o3, _ = sim.patch(t, np.random.default_rng(8), start=st)
    print("  máx |dif| imagen (misma semilla):", float((o1["image"] - o2["image"]).abs().max()), "| etiqueta idéntica:", bool((o1["label"] == o2["label"]).all()), "| parámetros idénticos:", P1 == P2)
    print("  máx |dif| imagen (otra semilla):", round(float((o1["image"] - o3["image"]).abs().max()), 1))

if "tercios" in a.que:
    titulo("third_map / metrics_by_thirds con un caso conocido, y Tooth.third_of_z")
    from metrics_canal import metrics_by_thirds, third_map
    info = {"separacion": {"canal_solo_z_vox": [100, 700]}, "orientacion": {"corona_en_z": "alto"}}
    tm = third_map(info, 0, 600, 128); z = (np.arange(128) + 0.5) * 600 / 128
    print("  corona alta: apical z", round(z[tm == 0].min()), "-", round(z[tm == 0].max()), "| medio", round(z[tm == 1].min()), "-", round(z[tm == 1].max()),
          "| coronal", round(z[tm == 2].min()), "-", round(z[tm == 2].max()), "| fuera:", int((tm == -1).sum()), "cortes")
    info["orientacion"]["corona_en_z"] = "bajo"
    print("  corona baja: apical z", round(z[third_map(info, 0, 600, 128) == 0].min()), "-", round(z[third_map(info, 0, 600, 128) == 0].max()))
    gt = np.zeros((128, 20, 20), bool); gt[10:120, 8:12, 8:12] = True; pr = np.roll(gt, 1, axis=1)
    r = metrics_by_thirds(pr, gt, third_map({"separacion": {"canal_solo_z_vox": [10, 120]}, "orientacion": {"corona_en_z": "alto"}}, 0, 128, 128), 0.075)
    print("  cuadrado 4x4 desplazado 1 vóxel: Dice %.3f (esperado 0.75) | HD95 %.3f mm (esperado 0.075)" % (r["total"]["dice"], r["total"]["hd95_mm"]))
    t = sc.Tooth(a.results / "Diente_35")
    zz = np.arange(t.z_canal[0] - 20, t.z_canal[1] + 20, 7)
    inf = {"separacion": {"canal_solo_z_vox": list(t.z_canal)}, "orientacion": {"corona_en_z": "alto" if t.crown_high else "bajo"}}
    # third_map toma la esquina del parche y el centro de cada corte; se comprueba con parches de 1 corte equivalentes
    ok = all(int(third_map(inf, float(q) - 0.5, 1, 1)[0]) == int(t.third_of_z(q)) for q in zz)
    print("  third_map coincide con Tooth.third_of_z en", len(zz), "cortes:", ok)
