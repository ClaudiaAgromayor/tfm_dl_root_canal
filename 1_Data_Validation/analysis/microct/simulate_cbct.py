"""Simulador de CBCT a 75 µm a partir de un µCT con máscara de canal (etapa 2 del TFM).

Coge un bloque aleatorio de un µCT ya preprocesado (results/microct/Diente_XX con norm.npy,
canal_solo_conducto.npy, camara_y_cavidades.npy e info.json; ver preprocess_microct.py y split_canal.py) y lo
degrada como lo haría un CBCT de 75 µm. Como el falso sale del µCT, la etiqueta ya está alineada: no hay registro.

El tamaño de vóxel de cada µCT se lee de su info.json (no se supone 16 µm): el bloque tiene
`crop_size(voxel)` vóxeles, los que cubren out_size × 75 µm.

Orden de los pasos (cada valor aleatorio sale de los rangos de simulate_cbct_config.yaml):
  1. recorte aleatorio de un bloque (a la resolución del µCT) alrededor del conducto; el centro se reparte entre
     los tercios apical, medio y coronal del conducto según `crop.third_weights` (por defecto, a partes iguales);
  2. grises: el µCT (aire = 0, dentina ≈ 1) se pasa a grises de CBCT; el conducto se rellena con un gris
     de pulpa (no aire) y alrededor del diente se pone "hueso / tejido blando" con textura y, a veces, una
     capa fina de ligamento periodontal;
  3. emborronado gaussiano (PSF) de anchura aleatoria, ANTES de reducir;
  4. reducción a 75 µm promediando cajas EXACTAS de 75 µm (`crop.reduccion: exacta`; la opción `adaptativa` es la
     antigua, con ventanas que se solapan y miden ~106 µm). Cada parche empieza en un vóxel distinto del µCT, así
     el desplazamiento respecto a la rejilla de 75 µm es aleatorio;
  5. ruido de tipo Poisson filtrado (correlacionado) + variación suave de brillo;
  6. etiqueta con exactamente la misma geometría: fracción de conducto por vóxel ("blanda") y binaria al 50 %.

Uso:
    python simulate_cbct.py results/microct/Diente_35 results/microct/Diente_51 --n 8 --seed 0 \
        --out results/sim_cbct --figs 4 --real-cbct <cbct.nrrd> --real-seg <seg.nrrd>
    python simulate_cbct.py results/microct/Diente_35 --check --calibration results/cbct_calibration/Isa_caso01/calibration.json

El mismo código sirve para generar al vuelo en un dataloader: `Simulator(cfg).patch(tooth, rng)`.
Con GPU (RTX 5080) `patch()` tarda ~0,5 s por parche de 128³ (medido; 87 s en CPU); el bucle de esta línea de
comandos, que además comprime y guarda cada parche, ~1-2 s. Ver diagnostico/tiempos_simulador.py.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- utilidades GPU
def gkernel(sigma: float) -> torch.Tensor:
    r = max(1, int(math.ceil(4 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=DEV)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur(x: torch.Tensor, sig) -> torch.Tensor:
    """Gaussiano separable sobre un tensor (D,H,W); sig = (sigma_z, sigma_y, sigma_x) en vóxeles."""
    x = x[None, None]
    for ax, s in enumerate(sig):
        if s < 0.05:
            continue
        k = gkernel(float(s))
        r = (len(k) - 1) // 2
        shape = [1, 1, 1, 1, 1]
        shape[2 + ax] = len(k)
        pad = [0] * 6
        pad[2 * (2 - ax)] = pad[2 * (2 - ax) + 1] = r
        x = F.conv3d(F.pad(x, pad, mode="replicate"), k.view(shape))
    return x[0, 0]


def noise_gain(sig) -> float:
    """sd de un ruido blanco unitario tras el filtro gaussiano separable de sigma `sig` por eje."""
    g = 1.0
    for s in sig:
        if s >= 0.05:
            g *= float((gkernel(float(s)) ** 2).sum()) ** 0.5
    return g


def smooth_field(extent_mm: float, sigma_mm: float, size, gen) -> torch.Tensor:
    """Campo aleatorio suave (media 0, sd 1) con correlación ~sigma_mm, generado en malla gruesa."""
    h = max(sigma_mm / 3, 0.05)
    n = max(int(math.ceil(extent_mm / h)), 4)
    f = torch.randn((n, n, n), generator=gen, device=DEV)
    s = sigma_mm / (extent_mm / n)
    f = blur(f, (s, s, s))
    f = (f - f.mean()) / (f.std() + 1e-6)
    return F.interpolate(f[None, None], size=tuple(size), mode="trilinear", align_corners=False)[0, 0]


def dilate_box(x: torch.Tensor, r: int) -> torch.Tensor:
    """Dilatación con un cubo de lado 2r+1 sobre un tensor (D,H,W). Separable (tres pasadas 1D): igual que un
    máximo 3D de lado 2r+1 pero cientos de veces más barato con r grande. Devuelve bool."""
    k = 2 * r + 1
    y = x[None, None]
    y = F.max_pool3d(y, (k, 1, 1), 1, (r, 0, 0))
    y = F.max_pool3d(y, (1, k, 1), 1, (0, r, 0))
    y = F.max_pool3d(y, (1, 1, k), 1, (0, 0, r))
    return y[0, 0] > 0


def box_weights(n_in: int, m_out: int) -> torch.Tensor:
    """Matriz (m_out, n_in) del promedio por cajas EXACTO: el vóxel de salida i cubre [i·n_in/m, (i+1)·n_in/m) en
    unidades de vóxel de entrada y cada vóxel de entrada pesa la fracción de su longitud que cae dentro (cada fila
    suma 1). Con n_in = 300 y m_out = 128 cada salida promedia 2,34 vóxeles de entrada (75 µm si el de entrada
    mide 32 µm), repartidos con pesos fraccionarios."""
    edges = torch.arange(m_out + 1, dtype=torch.float64, device=DEV) * n_in / m_out
    j = torch.arange(n_in, dtype=torch.float64, device=DEV)[None, :]
    overlap = (torch.minimum(edges[1:, None], j + 1) - torch.maximum(edges[:-1, None], j)).clamp(min=0)
    return (overlap / (n_in / m_out)).float()


def reduce_to(x: torch.Tensor, m: int, exact: bool = True) -> torch.Tensor:
    """Reduce (D,H,W) a (m,m,m) promediando. exact=True: cajas exactas (separable: tres productos con matrices de
    pesos, coste O(m·n·n²)); exact=False: `adaptive_avg_pool3d` de PyTorch, el comportamiento anterior, cuyas ventanas
    son de vóxeles enteros que se solapan (con 300 -> 128 miden 3 o 4 vóxeles, 106 µm de media, en vez de 75 µm)."""
    if not exact:
        return F.adaptive_avg_pool3d(x[None, None], m)[0, 0]
    y = torch.einsum("ia,abc->ibc", box_weights(x.shape[0], m), x)
    y = torch.einsum("jb,ibc->ijc", box_weights(x.shape[1], m), y)
    return torch.einsum("kc,ijc->ijk", box_weights(x.shape[2], m), y)


def pool_to(x: torch.Tensor, m: int, exact: bool = True) -> torch.Tensor:
    """µCT -> 75 µm: promedio por 2 y luego reducción a m vóxeles (las dos geometrías, la de imagen y la de etiqueta,
    pasan por esto mismo, por eso quedan alineadas)."""
    return reduce_to(F.avg_pool3d(x[None, None], 2)[0, 0], m, exact)


# ---------------------------------------------------------------------------------- datos
class Tooth:
    """Un µCT preprocesado: carpeta con norm.npy, info.json y las máscaras de split_canal.py.

    La etiqueta es `canal_solo_conducto.npy`. `camara_y_cavidades.npy` (lo quitado: cámara pulpar y
    cavidades de la corona) no es etiqueta, pero en un paciente esas zonas contienen pulpa, así que el
    simulador las rellena con gris de pulpa (ver config `labels`). Con `canal_name="canal.npy"` y
    `chamber_name=None` se recupera el comportamiento anterior (la máscara completa como etiqueta).
    """

    def __init__(self, folder: Path, canal_name="canal_solo_conducto.npy", chamber_name="camara_y_cavidades.npy"):
        self.folder = Path(folder)
        self.name = self.folder.name
        info = json.loads((self.folder / "info.json").read_text())
        self.info = info
        self.voxel = float(info["voxel_mm"])
        if not (self.folder / canal_name).exists():
            raise FileNotFoundError(f"{self.folder / canal_name} no existe: ejecuta split_canal.py en {self.folder}")
        self.norm = np.load(self.folder / "norm.npy", mmap_mode="r")
        self.canal = np.load(self.folder / canal_name, mmap_mode="r")
        self.chamber = (np.load(self.folder / chamber_name, mmap_mode="r")
                        if chamber_name and (self.folder / chamber_name).exists() else None)
        cache = self.folder / f"coords_{Path(canal_name).stem}_s6.npy"
        if not cache.exists():
            pts = []
            for z0 in range(0, self.canal.shape[0], 120):
                blk = np.asarray(self.canal[z0:z0 + 120:6, ::6, ::6])
                p = np.argwhere(blk > 0)
                p[:, 0] = p[:, 0] * 6 + z0
                p[:, 1:] *= 6
                pts.append(p)
            np.save(cache, np.concatenate(pts).astype(np.int32))
        self.coords = np.load(cache)
        sep, ori = info.get("separacion"), info.get("orientacion")          # los escribe split_canal.py
        self.z_canal = tuple(sep["canal_solo_z_vox"]) if sep and "canal_solo_z_vox" in sep else None
        self.crown_high = (ori["corona_en_z"] == "alto") if ori and "corona_en_z" in ori else None
        self._coords_by_third = None

    def third_of_z(self, z):
        """Tercio del conducto (0 apical, 1 medio, 2 coronal) de uno o varios cortes z del µCT, según la extensión del
        conducto de ESTE diente; -1 si está fuera de ella o si el info.json no tiene esa información. Es la misma regla
        que usa 2_UNet/metrics_canal.py (third_map) para las métricas."""
        z = np.asarray(z, float)
        if self.z_canal is None or self.crown_high is None:
            return np.full(z.shape, -1, int)
        z0, z1 = self.z_canal
        f = (z1 - z) / max(z1 - z0, 1) if self.crown_high else (z - z0) / max(z1 - z0, 1)   # 0 corona -> 1 ápice
        t = np.where(f >= 2 / 3, 0, np.where(f >= 1 / 3, 1, 2))
        return np.where((z >= z0) & (z <= z1), t, -1)

    @property
    def coords_by_third(self):
        if self._coords_by_third is None:
            t = self.third_of_z(self.coords[:, 0])
            self._coords_by_third = [self.coords[t == k] for k in range(3)]
        return self._coords_by_third

    def read(self, arr, start, n):
        """Bloque n³ desde `start` (z,y,x); lo que cae fuera del volumen se rellena con 0 (aire)."""
        out = np.zeros((n, n, n), arr.dtype)
        src, dst = [], []
        for s, size in zip(start, arr.shape):
            a0, a1 = max(s, 0), min(s + n, size)
            if a1 <= a0:
                return out
            src.append(slice(a0, a1))
            dst.append(slice(a0 - s, a1 - s))
        out[tuple(dst)] = arr[tuple(src)]
        return out


# -------------------------------------------------------------------------------- simulador
def U(rng, r):
    return float(rng.uniform(r[0], r[1]))


class Simulator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        c = cfg["crop"]
        self.m = int(c["out_size"])
        self.exact = c.get("reduccion", "exacta") == "exacta"
        self.stats = {"pick_start": 0, "sin_cumplir": 0}        # cuántos recortes no cumplieron lo pedido en 20 intentos

    def crop_size(self, voxel_in_mm: float) -> int:
        """Lado (en vóxeles del µCT) del bloque que cubre `out_size` vóxeles de salida de `voxel_out_mm`. Es par para
        poder promediar de 2 en 2; si el vóxel del µCT no divide exactamente, el vóxel de salida efectivo
        (P["voxel_out_efectivo_mm"]) difiere algo de 75 µm (con 16 µm: 600 vóxeles, exacto)."""
        n = self.m * self.cfg["crop"]["voxel_out_mm"] / voxel_in_mm
        return 2 * int(round(n / 2))

    # ---- parámetros aleatorios (todo lo que se sortea queda guardado junto al parche)
    def sample_params(self, rng, center=False):
        c = self.cfg
        u = (lambda r: 0.5 * (r[0] + r[1])) if center else (lambda r: U(rng, r))
        P = {"dentin_hu": u(c["intensity"]["dentin_hu"]), "enamel_gain": u(c["intensity"]["enamel_gain"]),
             "canal_hu": u(c["intensity"]["canal_hu"]),
             "chamber_offset_hu": u(c.get("labels", {}).get("chamber_offset_hu", [0, 0])),
             "bg_base": u(c["background"]["base_hu"]), "bg_amp": u(c["background"]["texture_amp_hu"]),
             "bg_sigma_mm": u(c["background"]["texture_sigma_mm"]),
             "pdl": bool(center or rng.random() < c["background"]["pdl"]["prob"]),
             "pdl_width_mm": u(c["background"]["pdl"]["width_mm"]), "pdl_hu": u(c["background"]["pdl"]["hu"]),
             "psf_sigma_mm": u(c["psf"]["sigma_mm"]), "psf_z_factor": u(c["psf"]["z_factor"]),
             "noise_sd_hu": u(c["noise"]["sd_hu"]), "noise_corr_vox": u(c["noise"]["corr_sigma_vox"]),
             "noise_alpha": u(c["noise"]["density_exponent"]), "poisson_counts": u(c["noise"]["poisson_counts"]),
             "shade_mult": u(c["shading"]["mult"]), "shade_add_hu": u(c["shading"]["add_hu"]),
             "shade_sigma_mm": u(c["shading"]["sigma_mm"])}
        return P

    def pick_start(self, tooth: Tooth, rng):
        """Esquina (z,y,x) del bloque (en vóxeles del µCT): centrado cerca de un vóxel de conducto, con desplazamiento.

        Con `crop.third_weights = [wa, wm, wc]` se sortea primero el tercio del conducto (apical, medio, coronal) con esas
        probabilidades y el centro sale de un vóxel de ese tercio; el desplazamiento aleatorio no puede sacar el centro
        de él (se vuelve a sortear). Sin eso (`third_weights: null`) el centro sale de un vóxel de conducto cualquiera,
        y como la parte coronal tiene mucho más volumen que la apical, el ápice quedaba infrarrepresentado."""
        cr = self.cfg["crop"]
        n = self.crop_size(tooth.voxel)
        empty = rng.random() < cr["p_empty"]
        jit = (8.0 if empty else cr["jitter_mm"]) / tooth.voxel
        pool, third = tooth.coords, None
        w = cr.get("third_weights")
        if w is not None and not empty and tooth.z_canal is not None:
            p = np.asarray(w, float)
            k = int(rng.choice(3, p=p / p.sum()))
            if len(tooth.coords_by_third[k]):
                pool, third = tooth.coords_by_third[k], k
        self.stats["pick_start"] += 1
        start = None
        for _ in range(20):
            c = pool[rng.integers(len(pool))] + rng.uniform(-jit, jit, 3)
            if third is not None and int(tooth.third_of_z(c[0])) != third:
                continue
            start = np.round(c - n / 2).astype(int)
            if empty:
                return start
            canal = tooth.read(tooth.canal, start, n)
            if canal.sum() * tooth.voxel ** 3 >= cr["min_canal_mm3"]:
                return start
        self.stats["sin_cumplir"] += 1
        return start if start is not None else np.round(pool[0] - n / 2).astype(int)   # ningún intento mantuvo el tercio

    # ---- el parche
    @torch.no_grad()
    def patch(self, tooth: Tooth, rng, start=None, params=None, center=False):
        cfg, m = self.cfg, self.m
        n = self.crop_size(tooth.voxel)
        ex = self.exact
        P = params or self.sample_params(rng, center)
        start = self.pick_start(tooth, rng) if start is None else np.asarray(start)
        gen = torch.Generator(device=DEV)
        gen.manual_seed(int(rng.integers(2 ** 31 - 1)))
        vx_in = tooth.voxel
        extent = n * vx_in
        norm = torch.from_numpy(tooth.read(tooth.norm, start, n)).to(DEV).float()      # float16 -> GPU y convertir allí
        canal = torch.from_numpy(tooth.read(tooth.canal, start, n)).to(DEV).float()
        chamber = (torch.from_numpy(tooth.read(tooth.chamber, start, n)).to(DEV).float()
                   if tooth.chamber is not None else None)
        lab_cfg = cfg.get("labels", {})

        # 2. grises. Lineal con aire = -1000: rho = densidad relativa a la dentina (dentina = 1)
        air = float(cfg["intensity"]["air_hu"])
        scale = P["dentin_hu"] - air
        ref = cfg["intensity"]["dentin_ref_norm"]
        a = norm.clamp(0, 1)
        rho = (a + P["enamel_gain"] * (norm - 1).clamp(min=0)) / ref
        # fondo (hueso / médula / tejido blando) en malla gruesa de 64 µm y luego a 16 µm
        f = max(int(round(0.064 / vx_in)), 1)          # malla gruesa de ~64 µm (4 vóxeles con un µCT de 16 µm)
        nc = max(int(round(n / f)), 4)
        bg = P["bg_base"] + P["bg_amp"] * smooth_field(extent, P["bg_sigma_mm"], (nc, nc, nc), gen)
        bg = bg.clamp(*cfg["background"]["clip_hu"])
        if P["pdl"]:
            solid = (norm > 0.5).float()[None, None]
            tooth_c = (F.avg_pool3d(solid, n // nc) if n % nc == 0 else F.adaptive_avg_pool3d(solid, nc))[0, 0]
            sg = P["pdl_width_mm"] / (extent / nc)
            halo = (blur(tooth_c, (sg, sg, sg)) > 0.15) & (tooth_c < 0.5)
            bg = torch.where(halo, torch.full_like(bg, P["pdl_hu"]), bg)
        bg = F.interpolate(bg[None, None], size=(n, n, n), mode="trilinear", align_corners=False)[0, 0]
        # el conducto y su borde de 3 vóxeles se rellenan con gris de pulpa. La cámara y las cavidades
        # (camara_y_cavidades.npy) también: en un paciente contienen pulpa, no "hueso con textura".
        pulp = torch.full_like(bg, P["canal_hu"])
        r_rim = max(int(round(0.048 / vx_in)), 1)               # borde de 48 µm (3 vóxeles con un µCT de 16 µm)
        fill = canal
        if chamber is not None and lab_cfg.get("chamber_fill", "pulp") == "pulp":
            fill = torch.maximum(canal, chamber)
            cham_d = dilate_box(chamber, r_rim)
            pulp = pulp + P["chamber_offset_hu"] * cham_d       # un poco distinta del conducto, no idéntica
            del cham_d
        cdil = dilate_box(fill, r_rim)
        bg = torch.where(cdil, pulp, bg)
        del pulp, fill
        rho = rho + (1 - a) * (bg - air) / scale
        del bg, cdil, a
        img16 = air + scale * rho
        del rho
        # 3-4. emborronado y reducción
        img32 = F.avg_pool3d(img16[None, None], 2)[0, 0]
        del img16
        s32 = P["psf_sigma_mm"] / (2 * vx_in)
        img32 = blur(img32, (s32 * P["psf_z_factor"], s32, s32))
        img = reduce_to(img32, m, ex)
        del img32
        # 5. ruido: Poisson antes de filtrar (asimetría), luego filtro gaussiano (correlación)
        d = ((img - air) / scale).clamp(0.05, 3.0)
        lam = P["poisson_counts"] * d ** (2 * P["noise_alpha"])
        e = (torch.poisson(lam, generator=gen) - lam) / math.sqrt(P["poisson_counts"])
        cs = (P["noise_corr_vox"],) * 3
        noise = blur(e, cs) * (P["noise_sd_hu"] / noise_gain(cs))
        ext_out = m * self.cfg["crop"]["voxel_out_mm"]
        mult = 1 + P["shade_mult"] * smooth_field(ext_out, P["shade_sigma_mm"], (m, m, m), gen)
        add = P["shade_add_hu"] * smooth_field(ext_out, P["shade_sigma_mm"], (m, m, m), gen)
        img = (img * mult + add + noise).clamp(min=air)
        # 6. etiqueta con la misma geometría
        soft = pool_to(canal, m, ex)
        toothf = pool_to((norm > 0.5).float(), m, ex)
        # peso para la pérdida: 0 = "ignorar". weight_mode: none (todo cuenta), band (franja alrededor del
        # orificio, donde conducto y cámara se tocan: ahí el corte es una convención de ±0.1 mm) o chamber
        # (toda la cámara y cavidades quitadas).
        mode = lab_cfg.get("weight_mode", "band")
        weight = torch.ones((m, m, m), device=DEV)
        cham_soft = torch.zeros((m, m, m), device=DEV)
        if chamber is not None:
            cham_soft = pool_to(chamber, m, ex)
            if mode == "chamber":
                weight = (cham_soft < 0.2).float()
            elif mode == "band":
                rb = int(round(lab_cfg.get("ignore_band_mm", 0.15) / vx_in))
                band = (dilate_box(canal, rb) & dilate_box(chamber, rb)).float()
                weight = (pool_to(band, m, ex) < 0.2).float()
                del band
        out = {"image": img, "label_soft": soft, "label": (soft >= 0.5).to(torch.uint8), "tooth": toothf,
               "weight": weight, "chamber": cham_soft}
        # aumentos en el plano xy (ejes 1 y 2); z no se toca
        if self.cfg["crop"].get("aug_xy", False) and not center:
            flips = [ax for ax in (1, 2) if rng.random() < 0.5]
            swap = rng.random() < 0.5
            for k, v in out.items():
                if flips:
                    v = torch.flip(v, flips)
                if swap:
                    v = v.transpose(1, 2).contiguous()
                out[k] = v
            P["aug"] = {"flips": flips, "swap_xy": bool(swap)}
        P["start_zyx_vox"] = [int(s) for s in start]
        P["voxel_in_mm"] = vx_in
        P["voxel_out_efectivo_mm"] = n * vx_in / m
        P["tercio_centro"] = int(tooth.third_of_z(start[0] + n / 2))
        return out, P


# ------------------------------------------------------------------------------ control visual
def figure(path: Path, tooth: Tooth, rec, real=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out, P = rec
    img = out["image"].cpu().numpy()
    lab = out["label"].cpu().numpy()
    soft = out["label_soft"].cpu().numpy()
    z, y, x = [int(round(c)) for c in np.argwhere(lab > 0).mean(0)] if lab.any() else [s // 2 for s in img.shape]
    # el µCT del mismo bloque, a la misma escala (se ve la geometría de la que sale): sin giro xy, por eso se
    # deshace el aumento solo para dibujar el µCT
    n = int(round(img.shape[0] * P["voxel_out_efectivo_mm"] / tooth.voxel))
    st = P["start_zyx_vox"]
    ct = tooth.read(tooth.norm, st, n).astype(np.float32)
    lc = tooth.read(tooth.canal, st, n)
    aug = P.get("aug", {})
    if aug:
        for ax in aug["flips"]:
            ct, lc = np.flip(ct, ax), np.flip(lc, ax)
        if aug["swap_xy"]:
            ct, lc = ct.transpose(0, 2, 1), lc.transpose(0, 2, 1)
    f = n / img.shape[0]
    cols = 3 if real is not None else 2
    fig, ax = plt.subplots(3, cols, figsize=(5 * cols, 15))
    views = (("axial", lambda a: a[z], lambda a: a[int(z * f)]), ("coronal", lambda a: a[:, y], lambda a: a[:, int(y * f)]),
             ("sagital", lambda a: a[:, :, x], lambda a: a[:, :, int(x * f)]))
    for r, (name, vs, vc) in enumerate(views):
        ax[r, 0].imshow(vc(ct)[::4, ::4], cmap="gray", vmin=0, vmax=1.6)
        ax[r, 0].contour(vc(lc)[::4, ::4], [0.5], colors="r", linewidths=0.6)
        ax[r, 0].set_title(f"µCT {tooth.name} {name} (16 µm, norm.) + etiqueta")
        ax[r, 1].imshow(vs(img), cmap="gray", vmin=-500, vmax=3500)
        ax[r, 1].contour(vs(soft), [0.5], colors="r", linewidths=0.6)
        cham = out["chamber"].cpu().numpy()
        if cham.max() > 0.5:
            ax[r, 1].contour(vs(cham), [0.5], colors="c", linewidths=0.6)
        ig = out["weight"].cpu().numpy() < 0.5
        if ig.any():
            ax[r, 1].contour(vs(ig.astype(np.float32)), [0.5], colors="y", linewidths=0.6)
        ax[r, 1].set_title(f"SIMULADO {name} (75 µm): rojo=etiqueta, cian=cámara, amarillo=ignorar")
        if real is not None:
            rimg, rseg = real
            rz, ry, rx = [int(round(c)) for c in np.argwhere(rseg).mean(0)]
            h = img.shape[0] // 2
            sl = (slice(rz - h, rz + h), slice(ry - h, ry + h), slice(rx - h, rx + h))
            cr, cs_ = rimg[sl], rseg[sl]
            pick = {"axial": lambda a: a[h], "coronal": lambda a: a[:, h], "sagital": lambda a: a[:, :, h]}[name]
            ax[r, 2].imshow(pick(cr), cmap="gray", vmin=-500, vmax=3500)
            ax[r, 2].contour(pick(cs_), [0.5], colors="r", linewidths=0.6)
            ax[r, 2].set_title(f"REAL (Isa, borrador de seg.) {name}")
        for a_ in ax[r]:
            a_.axis("off")
    plt.suptitle("dentina {:.0f} HU | canal {:.0f} | PSF {:.3f} mm | ruido sd {:.0f} corr {:.2f} | fondo {:.0f}±{:.0f}".format(
        P["dentin_hu"], P["canal_hu"], P["psf_sigma_mm"], P["noise_sd_hu"], P["noise_corr_vox"], P["bg_base"], P["bg_amp"]))
    plt.tight_layout()
    plt.savefig(path, dpi=45)
    plt.close(fig)


# ------------------------------------------------------------------------------------ main
def load_real(cbct, seg):
    import nrrd
    img = nrrd.read(str(cbct))[0].astype(np.float32)
    s = nrrd.read(str(seg))[0] > 0
    # el NRRD viene en (x,y,z); el simulador trabaja en (z,y,x)
    return img.transpose(2, 1, 0), s.transpose(2, 1, 0)


def check(sim: Simulator, teeth, rng, n, calib_path, center):
    """Mide en n parches simulados lo mismo que se midió en el CBCT real y lo compara."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import calibrate_cbct as cc
    imgs, segs = [], []
    for i in range(n):
        out, _ = sim.patch(teeth[i % len(teeth)], rng, center=center)
        imgs.append(out["image"].cpu().numpy().transpose(2, 1, 0).astype(np.float32))   # a (x,y,z) como el NRRD
        segs.append(out["label"].cpu().numpy().transpose(2, 1, 0) > 0)
    res = cc.measure(imgs, segs, sim.cfg["crop"]["voxel_out_mm"])
    ref = json.loads(Path(calib_path).read_text())
    rows = []
    def g(d, *k):
        for q in k:
            if d is None or q not in d:
                return None
            d = d[q]
        return d
    for name, path in (("gris dentina (mediana)", ("grises", "dentina", "mediana")),
                       ("gris esmalte (mediana)", ("grises", "esmalte", "mediana")),
                       ("gris tejido lejano (mediana)", ("grises", "tejido_lejano", "mediana")),
                       ("ruido sd dentina", ("ruido", "dentina", "sd_mediana")),
                       ("ruido sd tejido blando", ("ruido", "blando", "sd_mediana")),
                       ("autocorr lag1 dentina (x)", ("ruido", "dentina", "autocorr_lag1_x_y_z")),
                       ("asimetría ruido dentina", ("ruido", "dentina", "asimetria_mediana")),
                       ("curtosis ruido dentina", ("ruido", "dentina", "curtosis_exceso_mediana")),
                       ("borde dentina–exterior sigma (mm)", ("bordes", "dentina_exterior", "sigma_mm")),
                       ("borde esmalte–exterior sigma (mm)", ("bordes", "esmalte_exterior", "sigma_mm")),
                       ("borde dentina–conducto sigma (mm)", ("bordes", "dentina_conducto", "sigma_mm"))):
        a, b = g(ref, *path), g(res, *path)
        a = a[0] if isinstance(a, list) else a
        b = b[0] if isinstance(b, list) else b
        rows.append((name, a, b))
    print(f"{'magnitud':40s} {'Isa (real)':>12s} {'simulado':>12s}")
    for name, a, b in rows:
        fa = f"{a:12.3f}" if a is not None else f"{'-':>12s}"
        fb = f"{b:12.3f}" if b is not None else f"{'-':>12s}"
        print(f"{name:40s} {fa} {fb}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("teeth", type=Path, nargs="+", help="carpetas results/microct/Diente_XX")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("simulate_cbct_config.yaml"))
    ap.add_argument("--n", type=int, default=4, help="nº de parches (se reparten entre los dientes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/sim_cbct"))
    ap.add_argument("--figs", type=int, default=0, help="nº de imágenes de control a guardar")
    ap.add_argument("--real-cbct", type=Path, default=None)
    ap.add_argument("--real-seg", type=Path, default=None)
    ap.add_argument("--check", action="store_true", help="medir el simulador con las mismas funciones que el CBCT real (con muestreo proporcional, ver arriba)")
    ap.add_argument("--center", action="store_true", help="con --check: parámetros en el centro de cada rango")
    ap.add_argument("--calibration", type=Path, default=None)
    a = ap.parse_args()
    if DEV.type == "cuda":
        torch.set_default_device("cuda")
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    if a.check:
        # Isa se midió en un recorte que incluye la corona: la comprobación usa el muestreo proporcional al volumen
        # (como se calibró). Con parches repartidos por tercios entra menos esmalte y esas medidas cambian por el
        # contenido de los parches, no por el simulador.
        cfg["crop"]["third_weights"] = None
    sim = Simulator(cfg)
    teeth = [Tooth(t) for t in a.teeth]
    rng = np.random.default_rng(a.seed)
    if a.check:
        check(sim, teeth, rng, a.n, a.calibration, a.center)
        raise SystemExit
    a.out.mkdir(parents=True, exist_ok=True)
    real = load_real(a.real_cbct, a.real_seg) if a.real_cbct and a.figs else None
    for i in range(a.n):
        t = teeth[i % len(teeth)]
        t0 = time.time()
        rec = sim.patch(t, rng)
        out, P = rec
        tag = f"{t.name}_s{a.seed}_{i:03d}"
        np.savez_compressed(a.out / f"{tag}.npz", image=out["image"].cpu().numpy(),
                            label_soft=out["label_soft"].cpu().numpy().astype(np.float16),
                            label=out["label"].cpu().numpy(), tooth=out["tooth"].cpu().numpy().astype(np.float16),
                            weight=out["weight"].cpu().numpy().astype(np.uint8),
                            chamber=out["chamber"].cpu().numpy().astype(np.float16))
        (a.out / f"{tag}.json").write_text(json.dumps({"tooth": t.name, "seed": a.seed, "index": i, "params": P}, indent=1))
        if i < a.figs:
            figure(a.out / f"{tag}.png", t, rec, real)
        mem = f", GPU máx {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if DEV.type == "cuda" else ""
        print(f"{tag}: {time.time() - t0:.1f} s, canal {float(out['label'].sum()) * 0.075**3:.2f} mm3 en el parche{mem}")
