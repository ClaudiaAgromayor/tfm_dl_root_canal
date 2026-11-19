"""
Ordena la carpeta datasets/CBCT_75um (CBCT reales de 75 um de Isa; datos de paciente).

Por defecto es un ENSAYO (--dry-run): solo muestra "nombre antiguo -> nombre nuevo" y no
cambia nada. Para mover de verdad hay que pasar --aplicar.

    python ordenar_cbct_75um.py                      # ensayo (calcula hashes, tarda unos minutos)
    python ordenar_cbct_75um.py --rapido             # ensayo sin hashes
    python ordenar_cbct_75um.py --aplicar            # mueve, comprueba hashes y escribe LEEME.txt
    python ordenar_cbct_75um.py --deshacer mapeo_renombrado.csv

Garantias:
  * Los ficheros se MUEVEN dentro del mismo disco (os.rename): no se copian ni se reescriben.
  * El hash SHA-256 se calcula antes y se comprueba despues; si no coincide, se avisa.
  * Nunca se sobrescribe ni se borra un fichero. Los duplicados se apartan en _duplicados/.
  * El CSV (mapeo_renombrado.csv) guarda antiguo, nuevo, tamano y hash de cada movimiento,
    y sirve para --deshacer. Contiene los nombres antiguos de los zip (que llevan el nombre
    del paciente): queda dentro de CBCT_75um, que esta en .gitignore. No lo subas.
  * Ni este script ni los LEEME.txt escriben nombres de pacientes.
"""
import argparse
import csv
import hashlib
import os
import re
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
DEFAULT_ROOT = HERE.parents[2] / "datasets" / "CBCT_75um"

# ---------------------------------------------------------------------------------------
# Plan: (origen como patron relativo a la raiz, destino relativo). Cada patron debe
# encontrar exactamente 1 fichero. No hay nombres de pacientes: los zip se buscan por
# "*ORIG-CBCT<N>.zip".
# ---------------------------------------------------------------------------------------
def zip_de(n):
    return f"wetransfer_cbcts-originales-*/*ORIG-CBCT{n}.zip"


PLAN = [
    # El escaneo original (hoy Isa_caso01, origen desconocido) pasa a Isa_caso00 para liberar el 01.
    # Estos dos movimientos DEBEN ir primero: Isa_caso01/Segmentation.seg.nrrd es tambien el destino del caso 1.
    ("Isa_caso01/1 3D CBCT Image Set.nrrd", "Isa_caso00/CBCT.nrrd"),
    ("Isa_caso01/Segmentation.seg.nrrd", "Isa_caso00/Segmentation.seg.nrrd"),
    # caso 1 (3DSlice1.nrrd y 3DSlice1.dcm.nrrd tienen el mismo hash: el segundo es copia)
    ("wetransfer_caso-1-*/3DSlice1.nrrd", "Isa_caso01/CBCT.nrrd"),
    ("wetransfer_caso-1-*/Segmentation.seg.nrrd", "Isa_caso01/Segmentation.seg.nrrd"),
    ("wetransfer_caso-1-*/3DSlice1.dcm.nrrd", "_duplicados/wt_caso01_3DSlice1.dcm.nrrd"),
    (zip_de(1), "Isa_caso01/DICOM_original.zip"),
    # caso 2
    ("wetransfer_caso-2-*/1 3D CBCT Image Set.nrrd", "Isa_caso02/CBCT.nrrd"),
    ("wetransfer_caso-2-*/Segmentation.seg.nrrd", "Isa_caso02/Segmentation.seg.nrrd"),
    (zip_de(2), "Isa_caso02/DICOM_original.zip"),
    # caso 3: la pareja buena es la de 1161 cortes (Image_2 + Segmentation_1)
    ("wetransfer_caso-3-*/1 3D CBCT Image Set_2.nrrd", "Isa_caso03/CBCT.nrrd"),
    ("wetransfer_caso-3-*/Segmentation_1.seg.nrrd", "Isa_caso03/Segmentation.seg.nrrd"),
    (zip_de(3), "Isa_caso03/DICOM_original.zip"),
    # la otra pareja de la carpeta del caso 3: su imagen es IDENTICA a la del caso 4
    ("wetransfer_caso-3-*/1 3D CBCT Image Set_1.nrrd", "_duplicados/wt_caso03_Image_Set_1(=imagen_del_caso04).nrrd"),
    ("wetransfer_caso-3-*/Segmentation.seg.nrrd", "Isa_caso04/Segmentation_otra_encontrada_en_carpeta_caso3.seg.nrrd"),
    # caso 4
    ("wetransfer_caso-4-*/1 3D CBCT Image Set.nrrd", "Isa_caso04/CBCT.nrrd"),
    ("wetransfer_caso-4-*/Segmentation.seg.nrrd", "Isa_caso04/Segmentation.seg.nrrd"),
    (zip_de(4), "Isa_caso04/DICOM_original.zip"),
    # caso 5: solo hay DICOM
    (zip_de(5), "Isa_caso05/DICOM_original.zip"),
    # caso 6
    ("wetransfer_caso-6-*/1 3D CBCT Image Set.nrrd", "Isa_caso06/CBCT.nrrd"),
    ("wetransfer_caso-6-*/Segmentation.seg.nrrd", "Isa_caso06/Segmentation.seg.nrrd"),
    (zip_de(6), "Isa_caso06/DICOM_original.zip"),
    # caso 7
    ("wetransfer_caso-7-*/1 3D CBCT Image Set.nrrd", "Isa_caso07/CBCT.nrrd"),
    ("wetransfer_caso-7-*/Segmentation.seg.nrrd", "Isa_caso07/Segmentation.seg.nrrd"),
    (zip_de(7), "Isa_caso07/DICOM_original.zip"),
    # caso 8
    ("wetransfer_caso-8-*/1 3D CBCT Image Set.nrrd", "Isa_caso08/CBCT.nrrd"),
    ("wetransfer_caso-8-*/Segmentation.seg.nrrd", "Isa_caso08/Segmentation.seg.nrrd"),
    (zip_de(8), "Isa_caso08/DICOM_original.zip"),
    # casos 9 y 10: solo hay DICOM
    (zip_de(9), "Isa_caso09/DICOM_original.zip"),
    (zip_de(10), "Isa_caso10/DICOM_original.zip"),
]

# Notas para el LEEME.txt de cada carpeta (solo lo que se sabe; lo dudoso va marcado).
NOTAS = {
    "Isa_caso00": [
        "Es la entrega original (7-oct-2026) con la que se hizo la calibracion del simulador.",
        "NO coincide con ningun caso de la entrega de WeTransfer (9-11 nov): hash distinto de todos y, al",
        "comparar un corte con los DICOM de los zips 1, 2, 4, 5, 6, 7, 9 y 10, ninguno coincide (r < 0,82 o",
        "tamano distinto). Origen sin confirmar; preguntar a Isa. El segmento se llama 'Segment_1' (nombre",
        "automatico), no 'PULP'.",
    ],
    "Isa_caso03": [
        "La carpeta original traia dos parejas imagen+segmentacion. Aqui esta la de 1161 cortes (coincide con",
        "el zip CBCT3, que tiene 1161 cortes DICOM). La otra pareja tiene la imagen identica a la del caso 4",
        "y se ha borrado (copia exacta, comprobada por hash) (ver Isa_caso04/).",
    ],
    "Isa_caso04": [
        "Incluye 'Segmentation_otra_encontrada_en_carpeta_caso3.seg.nrrd': una segmentacion distinta (Dice 0,60",
        "con la de este caso) hecha sobre ESTA misma imagen, que Isa dejo en la carpeta del caso 3.",
        "Sin confirmar con Isa si es una version anterior, un error de carpeta u otra cosa.",
        "Hay dos zip DICOM (CBCT4 y CBCT5) del mismo paciente con ~1 min de diferencia; esta imagen sale del",
        "CBCT4 (corte 344 identico, r = 1,0). El CBCT5 es otro escaneo (551 columnas).",
    ],
    "Isa_caso05": ["Solo DICOM (551 columnas). Sin NRRD ni segmentacion. Mismo paciente que el caso 4 (escaneo repetido)."],
    "Isa_caso09": ["Solo DICOM. Sin NRRD ni segmentacion. Mismo paciente que el caso 10 (escaneo repetido)."],
    "Isa_caso10": ["Solo DICOM. Sin NRRD ni segmentacion. Mismo paciente que el caso 9 (escaneo repetido)."],
    "Isa_caso01": ["Imagen con 551 columnas (no 555) y origen x distinto: otro campo de vision que Isa_caso00."],
    "Isa_caso07": ["Imagen con 551 columnas (no 555) y origen x distinto: otro campo de vision que Isa_caso00."],
    "Isa_caso08": ["Volumen mas pequeno (489x531x552) y con otro origen."],
}
GENERAL = [
    "Emparejado caso N <-> DICOM 'CBCT N' por el nombre y comprobado comparando un corte de cada NRRD",
    "con el DICOM de su zip (correlacion 1,0) en los casos 1, 2, 3, 4, 6, 7 y 8 (17-nov-2026).",
    "Datos de paciente: no compartir, no subir a GitHub.",
]


# ---------------------------------------------------------------------------------------
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def leer_cabecera(p):
    """Devuelve (dict de campos NRRD, posicion donde empiezan los datos)."""
    with open(p, "rb") as f:
        buf = b""
        while b"\n\n" not in buf and len(buf) < 4_000_000:
            c = f.read(65536)
            if not c:
                break
            buf += c
    cab, _, _ = buf.partition(b"\n\n")
    d = {}
    for linea in cab.decode("latin-1").splitlines()[1:]:
        m = re.match(r"([^:=]+?)\s*(?::=|:)\s*(.*)", linea)
        if m:
            d[m.group(1)] = m.group(2)
    return d, len(cab) + 2


def info_nrrd(p):
    d, _ = leer_cabecera(p)
    sizes = d.get("sizes", "?")
    dirs = re.findall(r"\(([^)]*)\)", d.get("space directions", ""))
    vox = [float(x) for x in dirs[0].split(",")] if dirs else []
    pasos = [round(abs(max(v for v in vox if v != 0)) if vox else 0, 4)] if vox else []
    return d, sizes, pasos


def volumen_seg(p):
    """Vóxeles distintos de 0 y volumen en mm3 de una segmentacion .seg.nrrd (gzip, uint8)."""
    d, ini = leer_cabecera(p)
    if d.get("encoding") != "gzip" or d.get("type") != "unsigned char":
        return None, None
    with open(p, "rb") as f:
        f.seek(ini)
        datos = zlib.decompress(f.read(), 16 + zlib.MAX_WBITS)
    a = np.frombuffer(datos, dtype=np.uint8)
    n = int((a > 0).sum())
    dirs = re.findall(r"\(([^)]*)\)", d.get("space directions", ""))
    v = np.prod([max(abs(float(x)) for x in s.split(",")) for s in dirs]) if dirs else float("nan")
    return n, n * v


def texto_leeme(carpeta, raiz, origenes):
    """origenes: {nombre_destino: nombre_origen_antiguo (solo carpeta wetransfer)}."""
    L = [f"{carpeta.name}", "=" * len(carpeta.name), ""]
    cbct, seg = carpeta / "CBCT.nrrd", carpeta / "Segmentation.seg.nrrd"
    if cbct.exists():
        d, sizes, pasos = info_nrrd(cbct)
        L += [f"CBCT.nrrd: tamano {sizes} vox, voxel {pasos[0] if pasos else '?'} mm, "
              f"origen {d.get('space origin', '?')}, {cbct.stat().st_size / 1e6:.0f} MB"]
    else:
        L += ["CBCT.nrrd: no hay."]
    if seg.exists():
        d, sizes, _ = info_nrrd(seg)
        estado = re.search(r"Segmentation\.Status:([^|]*)", d.get("Segment0_Tags", ""))
        n, mm3 = volumen_seg(seg)
        estado = estado.group(1) if estado else "desconocido"
        L += [f"Segmentation.seg.nrrd: segmento '{d.get('Segment0_Name', '?')}', estado '{estado}'"
              + (" (BORRADOR: no es referencia fiable)" if estado == "inprogress" else ""),
              f"    {n} voxeles marcados = {mm3:.1f} mm3 (incluye lo que Isa haya marcado: no se ha comprobado",
              "    si esta la camara pulpar); hecha a mano por Isa en 3D Slicer."]
    else:
        L += ["Segmentation.seg.nrrd: no hay."]
    if (carpeta / "DICOM_original.zip").exists():
        L += ["DICOM_original.zip: DICOM original del escaneo (el zip lleva el nombre del paciente dentro)."]
    otros = sorted(p.name for p in carpeta.iterdir()
                   if p.name not in ("CBCT.nrrd", "Segmentation.seg.nrrd", "DICOM_original.zip", "LEEME.txt"))
    for o in otros:
        L += [f"{o}: fichero extra (ver notas)."]
    L += ["", "Origen: " + origenes.get(carpeta.name, "ver mapeo_renombrado.csv")]
    if carpeta.name in NOTAS:
        L += ["", "Notas:"] + ["  " + t for t in NOTAS[carpeta.name]]
    L += ["", "Aviso:"] + ["  " + t for t in GENERAL]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------
def resolver_plan(raiz):
    mov, errores = [], []
    for pat, dst in PLAN:
        enc = sorted(raiz.glob(pat))
        if len(enc) != 1:
            errores.append(f"El patron {pat!r} encuentra {len(enc)} ficheros (debe ser 1)")
            continue
        mov.append((enc[0], raiz / dst))
    # Un destino que ya existe solo vale si ese fichero se va a mover ANTES a otro sitio.
    origen_pos = {s.resolve(): i for i, (s, _) in enumerate(mov)}
    for i, (s, d) in enumerate(mov):
        if d.exists() and origen_pos.get(d.resolve(), len(mov)) >= i:
            errores.append(f"El destino ya existe: {d.relative_to(raiz)}")
    dsts = [d for _, d in mov]
    if len(set(dsts)) != len(dsts):
        errores.append("Hay destinos repetidos en el plan")
    srcs = [s for s, _ in mov]
    if len(set(srcs)) != len(srcs):
        errores.append("Hay un origen usado dos veces en el plan")
    return mov, errores


def ficheros_no_cubiertos(raiz, mov):
    cubiertos = {s.resolve() for s, _ in mov}
    ignorar = {"mapeo_renombrado.csv", "mapeo_renombrado_ensayo.csv"}
    return sorted(p.relative_to(raiz) for p in raiz.rglob("*")
                  if p.is_file() and p.resolve() not in cubiertos and p.name not in ignorar
                  and p.name != "LEEME.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raiz", nargs="?", type=Path, default=DEFAULT_ROOT, help="carpeta CBCT_75um")
    ap.add_argument("--dry-run", action="store_true", help="ensayo (es el comportamiento por defecto)")
    ap.add_argument("--aplicar", action="store_true", help="mover de verdad")
    ap.add_argument("--rapido", action="store_true", help="en el ensayo, no calcular hashes")
    ap.add_argument("--deshacer", type=Path, metavar="CSV", help="revertir usando el CSV de un --aplicar anterior")
    a = ap.parse_args()
    raiz = a.raiz.resolve()
    if not raiz.is_dir():
        sys.exit(f"No existe {raiz}")

    if a.deshacer:
        return deshacer(raiz, a.deshacer.resolve())

    mov, errores = resolver_plan(raiz)
    if errores:
        print("ERRORES (no se hace nada):")
        for e in errores:
            print("  -", e)
        sys.exit(1)
    extra = ficheros_no_cubiertos(raiz, mov)
    if extra:
        print("AVISO: ficheros de la raiz que el plan no toca:", *map(str, extra), sep="\n  ")
    aplicar = a.aplicar and not a.dry_run
    calcular = aplicar or not a.rapido

    filas = []
    print(f"\n{'MODO: APLICAR' if aplicar else 'MODO: ENSAYO (no se cambia nada)'} - raiz {raiz}\n")
    for i, (s, d) in enumerate(mov, 1):
        h = sha256(s) if calcular else ""
        filas.append({"antiguo": str(s.relative_to(raiz)), "nuevo": str(d.relative_to(raiz)),
                      "bytes": s.stat().st_size, "sha256": h, "accion": "mover"})
        print(f"[{i:2d}/{len(mov)}] {filas[-1]['antiguo']}\n        -> {filas[-1]['nuevo']}"
              f"  ({filas[-1]['bytes'] / 1e6:.1f} MB)", flush=True)

    csv_path = raiz / ("mapeo_renombrado.csv" if aplicar else "mapeo_renombrado_ensayo.csv")
    if not aplicar:
        escribir_csv(csv_path, filas)
        print(f"\nEnsayo terminado. Mapeo guardado en {csv_path}\nNo se ha movido nada.")
        return

    if (raiz / "mapeo_renombrado.csv").exists():
        sys.exit("Ya existe mapeo_renombrado.csv: parece que ya se aplico. Usa --deshacer primero.")
    # 1) mover (os.rename en el mismo disco)
    hechos = []
    try:
        for f, (s, d) in zip(filas, mov):
            d.parent.mkdir(parents=True, exist_ok=True)
            os.rename(s, d)
            hechos.append((s, d))
    except Exception as e:
        print("ERROR moviendo, se revierte lo hecho:", e)
        for s, d in reversed(hechos):
            os.rename(d, s)
        sys.exit(1)
    # 2) hashes despues
    mal = 0
    for f, (s, d) in zip(filas, mov):
        if sha256(d) != f["sha256"]:
            mal += 1
            print("HASH DISTINTO:", f["nuevo"])
    print("\nComprobacion de hashes:", "TODO IGUAL" if not mal else f"{mal} FICHEROS DISTINTOS (no sigas, avisa)")
    escribir_csv(csv_path, filas)  # ya, para poder deshacer aunque falle algo de lo que sigue
    # 3) quitar carpetas antiguas SOLO si quedaron vacias (en Windows pueden estar abiertas en el
    #    Explorador o en el editor: si no se pueden borrar, solo se avisa)
    for s, _ in mov:
        p = s.parent
        if p != raiz and p.exists() and not any(p.iterdir()):
            try:
                p.rmdir()
            except OSError as e:
                print(f"AVISO: no se pudo borrar la carpeta vacia {p.name}: {e}")
    # 4) LEEME.txt
    origenes = {}
    for f, (s, d) in zip(filas, mov):
        if s.parent.name.startswith("wetransfer_caso"):
            origenes[d.parent.name] = f"carpeta WeTransfer '{s.parent.name}' (entrega del 9-11 nov 2026)"
    for carpeta in sorted(p for p in raiz.glob("Isa_*") if p.is_dir()):
        t = carpeta / "LEEME.txt"
        t.write_text(texto_leeme(carpeta, raiz, origenes), encoding="utf-8")
        filas.append({"antiguo": "", "nuevo": str(t.relative_to(raiz)), "bytes": t.stat().st_size,
                      "sha256": sha256(t), "accion": "crear"})
    escribir_csv(csv_path, filas)
    print(f"Hecho. Mapeo guardado en {csv_path}. Para revertir: --deshacer {csv_path.name}")


def escribir_csv(path, filas):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["antiguo", "nuevo", "bytes", "sha256", "accion"])
        w.writeheader()
        w.writerows(filas)


def deshacer(raiz, csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    filas = [r for r in filas if r["accion"] in ("mover", "crear")]  # "duplicado_borrado": ya no existe, se ignora
    nuevos = {r["nuevo"] for r in filas}
    # comprobar todo antes de tocar nada
    for r in filas:
        nuevo = raiz / r["nuevo"]
        if not nuevo.exists():
            sys.exit(f"Falta {r['nuevo']}: no se deshace nada.")
        # el antiguo puede existir si es, a la vez, el nuevo de otro movimiento (se vacia antes)
        if r["accion"] == "mover" and (raiz / r["antiguo"]).exists() and r["antiguo"] not in nuevos:
            sys.exit(f"Ya existe {r['antiguo']}: no se deshace nada.")
        if r["accion"] == "mover" and sha256(nuevo) != r["sha256"]:
            sys.exit(f"El hash de {r['nuevo']} ya no coincide con el CSV: no se deshace nada.")
    # en orden inverso al de aplicar (los LEEME, creados al final, se borran primero)
    for r in reversed(filas):
        nuevo = raiz / r["nuevo"]
        if r["accion"] == "crear":
            if sha256(nuevo) == r["sha256"]:
                nuevo.unlink()
            else:
                print("LEEME modificado a mano, se deja:", r["nuevo"])
        else:
            antiguo = raiz / r["antiguo"]
            antiguo.parent.mkdir(parents=True, exist_ok=True)
            os.rename(nuevo, antiguo)
    for r in filas:  # carpetas nuevas que queden vacias
        p = (raiz / r["nuevo"]).parent
        if p != raiz and p.exists() and not any(p.iterdir()):
            p.rmdir()
    print("Deshecho. El CSV se conserva por si lo necesitas.")


if __name__ == "__main__":
    main()
