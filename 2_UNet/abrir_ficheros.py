"""
abrir_ficheros.py - Abre y revisa volumenes NRRD exportados desde 3D Slicer
(imagen CBCT + Segmentation.seg.nrrd).

Que hace:
  1. Busca en la carpeta la imagen (*.nrrd) y la mascara (*.seg.nrrd).
  2. Imprime forma, spacing, tipo, rango de intensidades y segmentos.
  3. Comprueba que imagen y mascara comparten geometria (tamano, spacing, origen).
  4. Abre un visor con cortes axial / coronal / sagital y la mascara superpuesta.
  5. (Opcional) Exporta a NIfTI con la estructura que espera dataloader/Pulpy.py.

Uso (copiar solo la linea del comando, sin la explicacion de debajo):
  python abrir_ficheros.py
      usa la RUTA de abajo
  python abrir_ficheros.py "datasets\\Ana\\diente_02_segmentation"
      abre otra carpeta
  python abrir_ficheros.py "datasets\\Ana\\diente_02_segmentation" --no-visor
      solo el informe, sin ventana
  python abrir_ficheros.py "datasets\\Ana\\diente_01_probando" --nifti "datasets\\Ana_nifti\\diente_01"
      exporta a data.nii.gz + gt_pulp.nii.gz

La carga tarda unos segundos: cada imagen ocupa 327 MB comprimida y ~1 GB en memoria.
"""
import argparse
from pathlib import Path

import nrrd
import numpy as np

# ---------------------------------------------------------------------------
# CAMBIA AQUI LA RUTA (carpeta con los .nrrd o directamente un fichero .nrrd)
RUTA = r"datasets\Ana\diente_01_probando"
# ---------------------------------------------------------------------------


def buscar_ficheros(ruta):
  ruta = Path(ruta)
  if ruta.is_file():
    ruta = ruta.parent
  segs = sorted(ruta.glob('*.seg.nrrd'))
  imgs = sorted(p for p in ruta.glob('*.nrrd') if not p.name.endswith('.seg.nrrd'))
  if not imgs:
    raise FileNotFoundError(f'No hay ninguna imagen .nrrd en {ruta}')
  if len(imgs) > 1:
    print(f'[aviso] Hay {len(imgs)} imagenes, uso la primera: {imgs[0].name}')
    print('        (si son copias del mismo volumen, puedes borrar las demas)')
  if len(segs) > 1:
    print(f'[aviso] Hay {len(segs)} segmentaciones, uso la primera: {segs[0].name}')
  return imgs[0], (segs[0] if segs else None)


def geometria(header):
  spacing = np.linalg.norm(np.asarray(header['space directions'], dtype=float), axis=1)
  origin = np.asarray(header.get('space origin', [0, 0, 0]), dtype=float)
  return spacing, origin


def describir(nombre, data, header):
  spacing, origin = geometria(header)
  tamano_mm = np.array(data.shape[:3]) * spacing
  print(f'\n=== {nombre}')
  print(f'  forma        : {data.shape}  ({data.dtype})')
  print(f'  spacing (mm) : {np.round(spacing, 4)}')
  print(f'  tamano (mm)  : {np.round(tamano_mm, 1)}')
  print(f'  origen       : {np.round(origin, 3)}')
  print(f'  espacio      : {header.get("space", "?")}')
  print(f'  rango        : [{data.min()}, {data.max()}]')


def segmentos(header):
  """Lista (nombre, label, capa) de los segmentos guardados por 3D Slicer."""
  out = []
  i = 0
  while f'Segment{i}_Name' in header:
    out.append((header[f'Segment{i}_Name'],
                int(header.get(f'Segment{i}_LabelValue', 1)),
                int(header.get(f'Segment{i}_Layer', 0))))
    i += 1
  return out


def mascara_en_espacio_imagen(seg, seg_header, img_shape):
  """Devuelve una mascara 3D con el mismo tamano que la imagen.

  3D Slicer puede guardar la segmentacion recortada (con un offset) o en 4D
  si hay segmentos solapados (una capa por segmento).
  """
  if seg.ndim == 4:  # capas: (capa, x, y, z)
    lab = np.zeros(seg.shape[1:], dtype=np.uint8)
    for k, capa in enumerate(seg):
      lab[capa > 0] = k + 1
    seg = lab
  if seg.shape == tuple(img_shape):
    return seg
  offset = [int(v) for v in seg_header.get('Segmentation_ReferenceImageExtentOffset', '0 0 0').split()]
  print(f'[info] Segmentacion recortada {seg.shape}, se coloca en la imagen con offset {offset}')
  full = np.zeros(img_shape, dtype=seg.dtype)
  sl = tuple(slice(o, o + s) for o, s in zip(offset, seg.shape))
  full[sl] = seg
  return full


def comprobar(img, img_h, seg, seg_h):
  ok = True
  sp_i, or_i = geometria(img_h)
  sp_s, or_s = geometria(seg_h)
  if img.shape != seg.shape:
    print(f'[ERROR] Tamano distinto: imagen {img.shape} vs mascara {seg.shape}'); ok = False
  if not np.allclose(sp_i, sp_s):
    print(f'[ERROR] Spacing distinto: {sp_i} vs {sp_s}'); ok = False
  offset = seg_h.get('Segmentation_ReferenceImageExtentOffset', '0 0 0')
  if offset.split() == ['0', '0', '0'] and not np.allclose(or_i, or_s, atol=1e-3):
    print(f'[ERROR] Origen distinto: {or_i} vs {or_s}'); ok = False

  fg = seg > 0
  n = int(fg.sum())
  vox_mm3 = float(np.prod(sp_i))
  print('\n=== Comprobaciones')
  print(f'  etiquetas en la mascara : {np.flatnonzero(np.bincount(seg.ravel())).tolist()}')
  print(f'  voxeles etiquetados     : {n}  ({100 * n / seg.size:.3f}% del volumen)')
  print(f'  volumen etiquetado      : {n * vox_mm3:.1f} mm3')
  if n == 0:
    print('[ERROR] La mascara esta vacia'); return False

  idx = np.argwhere(fg)
  lo, hi = idx.min(0), idx.max(0)
  print(f'  bounding box (voxeles)  : {lo.tolist()} -> {hi.tolist()}')
  dentro = img[fg]
  fuera = (img.sum(dtype=np.int64) - dentro.sum(dtype=np.int64)) / (img.size - n)
  print(f'  intensidad imagen dentro: media {dentro.mean():.0f}, p5-p95 {np.percentile(dentro, [5, 95]).round(0).tolist()}')
  print(f'  intensidad imagen fuera : media {fuera:.0f}')

  try:
    from scipy import ndimage
    # se trabaja solo en la caja que rodea la mascara (el volumen entero ocupa ~1 GB)
    caja = fg[tuple(slice(max(a - 1, 0), b + 2) for a, b in zip(lo, hi))]
    _, ncomp = ndimage.label(caja)
    huecos = int(ndimage.binary_fill_holes(caja).sum() - n)
    print(f'  componentes conexas     : {ncomp}' + ('  <- la pulpa deberia ser 1 pieza' if ncomp > 1 else ''))
    print(f'  voxeles de huecos       : {huecos}')
  except ImportError:
    pass

  print('\n  RESULTADO:', 'geometria correcta, se puede usar' if ok else 'HAY PROBLEMAS (ver arriba)')
  return ok


def visor(img, seg, titulo):
  import matplotlib.pyplot as plt
  from matplotlib.colors import ListedColormap
  from matplotlib.widgets import Slider

  vmin, vmax = np.percentile(img[::4, ::4, ::4], [0.5, 99.5])
  centro = np.argwhere(seg > 0).mean(0).astype(int) if seg is not None and seg.any() \
    else np.array(img.shape) // 2

  vistas = [('Axial (z)', 2), ('Coronal (y)', 1), ('Sagital (x)', 0)]
  fig, axes = plt.subplots(1, 3, figsize=(15, 6))
  fig.suptitle(titulo)
  plt.subplots_adjust(bottom=0.2)
  sliders = []

  def corte(vol, eje, i):
    # indexado directo (np.take copiaria el volumen entero, ~1 GB, en cada corte)
    sl = [slice(None)] * 3
    sl[eje] = i
    return np.ascontiguousarray(vol[tuple(sl)].T)  # .T para que la imagen salga derecha

  def capa_mascara(eje, i):
    m = corte(seg, eje, i)
    return np.ma.masked_where(m == 0, m)  # el fondo queda transparente

  # se crean las imagenes una vez y al mover la barra solo se cambian los datos
  for k, (nombre, eje) in enumerate(vistas):
    ax = axes[k]
    i0 = int(centro[eje])
    im = ax.imshow(corte(img, eje, i0).astype(np.float32), cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    ov = ax.imshow(capa_mascara(eje, i0), cmap=ListedColormap(['lime']), alpha=0.4,
                   interpolation='nearest', origin='lower') if seg is not None else None
    ax.set_title(f'{nombre}  corte {i0}')
    ax.set_axis_off()

    def actualizar(v, ax=ax, eje=eje, im=im, ov=ov, nombre=nombre):
      i = int(v)
      im.set_data(corte(img, eje, i).astype(np.float32))
      if ov is not None:
        ov.set_data(capa_mascara(eje, i))
      ax.set_title(f'{nombre}  corte {i}')
      fig.canvas.draw_idle()

    sax = fig.add_axes([0.05 + k * 0.33, 0.07, 0.25, 0.03])
    s = Slider(sax, '', 0, img.shape[eje] - 1, valinit=i0, valstep=1)
    s.on_changed(actualizar)
    sliders.append(s)

  fig._sliders = sliders  # evita que el recolector de basura borre los sliders
  plt.show()


def exportar_nifti(img, img_h, seg, destino, nombre_label='gt_pulp'):
  """Guarda data.nii.gz y <nombre_label>.nii.gz (formato de dataloader/Pulpy.py)."""
  import nibabel as nib

  dirs = np.asarray(img_h['space directions'], dtype=float)
  affine = np.eye(4)
  affine[:3, :3] = dirs.T
  affine[:3, 3] = np.asarray(img_h.get('space origin', [0, 0, 0]), dtype=float)
  if img_h.get('space', '').lower() in ('left-posterior-superior', 'lps'):
    affine[:2, :] *= -1  # NRRD usa LPS, NIfTI usa RAS

  destino = Path(destino)
  destino.mkdir(parents=True, exist_ok=True)
  nib.save(nib.Nifti1Image(img.astype(np.int16 if img.max() < 32768 else np.int32), affine),
           destino / 'data.nii.gz')
  nib.save(nib.Nifti1Image((seg > 0).astype(np.uint8), affine), destino / f'{nombre_label}.nii.gz')
  print(f'\nExportado a {destino}: data.nii.gz, {nombre_label}.nii.gz')
  print('Recuerda anadir el paciente a splits.json del dataset.')


def main():
  parser = argparse.ArgumentParser(description='Abrir y revisar NRRD de 3D Slicer')
  parser.add_argument('ruta', nargs='?', default=RUTA)
  parser.add_argument('--no-visor', action='store_true', help='no abrir la ventana con los cortes')
  parser.add_argument('--nifti', metavar='CARPETA', help='exportar a NIfTI en esta carpeta')
  parser.add_argument('--label', default='gt_pulp', help='nombre del fichero de etiqueta al exportar')
  args = parser.parse_args()

  img_path, seg_path = buscar_ficheros(args.ruta)
  print(f'Leyendo imagen: {img_path}')
  img, img_h = nrrd.read(str(img_path))
  describir(img_path.name, img, img_h)

  seg = None
  if seg_path is None:
    print('\n[aviso] No hay Segmentation.seg.nrrd en la carpeta')
  else:
    print(f'\nLeyendo mascara: {seg_path}')
    seg_raw, seg_h = nrrd.read(str(seg_path))
    describir(seg_path.name, seg_raw, seg_h)
    for nombre, label, capa in segmentos(seg_h):
      print(f'  segmento     : "{nombre}" (label {label}, capa {capa})')
    seg = mascara_en_espacio_imagen(seg_raw, seg_h, img.shape)
    comprobar(img, img_h, seg, seg_h)

  if args.nifti:
    if seg is None:
      print('[ERROR] No se puede exportar sin mascara')
    else:
      exportar_nifti(img, img_h, seg, args.nifti, args.label)

  if not args.no_visor:
    visor(img, seg, img_path.parent.name)


if __name__ == '__main__':
  main()
