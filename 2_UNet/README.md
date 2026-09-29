# 2_UNet — Entrenamiento de redes de segmentación 3D

Entrenamiento de UNet 3D, Attention UNet y variantes para segmentar pulpa y nervio alveolar inferior (IAN) en CBCT.

El código parte del framework de entrenamiento de Pulpy3D (su modelo de seeding está en [Pulpy3D-Seed](https://github.com/mahmoudgamal0/Pulpy3D-Seed)), que a su vez se basa en el ToothFairy Challenge (MICCAI 2023) y en [alveolar_canal](https://github.com/AImageLab-zip/alveolar_canal) de AImageLab. Viene de `1_Data_Validation` con los errores ya corregidos (ver `../1_Data_Validation/ORGANIZACION.md`, sección 4).

---

## Estructura

```
2_UNet/
├── main.py                 ← punto de entrada: python main.py -c configs/<x>.yml
├── eval.py                 ← métricas: IoU, DICE, HD95
├── requirements.txt
├── configs/                ← un .yml por experimento + augmentations y preprocesado
├── dataloader/             ← carga de pacientes (torchio) y transformaciones de etiquetas
├── experiments/            ← bucle de entrenamiento/test por tipo de tarea
├── models/                 ← UNet3D, PosPadUNet3D, Attention UNet, UNETR, VNet, MultiHead
├── losses/                 ← Jaccard, Dice + CrossEntropy
├── optimizers/, schedulers/
├── datasets/               ← DATOS (no van a git)
│   └── Segmentaciones 3DSlicer/   ← 33 pacientes: CBCT (1 data.nrrd) + segmentación manual (.seg.nrrd)
├── results/                ← salidas de cada entrenamiento (no van a git)
├── checkpoints/            ← pesos (no van a git)
└── .venv/                  ← entorno virtual (no va a git)
```

### Qué hay en cada parte

| Carpeta | Contenido | Se elige en el config con |
| --- | --- | --- |
| `experiments/` | `PulpSegmentation` (pulpa, 1 clase), `IANSegmentation` (nervio, 1 clase), `Semantic` (fondo/IAN/pulpa), `Instance` (un canal por diente), `MultiHead` (pulpa + IAN con dos cabezas) | `experiment.name` |
| `models/` | `UNet3D`, `PosPadUNet3D`, `AttentionPosPadUNet3D`, `AttentionInstancePosPadUNet3D`, `MultiHeadPosPadUNet3D`, `UNETR`, `InstanceUNETR`, `VNet3D` | `model.name` |
| `losses/` | `Jaccard` (binaria), `DiceCrossEntropy` (multiclase) | `loss.name` |
| `optimizers/`, `schedulers/` | `SGD`, `Adam`; `Plateau`, `MultiStepLR` | `optimizer.name`, `lr_scheduler.name` |

---

## ⚠️ Estado actual

El código está listo, pero **el dataloader todavía espera el formato de Pulpy3D**:
`datasets/Pulpy3D/<paciente>/data.nii.gz` + `gt_*.nii.gz`, más `splits.json` y `dataset.json`.

Los datos de este proyecto son las segmentaciones de 3D Slicer (`.nrrd`, con las etiquetas numeradas de forma distinta en cada paciente). **Siguiente paso: adaptar `dataloader/Pulpy.py`** para leerlas, definir qué etiquetas se entrenan y crear su `splits.json`.

---

## Entorno

### En este portátil (programar y probar en CPU; **aquí no se puede entrenar**, no hay GPU NVIDIA)
Ya está creado en `.venv/` (Python 3.13 + torch CPU):
```
.venv\Scripts\activate
```
Si hubiera que rehacerlo:
```
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### En la máquina con GPU NVIDIA (para entrenar)
1. Clonar este repositorio y copiar aparte `datasets/` (no está en git).
2. Crear el entorno:
   ```
   python -m venv .venv
   source .venv/bin/activate              # en Windows: .venv\Scripts\activate
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   ```
3. Comprobar la GPU: `python -c "import torch; print(torch.cuda.is_available())"` tiene que imprimir `True`.

---

## Entrenar

```
python main.py -c configs/pulp_segmentation.yml --verbose
```
Otros experimentos: `ian_segmentation.yml`, `semantic.yml`, `instance.yml`, `multihead.yml`.

- Todo se guarda en `results/entrenamientos/<titulo>_<hash>/`: `output.log`, copia del `config.yaml` y `checkpoints/` (`best.pth` = el mejor en validación, `last.pth`, `test__N.pth`).
- **Solo evaluar o predecir**: en el `.yml`, `do_train: False`, `do_test: True` y/o `do_predict: True`, y en `checkpoint:` la ruta a un `best.pth`. Las predicciones salen en `.../outputs/<paciente>.nii.gz`, con la misma geometría que el CBCT (se pueden abrir en 3D Slicer encima).
- **Continuar un entrenamiento cortado**: `reload: True` y `checkpoint:` apuntando a su `last.pth`.
- **wandb**: por defecto guarda en local (`wandb/`). Para la web: `wandb login` y `WANDB_MODE=online WANDB_PROJECT=<proyecto> python main.py ...`. Para desactivarlo: `--debug`.

**A tener en cuenta**
- Cada 5 épocas se evalúa también en test y se guarda `test__N.pth` (el mejor **en test**). Para resultados publicables usa `best.pth`, elegido con validación: elegir el modelo mirando el test infla los resultados.
- `configs/augmentation.yml` aplica `RandomGhosting` y `RandomSwap` a todas las muestras (no tienen `p`). Viene así del original.
- Todo el código usa `.cuda()`: fuera de una GPU NVIDIA solo sirve para programar o revisar, no para ejecutar el entrenamiento.
