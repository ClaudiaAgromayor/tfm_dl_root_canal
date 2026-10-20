# Contexto del proyecto (TFM de Claudia Agromayor, ICAI-Comillas)

Responde siempre en **español**, en un lenguaje sencillo: la autora viene de ingeniería y está empezando en endodoncia. Explica los términos médicos y de imagen la primera vez que aparezcan. No des nada por hecho: si algo no está verificado, dilo.

## Objetivo
A partir de un CBCT clínico, **segmentar los conductos radiculares del primer molar inferior (36/46)** y medir longitud, curvatura, volumen y, con cautela, diámetro (a 75 µm el diámetro apical es impreciso).

## Enfoque decidido (caso B)
No se usan Pulpy3D ni los datos de Fer: **descartados**. Se parte de µCT de dientes extraídos:
1. Normalizar cada µCT y obtener el canal por umbral (`analysis/microct/preprocess_microct.py`).
2. **Fabricar CBCT falsos a 75 µm** a partir del µCT: bajar resolución, desenfoque (PSF), ruido (Poisson, correlacionado, no gaussiano), contraste aleatorio. Como el falso sale del µCT, **la etiqueta ya está alineada y no hay registro**.
3. Entrenar un segmentador 3D con los falsos y la etiqueta del µCT.
4. Aplicarlo a los CBCT reales de 75 µm de Isa. Los reales sirven sobre todo para **calibrar el simulador y como prueba**, no para entrenar con pseudoetiquetas sin revisar (heredarían los fallos del primer modelo).
5. Medir sobre la segmentación: línea central + transformada de distancia (el diámetro de referencia sale del µCT por geometría, no hace falta una segunda red).

## Datos (nunca se suben a GitHub; carpetas en `.gitignore`)
- `1_Data_Validation/datasets/MicroCT/Diente_35`, `Diente_51`, `Diente_52`: Nikon XT, **vóxel 16 µm isotrópico**, `.raw` uint16 con orden (z, y, x); dimensiones en el `.xtekVolume`. Pesan ~2,4–2,9 GB. Los números son de muestra (no se sabe si son códigos FDI); parecen molares de dos raíces con varios conductos. Del diente 58 solo hay proyecciones, sin volumen.
- `1_Data_Validation/datasets/CBCT_75um/Isa_caso01`: CBCT real, 555×667×688, vóxel 0,075 mm, NRRD, más `Segmentation.seg.nrrd` hecha a mano por Isa (46 mm³, 15 componentes). Datos de paciente: no compartir.
- **No existe todavía una máscara del canal hecha por un experto sobre el µCT.** La que se usa es la del script, sin validar.

## Lo ya hecho y resultados de referencia
`preprocess_microct.py` normaliza con dos anclas (aire y mediana del tejido), umbral 0,5 (punto medio, correcto con efecto de volumen parcial) y define el canal como hueco encerrado por el diente en los cortes axiales (el eje z es el eje largo); descarta huecos cortos (`--min-len-mm 1.5`).
Con `--downsample 4`: diente 35 → 20,53 mm³ (2 componentes); 51 → 23,38 mm³ (1); 52 → 35,66 mm³ (2). Debe dar lo mismo en cualquier máquina.
Hallazgos: a 64 µm se pierden conductos finos (visto en el 52); las etiquetas definitivas deben hacerse a 16 µm y bajarse a 75 µm después, mejor como **etiquetas blandas** (fracción de conducto por vóxel) en el ápice. En la corona aparecen falsos positivos por caries/poros.

## Entorno
- Máquina del laboratorio (Mobios): RTX 5080 de 16 GB, driver 581.57, CUDA 13.0. El portátil de Claudia **no tiene GPU NVIDIA**: el entrenamiento va en la máquina del laboratorio.
- Repositorio en `D:\tfm_claudia`. Entorno conda `tfm` (Python 3.12, PyTorch 2.14.1+cu130). Anaconda en `C:\ProgramData\anaconda3`. Ejecutar con `(tfm)` activo o con `C:\Users\tfg\.conda\envs\tfm\python.exe`.
- No tocar `D:\DATA\lfsanchez` (originales de Luis).
- Para usar la GPU por defecto: `torch.set_default_device('cuda')` al inicio.

## Código existente
`2_UNet/` (UNet3D, PosPadUNet3D, Attention UNet, UNETR, VNet3D; pérdidas Dice+CE) viene del proyecto Pulpy3D y de ToothFairy: sus configs y dataloader son de Pulpy3D y hay que adaptarlos. `1_Data_Validation/analysis/` (salvo `microct/`) es análisis antiguo de Pulpy3D, solo de referencia.

## Reglas de trabajo
- Repartir entrenamiento/validación/test **por diente**, nunca por recortes (evita fugas de datos).
- Métricas: Dice y HD95 **por tercios** (coronal, medio, apical), distancia del final del conducto al ápice real, y sobre todo el **error de cada medida** frente al µCT.
- Todo script nuevo debe poder aplicarse a varios µCT (parametrizado por carpeta).
- No hacer `git commit` ni `git push` sin que Claudia lo pida. No subir datos.

## Pendiente / preguntas abiertas (Luis, Ana)
1. Máscara del canal del µCT hecha por expertos: ¿existe y dónde?
2. ¿Cuántos µCT hay? Con 3 dientes no se puede entrenar bien (en las diapositivas se hablaba de 40).
3. ¿Hay volumen reconstruido del diente 58? ¿Qué dientes son 35, 51 y 52?
4. ¿Se puede tener el CBCT de Isa en la máquina del laboratorio (datos de paciente)?

## Próximos pasos
1. Generar las máscaras a 16 µm (`--downsample 1`) de los tres dientes y revisarlas en 3D Slicer.
2. **Simulador de CBCT a 75 µm**, parametrizable y aleatorizado, calibrado con el CBCT real de Isa (perfil del borde dentina–canal y desviación típica del ruido en dentina).
3. Adaptar el entrenamiento de `2_UNet` y entrenar en la GPU.
4. Evaluar en el CBCT real de Isa y calcular las medidas.
