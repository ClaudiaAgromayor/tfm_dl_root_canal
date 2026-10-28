# diagnostico/

Scripts que reproducen los números del análisis (`docs/RESUMEN_PROYECTO.md`) y las pruebas del simulador. Todos:

- se ejecutan desde cualquier directorio (las rutas por defecto cuelgan de `1_Data_Validation/`: `results/`, `datasets/`);
- aceptan `--help`; los parámetros que importan se pasan por línea de comandos;
- **no escriben en `results/microct*`** (solo, si hace falta, en `results/diagnostico/` o en `--out`);
- se lanzan con el entorno conda `tfm` (`C:\Users\tfg\.conda\envs\tfm\python.exe`).

| Script | Qué reproduce / comprueba | Resultado esperado (2026-10-28) | Cuesta |
|---|---|---|---|
| `comparar_resultados.py` | compara regenerados con guardados (`pre`, `hash`, `json`, `calib`) | `idéntico` / hashes iguales | segundos |
| `barrido_gap.py` | barrido de `--gap-mm` (RESUMEN 3.3) | con 0 el diente 51 a 16 µm se parte; con 0,13 mm: 1 componente de ~23,1 mm³ | minutos por diente a 16 µm |
| `grosor_piezas.py` | grosor (radio inscrito) de grupos y piezas, motas | rayas de caries: radio p90 ≤ 0,03 mm; conductos ≥ 0,07 mm (sobre una máscara **sin** filtrar; ver el docstring) | ~1 min por diente |
| `secciones_y_orificio.py` | sección mayor por corte y sensibilidad del orificio al umbral de área (RESUMEN 3.5) | orificio 786/797/815; −0,11…−0,21 mm con 1,25 mm², +0,22…+0,59 mm con 2,0 mm² | ~1 min |
| `dice_32_vs_16.py` | volumen, componentes y Dice entre 16 y 32 µm (`resumen_etapa1.json`) | Dice (rejilla 32 µm) 0,9739 / 0,9773 / 0,9888 | ~1 min |
| `ver_apice.py` | cortes ampliados del final de un conducto hasta la punta (RESUMEN 3.4) | el hueco se abre al exterior ~0,06 mm bajo el final de la máscara | segundos |
| `regresion_simulador.py` | parches con recortes fijos: guardar una línea base / compararla | `--legacy` frente a una versión antigua: diferencia 0 | ~1 min |
| `muestreo_por_tercios.py` | reparto de los centros de parche por tercio y conducto por bloque, muestreo antiguo y nuevo | apical 13,9 % → 31,9 % | ~8 min |
| `calibracion_reduccion.py` | efecto de la reducción exacta y del muestreo sobre la comprobación contra Isa | reducción exacta: σ del borde 0,142 → 0,141 mm | ~10 min |
| `prueba_vox_distinto.py` | el simulador usa el vóxel de cada diente (copia a 32 µm) | mismo tamaño físico (9,599 mm), volumen de la etiqueta ±0,7 % | ~1 min |
| `prueba_alineacion_y_pesos.py` | relleno de la cámara, franja de "ignorar", alineación imagen–etiqueta | mínimo de correlación en desplazamiento 0 en los 3 ejes | ~1 min |
| `ruido_medida_sesgo.py` | por qué la medida de ruido sube con rangos aleatorios | correlación 0,76 con el gris de dentina sorteado | ~3 min |
| `tiempos_simulador.py` | tiempo por etapas y CPU frente a GPU | `patch()` ~0,5 s en GPU, ~87 s en CPU | segundos / ~2 min |
| `comprobaciones.py` | segmentación de Isa, ventanas de reducción, ganancia del ruido, conservación de la etiqueta, dilatación, semilla, tercios | ver su docstring | ~2 min |

## Lo que NO se copió

Los scripts temporales de la etapa 1 que seguían un caso concreto a través de una máscara **intermedia que ya no existe**
(la anterior a limpiar motas y láminas de caries): seguimiento del conducto del 51 por cortes (`diag51*.py`), imágenes
"antes/después" de los filtros (`before_after.py`, `control_joins.py`, `specks.py`) y sus copias de seguridad de máscaras.
No se pueden reproducir con la máscara actual; sí se puede regenerar la intermedia con
`preprocess_microct.py --min-thick-mm 0 --min-piece-mm3 0 --out <otra carpeta>` y aplicar `grosor_piezas.py`,
`barrido_gap.py` (con `--min-piece-mm3 0`) y `dice_32_vs_16.py` a ella. Sus resultados, citados en el RESUMEN, quedan
marcados como "no reproducible desde el repo" salvo lo que cubren estos scripts.
