# Bateria de experimentos con `make`

Este documento define el flujo practico para ejecutar simulaciones del entrenamiento Forward-Forward de una BNN dentro de `ff_bnn_stage/hope_ff`.

El objetivo es probar todos los binarios MNIST preprocesados contra cuatro grupos de hiperparametros. Con el dataset actual hay 9 archivos `.bin`, por lo que el barrido completo esperado es:

```text
9 datasets x 4 grupos = 36 simulaciones
```

Cada simulacion guarda sus resultados en:

```text
results/<nombre_del_binario>/<grupo_hiperparametros>/<arquitectura>/
```

Dentro de cada carpeta se generan:

```text
log.txt
metrics.csv
summary.md
status.txt
```

Al final del flujo se genera:

```text
results/summary_all.md
```

## Estructura esperada

Ejecutar siempre desde:

```bash
cd ff_bnn_stage/hope_ff
```

Estructura esperada:

```text
ff_bnn_stage/
  hope_ff/
    Makefile
    make_experiments.md
    include/
    src/
    scripts/
      run_one_experiment.py
      run_all_experiments.py
      summarize_results.py
      write_hls_config.py
      run_hls_csim.py
    testbench/
      train_tb.cpp
      hls_csim_tb.cpp
    build/
    results/
  mnist/
    data/
      processed/
        *.bin
```

La carpeta de datasets por defecto es:

```make
DATA_DIR ?= ../mnist/data/processed
DATASETS := $(wildcard $(DATA_DIR)/*.bin)
```

Si los `.bin` estan directamente en `mnist/data`, ejecutar con:

```bash
make run-all DATA_DIR=../mnist/data
```

## Grupos de hiperparametros

| Grupo | Train samples | Eval samples | Epochs | Eval cada epoca |
|---|---:|---:|---:|---:|
| `smoke` | 256 | 64 | 1 | 0 |
| `debug` | 4096 | 1024 | 5 | 0 |
| `comparacion` | 35700 | 6300 | 20 | 1 |
| `barrido_final` | 35700 | 6300 | 30 | 1 |

## Comandos principales

Compilar el testbench C++:

```bash
make build
```

Ver la configuracion efectiva:

```bash
make config
```

Listar los `.bin` detectados:

```bash
make list-datasets
```

Ejecutar una simulacion individual:

```bash
make run \
  DATASET=../mnist/data/processed/mnist_20x20_1b_packed.bin \
  GROUP=smoke
```

Ejecutar una simulacion individual cambiando arquitectura e hiperparametros:

```bash
make run \
  DATASET=../mnist/data/processed/mnist_28x28_4b_packed.bin \
  GROUP=debug \
  HIDDEN_LAYERS=2 \
  HIDDEN_VALUES="32 8" \
  PARALLEL_NEURONS=8 \
  THRESHOLD=128 \
  PIXEL_SCALE=8
```

Ejecutar todos los datasets con todos los grupos:

```bash
make run-all
```

Generar la tabla global:

```bash
make summarize
```

Flujo completo recomendado:

```bash
make clean-results
make build
make run-all
make summarize
```

## Pruebas rapidas

Para comprobar el Makefile sin lanzar las 36 simulaciones completas:

```bash
make run-all GROUPS=smoke RUN_ALL_EXTRA="TRAIN_SAMPLES=2 EVAL_SAMPLES=1 EPOCHS=1"
make summarize
```

Para probar solo un grupo sobre todos los `.bin`:

```bash
make run-smoke
make run-debug
make run-comparacion
make run-barrido-final
```

Los targets abreviados aceptan overrides:

```bash
make run-smoke RUN_ALL_EXTRA="TRAIN_SAMPLES=16 EVAL_SAMPLES=4 EPOCHS=1"
```

## Formato de llamada del testbench

El ejecutable principal acepta argumentos por linea de comandos:

```bash
./build/<variante>/train_tb.exe \
  --dataset <ruta_bin> \
  --train-samples <N> \
  --eval-samples <N> \
  --epochs <N> \
  --eval-every-epoch <0|1> \
  --run-name <nombre> \
  --group <grupo> \
  --output-dir <carpeta_resultado> \
  --seed <entero>
```

Ejemplo equivalente al target `make run`:

```bash
./build/20x20_1b_h64_p8_thr72_ls8_ps1_plr1_llr4/train_tb.exe \
  --dataset ../mnist/data/processed/mnist_20x20_1b_packed.bin \
  --train-samples 256 \
  --eval-samples 64 \
  --epochs 1 \
  --eval-every-epoch 0 \
  --run-name manual_smoke \
  --group smoke \
  --output-dir results/mnist_20x20_1b_packed/smoke \
  --seed 0x1234
```

## Targets implementados en el Makefile

El `Makefile` real ya contiene estos targets:

| Target | Funcion |
|---|---|
| `make build` | Compila `testbench/train_tb.cpp` con la variante seleccionada. |
| `make run` | Ejecuta una simulacion y guarda `log.txt`, `metrics.csv`, `summary.md` y `status.txt`. |
| `make run-all` | Ejecuta `DATASETS x GROUPS` usando `scripts/run_all_experiments.py`. |
| `make summarize` | Genera `results/summary_all.md` con `scripts/summarize_results.py`. |
| `make sweep100` | Ejecuta 108 simulaciones `barrido_final` y genera tabla ordenada por accuracy/complejidad. |
| `make sweep100-dry` | Imprime las 108 simulaciones sin ejecutarlas. |
| `make list-datasets` | Lista todos los `.bin` encontrados en `DATA_DIR`. |
| `make clean-results` | Borra `results/`. |
| `make run-smoke` | Ejecuta todos los datasets con el grupo `smoke`. |
| `make run-debug` | Ejecuta todos los datasets con el grupo `debug`. |
| `make run-comparacion` | Ejecuta todos los datasets con el grupo `comparacion`. |
| `make run-barrido-final` | Ejecuta todos los datasets con el grupo `barrido_final`. |
| `make csim-run` | Lanza C simulation HLS con Vitis 2024.2 para el top `ff_train_kernel`. |

Variables importantes:

```make
DATA_DIR=../mnist/data/processed
DATASET=../mnist/data/processed/mnist_20x20_1b_packed.bin
GROUP=smoke
GROUPS="smoke debug comparacion barrido_final"
RESULTS_DIR=results
TRAIN_SAMPLES=256
EVAL_SAMPLES=64
EPOCHS=1
EVAL_EVERY_EPOCH=0
IMG_W=20
IMG_H=20
PIX_BITS=1
HIDDEN_LAYERS=1
HIDDEN_VALUES=64
PARALLEL_NEURONS=8
THRESHOLD=72
```

El nombre del dataset se usa para inferir `IMG_W`, `IMG_H` y `PIX_BITS` cuando se pasa `DATASET=...` desde la linea de comandos. La arquitectura se compila desde `HIDDEN_LAYERS`, `HIDDEN_VALUES` y `PARALLEL_NEURONS`; por ejemplo `HIDDEN_LAYERS=2 HIDDEN_VALUES="32 8"` genera macros `FF_HIDDEN_LAYERS=2`, `FF_HIDDEN_0=32`, `FF_HIDDEN_1=8`.

## Barrido de 108 simulaciones

Para buscar mejor relacion accuracy/complejidad:

```bash
make sweep100
```

Por defecto ejecuta:

```text
9 datasets x 12 perfiles = 108 simulaciones
TRAIN_SAMPLES=35700
EVAL_SAMPLES=6300
GROUP=barrido_final
```

Varia arquitectura, capas, neuronas, `PARALLEL_NEURONS`, `THRESHOLD`, `EPOCHS` y `EVAL_EVERY_EPOCH`.

Salida:

```text
results/sweep100/<dataset>/barrido_final/<arquitectura>/<threshold_epochs_eval>/
results/sweep100/summary_sweep100.md
```

La tabla se ordena por:

```text
accuracy desc, complexity_score asc
```

Complejidad estimada:

```text
complexity_score = sum(input_dim_layer * output_dim_layer)
params = weights + biases
weight_bits_est = params * 16
BRAM18_est = ceil(weight_bits_est / 18432)
```

Pruebas rapidas:

```bash
make sweep100-dry
make sweep100 SWEEP100_LIMIT=2 SWEEP100_EXTRA="TRAIN_SAMPLES=1 EVAL_SAMPLES=1 EPOCHS=1"
```

## Scripts auxiliares

El flujo de experimentos usa tres scripts simples:

```text
scripts/run_one_experiment.py
scripts/run_all_experiments.py
scripts/summarize_results.py
```

`run_one_experiment.py` ejecuta el binario, captura toda la salida y escribe:

```text
results/<dataset>/<grupo>/<arquitectura>/log.txt
results/<dataset>/<grupo>/<arquitectura>/status.txt
```

`train_tb.cpp` escribe:

```text
results/<dataset>/<grupo>/<arquitectura>/metrics.csv
results/<dataset>/<grupo>/<arquitectura>/summary.md
```

`summarize_results.py` lee todos los `summary.md` y `metrics.csv` para construir:

```text
results/summary_all.md
```

Los targets HLS usan:

```text
scripts/write_hls_config.py
scripts/run_hls_csim.py
```

## Tabla resumen global

La tabla final tiene este formato:

```markdown
# Resumen global de experimentos

Total de experimentos encontrados: 36

| ID | Dataset | Resolucion | Bits | Grupo | Arquitectura | Hidden layers | Hidden values | Parallel neurons | Train samples | Eval samples | Epochs | Accuracy | Goodness positiva | Goodness negativa | Goodness gap | Tiempo | Estado |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mnist_20x20_4b | 20x20 | 4b | smoke | 410 -> 32 -> 8 | 2 | 32 -> 8 | 8 | 16 | 8 | 1 | 0.00% | 363 | 313 | 50 | 0.05s | OK |
```

Las columnas reales generadas son:

| Columna | Significado |
|---|---|
| `ID` | Indice de experimento encontrado. |
| `Dataset` | Nombre base del binario. |
| `Resolucion` | Resolucion inferida del dataset o del header. |
| `Bits` | Bits inferidos del dataset o del header. |
| `Grupo` | Grupo de hiperparametros. |
| `Arquitectura` | Cadena completa, por ejemplo `410 -> 32 -> 8`. |
| `Hidden layers` | Cantidad de capas activas. |
| `Hidden values` | Neuronas por capa. |
| `Parallel neurons` | Neuronas procesadas en paralelo. |
| `Train samples` | Muestras usadas para entrenamiento. |
| `Eval samples` | Muestras usadas para evaluacion. |
| `Epochs` | Epocas ejecutadas. |
| `Accuracy` | Accuracy final formateada en porcentaje. |
| `Goodness positiva` | Goodness promedio de ejemplos positivos. |
| `Goodness negativa` | Goodness promedio de ejemplos negativos. |
| `Goodness gap` | Diferencia promedio `g_pos - g_neg`. |
| `Tiempo` | Tiempo de la corrida. |
| `Estado` | `OK`, `FAIL` o `UNKNOWN`. |

## Compatibilidad HLS / Vitis Unified IDE 2024.2

Para generar y ejecutar C simulation HLS:

```bash
make csim-run \
  DATASET=../mnist/data/processed/mnist_20x20_4b_packed.bin \
  GROUP=smoke \
  TRAIN_SAMPLES=16 \
  EVAL_SAMPLES=8 \
  EPOCHS=1 \
  HIDDEN_LAYERS=2 \
  HIDDEN_VALUES="32 8" \
  PARALLEL_NEURONS=8 \
  HLS_PART=xc7a100tcsg324-1
```

El target genera:

```text
build/<variante>/hls_config.cfg
runs/<run_name>_csim.txt
hls_work/<variante>/
```

La top function usada para HLS es:

```cpp
ff_train_kernel(...)
```

La parte seleccionada por defecto es:

```make
HLS_PART ?= xc7a100tcsg324-1
```

Ese part corresponde a la Nexys A7-100T basada en Artix-7.

Si Vitis esta instalado en otra ruta:

```bash
make csim-run VITIS_ROOT=C:/Xilinx/Vitis/2024.2
```

o:

```bash
make csim-run VITIS_RUN=C:/Xilinx/Vitis/2024.2/bin/vitis-run.bat
```

## Windows, Git Bash, MSYS2 y Linux

El flujo principal esta pensado para Windows usando Git Bash, MSYS2 o PowerShell:

```bash
cd /d/TFG/hardware_accelerator_for_DL/ff_bnn_stage/hope_ff
make run-all
```

Tambien evita bucles complejos de shell: el barrido completo se hace en Python, por lo que el mismo target `run-all` es facil de depurar y portable.

En Linux, usar rutas relativas normales:

```bash
cd ff_bnn_stage/hope_ff
make run-all DATA_DIR=../mnist/data/processed
```

Para HLS en Linux, si no existe `vitis-run.bat`, pasar explicitamente el ejecutable de Vitis:

```bash
make csim-run VITIS_RUN=vitis-run
```

## Cambios minimos si el testbench no aceptara argumentos

El `testbench/train_tb.cpp` actual ya acepta argumentos por terminal. Si en una rama futura se pierde este soporte, los cambios minimos son:

1. Cambiar `int main()` por `int main(int argc, char **argv)`.
2. Parsear `--dataset`, `--train-samples`, `--eval-samples`, `--epochs`, `--eval-every-epoch`, `--run-name`, `--group`, `--output-dir` y `--seed`.
3. Mantener variables de entorno como fallback para `csim-run`: `FF_DATASET_BIN`, `FF_TRAIN_SAMPLES`, `FF_EVAL_SAMPLES`, `FF_EPOCHS`, `FF_EVAL_EVERY_EPOCH` y `FF_SEED`.
4. Crear `output_dir` antes de escribir archivos.
5. Escribir `metrics.csv` con una fila por epoca evaluada.
6. Escribir `summary.md` con pares `clave: valor` para que `scripts/summarize_results.py` pueda leerlo.
7. Devolver codigo distinto de cero si el entrenamiento no cambia pesos ni bias.

## Ejemplos completos

Una corrida individual:

```bash
make clean-results
make run DATASET=../mnist/data/processed/mnist_20x20_1b_packed.bin GROUP=smoke
make summarize
```

Barrido completo de 36 simulaciones:

```bash
make clean-results
make run-all
make summarize
```

Barrido rapido de depuracion con todos los datasets:

```bash
make clean-results
make run-all GROUPS=smoke RUN_ALL_EXTRA="TRAIN_SAMPLES=2 EVAL_SAMPLES=1 EPOCHS=1"
make summarize
```

Probar varias arquitecturas sobre el dataset por defecto:

```bash
make run-archs DATASET=../mnist/data/processed/mnist_20x20_1b_packed.bin GROUP=smoke
```

Probar varios binarios con la arquitectura actual:

```bash
make run-bins GROUP=smoke
```

## Criterio de aceptacion del flujo

Un experimento se considera valido si:

```text
results/<dataset>/<grupo>/<arquitectura>/log.txt existe
results/<dataset>/<grupo>/<arquitectura>/metrics.csv existe
results/<dataset>/<grupo>/<arquitectura>/summary.md existe
results/<dataset>/<grupo>/<arquitectura>/status.txt contiene OK
summary.md reporta changed_weights > 0 o changed_biases > 0
```

El barrido completo se considera listo para comparar si:

```bash
make run-all
make summarize
```

termina sin error y `results/summary_all.md` contiene 36 filas.
