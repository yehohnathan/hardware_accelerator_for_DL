# hope_ff

Implementacion minima y modular de una BNN entrenada con Forward-Forward supervisado para MNIST empaquetado. El objetivo es tener un flujo C++ verificable y con una ruta razonable hacia Vitis/Vivado HLS 2024.2 en Nexys A7-100T.

## Arquitectura inicial

- Dataset por defecto: `../mnist/data/processed/mnist_28x28_1b_packed.bin`.
- Layout asumido: cabecera de 16 words de 32 bits y payload `[label_onehot | imagen]` LSB-first.
- Entrada logica del modelo: `10 + 20*20 = 410` features.
- Capa entrenable: `410 -> 64`.
- Pesos de forward: binarios `{-1,+1}` derivados del signo del peso latente.
- Pesos latentes: `int16_t`, clip `[-127,127]`.
- Bias: `int16_t`, clip `[-2047,2047]`.
- Activacion: ReLU entera con clip a `255`.
- Goodness: promedio de activaciones al cuadrado.
- Paralelismo inicial: 8 neuronas por tile.

Se eligio `20x20_1b` porque reduce el estado latente frente a 28x28 (`64*410` pesos en lugar de `64*794`) sin caer tan fuerte en perdida espacial como 16x16. Para una Nexys A7-100T, esta red cabe como punto de partida: unos 26 240 pesos latentes, acumuladores enteros y forward por suma/resta, sin DSP obligatorios.

## Como compilar y correr

Desde `ff_bnn_stage/hope_ff`:

```powershell
New-Item -ItemType Directory -Force build
g++ -std=c++17 -O2 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label -Iinclude src/ff_dataset.cpp src/ff_layer.cpp src/ff_train.cpp src/ff_infer.cpp testbench/train_tb.cpp -o build/train_tb.exe
.\build\train_tb.exe
```

Tambien puede usarse `make`. En esta version, `make` compila, ejecuta y
guarda la salida completa en `runs/*.txt`:

```powershell
make
make build
make run TRAIN_SAMPLES=16 EPOCHS=1 HIDDEN_LAYERS=2 HIDDEN_VALUES="32 8" PARALLEL_NEURONS=8
```

Variables utiles:

- `FF_DATASET_BIN`: ruta explicita al `.bin`.
- `FF_TRAIN_SAMPLES`: muestras de entrenamiento online, por defecto `1024`.
- `FF_EVAL_SAMPLES`: muestras de evaluacion, por defecto `512`.
- `FF_EPOCHS`: epocas, por defecto `5`.
- `FF_SEED`: semilla entera.

El testbench imprime goodness positiva media, goodness negativa media, gap, eventos de actualizacion, pesos/bias cambiados, cambios de signo binario, prediccion y accuracy parcial.

## Makefile experimental

El `Makefile` permite compilar variantes con macros `-D` y ejecutar barridos sin editar headers:

```powershell
make config
make run IMG_W=28 IMG_H=28 PIX_BITS=1 DATASET=../mnist/data/processed/mnist_28x28_1b_packed.bin
make run HIDDEN_LAYERS=2 HIDDEN_VALUES="32 8" PARALLEL_NEURONS=8 TRAIN_SAMPLES=16 EPOCHS=1
make csim-run TRAIN_SAMPLES=16 EVAL_SAMPLES=8 EPOCHS=1
make sweep100
make run-bins
make run-archs
make sweep
```

Targets principales:

- `make`: compila, ejecuta y guarda un `.txt`.
- `make build`: solo compila la variante actual.
- `make run`: ejecuta la variante actual.
- `make csim-run`: genera `hls_config.cfg` de la variante y ejecuta C-simulation con Vitis HLS 2024.2.
- `make sweep100`: ejecuta 108 simulaciones con samples de `barrido_final` y genera tabla ordenada por accuracy/complejidad.
- `make run-bins`: prueba varios `.bin`.
- `make run-archs`: prueba varias arquitecturas sobre el `.bin` actual.
- `make sweep`: barrido corto de datasets y arquitecturas.
- `make clean`: borra `build/`.
- `make clean-runs`: borra `runs/`.

Cada ejecucion guarda `log.txt`, `metrics.csv`, `summary.md` y `status.txt` en `results/<dataset>/<grupo>/<arquitectura>/`.

## Barrido grande de arquitecturas

Comando principal:

```powershell
make sweep100
```

Genera 108 simulaciones:

```text
9 datasets x 12 perfiles de arquitectura/threshold/epochs/eval_every_epoch
```

Cada corrida usa por defecto:

```make
GROUP=barrido_final
TRAIN_SAMPLES=35700
EVAL_SAMPLES=6300
```

La salida queda en:

```text
results/sweep100/<dataset>/barrido_final/<arquitectura>/<threshold_epochs_eval>/
results/sweep100/summary_sweep100.md
```

La tabla se ordena por accuracy descendente y complejidad ascendente. Complejidad estimada:

```text
complexity_score = sum(input_dim_layer * output_dim_layer)
params = weights + biases
weight_bits_est = params * 16
BRAM18_est = ceil(weight_bits_est / 18432)
```

Prueba rapida sin lanzar 108 entrenamientos completos:

```powershell
make sweep100-dry
make sweep100 SWEEP100_LIMIT=2 SWEEP100_EXTRA="TRAIN_SAMPLES=1 EVAL_SAMPLES=1 EPOCHS=1"
```

`csim-run` usa por defecto:

- `VITIS_RUN=C:/Xilinx/Vitis/2024.2/bin/vitis-run.bat`
- `HLS_PART=xc7a100tcsg324-1`
- `flow_target=vivado`
- `syn.top=ff_train_kernel`
- testbench HLS: `testbench/hls_csim_tb.cpp`

Esto valida C-simulation para una ruta compatible con Vivado/Vitis Unified IDE 2024.2 y Nexys A7-100T. El part puede cambiarse con `HLS_PART=...` si se desea usar otro speed grade.

## Ruta HLS

Archivos sintetizables principales:

- `include/ff_config.hpp`
- `include/ff_types.hpp`
- `include/ff_dataset.hpp` solo helpers inline de bits
- `include/ff_layer.hpp`
- `include/ff_train.hpp`
- `include/ff_infer.hpp`
- `src/ff_dataset.cpp` solo `ff_decode_*` y `ff_copy_sample_words`; las funciones con `ifstream/vector/string` estan protegidas con `#ifndef __SYNTHESIS__`
- `src/ff_layer.cpp`
- `src/ff_train.cpp`
- `src/ff_infer.cpp`

Solo host/testbench:

- lectura de archivo `.bin` con `ifstream`;
- `std::vector` para cargar el payload;
- impresiones y parsing de variables de entorno;
- `testbench/train_tb.cpp`.

Top functions propuestas:

- `ff_train_kernel(...)`: entrena online sobre `dataset_words`, donde el puntero debe apuntar al payload y no a la cabecera de 64 bytes.
- `ff_infer_kernel(...)`: carga pesos/bias planos y prueba las 10 etiquetas por muestra.

Las interfaces incluyen pragmas `m_axi`, `s_axilite`, `PIPELINE`, `UNROLL` y `ARRAY_PARTITION`. En un componente HLS, usar `ff_train_kernel` como top para entrenamiento o `ff_infer_kernel` como top para inferencia. Para C-sim automatizado, usar `make csim-run`.

## Aproximaciones hardware

El algoritmo original usa funciones suaves tipo sigmoid/softplus para separar goodness positiva y negativa. Aqui se reemplazan por una regla hinge sintetizable:

- si `g_pos < threshold`, se refuerzan conexiones activas del ejemplo positivo;
- si `g_neg > threshold`, se debilitan conexiones activas del ejemplo negativo.

Esto evita exponenciales, logaritmos y punto flotante. Tambien se usa entrenamiento online por muestra en lugar de mini-batches grandes para evitar buffers de gradientes en BRAM.

## Trade-offs esperados

- LUT/FF: dominados por 8 lanes de acumulacion y control de update.
- BRAM: dominada por `64*410` pesos latentes `int16_t`; puede reducirse migrando a `int8_t`.
- DSP: el forward usa suma/resta por peso binario, por lo que no necesita multiplicadores.
- Latencia forward: aproximadamente `(64/8)*410 = 3280` ciclos mas overhead.
- Latencia train/sample: dos forwards mas update local de `64*410`; es simple, pero no es aun throughput-optimo.
- Precision: enteros pequenos favorecen sintesis y depuracion, con accuracy inicial limitada.
- Sintesis: una capa densa pequeña es mas facil de cerrar que una red profunda o CNN.

## Parametros para experimentar

Editar `include/ff_config.hpp` o compilar con `-D`:

- `FF_HIDDEN_LAYERS`
- `FF_HIDDEN_0`, `FF_HIDDEN_1`, `FF_HIDDEN_2`, `FF_HIDDEN_3`
- `FF_PARALLEL_NEURONS`
- `FF_GOODNESS_THRESHOLD`
- `FF_LABEL_SCALE`
- `FF_PIXEL_LR_STEP`
- `FF_LABEL_LR_STEP`
- `FF_IMAGE_WIDTH`, `FF_IMAGE_HEIGHT`, `FF_PIXEL_BITS`

Si se cambia resolucion o bits por pixel, el `.bin` debe coincidir con la cabecera validada.

## Proximos pasos

1. Barrer `threshold`, `label_scale` y learning steps con `FF_TRAIN_SAMPLES` mayor.
2. Medir C-synthesis y ajustar `ARRAY_PARTITION`/`BIND_STORAGE`.
3. Probar `int8_t` para pesos latentes si BRAM queda alta.
4. Separar entrenamiento e inferencia en kernels distintos si se prioriza throughput.
5. Agregar una segunda capa FF solo despues de validar convergencia y recursos.
6. Evaluar 28x28_1b y 20x20_4b cuando el flujo base cierre sintesis.
