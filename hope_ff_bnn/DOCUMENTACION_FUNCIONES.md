# DOCUMENTACIÓN DE FUNCIONES - `hope_ff`

## Introducción

`hope_ff` implementa una versión mínima, configurable y orientada a HLS del
algoritmo Forward-Forward aplicado a una red neuronal binaria para MNIST. El
objetivo del módulo es entrenar y validar una BNN en C++ con restricciones
realistas para una futura implementación en una FPGA Nexys A7-100T basada en
Artix-7.

El diseño separa dos niveles:

- **Núcleo C++/HLS**: funciones con arreglos estáticos, enteros de ancho fijo,
  pragmas HLS y ausencia de memoria dinámica. Este nivel contiene el forward,
  cálculo de goodness, actualización local, inferencia y top functions.
- **Host/testbench/scripts**: carga de binarios, parseo de argumentos, ejecución
  de Make, generación de logs y análisis de resultados. Este nivel puede usar
  STL, Python, archivos y procesos externos porque no se sintetiza.

## Flujo completo del proyecto

1. **Carga del dataset**
   - Los archivos `.bin` contienen una cabecera de 16 palabras y muestras
     empaquetadas como `[label_onehot | pixels]`.
   - El host valida que resolución, bits por pixel y número de clases coincidan
     con las macros de compilación.

2. **Construcción de entradas positivas y negativas**
   - Para una muestra MNIST, el ejemplo positivo usa la etiqueta real.
   - El ejemplo negativo usa la misma imagen con una etiqueta incorrecta
     generada por LFSR.

3. **Forward local**
   - Cada capa usa pesos binarios derivados del signo de pesos latentes.
   - La acumulación suma o resta la característica según el signo del peso.
   - La activación es ReLU saturada, una alternativa simple y sintetizable.

4. **Cálculo de goodness**
   - La goodness de una capa es el promedio de activaciones al cuadrado.
   - El entrenamiento busca que ejemplos positivos tengan mayor goodness y que
     negativos queden por debajo del umbral.

5. **Señal de aprendizaje**
   - Si la goodness positiva está bajo el umbral, se refuerzan entradas activas.
   - Si la goodness negativa supera el umbral, se penalizan entradas activas.
   - La regla es local por capa y no usa backpropagation global.

6. **Actualización de pesos y bias**
   - Los pesos latentes se ajustan con pasos enteros y clipping.
   - El forward sigue usando solo el signo del peso latente.
   - Los bias se actualizan por neurona activa.

7. **Métricas**
   - Se registran goodness positiva, goodness negativa, gap, eventos de update,
     pesos/bias modificados, cambios de signo y accuracy.

8. **Kernels HLS**
   - `ff_train_kernel` entrena y exporta pesos/bias planos.
   - `ff_infer_kernel` carga pesos/bias y predice clases por goodness.
   - Las interfaces AXI separan dataset, modelo y métricas/resultados.

9. **Testbench y validación**
   - `train_tb.cpp` ejecuta simulaciones de entrenamiento completas.
   - `hls_csim_tb.cpp` valida los top functions en la ruta de Vitis HLS csim.
   - Los scripts Python automatizan experimentos, barridos y resúmenes.

## Carpeta `include`

### `ff_copy_sample_words`

- **Lenguaje:** C++
- **Archivo:** `include/ff_dataset.hpp`
- **Propósito:** copiar una muestra empaquetada desde el payload global hacia un
  buffer local de tamaño fijo.
- **Parámetros:** `dataset_words` es el payload sin cabecera; `sample_index`
  selecciona la muestra; `sample_words` recibe la copia local.
- **Retorno:** no retorna valor.
- **Efectos laterales:** escribe el buffer local `sample_words`.
- **Relación con FF:** prepara la muestra que luego se transforma en ejemplo
  positivo, negativo o entrada de inferencia.
- **HLS/FPGA:** sintetizable; reduce aritmética repetida de direcciones AXI.
- **Sintetizable:** sí.

### `ff_decode_label_from_words`

- **Lenguaje:** C++
- **Archivo:** `include/ff_dataset.hpp`
- **Propósito:** recuperar la etiqueta real one-hot desde la muestra empaquetada.
- **Parámetros:** `sample_words` contiene la muestra local.
- **Retorno:** índice de clase MNIST.
- **Efectos laterales:** ninguno.
- **Relación con FF:** define el ejemplo positivo y permite medir accuracy.
- **HLS/FPGA:** sintetizable; recorre diez bits fijos y puede desenrollarse.
- **Sintetizable:** sí.

### `ff_decode_pixel_from_words`

- **Lenguaje:** C++
- **Archivo:** `include/ff_dataset.hpp`
- **Propósito:** reconstruir un pixel cuantizado desde bits LSB-first.
- **Parámetros:** `sample_words` contiene la muestra; `pixel_index` indica el
  pixel dentro de la imagen redimensionada.
- **Retorno:** intensidad cuantizada del pixel.
- **Efectos laterales:** ninguno.
- **Relación con FF:** alimenta el vector de características de la primera capa.
- **HLS/FPGA:** sintetizable; el coste depende de `FF_PIXEL_BITS`.
- **Sintetizable:** sí.

### `ff_default_dataset_path`

- **Lenguaje:** C++
- **Archivo:** `include/ff_dataset.hpp`
- **Propósito:** resolver una ruta de dataset para simulación.
- **Parámetros:** ninguno.
- **Retorno:** ruta del binario MNIST.
- **Efectos laterales:** puede consultar variables de entorno y el sistema de
  archivos.
- **Relación con FF:** selecciona el dataset que alimenta entrenamiento.
- **HLS/FPGA:** no aplica a hardware; usa STL/I/O de host.
- **Sintetizable:** no, solo host/testbench.

### `ff_evaluate_dataset`

- **Lenguaje:** C++
- **Archivo:** `include/ff_infer.hpp`
- **Propósito:** evaluar un bloque de muestras y acumular accuracy.
- **Parámetros:** `model`, `dataset_words`, `n_samples`, `eval_metrics`.
- **Retorno:** no retorna valor.
- **Efectos laterales:** escribe `eval_metrics`.
- **Relación con FF:** mide desempeño final usando inferencia por goodness.
- **HLS/FPGA:** usa funciones sintetizables, aunque también se llama desde host.
- **Sintetizable:** sí.

### `ff_forward_layer`

- **Lenguaje:** C++
- **Archivo:** `include/ff_layer.hpp`
- **Propósito:** calcular activaciones y goodness de una capa BNN.
- **Parámetros:** modelo, índice de capa, features, activaciones y goodness.
- **Retorno:** no retorna valor.
- **Efectos laterales:** escribe activaciones y goodness.
- **Relación con FF:** es el bloque central que compara ejemplos positivos y
  negativos.
- **HLS/FPGA:** sintetizable; `FF_PARALLEL_NEURONS` controla recursos y
  throughput.
- **Sintetizable:** sí.

### `ff_get_packed_bit`

- **Lenguaje:** C++
- **Archivo:** `include/ff_dataset.hpp`
- **Propósito:** leer un bit físico de una muestra empaquetada.
- **Parámetros:** `sample_words` y `bit_index`.
- **Retorno:** bit 0/1.
- **Efectos laterales:** ninguno.
- **Relación con FF:** base común para decodificar etiquetas y pixeles.
- **HLS/FPGA:** sintetizable e inline; evita duplicar lógica de desempaquetado.
- **Sintetizable:** sí.

### `ff_infer_kernel`

- **Lenguaje:** C++
- **Archivo:** `include/ff_infer.hpp`
- **Propósito:** top function HLS de inferencia.
- **Parámetros:** dataset, pesos, bias, predicciones, conteo de aciertos y
  número de muestras.
- **Retorno:** no retorna valor.
- **Efectos laterales:** escribe predicciones y conteo de aciertos.
- **Relación con FF:** clasifica por mayor goodness entre diez etiquetas.
- **HLS/FPGA:** top sintetizable con interfaces AXI master y AXI-Lite.
- **Sintetizable:** sí.

### `ff_init_model`

- **Lenguaje:** C++
- **Archivo:** `include/ff_layer.hpp`
- **Propósito:** inicializar pesos latentes y bias.
- **Parámetros:** modelo y semilla.
- **Retorno:** no retorna valor.
- **Efectos laterales:** sobrescribe el modelo.
- **Relación con FF:** fija el estado inicial antes del entrenamiento local.
- **HLS/FPGA:** sintetizable; usa LFSR en lugar de generadores aleatorios de
  host.
- **Sintetizable:** sí.

### `ff_layer_input_dim`

- **Lenguaje:** C++
- **Archivo:** `include/ff_layer.hpp`
- **Propósito:** obtener la dimensión de entrada activa de una capa.
- **Parámetros:** índice de capa.
- **Retorno:** dimensión de entrada.
- **Efectos laterales:** ninguno.
- **Relación con FF:** conecta salida de una capa con entrada de la siguiente.
- **HLS/FPGA:** sintetizable; ayuda a usar máximos estáticos con capas activas.
- **Sintetizable:** sí.

### `ff_layer_neurons`

- **Lenguaje:** C++
- **Archivo:** `include/ff_layer.hpp`
- **Propósito:** obtener neuronas activas de una capa.
- **Parámetros:** índice de capa.
- **Retorno:** número de neuronas.
- **Efectos laterales:** ninguno.
- **Relación con FF:** limita forward/update a la arquitectura compilada.
- **HLS/FPGA:** sintetizable; evita memoria dinámica.
- **Sintetizable:** sí.

### `ff_load_model_flat`

- **Lenguaje:** C++
- **Archivo:** `include/ff_layer.hpp`
- **Propósito:** reconstruir el modelo estático desde buffers planos.
- **Parámetros:** modelo destino, pesos planos y bias planos.
- **Retorno:** no retorna valor.
- **Efectos laterales:** sobrescribe el modelo.
- **Relación con FF:** permite usar en inferencia un modelo entrenado/exportado.
- **HLS/FPGA:** sintetizable; adapta memoria AXI a arreglos internos.
- **Sintetizable:** sí.

### `ff_make_negative_label`

- **Lenguaje:** C++
- **Archivo:** `include/ff_layer.hpp`
- **Propósito:** generar una etiqueta incorrecta para el ejemplo negativo.
- **Parámetros:** etiqueta real y estado LFSR.
- **Retorno:** etiqueta distinta de la real.
- **Efectos laterales:** actualiza el estado LFSR.
- **Relación con FF:** crea la comparación supervisada positiva/negativa.
- **HLS/FPGA:** sintetizable; evita aleatoriedad de librerías no soportadas.
- **Sintetizable:** sí.

### `ff_predict_sample`

- **Lenguaje:** C++
- **Archivo:** `include/ff_infer.hpp`
- **Propósito:** predecir una muestra probando las diez etiquetas.
- **Parámetros:** modelo, muestra y arreglo de goodness por clase.
- **Retorno:** clase con mayor goodness.
- **Efectos laterales:** escribe `class_goodness`.
- **Relación con FF:** implementa clasificación sin softmax.
- **HLS/FPGA:** sintetizable; repite forward para cada etiqueta.
- **Sintetizable:** sí.

### `ff_print_header_summary`

- **Lenguaje:** C++
- **Archivo:** `include/ff_dataset.hpp`
- **Propósito:** imprimir metadatos del dataset.
- **Parámetros:** ruta y cabecera.
- **Retorno:** no retorna valor.
- **Efectos laterales:** escribe en consola.
- **Relación con FF:** documenta qué datos alimentan el entrenamiento.
- **HLS/FPGA:** no sintetizable por uso de `std::cout`.
- **Sintetizable:** no, solo host/testbench.

### `ff_read_dataset`

- **Lenguaje:** C++
- **Archivo:** `include/ff_dataset.hpp`
- **Propósito:** cargar y validar un `.bin` de MNIST.
- **Parámetros:** ruta, cabecera de salida, payload de salida y error.
- **Retorno:** `true` si la lectura fue válida.
- **Efectos laterales:** llena cabecera/payload o error.
- **Relación con FF:** prepara los datos que entrenan/evalúan la BNN.
- **HLS/FPGA:** no sintetizable por uso de archivos y `std::vector`.
- **Sintetizable:** no, solo host/testbench.

### `ff_store_model_flat`

- **Lenguaje:** C++
- **Archivo:** `include/ff_layer.hpp`
- **Propósito:** exportar el modelo a buffers planos.
- **Parámetros:** modelo, pesos de salida y bias de salida.
- **Retorno:** no retorna valor.
- **Efectos laterales:** escribe buffers externos.
- **Relación con FF:** guarda el modelo entrenado para inferencia.
- **HLS/FPGA:** sintetizable; prepara transferencia por AXI.
- **Sintetizable:** sí.

### `ff_train_epoch`

- **Lenguaje:** C++
- **Archivo:** `include/ff_train.hpp`
- **Propósito:** ejecutar una época online de entrenamiento.
- **Parámetros:** modelo, dataset, muestras, semilla, modelo inicial y métricas.
- **Retorno:** no retorna valor.
- **Efectos laterales:** modifica el modelo y escribe métricas.
- **Relación con FF:** acumula goodness y aplica updates muestra a muestra.
- **HLS/FPGA:** sintetizable; evita mini-batches grandes para ahorrar BRAM.
- **Sintetizable:** sí.

### `ff_train_kernel`

- **Lenguaje:** C++
- **Archivo:** `include/ff_train.hpp`
- **Propósito:** top function HLS de entrenamiento.
- **Parámetros:** dataset, salidas de modelo, salidas de métricas, muestras,
  épocas, semilla y bandera de reset.
- **Retorno:** no retorna valor.
- **Efectos laterales:** mantiene/modifica modelo estático y escribe salidas.
- **Relación con FF:** ejecuta entrenamiento completo por epochs.
- **HLS/FPGA:** top sintetizable con AXI master para datos y AXI-Lite para
  control.
- **Sintetizable:** sí.

### `ff_train_one_sample`

- **Lenguaje:** C++
- **Archivo:** `include/ff_train.hpp`
- **Propósito:** entrenar usando un par positivo/negativo.
- **Parámetros:** modelo, muestra, LFSR, métricas y contadores de update.
- **Retorno:** no retorna valor.
- **Efectos laterales:** modifica pesos/bias y avanza LFSR.
- **Relación con FF:** unidad básica de aprendizaje local.
- **HLS/FPGA:** sintetizable; opera con arreglos locales estáticos.
- **Sintetizable:** sí.

### `ff_update_layer_local`

- **Lenguaje:** C++
- **Archivo:** `include/ff_layer.hpp`
- **Propósito:** aplicar la regla local de aprendizaje a una capa.
- **Parámetros:** modelo, capa, features/activaciones positivas y negativas,
  goodness y contadores.
- **Retorno:** no retorna valor.
- **Efectos laterales:** modifica pesos latentes y bias.
- **Relación con FF:** separa goodness positiva y negativa sin backpropagation.
- **HLS/FPGA:** sintetizable; usa operaciones enteras y clipping.
- **Sintetizable:** sí.

### `ff_validate_header_for_build`

- **Lenguaje:** C++
- **Archivo:** `include/ff_dataset.hpp`
- **Propósito:** validar layout del dataset contra macros compiladas.
- **Parámetros:** cabecera y texto de error.
- **Retorno:** `true` si el dataset es compatible.
- **Efectos laterales:** escribe `error` si falla.
- **Relación con FF:** evita entrenar con entradas mal dimensionadas.
- **HLS/FPGA:** no sintetizable por `std::string`; es validación de host.
- **Sintetizable:** no, solo host/testbench.

## Carpeta `scripts`

### `arch_tag`

- **Lenguaje:** Python
- **Archivo:** `scripts/run_sweep100.py`
- **Propósito:** construir una etiqueta de arquitectura.
- **Parámetros:** `profile`, diccionario con capas, neuronas y paralelismo.
- **Retorno:** cadena como `L2_32_8_P8`.
- **Efectos laterales:** ninguno.
- **Relación con FF:** separa resultados por red entrenada.
- **HLS/FPGA:** host; codifica paralelismo usado por la build.
- **Sintetizable:** no, script de soporte.

### `build_rows`

- **Lenguaje:** Python
- **Archivo:** `scripts/summarize_results.py`
- **Propósito:** recolectar resultados individuales en filas normalizadas.
- **Parámetros:** `results_dir`.
- **Retorno:** lista de diccionarios.
- **Efectos laterales:** lee archivos de resultados.
- **Relación con FF:** agrupa métricas de goodness y accuracy.
- **HLS/FPGA:** host; analiza coste estimado, no hardware real.
- **Sintetizable:** no.

### `estimate_complexity`

- **Lenguaje:** Python
- **Archivo:** `scripts/summarize_results.py`
- **Propósito:** estimar parámetros, bits, BRAM18 y operaciones.
- **Parámetros:** cadena de arquitectura.
- **Retorno:** diccionario con métricas de complejidad.
- **Efectos laterales:** ninguno.
- **Relación con FF:** permite comparar calidad contra coste de red.
- **HLS/FPGA:** host; aproximación útil para selección previa a síntesis.
- **Sintetizable:** no.

### `format_accuracy`

- **Lenguaje:** Python
- **Archivo:** `scripts/summarize_results.py`
- **Propósito:** convertir accuracy cruda en porcentaje.
- **Parámetros:** texto numérico.
- **Retorno:** texto formateado.
- **Efectos laterales:** ninguno.
- **Relación con FF:** presenta la métrica de clasificación final.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `format_time`

- **Lenguaje:** Python
- **Archivo:** `scripts/summarize_results.py`
- **Propósito:** formatear segundos de ejecución.
- **Parámetros:** texto numérico.
- **Retorno:** texto con sufijo `s`.
- **Efectos laterales:** ninguno.
- **Relación con FF:** ayuda a comparar coste temporal de experimentos.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `main`

- **Lenguaje:** Python
- **Archivo:** `scripts/run_all_experiments.py`
- **Propósito:** ejecutar todos los grupos sobre todos los datasets.
- **Parámetros:** argumentos CLI.
- **Retorno:** código de salida.
- **Efectos laterales:** lanza procesos `make run`.
- **Relación con FF:** automatiza comparación de entrenamiento.
- **HLS/FPGA:** host; recompila macros para cada dataset.
- **Sintetizable:** no.

### `main`

- **Lenguaje:** Python
- **Archivo:** `scripts/run_hls_csim.py`
- **Propósito:** preparar entorno y lanzar Vitis HLS csim.
- **Parámetros:** argumentos CLI.
- **Retorno:** código de Vitis o error 127.
- **Efectos laterales:** crea directorios y log.
- **Relación con FF:** valida kernels de training/infer.
- **HLS/FPGA:** host de validación HLS.
- **Sintetizable:** no.

### `main`

- **Lenguaje:** Python
- **Archivo:** `scripts/run_one_experiment.py`
- **Propósito:** ejecutar un `train_tb` y guardar su log.
- **Parámetros:** argumentos CLI.
- **Retorno:** código del testbench.
- **Efectos laterales:** escribe `log.txt` y `status.txt`.
- **Relación con FF:** conserva evidencia de una corrida.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `main`

- **Lenguaje:** Python
- **Archivo:** `scripts/run_sweep100.py`
- **Propósito:** ejecutar el barrido grande de arquitecturas.
- **Parámetros:** argumentos CLI.
- **Retorno:** código de salida.
- **Efectos laterales:** lanza muchas ejecuciones Make.
- **Relación con FF:** explora capacidad de aprendizaje por arquitectura.
- **HLS/FPGA:** host; estima viabilidad antes de síntesis.
- **Sintetizable:** no.

### `main`

- **Lenguaje:** Python
- **Archivo:** `scripts/summarize_results.py`
- **Propósito:** generar tabla global Markdown.
- **Parámetros:** argumentos CLI.
- **Retorno:** código de salida.
- **Efectos laterales:** escribe `summary_all.md`.
- **Relación con FF:** compara goodness, accuracy y complejidad.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `main`

- **Lenguaje:** Python
- **Archivo:** `scripts/write_hls_config.py`
- **Propósito:** escribir configuración para Vitis HLS.
- **Parámetros:** argumentos CLI.
- **Retorno:** código de salida.
- **Efectos laterales:** escribe archivo `.cfg`.
- **Relación con FF:** fija top y macros del kernel a simular.
- **HLS/FPGA:** host de preparación HLS.
- **Sintetizable:** no.

### `md_escape`

- **Lenguaje:** Python
- **Archivo:** `scripts/summarize_results.py`
- **Propósito:** escapar celdas de tabla Markdown.
- **Parámetros:** valor de celda.
- **Retorno:** texto seguro para tabla.
- **Efectos laterales:** ninguno.
- **Relación con FF:** mantiene legibles resultados de experimentos.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `parse_arch_dims`

- **Lenguaje:** Python
- **Archivo:** `scripts/summarize_results.py`
- **Propósito:** extraer dimensiones de una arquitectura.
- **Parámetros:** texto de arquitectura.
- **Retorno:** lista de enteros.
- **Efectos laterales:** ninguno.
- **Relación con FF:** traduce `input -> hidden...` a coste estimado.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `parse_dataset_name`

- **Lenguaje:** Python
- **Archivo:** `scripts/summarize_results.py`
- **Propósito:** inferir resolución y bits desde nombre de dataset.
- **Parámetros:** nombre del dataset.
- **Retorno:** `(resolucion, bits)`.
- **Efectos laterales:** ninguno.
- **Relación con FF:** completa columnas de comparación.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `parse_dataset_shape`

- **Lenguaje:** Python
- **Archivo:** `scripts/run_all_experiments.py`
- **Propósito:** extraer macros de imagen desde el nombre `.bin`.
- **Parámetros:** ruta del dataset.
- **Retorno:** `(ancho, alto, bits)`.
- **Efectos laterales:** ninguno.
- **Relación con FF:** evita mismatch entre binario y build.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `parse_dataset_shape`

- **Lenguaje:** Python
- **Archivo:** `scripts/run_sweep100.py`
- **Propósito:** extraer macros de imagen para el barrido.
- **Parámetros:** ruta del dataset.
- **Retorno:** `(ancho, alto, bits)`.
- **Efectos laterales:** ninguno.
- **Relación con FF:** compila cada perfil con layout correcto.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `parse_float`

- **Lenguaje:** Python
- **Archivo:** `scripts/summarize_results.py`
- **Propósito:** convertir texto a flotante tolerante a fallos.
- **Parámetros:** texto.
- **Retorno:** flotante.
- **Efectos laterales:** ninguno.
- **Relación con FF:** permite ordenar resultados incompletos.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `read_last_metrics`

- **Lenguaje:** Python
- **Archivo:** `scripts/summarize_results.py`
- **Propósito:** leer la última fila de `metrics.csv`.
- **Parámetros:** ruta del CSV.
- **Retorno:** diccionario de métricas.
- **Efectos laterales:** lee archivo.
- **Relación con FF:** recupera el estado final de entrenamiento.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `read_summary`

- **Lenguaje:** Python
- **Archivo:** `scripts/summarize_results.py`
- **Propósito:** leer pares clave/valor desde `summary.md`.
- **Parámetros:** ruta del resumen.
- **Retorno:** diccionario normalizado.
- **Efectos laterales:** lee archivo.
- **Relación con FF:** convierte reportes del testbench en datos tabulares.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `render_markdown`

- **Lenguaje:** Python
- **Archivo:** `scripts/summarize_results.py`
- **Propósito:** construir la tabla Markdown final.
- **Parámetros:** filas normalizadas.
- **Retorno:** texto Markdown.
- **Efectos laterales:** ninguno.
- **Relación con FF:** presenta comparación global de experimentos.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `resolve_existing`

- **Lenguaje:** Python
- **Archivo:** `scripts/run_hls_csim.py`
- **Propósito:** normalizar rutas existentes.
- **Parámetros:** texto de ruta.
- **Retorno:** ruta absoluta o texto original.
- **Efectos laterales:** consulta sistema de archivos.
- **Relación con FF:** ayuda a csim a encontrar el dataset correcto.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `row_sort_key`

- **Lenguaje:** Python
- **Archivo:** `scripts/summarize_results.py`
- **Propósito:** ordenar filas por ruta o por accuracy/complejidad.
- **Parámetros:** fila y modo.
- **Retorno:** tupla de ordenamiento.
- **Efectos laterales:** ninguno.
- **Relación con FF:** prioriza modelos con mejor trade-off.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `run_command`

- **Lenguaje:** Python
- **Archivo:** `scripts/run_hls_csim.py`
- **Propósito:** ejecutar Vitis HLS csim.
- **Parámetros:** ejecutable Vitis, config, work dir, entorno y log.
- **Retorno:** código de salida de Vitis.
- **Efectos laterales:** lanza proceso externo y escribe log.
- **Relación con FF:** valida kernels HLS del entrenamiento/inferencia.
- **HLS/FPGA:** host de validación.
- **Sintetizable:** no.

### `run_tag`

- **Lenguaje:** Python
- **Archivo:** `scripts/run_sweep100.py`
- **Propósito:** etiquetar threshold, épocas y evaluación.
- **Parámetros:** perfil de barrido.
- **Retorno:** texto como `thr64_ep20_ev1`.
- **Efectos laterales:** ninguno.
- **Relación con FF:** separa variantes de regla de entrenamiento.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

## Carpeta `src`

Las funciones en `src` son implementaciones de las interfaces descritas en
`include`. Las funciones protegidas por `__SYNTHESIS__` o que usan STL son de
host; las funciones restantes están diseñadas para ser compatibles con HLS.

### `ff_binary_weight`

- **Lenguaje:** C++
- **Archivo:** `src/ff_layer.cpp`
- **Propósito:** convertir peso latente a signo binario.
- **Parámetros:** peso latente.
- **Retorno:** `+1` o `-1`.
- **Efectos laterales:** ninguno.
- **Relación con FF:** define el forward de la BNN.
- **HLS/FPGA:** sintetizable; reemplaza multiplicación por suma/resta.
- **Sintetizable:** sí.

### `ff_build_features`

- **Lenguaje:** C++
- **Archivo:** `src/ff_layer.cpp`
- **Propósito:** construir entradas con etiqueta e imagen.
- **Parámetros:** muestra, etiqueta candidata y vector de salida.
- **Retorno:** no retorna valor.
- **Efectos laterales:** escribe `features`.
- **Relación con FF:** crea positivos, negativos e hipótesis de inferencia.
- **HLS/FPGA:** sintetizable; usa enteros y bucles acotados.
- **Sintetizable:** sí.

### `ff_clip_bias`

- **Lenguaje:** C++
- **Archivo:** `src/ff_layer.cpp`
- **Propósito:** limitar bias entrenable.
- **Parámetros:** valor acumulado.
- **Retorno:** bias saturado.
- **Efectos laterales:** ninguno.
- **Relación con FF:** estabiliza updates locales.
- **HLS/FPGA:** sintetizable; controla ancho efectivo de acumulación.
- **Sintetizable:** sí.

### `ff_clip_latent`

- **Lenguaje:** C++
- **Archivo:** `src/ff_layer.cpp`
- **Propósito:** limitar peso latente entrenable.
- **Parámetros:** valor acumulado.
- **Retorno:** peso saturado.
- **Efectos laterales:** ninguno.
- **Relación con FF:** mantiene pesos dentro de rango durante aprendizaje.
- **HLS/FPGA:** sintetizable; reduce riesgo de overflow.
- **Sintetizable:** sí.

### `ff_copy_sample_words`

- **Lenguaje:** C++
- **Archivo:** `src/ff_dataset.cpp`
- **Propósito:** copiar muestras desde payload global.
- **Parámetros:** payload, índice y buffer local.
- **Retorno:** no retorna valor.
- **Efectos laterales:** escribe buffer local.
- **Relación con FF:** prepara cada muestra para entrenamiento/inferencia.
- **HLS/FPGA:** sintetizable; puede pipelinearse.
- **Sintetizable:** sí.

### `ff_count_changed_biases`

- **Lenguaje:** C++
- **Archivo:** `src/ff_layer.cpp`
- **Propósito:** contar bias modificados.
- **Parámetros:** modelo antes y después.
- **Retorno:** número de bias cambiados.
- **Efectos laterales:** ninguno.
- **Relación con FF:** valida que hubo aprendizaje.
- **HLS/FPGA:** sintetizable, aunque su uso principal es métrica.
- **Sintetizable:** sí.

### `ff_count_changed_weights`

- **Lenguaje:** C++
- **Archivo:** `src/ff_layer.cpp`
- **Propósito:** contar pesos latentes modificados.
- **Parámetros:** modelo antes y después.
- **Retorno:** número de pesos cambiados.
- **Efectos laterales:** ninguno.
- **Relación con FF:** valida updates locales.
- **HLS/FPGA:** sintetizable, pero consume ciclos proporcionales al modelo.
- **Sintetizable:** sí.

### `ff_decode_label_from_words`

- **Lenguaje:** C++
- **Archivo:** `src/ff_dataset.cpp`
- **Propósito:** decodificar etiqueta one-hot.
- **Parámetros:** muestra empaquetada.
- **Retorno:** clase MNIST.
- **Efectos laterales:** ninguno.
- **Relación con FF:** identifica el ejemplo positivo.
- **HLS/FPGA:** sintetizable; el bucle de 10 clases se desenrolla.
- **Sintetizable:** sí.

### `ff_decode_pixel_from_words`

- **Lenguaje:** C++
- **Archivo:** `src/ff_dataset.cpp`
- **Propósito:** decodificar un pixel cuantizado.
- **Parámetros:** muestra e índice de pixel.
- **Retorno:** intensidad cuantizada.
- **Efectos laterales:** ninguno.
- **Relación con FF:** alimenta características de la imagen.
- **HLS/FPGA:** sintetizable; coste proporcional a bits por pixel.
- **Sintetizable:** sí.

### `ff_default_dataset_path`

- **Lenguaje:** C++
- **Archivo:** `src/ff_dataset.cpp`
- **Propósito:** resolver ruta de dataset por entorno o rutas conocidas.
- **Parámetros:** ninguno.
- **Retorno:** ruta.
- **Efectos laterales:** consulta sistema de archivos.
- **Relación con FF:** selecciona datos de entrenamiento.
- **HLS/FPGA:** no sintetizable; usa host I/O.
- **Sintetizable:** no.

### `ff_evaluate_dataset`

- **Lenguaje:** C++
- **Archivo:** `src/ff_infer.cpp`
- **Propósito:** calcular accuracy sobre muestras.
- **Parámetros:** modelo, dataset, cantidad y métricas.
- **Retorno:** no retorna valor.
- **Efectos laterales:** escribe métricas.
- **Relación con FF:** evalúa predicción por goodness.
- **HLS/FPGA:** sintetizable.
- **Sintetizable:** sí.

### `ff_forward_layer`

- **Lenguaje:** C++
- **Archivo:** `src/ff_layer.cpp`
- **Propósito:** ejecutar capa BNN y goodness.
- **Parámetros:** modelo, capa, features, activaciones, goodness.
- **Retorno:** no retorna valor.
- **Efectos laterales:** escribe activaciones/goodness.
- **Relación con FF:** núcleo de comparación positivo/negativo.
- **HLS/FPGA:** sintetizable; paralelismo parametrizable.
- **Sintetizable:** sí.

### `ff_header_from_words`

- **Lenguaje:** C++
- **Archivo:** `src/ff_dataset.cpp`
- **Propósito:** convertir cabecera cruda a estructura.
- **Parámetros:** 16 palabras de cabecera.
- **Retorno:** cabecera tipada.
- **Efectos laterales:** ninguno.
- **Relación con FF:** valida datos antes de entrenar.
- **HLS/FPGA:** host; no participa en kernel.
- **Sintetizable:** no.

### `ff_infer_kernel`

- **Lenguaje:** C++
- **Archivo:** `src/ff_infer.cpp`
- **Propósito:** top HLS de inferencia.
- **Parámetros:** dataset, modelo plano, salidas y muestras.
- **Retorno:** no retorna valor.
- **Efectos laterales:** escribe predicciones y aciertos.
- **Relación con FF:** predice por mayor goodness.
- **HLS/FPGA:** top sintetizable con AXI.
- **Sintetizable:** sí.

### `ff_initial_latent`

- **Lenguaje:** C++
- **Archivo:** `src/ff_layer.cpp`
- **Propósito:** generar peso inicial pequeño.
- **Parámetros:** estado LFSR.
- **Retorno:** peso latente inicial.
- **Efectos laterales:** ninguno.
- **Relación con FF:** define punto de partida del entrenamiento.
- **HLS/FPGA:** sintetizable; no usa aleatoriedad de host.
- **Sintetizable:** sí.

### `ff_init_model`

- **Lenguaje:** C++
- **Archivo:** `src/ff_layer.cpp`
- **Propósito:** inicializar modelo completo.
- **Parámetros:** modelo y semilla.
- **Retorno:** no retorna valor.
- **Efectos laterales:** sobrescribe pesos/bias.
- **Relación con FF:** prepara estado inicial.
- **HLS/FPGA:** sintetizable; inicialización secuencial pipelineada.
- **Sintetizable:** sí.

### `ff_layer_input_dim`

- **Lenguaje:** C++
- **Archivo:** `src/ff_layer.cpp`
- **Propósito:** obtener entrada activa por capa.
- **Parámetros:** índice de capa.
- **Retorno:** dimensión.
- **Efectos laterales:** ninguno.
- **Relación con FF:** conecta capas.
- **HLS/FPGA:** sintetizable.
- **Sintetizable:** sí.

### `ff_layer_neurons`

- **Lenguaje:** C++
- **Archivo:** `src/ff_layer.cpp`
- **Propósito:** obtener neuronas activas por capa.
- **Parámetros:** índice de capa.
- **Retorno:** neuronas.
- **Efectos laterales:** ninguno.
- **Relación con FF:** limita bucles a arquitectura activa.
- **HLS/FPGA:** sintetizable.
- **Sintetizable:** sí.

### `ff_lfsr_next`

- **Lenguaje:** C++
- **Archivo:** `src/ff_layer.cpp`
- **Propósito:** avanzar generador pseudoaleatorio.
- **Parámetros:** estado actual.
- **Retorno:** nuevo estado.
- **Efectos laterales:** ninguno.
- **Relación con FF:** inicializa pesos y genera negativos.
- **HLS/FPGA:** sintetizable.
- **Sintetizable:** sí.

### `ff_load_model_flat`

- **Lenguaje:** C++
- **Archivo:** `src/ff_layer.cpp`
- **Propósito:** cargar modelo desde buffers planos.
- **Parámetros:** modelo, pesos y bias.
- **Retorno:** no retorna valor.
- **Efectos laterales:** sobrescribe modelo.
- **Relación con FF:** prepara inferencia con modelo entrenado.
- **HLS/FPGA:** sintetizable; adapta AXI a memoria local.
- **Sintetizable:** sí.

### `ff_lr_step_for_input`

- **Lenguaje:** C++
- **Archivo:** `src/ff_layer.cpp`
- **Propósito:** elegir paso de aprendizaje por entrada.
- **Parámetros:** capa e índice de entrada.
- **Retorno:** paso entero.
- **Efectos laterales:** ninguno.
- **Relación con FF:** refuerza etiquetas frente a pixeles.
- **HLS/FPGA:** sintetizable.
- **Sintetizable:** sí.

### `ff_make_negative_label`

- **Lenguaje:** C++
- **Archivo:** `src/ff_layer.cpp`
- **Propósito:** crear clase negativa supervisada.
- **Parámetros:** etiqueta real y estado LFSR.
- **Retorno:** etiqueta incorrecta.
- **Efectos laterales:** avanza LFSR.
- **Relación con FF:** genera ejemplo negativo.
- **HLS/FPGA:** sintetizable.
- **Sintetizable:** sí.

### `ff_predict_sample`

- **Lenguaje:** C++
- **Archivo:** `src/ff_infer.cpp`
- **Propósito:** predecir una muestra por goodness.
- **Parámetros:** modelo, muestra y scores.
- **Retorno:** clase predicha.
- **Efectos laterales:** escribe scores por clase.
- **Relación con FF:** reemplaza softmax por comparación de etiquetas.
- **HLS/FPGA:** sintetizable.
- **Sintetizable:** sí.

### `ff_print_header_summary`

- **Lenguaje:** C++
- **Archivo:** `src/ff_dataset.cpp`
- **Propósito:** imprimir metadatos del dataset.
- **Parámetros:** ruta y cabecera.
- **Retorno:** no retorna valor.
- **Efectos laterales:** escribe consola.
- **Relación con FF:** documenta entrada experimental.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `ff_read_dataset`

- **Lenguaje:** C++
- **Archivo:** `src/ff_dataset.cpp`
- **Propósito:** leer dataset completo.
- **Parámetros:** ruta, cabecera, payload y error.
- **Retorno:** booleano de éxito.
- **Efectos laterales:** llena estructuras de host.
- **Relación con FF:** alimenta entrenamiento/evaluación.
- **HLS/FPGA:** host; usa archivos y vector.
- **Sintetizable:** no.

### `ff_relu_clip`

- **Lenguaje:** C++
- **Archivo:** `src/ff_layer.cpp`
- **Propósito:** aplicar ReLU saturada.
- **Parámetros:** acumulación.
- **Retorno:** activación.
- **Efectos laterales:** ninguno.
- **Relación con FF:** define activación usada en goodness.
- **HLS/FPGA:** sintetizable; evita funciones no lineales costosas.
- **Sintetizable:** sí.

### `ff_store_model_flat`

- **Lenguaje:** C++
- **Archivo:** `src/ff_layer.cpp`
- **Propósito:** exportar modelo a memoria plana.
- **Parámetros:** modelo, pesos de salida y bias de salida.
- **Retorno:** no retorna valor.
- **Efectos laterales:** escribe buffers.
- **Relación con FF:** conserva modelo entrenado.
- **HLS/FPGA:** sintetizable; orientado a AXI.
- **Sintetizable:** sí.

### `ff_train_epoch`

- **Lenguaje:** C++
- **Archivo:** `src/ff_train.cpp`
- **Propósito:** entrenar una época online.
- **Parámetros:** modelo, dataset, muestras, semilla, modelo inicial y métricas.
- **Retorno:** no retorna valor.
- **Efectos laterales:** modifica modelo y métricas.
- **Relación con FF:** aplica regla local muestra a muestra.
- **HLS/FPGA:** sintetizable; ahorra BRAM frente a mini-batch.
- **Sintetizable:** sí.

### `ff_train_kernel`

- **Lenguaje:** C++
- **Archivo:** `src/ff_train.cpp`
- **Propósito:** top function de entrenamiento.
- **Parámetros:** buffers AXI, muestras, épocas, semilla y reset.
- **Retorno:** no retorna valor.
- **Efectos laterales:** entrena modelo estático y escribe salidas.
- **Relación con FF:** ejecuta el flujo de aprendizaje completo.
- **HLS/FPGA:** top sintetizable.
- **Sintetizable:** sí.

### `ff_train_one_sample`

- **Lenguaje:** C++
- **Archivo:** `src/ff_train.cpp`
- **Propósito:** entrenar una muestra.
- **Parámetros:** modelo, muestra, LFSR, métricas y contadores.
- **Retorno:** no retorna valor.
- **Efectos laterales:** cambia pesos/bias.
- **Relación con FF:** crea positivo/negativo y actualiza capas.
- **HLS/FPGA:** sintetizable.
- **Sintetizable:** sí.

### `ff_update_layer_local`

- **Lenguaje:** C++
- **Archivo:** `src/ff_layer.cpp`
- **Propósito:** actualizar pesos/bias de una capa.
- **Parámetros:** modelo, capa, features/activaciones, goodness y contadores.
- **Retorno:** no retorna valor.
- **Efectos laterales:** modifica parámetros entrenables.
- **Relación con FF:** materializa la señal de aprendizaje local.
- **HLS/FPGA:** sintetizable; usa enteros y clipping.
- **Sintetizable:** sí.

### `ff_validate_header_for_build`

- **Lenguaje:** C++
- **Archivo:** `src/ff_dataset.cpp`
- **Propósito:** validar cabecera contra macros.
- **Parámetros:** cabecera y error.
- **Retorno:** booleano.
- **Efectos laterales:** escribe error.
- **Relación con FF:** asegura entrada consistente.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

## Carpeta `testbench`

### `arg_eq`

- **Lenguaje:** C++
- **Archivo:** `testbench/train_tb.cpp`
- **Propósito:** comparar tokens CLI.
- **Parámetros:** argumento y texto esperado.
- **Retorno:** booleano.
- **Efectos laterales:** ninguno.
- **Relación con FF:** soporte para configurar experimentos.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `architecture_string`

- **Lenguaje:** C++
- **Archivo:** `testbench/train_tb.cpp`
- **Propósito:** imprimir arquitectura completa.
- **Parámetros:** ninguno.
- **Retorno:** cadena `input -> hidden...`.
- **Efectos laterales:** ninguno.
- **Relación con FF:** documenta red entrenada.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `architecture_tag`

- **Lenguaje:** C++
- **Archivo:** `testbench/train_tb.cpp`
- **Propósito:** crear etiqueta de carpeta por arquitectura.
- **Parámetros:** ninguno.
- **Retorno:** texto como `L2_32_8_P8`.
- **Efectos laterales:** ninguno.
- **Relación con FF:** separa resultados.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `binary_sign`

- **Lenguaje:** C++
- **Archivo:** `testbench/train_tb.cpp`
- **Propósito:** obtener signo binario de un peso.
- **Parámetros:** peso latente.
- **Retorno:** `+1` o `-1`.
- **Efectos laterales:** ninguno.
- **Relación con FF:** mide cambios funcionales de la BNN.
- **HLS/FPGA:** host/testbench.
- **Sintetizable:** no.

### `compiled_hidden_values_string`

- **Lenguaje:** C++
- **Archivo:** `testbench/train_tb.cpp`
- **Propósito:** formatear capas ocultas compiladas.
- **Parámetros:** ninguno.
- **Retorno:** texto de valores ocultos.
- **Efectos laterales:** ninguno.
- **Relación con FF:** valida arquitectura.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `count_binary_sign_changes`

- **Lenguaje:** C++
- **Archivo:** `testbench/train_tb.cpp`
- **Propósito:** contar cambios de signo de pesos.
- **Parámetros:** modelo antes y después.
- **Retorno:** número de cambios de signo.
- **Efectos laterales:** ninguno.
- **Relación con FF:** indica cambios reales del forward binario.
- **HLS/FPGA:** host/testbench.
- **Sintetizable:** no.

### `dataset_stem_from_path`

- **Lenguaje:** C++
- **Archivo:** `testbench/train_tb.cpp`
- **Propósito:** extraer stem de ruta.
- **Parámetros:** ruta.
- **Retorno:** stem.
- **Efectos laterales:** ninguno.
- **Relación con FF:** nombra resultados por dataset.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `dataset_stem_without_packed`

- **Lenguaje:** C++
- **Archivo:** `testbench/train_tb.cpp`
- **Propósito:** normalizar nombre de dataset.
- **Parámetros:** ruta.
- **Retorno:** nombre sin `_packed`.
- **Efectos laterales:** ninguno.
- **Relación con FF:** agrupa corridas comparables.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `hidden_arrow_string`

- **Lenguaje:** C++
- **Archivo:** `testbench/train_tb.cpp`
- **Propósito:** formatear capas ocultas.
- **Parámetros:** ninguno.
- **Retorno:** texto con flechas.
- **Efectos laterales:** ninguno.
- **Relación con FF:** documenta estructura entrenada.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `main`

- **Lenguaje:** C++
- **Archivo:** `testbench/hls_csim_tb.cpp`
- **Propósito:** ejecutar csim de kernels HLS.
- **Parámetros:** ninguno.
- **Retorno:** código de salida.
- **Efectos laterales:** carga dataset, llama kernels e imprime métricas.
- **Relación con FF:** valida training e inferencia top-level.
- **HLS/FPGA:** host de csim, no hardware.
- **Sintetizable:** no.

### `main`

- **Lenguaje:** C++
- **Archivo:** `testbench/train_tb.cpp`
- **Propósito:** ejecutar simulación completa.
- **Parámetros:** `argc`, `argv`.
- **Retorno:** código de salida.
- **Efectos laterales:** entrena, evalúa y escribe resultados.
- **Relación con FF:** flujo experimental principal.
- **HLS/FPGA:** host/testbench.
- **Sintetizable:** no.

### `parse_args`

- **Lenguaje:** C++
- **Archivo:** `testbench/train_tb.cpp`
- **Propósito:** parsear y validar CLI.
- **Parámetros:** `argc`, `argv`.
- **Retorno:** opciones de corrida.
- **Efectos laterales:** puede terminar el proceso ante error.
- **Relación con FF:** garantiza consistencia entre comando y build.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `parse_hidden_values_list`

- **Lenguaje:** C++
- **Archivo:** `testbench/train_tb.cpp`
- **Propósito:** convertir `"32 8"` a lista.
- **Parámetros:** texto.
- **Retorno:** vector de enteros.
- **Efectos laterales:** ninguno.
- **Relación con FF:** valida capas ocultas.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `print_model_probe`

- **Lenguaje:** C++
- **Archivo:** `testbench/train_tb.cpp`
- **Propósito:** imprimir una muestra de parámetros.
- **Parámetros:** prefijo y modelo.
- **Retorno:** no retorna valor.
- **Efectos laterales:** escribe consola.
- **Relación con FF:** evidencia cambios tras entrenamiento.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `print_network_config`

- **Lenguaje:** C++
- **Archivo:** `testbench/train_tb.cpp`
- **Propósito:** imprimir configuración efectiva.
- **Parámetros:** ruta de dataset.
- **Retorno:** no retorna valor.
- **Efectos laterales:** escribe consola.
- **Relación con FF:** documenta entrada y arquitectura.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `print_scores_for_first_eval`

- **Lenguaje:** C++
- **Archivo:** `testbench/train_tb.cpp`
- **Propósito:** imprimir goodness de la primera muestra.
- **Parámetros:** modelo y bloque de evaluación.
- **Retorno:** no retorna valor.
- **Efectos laterales:** escribe consola.
- **Relación con FF:** muestra inferencia por comparación de etiquetas.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `print_usage`

- **Lenguaje:** C++
- **Archivo:** `testbench/train_tb.cpp`
- **Propósito:** mostrar ayuda CLI.
- **Parámetros:** nombre del programa.
- **Retorno:** no retorna valor.
- **Efectos laterales:** escribe consola.
- **Relación con FF:** documenta cómo ejecutar experimentos.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `read_env_int`

- **Lenguaje:** C++
- **Archivo:** `testbench/hls_csim_tb.cpp` y `testbench/train_tb.cpp`
- **Propósito:** leer enteros desde entorno.
- **Parámetros:** nombre y valor por defecto.
- **Retorno:** entero validado.
- **Efectos laterales:** consulta entorno.
- **Relación con FF:** configura muestras/épocas sin recompilar.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `read_env_u32`

- **Lenguaje:** C++
- **Archivo:** `testbench/hls_csim_tb.cpp` y `testbench/train_tb.cpp`
- **Propósito:** leer semilla desde entorno.
- **Parámetros:** nombre y semilla por defecto.
- **Retorno:** semilla validada.
- **Efectos laterales:** consulta entorno.
- **Relación con FF:** reproduce inicialización y negativos.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `validate_dataset_matches_compile`

- **Lenguaje:** C++
- **Archivo:** `testbench/train_tb.cpp`
- **Propósito:** validar dataset contra macros.
- **Parámetros:** cabecera.
- **Retorno:** booleano.
- **Efectos laterales:** ninguno.
- **Relación con FF:** evita experimentos inválidos.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `write_metrics_csv`

- **Lenguaje:** C++
- **Archivo:** `testbench/train_tb.cpp`
- **Propósito:** escribir métricas por época.
- **Parámetros:** carpeta y filas.
- **Retorno:** no retorna valor.
- **Efectos laterales:** escribe `metrics.csv`.
- **Relación con FF:** conserva goodness, updates y accuracy.
- **HLS/FPGA:** host.
- **Sintetizable:** no.

### `write_summary_md`

- **Lenguaje:** C++
- **Archivo:** `testbench/train_tb.cpp`
- **Propósito:** escribir resumen final.
- **Parámetros:** carpeta, opciones, cabecera, métricas y estado.
- **Retorno:** no retorna valor.
- **Efectos laterales:** escribe `summary.md`.
- **Relación con FF:** deja evidencia final de cada corrida.
- **HLS/FPGA:** host.
- **Sintetizable:** no.
