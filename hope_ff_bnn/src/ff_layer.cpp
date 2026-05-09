
/**
 * @file ff_layer.cpp
 * @brief Implementación de la capa BNN usada por Forward-Forward.
 *
 * Este archivo contiene la parte central sintetizable: pesos latentes, pesos
 * binarios por signo, ReLU saturada, cálculo de goodness y actualización local
 * sin backpropagation global. Las decisiones numéricas favorecen operaciones
 * enteras simples para una FPGA Artix-7.
 */
#include "ff_layer.hpp"

/**
 * @brief Satura un peso latente al rango entrenable permitido.
 *
 * @param value Nuevo valor acumulado antes del clipping.
 * @return Peso latente limitado a `[-FF_LATENT_CLIP, FF_LATENT_CLIP]`.
 *
 * @note Sintetizable. El clipping evita overflow y mantiene acotado el ancho
 * efectivo de la memoria de pesos.
 */
static inline ff_latent_t ff_clip_latent(ff_accum_t value) {
/* Inline elimina llamadas en el bucle de actualización de todos los pesos. */
#pragma HLS INLINE
    if (value > FF_LATENT_CLIP) {
        return (ff_latent_t)FF_LATENT_CLIP;
    }

    if (value < -FF_LATENT_CLIP) {
        return (ff_latent_t)(-FF_LATENT_CLIP);
    }

    return (ff_latent_t)value;
}

/**
 * @brief Satura un bias al rango permitido por la configuración.
 *
 * @param value Bias acumulado antes del clipping.
 * @return Bias limitado al rango de entrenamiento.
 *
 * @note Sintetizable. Controla crecimiento durante entrenamientos largos y
 * ayuda a mantener acotados acumuladores y registros.
 */
static inline ff_bias_t ff_clip_bias(ff_accum_t value) {
/* Inline reduce lógica de control dentro del update por neurona. */
#pragma HLS INLINE
    if (value > FF_BIAS_CLIP) {
        return (ff_bias_t)FF_BIAS_CLIP;
    }

    if (value < -FF_BIAS_CLIP) {
        return (ff_bias_t)(-FF_BIAS_CLIP);
    }

    return (ff_bias_t)value;
}

/**
 * @brief Convierte el peso latente en peso binario de la BNN.
 *
 * @param latent_weight Peso entrenable almacenado en memoria.
 * @return `+1` si el peso latente es no negativo, `-1` en caso contrario.
 *
 * El modelo entrena pesos latentes pero el forward usa únicamente el signo.
 * Esto aproxima una BNN sin requerir multiplicadores generales.
 */
static inline int8_t ff_binary_weight(ff_latent_t latent_weight) {
/* Inline hace que el signo se evalúe junto al producto suma/resta del forward. */
#pragma HLS INLINE
    return (latent_weight >= 0) ? (int8_t)1 : (int8_t)-1;
}

/**
 * @brief Aplica ReLU y saturación a una acumulación de neurona.
 *
 * @param value Suma ponderada más bias.
 * @return Activación no negativa limitada por `FF_ACTIVATION_CLIP`.
 *
 * La saturación reemplaza funciones no lineales costosas y estabiliza el
 * cálculo de goodness en enteros.
 */
static inline ff_activation_t ff_relu_clip(ff_accum_t value) {
/* Inline evita jerarquía adicional en la ruta crítica del forward. */
#pragma HLS INLINE
    if (value <= 0) {
        return 0;
    }

    if (value > FF_ACTIVATION_CLIP) {
        return FF_ACTIVATION_CLIP;
    }

    return (ff_activation_t)value;
}

/**
 * @brief Genera un peso latente inicial pequeño desde el estado LFSR.
 *
 * @param state Estado pseudoaleatorio actual.
 * @return Peso inicial con signo y magnitud deterministas.
 *
 * La inicialización no usa `rand()` ni distribución flotante, por lo que puede
 * reproducirse en csim y sintetizarse.
 */
static inline ff_latent_t ff_initial_latent(uint32_t state) {
/* Inline integra la inicialización dentro del bucle que llena el modelo. */
#pragma HLS INLINE
    ff_latent_t magnitude =
        (state & 0x2u) ? (ff_latent_t)FF_LATENT_INIT_MAG
                       : (ff_latent_t)(FF_LATENT_INIT_MAG + 1);
    return (state & 0x1u) ? magnitude : (ff_latent_t)(-magnitude);
}

/**
 * @brief Selecciona el paso de aprendizaje según el tipo de entrada.
 *
 * @param layer_idx Índice de capa entrenada.
 * @param input_idx Índice de entrada dentro de la capa.
 * @return Paso entero aplicado al peso latente.
 *
 * La primera capa da más fuerza a los bits de etiqueta porque solo hay 10
 * entradas one-hot frente a cientos de pixeles.
 */
static inline int ff_lr_step_for_input(int layer_idx, int input_idx) {
/* Inline evita una llamada dentro del bucle de actualización de pesos. */
#pragma HLS INLINE
    if (layer_idx == 0 && input_idx < FF_NUM_CLASSES) {
        return FF_LABEL_LR_STEP;
    }
    return FF_PIXEL_LR_STEP;
}

/**
 * @brief Avanza un LFSR determinista de 16 bits.
 *
 * @param state Estado actual; si llega cero se reemplaza por una semilla fija.
 * @return Nuevo estado pseudoaleatorio.
 *
 * @note Sintetizable. Se usa para inicialización y generación de ejemplos
 * negativos sin depender de librerías no sintetizables.
 */
uint32_t ff_lfsr_next(uint32_t state) {
/* Inline permite usar el LFSR dentro de inicialización y entrenamiento sin
 * crear una unidad separada.
 */
#pragma HLS INLINE
    if (state == 0u) {
        state = 0xACE1u;
    }

    uint32_t bit = ((state >> 0) ^ (state >> 2) ^
                    (state >> 3) ^ (state >> 5)) & 1u;
    return (state >> 1) | (bit << 15);
}

/**
 * @brief Construye una etiqueta negativa supervisada.
 *
 * @param true_label Etiqueta real de MNIST.
 * @param rng_state Estado LFSR que se actualiza por referencia.
 * @return Clase incorrecta usada para el ejemplo negativo.
 *
 * La aritmética con offset garantiza que el negativo no coincida con la etiqueta
 * verdadera, condición básica del entrenamiento Forward-Forward supervisado.
 */
uint8_t ff_make_negative_label(
    uint8_t true_label,
    uint32_t &rng_state
) {
/* Inline reduce overhead porque se ejecuta una vez por muestra entrenada. */
#pragma HLS INLINE
    rng_state = ff_lfsr_next(rng_state);
    uint8_t offset = (uint8_t)(1u + (rng_state % (FF_NUM_CLASSES - 1)));
    return (uint8_t)((true_label + offset) % FF_NUM_CLASSES);
}

/**
 * @brief Devuelve el ancho de entrada activo de una capa.
 *
 * @param layer_idx Capa consultada.
 * @return `FF_INPUT_DIM` para la primera capa o neuronas de la capa anterior.
 */
int ff_layer_input_dim(int layer_idx) {
/* Inline permite que HLS propague constantes cuando `layer_idx` se conoce en
 * bucles acotados por `FF_HIDDEN_LAYERS`.
 */
#pragma HLS INLINE
    return (layer_idx == 0) ? FF_INPUT_DIM : FF_LAYER_NEURONS[layer_idx - 1];
}

/**
 * @brief Devuelve el número de neuronas activas de una capa.
 *
 * @param layer_idx Capa consultada.
 * @return Neuronas configuradas por macros de arquitectura.
 */
int ff_layer_neurons(int layer_idx) {
/* Inline evita una pequeña función de consulta en bucles de capa. */
#pragma HLS INLINE
    return FF_LAYER_NEURONS[layer_idx];
}

/**
 * @brief Inicializa pesos latentes y bias del modelo.
 *
 * @param model Modelo que se sobrescribe.
 * @param seed Semilla reproducible para el LFSR.
 *
 * @sideeffect Escribe todos los pesos y bias. Las posiciones fuera de la
 * arquitectura activa quedan a cero para que exportación, conteos y síntesis no
 * dependan de memoria sin inicializar.
 */
void ff_init_model(
    ff_model_t &model,
    uint32_t seed
) {
    uint32_t state = (seed == 0u) ? 0x1D2Bu : seed;

init_layer_loop:
    for (int layer = 0; layer < FF_HIDDEN_LAYERS; layer++) {
        int neurons = ff_layer_neurons(layer);
        int input_dim = ff_layer_input_dim(layer);

init_neuron_loop:
        for (int neuron = 0; neuron < FF_MAX_LAYER_NEURONS; neuron++) {
            model.biases[layer][neuron] = 0;

init_input_loop:
            for (int input_idx = 0; input_idx < FF_MAX_LAYER_INPUT_DIM; input_idx++) {
/* II=1 busca inicializar una celda de peso por ciclo. El coste de latencia es
 * proporcional al tamaño máximo del modelo, no solo a la arquitectura activa.
 */
#pragma HLS PIPELINE II=1
                state = ff_lfsr_next(state);
                if (neuron < neurons && input_idx < input_dim) {
                    model.weights[layer][neuron][input_idx] =
                        ff_initial_latent(state);
                } else {
                    model.weights[layer][neuron][input_idx] = 0;
                }
            }
        }
    }
}

/**
 * @brief Construye características para una etiqueta candidata.
 *
 * @param sample_words Muestra MNIST empaquetada.
 * @param label_index Etiqueta insertada como hipótesis.
 * @param features Vector de salida para la primera capa.
 *
 * Se usa tres veces en el flujo: ejemplo positivo con etiqueta real, ejemplo
 * negativo con etiqueta falsa e inferencia multiclase con cada etiqueta 0..9.
 */
void ff_build_features(
    const ff_word_t sample_words[FF_WORDS_PER_SAMPLE],
    uint8_t label_index,
    ff_feature_t features[FF_MAX_LAYER_INPUT_DIM]
) {
build_clear_features_loop:
    for (int i = 0; i < FF_MAX_LAYER_INPUT_DIM; i++) {
/* La limpieza completa evita que entradas sobrantes de arquitecturas previas o
 * capas más pequeñas contaminen la siguiente evaluación.
 */
#pragma HLS PIPELINE II=1
        features[i] = 0;
    }

build_label_features_loop:
    for (int c = 0; c < FF_NUM_CLASSES; c++) {
/* Las 10 clases de MNIST son pocas; desenrollarlas crea en paralelo el prefijo
 * one-hot y elimina latencia apreciable.
 */
#pragma HLS UNROLL
        features[c] = (c == (int)label_index) ?
            (ff_feature_t)FF_LABEL_SCALE : (ff_feature_t)0;
    }

build_pixel_features_loop:
    for (int pixel = 0; pixel < FF_NUM_PIXELS; pixel++) {
/* Se intenta producir un pixel escalado por ciclo. La memoria local de muestra
 * y el decodificador de bits determinan si HLS alcanza II=1.
 */
#pragma HLS PIPELINE II=1
        uint8_t raw_pixel = ff_decode_pixel_from_words(sample_words, pixel);

        if (raw_pixel == 0u) {
            features[FF_NUM_CLASSES + pixel] = 0;
        } else if (FF_PIXEL_BITS == 1) {
            features[FF_NUM_CLASSES + pixel] = (ff_feature_t)FF_PIXEL_SCALE;
        } else {
            int max_value = (1 << FF_PIXEL_BITS) - 1;
/* El redondeo entero reemplaza normalización flotante. Si el pixel cuantizado
 * es no cero, se fuerza al menos 1 para no perder información tenue.
 */
            int scaled = ((int)raw_pixel * FF_PIXEL_SCALE + (max_value / 2)) /
                max_value;
            if (scaled <= 0) {
                scaled = 1;
            }
            features[FF_NUM_CLASSES + pixel] = (ff_feature_t)scaled;
        }
    }
}

/**
 * @brief Ejecuta una capa BNN y calcula goodness local.
 *
 * @param model Modelo con pesos latentes y bias.
 * @param layer_idx Capa evaluada.
 * @param features Entrada activa de la capa.
 * @param activations Activaciones ReLU de salida.
 * @param goodness Promedio de cuadrados de activación.
 *
 * @note Sintetizable. Cada tile procesa `FF_PARALLEL_NEURONS` neuronas. Aumentar
 * ese valor mejora throughput, pero incrementa sumadores, comparadores y
 * presión sobre la memoria de pesos.
 */
void ff_forward_layer(
    const ff_model_t &model,
    int layer_idx,
    const ff_feature_t features[FF_MAX_LAYER_INPUT_DIM],
    ff_activation_t activations[FF_MAX_LAYER_NEURONS],
    ff_goodness_t &goodness
) {
    ff_metric_accum_t sum_sq = 0;
    int neurons = ff_layer_neurons(layer_idx);
    int input_dim = ff_layer_input_dim(layer_idx);

forward_clear_loop:
    for (int neuron = 0; neuron < FF_MAX_LAYER_NEURONS; neuron++) {
/* Inicializar todas las salidas evita que neuronas inactivas propaguen valores
 * antiguos hacia capas posteriores o métricas.
 */
#pragma HLS PIPELINE II=1
        activations[neuron] = 0;
    }

forward_tile_loop:
    for (int tile = 0; tile < FF_MAX_LAYER_NEURONS; tile += FF_PARALLEL_NEURONS) {
        ff_accum_t acc[FF_PARALLEL_NEURONS];
/* Cada lane necesita su acumulador independiente. La partición completa permite
 * sumar varias neuronas en paralelo a costa de más registros/LUT.
 */
#pragma HLS ARRAY_PARTITION variable=acc complete

forward_init_lane_loop:
        for (int lane = 0; lane < FF_PARALLEL_NEURONS; lane++) {
/* Desenrollar lanes instancia hardware paralelo para el tile de neuronas. */
#pragma HLS UNROLL
            int neuron = tile + lane;
            acc[lane] = (neuron < neurons) ?
                model.biases[layer_idx][neuron] : 0;
        }

forward_input_loop:
        for (int input_idx = 0; input_idx < FF_MAX_LAYER_INPUT_DIM; input_idx++) {
/* El pipeline intenta consumir una característica por ciclo para todo el tile.
 * La latencia por tile depende de la dimensión máxima de entrada.
 */
#pragma HLS PIPELINE II=1
            ff_feature_t x = (input_idx < input_dim) ? features[input_idx] : 0;

forward_lane_loop:
            for (int lane = 0; lane < FF_PARALLEL_NEURONS; lane++) {
/* El desenrollado aplica la misma característica a varias neuronas del tile. */
#pragma HLS UNROLL
                int neuron = tile + lane;
                if (neuron < neurons && input_idx < input_dim) {
                    int8_t sign =
                        ff_binary_weight(model.weights[layer_idx][neuron][input_idx]);
                    acc[lane] += (sign > 0) ? (ff_accum_t)x : (ff_accum_t)(-x);
                }
            }
        }

forward_store_lane_loop:
        for (int lane = 0; lane < FF_PARALLEL_NEURONS; lane++) {
/* Guardar lanes en paralelo reduce la cola del tile y acumula goodness sin un
 * bucle secuencial por neurona.
 */
#pragma HLS UNROLL
            int neuron = tile + lane;
            if (neuron < neurons) {
                ff_activation_t act = ff_relu_clip(acc[lane]);
                activations[neuron] = act;
                sum_sq += (ff_metric_accum_t)act * (ff_metric_accum_t)act;
            }
        }
    }

    goodness = (neurons > 0) ? (ff_goodness_t)(sum_sq / neurons) : 0;
}

/**
 * @brief Aplica la actualización local de una capa Forward-Forward.
 *
 * @param model Modelo modificado in-place.
 * @param layer_idx Capa a entrenar.
 * @param pos_features Entrada del ejemplo con etiqueta correcta.
 * @param neg_features Entrada del ejemplo con etiqueta incorrecta.
 * @param pos_acts Activaciones positivas de la capa.
 * @param neg_acts Activaciones negativas de la capa.
 * @param g_pos Goodness positiva.
 * @param g_neg Goodness negativa.
 * @param pos_update_events Contador de neuronas reforzadas.
 * @param neg_update_events Contador de neuronas penalizadas.
 *
 * @sideeffect Cambia pesos latentes y bias. La regla es local: no necesita
 * almacenar gradientes globales ni recorrer capas en sentido inverso.
 */
void ff_update_layer_local(
    ff_model_t &model,
    int layer_idx,
    const ff_feature_t pos_features[FF_MAX_LAYER_INPUT_DIM],
    const ff_feature_t neg_features[FF_MAX_LAYER_INPUT_DIM],
    const ff_activation_t pos_acts[FF_MAX_LAYER_NEURONS],
    const ff_activation_t neg_acts[FF_MAX_LAYER_NEURONS],
    ff_goodness_t g_pos,
    ff_goodness_t g_neg,
    uint32_t &pos_update_events,
    uint32_t &neg_update_events
) {
/* La señal de aprendizaje es una comparación por tramos contra el umbral. Esta
 * aproximación evita sigmoid/softplus y reduce coste en FPGA.
 */
    bool push_positive_up = (g_pos < (ff_goodness_t)FF_GOODNESS_THRESHOLD);
    bool push_negative_down = (g_neg > (ff_goodness_t)FF_GOODNESS_THRESHOLD);
    int neurons = ff_layer_neurons(layer_idx);
    int input_dim = ff_layer_input_dim(layer_idx);

update_neuron_loop:
    for (int neuron = 0; neuron < FF_MAX_LAYER_NEURONS; neuron++) {
        bool active_neuron = (neuron < neurons);
        bool pos_active = active_neuron && push_positive_up && (pos_acts[neuron] > 0);
        bool neg_active = active_neuron && push_negative_down && (neg_acts[neuron] > 0);

        if (pos_active) {
            pos_update_events++;
            model.biases[layer_idx][neuron] =
                ff_clip_bias((ff_accum_t)model.biases[layer_idx][neuron] +
                             FF_BIAS_LR_STEP);
        }

        if (neg_active) {
            neg_update_events++;
            model.biases[layer_idx][neuron] =
                ff_clip_bias((ff_accum_t)model.biases[layer_idx][neuron] -
                             FF_BIAS_LR_STEP);
        }

update_input_loop:
        for (int input_idx = 0; input_idx < FF_MAX_LAYER_INPUT_DIM; input_idx++) {
/* El update recorre pesos de forma streaming. II=1 reduce latencia, aunque las
 * dependencias de escritura sobre `model.weights` pueden limitar el resultado.
 */
#pragma HLS PIPELINE II=1
            if (!active_neuron || input_idx >= input_dim) {
                continue;
            }

            ff_accum_t next_weight = model.weights[layer_idx][neuron][input_idx];
            int step = ff_lr_step_for_input(layer_idx, input_idx);

            if (pos_active && pos_features[input_idx] != 0) {
                next_weight += step;
            }

            if (neg_active && neg_features[input_idx] != 0) {
                next_weight -= step;
            }

            model.weights[layer_idx][neuron][input_idx] =
                ff_clip_latent(next_weight);
        }
    }
}

/**
 * @brief Cuenta pesos latentes que cambiaron respecto a una copia inicial.
 *
 * @param before Modelo antes del entrenamiento.
 * @param after Modelo después del entrenamiento.
 * @return Número de celdas de peso modificadas en capas activas.
 *
 * Se usa como métrica de verificación: un entrenamiento que no modifica pesos
 * probablemente no está ejerciendo la regla Forward-Forward.
 */
uint32_t ff_count_changed_weights(
    const ff_model_t &before,
    const ff_model_t &after
) {
    uint32_t changed = 0;

count_layer_loop:
    for (int layer = 0; layer < FF_HIDDEN_LAYERS; layer++) {
        int neurons = ff_layer_neurons(layer);
        int input_dim = ff_layer_input_dim(layer);

count_neuron_loop:
        for (int neuron = 0; neuron < FF_MAX_LAYER_NEURONS; neuron++) {
count_input_loop:
            for (int input_idx = 0; input_idx < FF_MAX_LAYER_INPUT_DIM; input_idx++) {
/* El conteo se pipelinea para revisar modelos grandes sin convertir la
 * validación en el cuello de botella de simulación/HLS.
 */
#pragma HLS PIPELINE II=1
                if (neuron < neurons && input_idx < input_dim &&
                    before.weights[layer][neuron][input_idx] !=
                    after.weights[layer][neuron][input_idx]) {
                    changed++;
                }
            }
        }
    }

    return changed;
}

/**
 * @brief Cuenta bias que cambiaron respecto a una copia inicial.
 *
 * @param before Modelo antes del entrenamiento.
 * @param after Modelo después del entrenamiento.
 * @return Número de bias modificados en neuronas activas.
 */
uint32_t ff_count_changed_biases(
    const ff_model_t &before,
    const ff_model_t &after
) {
    uint32_t changed = 0;

count_bias_layer_loop:
    for (int layer = 0; layer < FF_HIDDEN_LAYERS; layer++) {
        int neurons = ff_layer_neurons(layer);

count_bias_neuron_loop:
        for (int neuron = 0; neuron < FF_MAX_LAYER_NEURONS; neuron++) {
/* Pipelinear el conteo mantiene bajo el coste de revisar todas las neuronas. */
#pragma HLS PIPELINE II=1
            if (neuron < neurons &&
                before.biases[layer][neuron] != after.biases[layer][neuron]) {
                changed++;
            }
        }
    }

    return changed;
}

/**
 * @brief Serializa el modelo estático hacia buffers planos.
 *
 * @param model Modelo interno.
 * @param weights_out Buffer plano de pesos latentes.
 * @param biases_out Buffer plano de bias.
 *
 * @sideeffect Escribe buffers de salida que pueden residir en memoria externa.
 * Esta forma permite que el host guarde el modelo o lo entregue al kernel de
 * inferencia.
 */
void ff_store_model_flat(
    const ff_model_t &model,
    ff_latent_t *weights_out,
    ff_bias_t *biases_out
) {
store_layer_loop:
    for (int layer = 0; layer < FF_HIDDEN_LAYERS; layer++) {
store_neuron_loop:
        for (int neuron = 0; neuron < FF_MAX_LAYER_NEURONS; neuron++) {
store_input_loop:
            for (int input_idx = 0; input_idx < FF_MAX_LAYER_INPUT_DIM; input_idx++) {
/* La exportación se organiza como streaming lineal para facilitar ráfagas de
 * memoria externa cuando el kernel se conecte a AXI.
 */
#pragma HLS PIPELINE II=1
                int flat = layer * FF_MODEL_WEIGHT_STRIDE +
                    neuron * FF_MAX_LAYER_INPUT_DIM + input_idx;
                weights_out[flat] = model.weights[layer][neuron][input_idx];
            }
            biases_out[layer * FF_MAX_LAYER_NEURONS + neuron] =
                model.biases[layer][neuron];
        }
    }
}

/**
 * @brief Reconstruye el modelo estático desde buffers planos.
 *
 * @param model Modelo interno escrito.
 * @param weights_in Buffer plano de pesos latentes.
 * @param biases_in Buffer plano de bias.
 *
 * @sideeffect Sobrescribe el modelo local antes de inferencia o continuación de
 * entrenamiento.
 */
void ff_load_model_flat(
    ff_model_t &model,
    const ff_latent_t *weights_in,
    const ff_bias_t *biases_in
) {
load_layer_loop:
    for (int layer = 0; layer < FF_HIDDEN_LAYERS; layer++) {
load_neuron_loop:
        for (int neuron = 0; neuron < FF_MAX_LAYER_NEURONS; neuron++) {
load_input_loop:
            for (int input_idx = 0; input_idx < FF_MAX_LAYER_INPUT_DIM; input_idx++) {
/* La carga secuencial desde un buffer plano simplifica el contrato AXI y evita
 * estructuras anidadas en la interfaz del kernel.
 */
#pragma HLS PIPELINE II=1
                int flat = layer * FF_MODEL_WEIGHT_STRIDE +
                    neuron * FF_MAX_LAYER_INPUT_DIM + input_idx;
                model.weights[layer][neuron][input_idx] = weights_in[flat];
            }
            model.biases[layer][neuron] =
                biases_in[layer * FF_MAX_LAYER_NEURONS + neuron];
        }
    }
}
