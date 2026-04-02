#include "forward_fw.hpp"  // Se incluye el header principal del kernel FF.

#include <hls_math.h>      // Se incluye la libreria matematica sintetizable de HLS.
#include <cmath>           // Se incluye cmath para permitir pruebas host usando std::exp y std::log.

// ============================================================
// Utilidades locales de saturacion y conversion
// ============================================================
static latent_t clip_latent(latent_t value) {
    if (value > LATENT_WEIGHT_CLIP) {
        return LATENT_WEIGHT_CLIP;
        // Se satura cualquier peso por encima del clip positivo.
    }

    if (value < -LATENT_WEIGHT_CLIP) {
        return -LATENT_WEIGHT_CLIP;
        // Se satura cualquier peso por debajo del clip negativo.
    }

    return value;
    // Se retorna el peso original cuando ya esta dentro del rango permitido.
}

static bias_t clip_bias(bias_t value) {
    if (value > LATENT_BIAS_CLIP) {
        return LATENT_BIAS_CLIP;
        // Se satura cualquier bias por encima del clip positivo.
    }

    if (value < -LATENT_BIAS_CLIP) {
        return -LATENT_BIAS_CLIP;
        // Se satura cualquier bias por debajo del clip negativo.
    }

    return value;
    // Se retorna el bias original cuando ya esta dentro del rango permitido.
}

static activation_t relu_local(preact_t value) {
    if (value > 0) {
        return (activation_t)value;
        // Se conserva el preactivado cuando la salida ReLU es positiva.
    }

    return (activation_t)0;
    // Se anula la salida cuando el preactivado es negativo o cero.
}

static goodness_t clip_nonlinear_margin(goodness_t value) {
    if (value > NONLINEAR_CLIP_HW) {
        return NONLINEAR_CLIP_HW;
        // Se limita el margen positivo para abaratar las no linealidades.
    }

    if (value < -NONLINEAR_CLIP_HW) {
        return -NONLINEAR_CLIP_HW;
        // Se limita el margen negativo para abaratar las no linealidades.
    }

    return value;
    // Se retorna el margen sin cambios cuando ya esta en el rango util.
}

static feature_t get_feature_value(ff_input_t ff_input, int input_idx) {
    if (ff_input[input_idx] == 0) {
        return (feature_t)0;
        // Se retorna cero cuando el bit de entrada esta inactivo.
    }

    if (input_idx < LABEL_BITS) {
        return LABEL_SCALE_HW;
        // Se aplica label_scale al segmento de etiqueta incrustada.
    }

    return PIXEL_SCALE_HW;
    // Se aplica escala unitaria al segmento de pixeles binarios.
}

static float exp_host_or_hls(float value) {
#ifdef __SYNTHESIS__
    return hls::expf(value);
    // En sintesis se usa la primitiva matematica sintetizable provista por HLS.
#else
    return std::exp(value);
    // En ejecucion host se usa std::exp para permitir barridos rapidos fuera de vitis-run.
#endif
}

static float log_host_or_hls(float value) {
#ifdef __SYNTHESIS__
    return hls::logf(value);
    // En sintesis se usa la primitiva matematica sintetizable provista por HLS.
#else
    return std::log(value);
    // En ejecucion host se usa std::log para permitir barridos rapidos fuera de vitis-run.
#endif
}

static scale_t sigmoid_hw(goodness_t margin) {
    goodness_t clipped_margin = clip_nonlinear_margin(margin);
    // Se acota el margen antes de entrar a la no linealidad suave.

    float x = clipped_margin.to_float();
    // Se convierte el margen a float para usar la libreria sintetizable de HLS.

    float exp_value = exp_host_or_hls(-x);
    // Se calcula la exponencial negativa requerida por la sigmoide.

    float sigmoid_value = 1.0f / (1.0f + exp_value);
    // Se evalua la sigmoide suave usada por el notebook para ponderar la actualizacion.

    return (scale_t)sigmoid_value;
    // Se retorna la escala de actualizacion en punto fijo.
}

static loss_t softplus_hw(goodness_t margin) {
    goodness_t clipped_margin = clip_nonlinear_margin(margin);
    // Se acota el margen antes de evaluar la perdida FF suave.

    float x = clipped_margin.to_float();
    // Se convierte el margen a float para reutilizar operadores sintetizables de HLS.

    float softplus_value = 0.0f;
    // Se reserva la variable local de la softplus estable.

    if (x > 0.0f) {
        softplus_value = x + log_host_or_hls(1.0f + exp_host_or_hls(-x));
        // Se usa la forma estable para margenes positivos.
    } else {
        softplus_value = log_host_or_hls(1.0f + exp_host_or_hls(x));
        // Se usa la forma estable para margenes negativos o pequenos.
    }

    return (loss_t)softplus_value;
    // Se retorna la perdida FF en punto fijo para trazas del testbench.
}

static learning_rate_t compute_batch_learning_rate(int batch_size) {
    if (batch_size <= 0) {
        return (learning_rate_t)0;
        // Se evita cualquier division invalida cuando el batch esta vacio.
    }

    float lr = LEARNING_RATE_HW.to_float();
    // Se extrae la tasa de aprendizaje base en formato flotante.

    float normalized_lr = lr / (float)batch_size;
    // Se normaliza la tasa de aprendizaje segun el tamano real del batch.

    return (learning_rate_t)normalized_lr;
    // Se retorna la tasa de aprendizaje efectiva del batch.
}

static loss_t compute_mean_loss(stat_accum_t sum_value, int count) {
    if (count <= 0) {
        return (loss_t)0;
        // Se retorna cero cuando no hubo muestras validas en la epoca.
    }

    float mean_value = sum_value.to_float() / (float)count;
    // Se calcula la media muestral para dejar el historial comparable entre corridas.

    return (loss_t)mean_value;
    // Se retorna la media en formato fijo compacto.
}

static goodness_t compute_mean_goodness(stat_accum_t sum_value, int count) {
    if (count <= 0) {
        return (goodness_t)0;
        // Se retorna cero cuando no hubo muestras validas en la epoca.
    }

    float mean_value = sum_value.to_float() / (float)count;
    // Se calcula la media muestral de goodness sobre la epoca completa.

    return (goodness_t)mean_value;
    // Se retorna la media en formato fijo de la traza hardware.
}

// ============================================================
// LFSR de 16 bits
// ============================================================
lfsr_t lfsr16_next(lfsr_t state) {
    bool new_bit = state[0] ^ state[2] ^ state[3] ^ state[5];
    // Se calcula el nuevo bit de realimentacion con taps de bajo costo.

    lfsr_t next_state = (state >> 1);
    // Se desplaza el estado una posicion a la derecha.

    next_state[15] = new_bit;
    // Se inserta el nuevo bit en la posicion mas significativa.

    return next_state;
    // Se retorna el siguiente estado del generador pseudoaleatorio.
}

// ============================================================
// Validacion de vectors one-hot
// ============================================================
bool is_valid_onehot(label_oh_t label_onehot) {
    ap_uint<4> count_ones = 0;
    // Se crea un contador pequeno porque solo hay 10 posiciones posibles.

onehot_count_loop:
    for (int i = 0; i < NUM_CLASSES; i++) {
#pragma HLS UNROLL
        // Se desenrolla completamente porque el numero de clases es fijo y pequeno.

        if (label_onehot[i] == 1) {
            count_ones++;
            // Se incrementa el contador cuando se detecta un bit activo.
        }
    }

    return (count_ones == 1);
    // Se retorna true solo cuando existe exactamente un uno.
}

// ============================================================
// Conversion one-hot -> indice
// ============================================================
label_idx_t decode_onehot(label_oh_t label_onehot) {
    label_idx_t label_idx = 0;
    // Se inicializa el indice en cero como valor por defecto.

onehot_decode_loop:
    for (int i = 0; i < NUM_CLASSES; i++) {
#pragma HLS UNROLL
        // Se desenrolla completamente porque el recorrido tiene longitud fija.

        if (label_onehot[i] == 1) {
            label_idx = (label_idx_t)i;
            // Se captura el indice de la unica posicion activa.
        }
    }

    return label_idx;
    // Se retorna el indice correspondiente al vector one-hot.
}

// ============================================================
// Conversion indice -> one-hot
// ============================================================
label_oh_t encode_onehot(label_idx_t label_idx) {
    label_oh_t label_onehot = 0;
    // Se limpia completamente el vector destino.

    label_onehot[label_idx] = 1;
    // Se activa unicamente la clase solicitada.

    return label_onehot;
    // Se retorna el vector one-hot ya construido.
}

// ============================================================
// Generacion de etiqueta negativa excluyente
// ============================================================
label_idx_t generate_negative_label(label_idx_t true_label, lfsr_t &state) {
    state = lfsr16_next(state);
    // Se avanza el LFSR antes de generar la nueva etiqueta.

    ap_uint<4> offset = (state.range(3, 0) % 9) + 1;
    // Se produce un desplazamiento entre 1 y 9 para excluir la clase verdadera.

    ap_uint<5> temp = true_label + offset;
    // Se suma el desplazamiento en un contenedor que admite acarreo.

    label_idx_t negative_label = (temp >= 10) ? (label_idx_t)(temp - 10) : (label_idx_t)temp;
    // Se aplica un modulo 10 manual para regresar al rango valido de clases.

    return negative_label;
    // Se retorna una clase incorrecta y distinta de la verdadera.
}

// ============================================================
// Carga y almacenamiento de muestras empaquetadas
// ============================================================
raw_sample_t load_sample_from_words(const word_t *mem, int sample_idx) {
    raw_sample_t sample = 0;
    // Se inicializa el contenedor fisico de 800 bits.

    int base = sample_idx * WORDS_PER_SAMPLE;
    // Se calcula la posicion base de la muestra dentro del buffer lineal.

load_words_loop:
    for (int w = 0; w < WORDS_PER_SAMPLE; w++) {
#pragma HLS UNROLL
        // Se desenrolla porque siempre se cargan exactamente 25 palabras.

        sample.range((w + 1) * WORD_BITS - 1, w * WORD_BITS) = mem[base + w];
        // Se copia cada palabra de 32 bits al rango correcto del vector ancho.
    }

    return sample;
    // Se retorna la muestra reconstruida como un unico vector de 800 bits.
}

void store_sample_to_words(word_t *mem, int sample_idx, raw_sample_t sample) {
    int base = sample_idx * WORDS_PER_SAMPLE;
    // Se calcula la posicion base de escritura dentro del buffer lineal.

store_words_loop:
    for (int w = 0; w < WORDS_PER_SAMPLE; w++) {
#pragma HLS UNROLL
        // Se desenrolla porque siempre se almacenan exactamente 25 palabras.

        mem[base + w] = sample.range((w + 1) * WORD_BITS - 1, w * WORD_BITS);
        // Se extrae cada palabra de 32 bits del vector ancho y se almacena externamente.
    }
}

// ============================================================
// Empaquetado y desempaquetado fisico
// ============================================================
void unpack_sample(raw_sample_t sample, label_oh_t &label_onehot, pixels_t &pixels, padding_t &padding) {
    label_onehot = sample.range(LABEL_BITS - 1, 0);
    // Se recupera el segmento de etiqueta one-hot.

    pixels = sample.range(LABEL_BITS + PIXEL_BITS - 1, LABEL_BITS);
    // Se recupera el segmento de pixeles binarios.

    padding = sample.range(TOTAL_BITS - 1, LABEL_BITS + PIXEL_BITS);
    // Se recupera el segmento de padding fisico.
}

raw_sample_t pack_sample(label_oh_t label_onehot, pixels_t pixels, padding_t padding) {
    raw_sample_t sample = 0;
    // Se limpia completamente la muestra fisica antes de insertar campos.

    sample.range(LABEL_BITS - 1, 0) = label_onehot;
    // Se inserta la etiqueta one-hot en el segmento bajo.

    sample.range(LABEL_BITS + PIXEL_BITS - 1, LABEL_BITS) = pixels;
    // Se inserta la imagen binaria en el segmento central.

    sample.range(TOTAL_BITS - 1, LABEL_BITS + PIXEL_BITS) = padding;
    // Se inserta el padding fisico en el segmento alto.

    return sample;
    // Se retorna la muestra fisica ya empaquetada.
}

// ============================================================
// Construccion de la entrada logica FF
// ============================================================
ff_input_t build_ff_input(label_oh_t label_onehot, pixels_t pixels) {
    ff_input_t ff_input = 0;
    // Se limpia el vector logico de entrada FF.

    ff_input.range(LABEL_BITS - 1, 0) = label_onehot;
    // Se coloca la etiqueta incrustada en la zona baja.

    ff_input.range(MODEL_INPUT_BITS - 1, LABEL_BITS) = pixels;
    // Se coloca la imagen binaria a continuacion de la etiqueta.

    return ff_input;
    // Se retorna la entrada logica final de 794 bits.
}

void unpack_ff_input(ff_input_t ff_input, label_oh_t &label_onehot, pixels_t &pixels) {
    label_onehot = ff_input.range(LABEL_BITS - 1, 0);
    // Se recupera la etiqueta incrustada.

    pixels = ff_input.range(MODEL_INPUT_BITS - 1, LABEL_BITS);
    // Se recupera la imagen binaria.
}

// ============================================================
// Inicializacion del modelo latente por capa
// ============================================================
template <int INPUT_DIM, int OUTPUT_DIM>
static void initialize_dense_layer(
    latent_t weights[OUTPUT_DIM][INPUT_DIM],
    bias_t biases[OUTPUT_DIM],
    lfsr_t &state
) {
#pragma HLS inline off
    // Se evita inline para no replicar el bloque de inicializacion de una capa completa.

init_dense_neuron_loop:
    for (int neuron = 0; neuron < OUTPUT_DIM; neuron++) {
        // Se recorre cada neurona latente de la capa actual.

        biases[neuron] = (bias_t)0;
        // Se inicializa cada bias en cero para arrancar desde una base simetrica.

init_dense_weight_loop:
        for (int input_idx = 0; input_idx < INPUT_DIM; input_idx++) {
#pragma HLS PIPELINE II=1
            // Se pipelinea la inicializacion de pesos para no inflar la latencia de arranque.

            state = lfsr16_next(state);
            // Se avanza el generador pseudoaleatorio antes de fijar el nuevo peso.

            latent_t magnitude = state[1] ? LATENT_INIT_MAG : (latent_t)(LATENT_INIT_MAG * (latent_t)2);
            // Se alterna entre dos magnitudes pequenas para romper simetrias de inicializacion.

            weights[neuron][input_idx] = state[0] ? magnitude : (latent_t)(-magnitude);
            // Se asigna un signo pseudoaleatorio y una magnitud pequena al peso latente.
        }
    }
}

// ============================================================
// Forward local de la primera capa desde la entrada FF
// ============================================================
static void ff_layer1_forward(
    ff_input_t ff_input,
    latent_t weights[MODEL_LAYER1_NEURONS][MODEL_LAYER1_INPUT_BITS],
    bias_t biases[MODEL_LAYER1_NEURONS],
    activation_t activations[MODEL_LAYER1_NEURONS],
    goodness_t &goodness
) {
#pragma HLS inline off
    // Se evita inline para conservar una sola instancia clara del forward de la capa 1.

    stat_accum_t sum_sq = 0;
    // Se inicializa el acumulador de suma de cuadrados para la goodness local.

layer1_tile_forward_loop:
    for (int tile = 0; tile < MODEL_LAYER1_TILES; tile++) {
        // Se recorre cada tile de la primera capa respetando el paralelismo de 8 neuronas.

        preact_t z_lane[MODEL_PARALLEL_NEURONS];
        // Se reservan los acumuladores locales del preactivado por lane.

#pragma HLS ARRAY_PARTITION variable=z_lane complete
        // Se particionan completamente los acumuladores locales para operar en paralelo.

layer1_lane_init_loop:
        for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
            // Se inicializan en paralelo los 8 lanes del tile actual.

            int neuron = tile * MODEL_PARALLEL_NEURONS + lane;
            // Se calcula el indice absoluto de la neurona del lane actual.

            z_lane[lane] = (preact_t)biases[neuron];
            // Se arranca el preactivado desde el bias latente de esa neurona.
        }

layer1_input_forward_loop:
        for (int input_idx = 0; input_idx < MODEL_LAYER1_INPUT_BITS; input_idx++) {
#pragma HLS PIPELINE II=1
            // Se pipelinea el recorrido sobre las 794 entradas de la primera capa.

            feature_t feature_value = get_feature_value(ff_input, input_idx);
            // Se reconstruye el valor numerico de la caracteristica actual.

            if (feature_value != 0) {
                // Solo las entradas activas aportan al preactivado de la capa.

layer1_lane_acc_loop:
                for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
                    // Se actualizan en paralelo las 8 neuronas activas del tile.

                    int neuron = tile * MODEL_PARALLEL_NEURONS + lane;
                    // Se calcula el indice absoluto de la neurona del lane actual.

                    if (weights[neuron][input_idx] >= 0) {
                        z_lane[lane] = z_lane[lane] + (preact_t)feature_value;
                        // Se suma la caracteristica cuando el peso binarizado vale +1.
                    } else {
                        z_lane[lane] = z_lane[lane] - (preact_t)feature_value;
                        // Se resta la caracteristica cuando el peso binarizado vale -1.
                    }
                }
            }
        }

layer1_lane_output_loop:
        for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
            // Se materializan en paralelo las activaciones y la suma de cuadrados del tile.

            int neuron = tile * MODEL_PARALLEL_NEURONS + lane;
            // Se calcula el indice absoluto de la neurona del lane actual.

            activation_t activation_value = relu_local(z_lane[lane]);
            // Se aplica la activacion ReLU definida por el notebook.

            activations[neuron] = activation_value;
            // Se almacena la activacion para la segunda capa y para la actualizacion local.

            sum_sq = sum_sq + (stat_accum_t)(activation_value * activation_value);
            // Se acumula el cuadrado de la activacion para la goodness de esta capa.
        }
    }

    float mean_sq = sum_sq.to_float() / (float)MODEL_LAYER1_NEURONS;
    // Se convierte la suma a media cuadratica por neurona de la primera capa.

    goodness = (goodness_t)mean_sq;
    // Se retorna la goodness local de la primera capa.
}

// ============================================================
// Forward local de una capa densa activacion -> activacion
// ============================================================
template <int INPUT_DIM, int OUTPUT_DIM, int TILE_COUNT>
static void ff_layer_forward_dense(
    const activation_t layer_input[INPUT_DIM],
    latent_t weights[OUTPUT_DIM][INPUT_DIM],
    bias_t biases[OUTPUT_DIM],
    activation_t activations[OUTPUT_DIM],
    goodness_t &goodness
) {
#pragma HLS inline off
    // Se evita inline para mantener una unica instancia clara del forward denso.

    stat_accum_t sum_sq = 0;
    // Se inicializa el acumulador de suma de cuadrados de la capa actual.

dense_tile_forward_loop:
    for (int tile = 0; tile < TILE_COUNT; tile++) {
        // Se recorre cada tile de la capa densa respetando el paralelismo fijo.

        preact_t z_lane[MODEL_PARALLEL_NEURONS];
        // Se reservan los acumuladores locales del preactivado por lane.

#pragma HLS ARRAY_PARTITION variable=z_lane complete
        // Se particionan completamente los acumuladores del tile para operar en paralelo.

dense_lane_init_loop:
        for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
            // Se inicializan en paralelo los 8 lanes del tile actual.

            int neuron = tile * MODEL_PARALLEL_NEURONS + lane;
            // Se calcula el indice absoluto de la neurona del lane actual.

            z_lane[lane] = (preact_t)biases[neuron];
            // Se inicializa el preactivado con el bias latente correspondiente.
        }

dense_input_forward_loop:
        for (int input_idx = 0; input_idx < INPUT_DIM; input_idx++) {
#pragma HLS PIPELINE II=1
            // Se pipelinea el recorrido sobre la entrada densa de la capa actual.

            activation_t input_value = layer_input[input_idx];
            // Se recupera la activacion de entrada que alimenta esta capa.

            if (input_value != 0) {
                // Solo las entradas activas aportan al preactivado local.

dense_lane_acc_loop:
                for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
                    // Se actualizan en paralelo las 8 neuronas del tile actual.

                    int neuron = tile * MODEL_PARALLEL_NEURONS + lane;
                    // Se calcula el indice absoluto de la neurona del lane actual.

                    if (weights[neuron][input_idx] >= 0) {
                        z_lane[lane] = z_lane[lane] + (preact_t)input_value;
                        // Se suma la entrada cuando el peso binarizado vale +1.
                    } else {
                        z_lane[lane] = z_lane[lane] - (preact_t)input_value;
                        // Se resta la entrada cuando el peso binarizado vale -1.
                    }
                }
            }
        }

dense_lane_output_loop:
        for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
            // Se materializan en paralelo las activaciones y la suma de cuadrados del tile.

            int neuron = tile * MODEL_PARALLEL_NEURONS + lane;
            // Se calcula el indice absoluto de la neurona del lane actual.

            activation_t activation_value = relu_local(z_lane[lane]);
            // Se aplica la activacion ReLU definida por el notebook.

            activations[neuron] = activation_value;
            // Se almacena la activacion para la siguiente etapa de entrenamiento o inferencia.

            sum_sq = sum_sq + (stat_accum_t)(activation_value * activation_value);
            // Se acumula el cuadrado de la activacion para la goodness local.
        }
    }

    float mean_sq = sum_sq.to_float() / (float)OUTPUT_DIM;
    // Se convierte la suma a media cuadratica por neurona de la capa actual.

    goodness = (goodness_t)mean_sq;
    // Se retorna la goodness local de la capa actual.
}

// ============================================================
// Acumulacion local de la actualizacion de la primera capa
// ============================================================
static void accumulate_layer1_tile_update(
    ff_input_t pos_input,
    ff_input_t neg_input,
    const activation_t h_pos[MODEL_LAYER1_NEURONS],
    const activation_t h_neg[MODEL_LAYER1_NEURONS],
    scale_t pos_scale,
    scale_t neg_scale,
    update_accum_t weight_delta[MODEL_PARALLEL_NEURONS][MODEL_LAYER1_INPUT_BITS],
    update_accum_t bias_delta[MODEL_PARALLEL_NEURONS],
    int tile
) {
#pragma HLS inline off
    // Se evita inline para mantener acotado el bloque de actualizacion de la capa 1.

    update_accum_t pos_scaled_activation[MODEL_PARALLEL_NEURONS];
    // Se reservan las activaciones positivas ya ponderadas por sigmoide para el tile actual.

    update_accum_t neg_scaled_activation[MODEL_PARALLEL_NEURONS];
    // Se reservan las activaciones negativas ya ponderadas por sigmoide para el tile actual.

#pragma HLS ARRAY_PARTITION variable=pos_scaled_activation complete
    // Se particionan completamente las activaciones positivas ponderadas del tile.

#pragma HLS ARRAY_PARTITION variable=neg_scaled_activation complete
    // Se particionan completamente las activaciones negativas ponderadas del tile.

layer1_scaled_activation_loop:
    for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
        // Se calculan en paralelo las activaciones ponderadas de las 8 neuronas del tile.

        int neuron = tile * MODEL_PARALLEL_NEURONS + lane;
        // Se calcula el indice absoluto de la neurona del lane actual.

        pos_scaled_activation[lane] =
            (update_accum_t)h_pos[neuron] * (update_accum_t)pos_scale;
        // Se pondera la activacion positiva con la sigmoide del margen positivo.

        neg_scaled_activation[lane] =
            (update_accum_t)h_neg[neuron] * (update_accum_t)neg_scale;
        // Se pondera la activacion negativa con la sigmoide del margen negativo.

        bias_delta[lane] = bias_delta[lane] + pos_scaled_activation[lane] - neg_scaled_activation[lane];
        // Se acumula la actualizacion de bias como evidencia positiva menos negativa.
    }

layer1_accumulate_input_loop:
    for (int input_idx = 0; input_idx < MODEL_LAYER1_INPUT_BITS; input_idx++) {
#pragma HLS PIPELINE II=1
        // Se recorre cada entrada de la primera capa con pipeline sintetizable.

        feature_t pos_value = get_feature_value(pos_input, input_idx);
        // Se reconstruye el valor numerico de la entrada positiva actual.

        feature_t neg_value = get_feature_value(neg_input, input_idx);
        // Se reconstruye el valor numerico de la entrada negativa actual.

layer1_accumulate_lane_loop:
        for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
            // Se acumula en paralelo el delta de las 8 neuronas del tile.

            weight_delta[lane][input_idx] =
                weight_delta[lane][input_idx]
                + (update_accum_t)pos_value * pos_scaled_activation[lane]
                - (update_accum_t)neg_value * neg_scaled_activation[lane];
            // Se aplica la correlacion ponderada del notebook usando x_pos y x_neg por separado.
        }
    }
}

// ============================================================
// Acumulacion local de la actualizacion de una capa densa
// ============================================================
template <int INPUT_DIM, int OUTPUT_DIM>
static void accumulate_dense_tile_update(
    const activation_t pos_input[INPUT_DIM],
    const activation_t neg_input[INPUT_DIM],
    const activation_t h_pos[OUTPUT_DIM],
    const activation_t h_neg[OUTPUT_DIM],
    scale_t pos_scale,
    scale_t neg_scale,
    update_accum_t weight_delta[MODEL_PARALLEL_NEURONS][INPUT_DIM],
    update_accum_t bias_delta[MODEL_PARALLEL_NEURONS],
    int tile
) {
#pragma HLS inline off
    // Se evita inline para mantener una sola rutina de actualizacion densa.

    update_accum_t pos_scaled_activation[MODEL_PARALLEL_NEURONS];
    // Se reservan las activaciones positivas ponderadas del tile actual.

    update_accum_t neg_scaled_activation[MODEL_PARALLEL_NEURONS];
    // Se reservan las activaciones negativas ponderadas del tile actual.

#pragma HLS ARRAY_PARTITION variable=pos_scaled_activation complete
    // Se particionan completamente las activaciones positivas ponderadas.

#pragma HLS ARRAY_PARTITION variable=neg_scaled_activation complete
    // Se particionan completamente las activaciones negativas ponderadas.

dense_scaled_activation_loop:
    for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
        // Se calculan en paralelo las activaciones ponderadas de las 8 neuronas del tile.

        int neuron = tile * MODEL_PARALLEL_NEURONS + lane;
        // Se calcula el indice absoluto de la neurona del lane actual.

        pos_scaled_activation[lane] =
            (update_accum_t)h_pos[neuron] * (update_accum_t)pos_scale;
        // Se pondera la activacion positiva con la sigmoide del margen positivo.

        neg_scaled_activation[lane] =
            (update_accum_t)h_neg[neuron] * (update_accum_t)neg_scale;
        // Se pondera la activacion negativa con la sigmoide del margen negativo.

        bias_delta[lane] = bias_delta[lane] + pos_scaled_activation[lane] - neg_scaled_activation[lane];
        // Se acumula la actualizacion de bias como evidencia positiva menos negativa.
    }

dense_accumulate_input_loop:
    for (int input_idx = 0; input_idx < INPUT_DIM; input_idx++) {
#pragma HLS PIPELINE II=1
        // Se recorre cada entrada de la capa actual con pipeline sintetizable.

        activation_t pos_value = pos_input[input_idx];
        // Se recupera la activacion positiva de entrada del batch actual.

        activation_t neg_value = neg_input[input_idx];
        // Se recupera la activacion negativa de entrada del batch actual.

dense_accumulate_lane_loop:
        for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
            // Se acumula en paralelo el delta de las 8 neuronas del tile actual.

            weight_delta[lane][input_idx] =
                weight_delta[lane][input_idx]
                + (update_accum_t)pos_value * pos_scaled_activation[lane]
                - (update_accum_t)neg_value * neg_scaled_activation[lane];
            // Se aplica la correlacion ponderada del notebook sobre la capa densa actual.
        }
    }
}

// ============================================================
// Aplicacion del update acumulado a una capa densa
// ============================================================
template <int INPUT_DIM, int OUTPUT_DIM>
static void apply_dense_tile_update(
    latent_t weights[OUTPUT_DIM][INPUT_DIM],
    bias_t biases[OUTPUT_DIM],
    update_accum_t weight_delta[MODEL_PARALLEL_NEURONS][INPUT_DIM],
    update_accum_t bias_delta[MODEL_PARALLEL_NEURONS],
    learning_rate_t batch_lr,
    int tile
) {
#pragma HLS inline off
    // Se evita inline para mantener acotada la logica de escritura sobre el modelo latente.

apply_dense_weight_loop:
    for (int input_idx = 0; input_idx < INPUT_DIM; input_idx++) {
#pragma HLS PIPELINE II=1
        // Se aplica el batch update a cada conexion del tile con pipeline.

apply_dense_lane_loop:
        for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
            // Se actualizan en paralelo las 8 neuronas del tile actual.

            int neuron = tile * MODEL_PARALLEL_NEURONS + lane;
            // Se calcula el indice absoluto de la neurona del lane actual.

            latent_t delta_weight = (latent_t)((update_accum_t)batch_lr * weight_delta[lane][input_idx]);
            // Se normaliza el delta acumulado del batch y se convierte al tipo del peso latente.

            weights[neuron][input_idx] = clip_latent(weights[neuron][input_idx] + delta_weight);
            // Se aplica la actualizacion local y luego se hace weight clipping.
        }
    }

apply_dense_bias_loop:
    for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
        // Se actualizan en paralelo los bias de las 8 neuronas del tile.

        int neuron = tile * MODEL_PARALLEL_NEURONS + lane;
        // Se calcula el indice absoluto de la neurona del lane actual.

        bias_t delta_bias = (bias_t)((update_accum_t)batch_lr * bias_delta[lane]);
        // Se normaliza el delta acumulado del bias segun el tamano real del batch.

        biases[neuron] = clip_bias(biases[neuron] + delta_bias);
        // Se aplica la actualizacion local y luego se satura el bias.
    }
}

// ============================================================
// Entrenamiento local por mini-batch
// ============================================================
static void train_batch_local(
    const word_t *in_mem,
    int batch_start,
    int batch_size,
    latent_t weights_l1[MODEL_LAYER1_NEURONS][MODEL_LAYER1_INPUT_BITS],
    bias_t biases_l1[MODEL_LAYER1_NEURONS],
    latent_t weights_l2[MODEL_LAYER2_NEURONS][MODEL_LAYER2_INPUT_BITS],
    bias_t biases_l2[MODEL_LAYER2_NEURONS],
    lfsr_t &train_state,
    bool capture_sample_metrics,
    goodness_t *g_pos_mem,
    goodness_t *g_neg_mem,
    goodness_t *gap_mem,
    stat_accum_t &epoch_loss_pos_sum,
    stat_accum_t &epoch_loss_neg_sum,
    stat_accum_t &epoch_g_pos_sum,
    stat_accum_t &epoch_g_neg_sum
) {
#pragma HLS inline off
    // Se evita inline para no replicar toda la logica de batch dentro del top.

    ff_input_t batch_pos_inputs[MODEL_BATCH_SIZE];
    // Se reserva el buffer del batch de ejemplos positivos ya incrustados.

    ff_input_t batch_neg_inputs[MODEL_BATCH_SIZE];
    // Se reserva el buffer del batch de ejemplos negativos ya incrustados.

    activation_t batch_h1_pos[MODEL_BATCH_SIZE][MODEL_LAYER1_NEURONS];
    // Se reserva el buffer de activaciones positivas de la primera capa.

    activation_t batch_h1_neg[MODEL_BATCH_SIZE][MODEL_LAYER1_NEURONS];
    // Se reserva el buffer de activaciones negativas de la primera capa.

    activation_t batch_h2_pos[MODEL_BATCH_SIZE][MODEL_LAYER2_NEURONS];
    // Se reserva el buffer de activaciones positivas de la segunda capa.

    activation_t batch_h2_neg[MODEL_BATCH_SIZE][MODEL_LAYER2_NEURONS];
    // Se reserva el buffer de activaciones negativas de la segunda capa.

    goodness_t batch_g1_pos[MODEL_BATCH_SIZE];
    // Se reserva la goodness positiva de la primera capa por muestra.

    goodness_t batch_g1_neg[MODEL_BATCH_SIZE];
    // Se reserva la goodness negativa de la primera capa por muestra.

    goodness_t batch_g2_pos[MODEL_BATCH_SIZE];
    // Se reserva la goodness positiva de la segunda capa por muestra.

    goodness_t batch_g2_neg[MODEL_BATCH_SIZE];
    // Se reserva la goodness negativa de la segunda capa por muestra.

    scale_t batch_pos_scale_l1[MODEL_BATCH_SIZE];
    // Se reserva la escala sigmoidal positiva de la primera capa.

    scale_t batch_neg_scale_l1[MODEL_BATCH_SIZE];
    // Se reserva la escala sigmoidal negativa de la primera capa.

    scale_t batch_pos_scale_l2[MODEL_BATCH_SIZE];
    // Se reserva la escala sigmoidal positiva de la segunda capa.

    scale_t batch_neg_scale_l2[MODEL_BATCH_SIZE];
    // Se reserva la escala sigmoidal negativa de la segunda capa.

#pragma HLS ARRAY_PARTITION variable=batch_h1_pos cyclic factor=MODEL_PARALLEL_NEURONS dim=2
    // Se particiona la dimension de neuronas de la capa 1 para reutilizar 8 lanes.

#pragma HLS ARRAY_PARTITION variable=batch_h1_neg cyclic factor=MODEL_PARALLEL_NEURONS dim=2
    // Se particiona la dimension de neuronas negativas de la capa 1 con la misma estrategia.

#pragma HLS ARRAY_PARTITION variable=batch_h2_pos cyclic factor=MODEL_PARALLEL_NEURONS dim=2
    // Se particiona la dimension de neuronas de la capa 2 para reutilizar 8 lanes.

#pragma HLS ARRAY_PARTITION variable=batch_h2_neg cyclic factor=MODEL_PARALLEL_NEURONS dim=2
    // Se particiona la dimension de neuronas negativas de la capa 2 con la misma estrategia.

#pragma HLS BIND_STORAGE variable=batch_pos_inputs type=ram_2p impl=bram
    // Se fuerza almacenamiento en BRAM para el buffer positivo del batch.

#pragma HLS BIND_STORAGE variable=batch_neg_inputs type=ram_2p impl=bram
    // Se fuerza almacenamiento en BRAM para el buffer negativo del batch.

#pragma HLS BIND_STORAGE variable=batch_h1_pos type=ram_2p impl=bram
    // Se almacena el buffer de activaciones positivas de la capa 1 en BRAM.

#pragma HLS BIND_STORAGE variable=batch_h1_neg type=ram_2p impl=bram
    // Se almacena el buffer de activaciones negativas de la capa 1 en BRAM.

#pragma HLS BIND_STORAGE variable=batch_h2_pos type=ram_2p impl=bram
    // Se almacena el buffer de activaciones positivas de la capa 2 en BRAM.

#pragma HLS BIND_STORAGE variable=batch_h2_neg type=ram_2p impl=bram
    // Se almacena el buffer de activaciones negativas de la capa 2 en BRAM.

prepare_layer1_batch_loop:
    for (int batch_offset = 0; batch_offset < batch_size; batch_offset++) {
        // Se recorre cada muestra del batch para ejecutar la capa 1 y guardar sus activaciones.

        int sample_idx = batch_start + batch_offset;
        // Se calcula el indice absoluto de la muestra dentro del subset de entrenamiento.

        raw_sample_t input_sample = load_sample_from_words(in_mem, sample_idx);
        // Se lee la muestra fisica actual desde memoria externa.

        label_oh_t true_onehot;
        // Se reserva el contenedor de la etiqueta verdadera.

        pixels_t pixels;
        // Se reserva el contenedor de la imagen binaria.

        padding_t padding;
        // Se reserva el contenedor del padding fisico, aunque aqui no se use.

        unpack_sample(input_sample, true_onehot, pixels, padding);
        // Se desempaqueta la muestra para recuperar label y pixeles.

        label_idx_t true_label_idx = decode_onehot(true_onehot);
        // Se decodifica la etiqueta correcta de la muestra.

        label_idx_t neg_label_idx = generate_negative_label(true_label_idx, train_state);
        // Se genera una etiqueta negativa excluyente para esta misma imagen.

        label_oh_t neg_onehot = encode_onehot(neg_label_idx);
        // Se vuelve a codificar la etiqueta negativa en formato one-hot.

        batch_pos_inputs[batch_offset] = build_ff_input(true_onehot, pixels);
        // Se construye la entrada positiva con label correcto y label_scale implicito.

        batch_neg_inputs[batch_offset] = build_ff_input(neg_onehot, pixels);
        // Se construye la entrada negativa con la misma imagen y label incorrecto.

        ff_layer1_forward(
            batch_pos_inputs[batch_offset],
            weights_l1,
            biases_l1,
            batch_h1_pos[batch_offset],
            batch_g1_pos[batch_offset]
        );
        // Se ejecuta el forward positivo de la primera capa.

        ff_layer1_forward(
            batch_neg_inputs[batch_offset],
            weights_l1,
            biases_l1,
            batch_h1_neg[batch_offset],
            batch_g1_neg[batch_offset]
        );
        // Se ejecuta el forward negativo de la primera capa.

        goodness_t pos_margin_l1 = GOODNESS_THRESHOLD_HW - batch_g1_pos[batch_offset];
        // Se calcula el margen positivo usado por sigmoid y softplus en la capa 1.

        goodness_t neg_margin_l1 = batch_g1_neg[batch_offset] - GOODNESS_THRESHOLD_HW;
        // Se calcula el margen negativo usado por sigmoid y softplus en la capa 1.

        batch_pos_scale_l1[batch_offset] = sigmoid_hw(pos_margin_l1);
        // Se calcula la escala suave para reforzar positivos por debajo del threshold en la capa 1.

        batch_neg_scale_l1[batch_offset] = sigmoid_hw(neg_margin_l1);
        // Se calcula la escala suave para suprimir negativos por encima del threshold en la capa 1.

        loss_t loss_pos_l1 = softplus_hw(pos_margin_l1);
        // Se calcula la perdida FF suave del ejemplo positivo de la capa 1.

        loss_t loss_neg_l1 = softplus_hw(neg_margin_l1);
        // Se calcula la perdida FF suave del ejemplo negativo de la capa 1.

        epoch_loss_pos_sum = epoch_loss_pos_sum + (stat_accum_t)loss_pos_l1;
        // Se acumula la perdida positiva local de la capa 1 sobre la epoca.

        epoch_loss_neg_sum = epoch_loss_neg_sum + (stat_accum_t)loss_neg_l1;
        // Se acumula la perdida negativa local de la capa 1 sobre la epoca.

        epoch_g_pos_sum = epoch_g_pos_sum + (stat_accum_t)batch_g1_pos[batch_offset];
        // Se acumula la goodness positiva de la capa 1 dentro del score total por muestra.

        epoch_g_neg_sum = epoch_g_neg_sum + (stat_accum_t)batch_g1_neg[batch_offset];
        // Se acumula la goodness negativa de la capa 1 dentro del score total por muestra.
    }

    learning_rate_t batch_lr = compute_batch_learning_rate(batch_size);
    // Se calcula la tasa efectiva normalizada por el tamano real del batch.

update_layer1_tile_loop:
    for (int tile = 0; tile < MODEL_LAYER1_TILES; tile++) {
        // Se recorre cada tile de la primera capa para acumular y aplicar la actualizacion del batch.

        update_accum_t weight_delta[MODEL_PARALLEL_NEURONS][MODEL_LAYER1_INPUT_BITS];
        // Se reserva el acumulador local de dW del tile actual de la capa 1.

        update_accum_t bias_delta[MODEL_PARALLEL_NEURONS];
        // Se reserva el acumulador local de db del tile actual de la capa 1.

#pragma HLS ARRAY_PARTITION variable=weight_delta complete dim=1
        // Se particiona completamente la dimension de lanes del delta de pesos.

#pragma HLS ARRAY_PARTITION variable=bias_delta complete
        // Se particiona completamente el delta de bias del tile.

#pragma HLS BIND_STORAGE variable=weight_delta type=ram_2p impl=bram
        // Se almacena el delta de pesos del tile en BRAM para reducir registros.

zero_layer1_lane_loop:
        for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
            // Se inicializan en paralelo los acumuladores de bias de cada lane.

            bias_delta[lane] = (update_accum_t)0;
            // Se limpia el acumulador de bias del lane actual.

zero_layer1_input_loop:
            for (int input_idx = 0; input_idx < MODEL_LAYER1_INPUT_BITS; input_idx++) {
#pragma HLS PIPELINE II=1
                // Se limpia el acumulador de pesos con pipeline para no inflar la latencia.

                weight_delta[lane][input_idx] = (update_accum_t)0;
                // Se limpia el acumulador de la conexion actual del tile.
            }
        }

accumulate_layer1_batch_loop:
        for (int batch_offset = 0; batch_offset < batch_size; batch_offset++) {
            // Se recorre el batch y se acumulan las correlaciones ponderadas de la capa 1.

            accumulate_layer1_tile_update(
                batch_pos_inputs[batch_offset],
                batch_neg_inputs[batch_offset],
                batch_h1_pos[batch_offset],
                batch_h1_neg[batch_offset],
                batch_pos_scale_l1[batch_offset],
                batch_neg_scale_l1[batch_offset],
                weight_delta,
                bias_delta,
                tile
            );
            // Se aplica la regla local del notebook sobre la primera capa.
        }

        apply_dense_tile_update<MODEL_LAYER1_INPUT_BITS, MODEL_LAYER1_NEURONS>(
            weights_l1,
            biases_l1,
            weight_delta,
            bias_delta,
            batch_lr,
            tile
        );
        // Se escribe la actualizacion acumulada sobre la primera capa.
    }

prepare_layer2_batch_loop:
    for (int batch_offset = 0; batch_offset < batch_size; batch_offset++) {
        // Se recorre cada muestra del batch para ejecutar la capa 2 usando las activaciones de la capa 1.

        ff_layer_forward_dense<MODEL_LAYER2_INPUT_BITS, MODEL_LAYER2_NEURONS, MODEL_LAYER2_TILES>(
            batch_h1_pos[batch_offset],
            weights_l2,
            biases_l2,
            batch_h2_pos[batch_offset],
            batch_g2_pos[batch_offset]
        );
        // Se ejecuta el forward positivo de la segunda capa.

        ff_layer_forward_dense<MODEL_LAYER2_INPUT_BITS, MODEL_LAYER2_NEURONS, MODEL_LAYER2_TILES>(
            batch_h1_neg[batch_offset],
            weights_l2,
            biases_l2,
            batch_h2_neg[batch_offset],
            batch_g2_neg[batch_offset]
        );
        // Se ejecuta el forward negativo de la segunda capa.

        goodness_t pos_margin_l2 = GOODNESS_THRESHOLD_HW - batch_g2_pos[batch_offset];
        // Se calcula el margen positivo usado por sigmoid y softplus en la capa 2.

        goodness_t neg_margin_l2 = batch_g2_neg[batch_offset] - GOODNESS_THRESHOLD_HW;
        // Se calcula el margen negativo usado por sigmoid y softplus en la capa 2.

        batch_pos_scale_l2[batch_offset] = sigmoid_hw(pos_margin_l2);
        // Se calcula la escala suave para reforzar positivos por debajo del threshold en la capa 2.

        batch_neg_scale_l2[batch_offset] = sigmoid_hw(neg_margin_l2);
        // Se calcula la escala suave para suprimir negativos por encima del threshold en la capa 2.

        loss_t loss_pos_l2 = softplus_hw(pos_margin_l2);
        // Se calcula la perdida FF suave del ejemplo positivo de la capa 2.

        loss_t loss_neg_l2 = softplus_hw(neg_margin_l2);
        // Se calcula la perdida FF suave del ejemplo negativo de la capa 2.

        epoch_loss_pos_sum = epoch_loss_pos_sum + (stat_accum_t)loss_pos_l2;
        // Se acumula la perdida positiva local de la capa 2 sobre la epoca.

        epoch_loss_neg_sum = epoch_loss_neg_sum + (stat_accum_t)loss_neg_l2;
        // Se acumula la perdida negativa local de la capa 2 sobre la epoca.

        epoch_g_pos_sum = epoch_g_pos_sum + (stat_accum_t)batch_g2_pos[batch_offset];
        // Se acumula la goodness positiva de la capa 2 dentro del score total por muestra.

        epoch_g_neg_sum = epoch_g_neg_sum + (stat_accum_t)batch_g2_neg[batch_offset];
        // Se acumula la goodness negativa de la capa 2 dentro del score total por muestra.

        if (capture_sample_metrics == true) {
            int sample_idx = batch_start + batch_offset;
            // Se calcula el indice absoluto de la muestra cuando se guardan trazas de la ultima epoca.

            goodness_t total_g_pos = batch_g1_pos[batch_offset] + batch_g2_pos[batch_offset];
            // Se forma la goodness total positiva como g1 + g2.

            goodness_t total_g_neg = batch_g1_neg[batch_offset] + batch_g2_neg[batch_offset];
            // Se forma la goodness total negativa como g1 + g2.

            g_pos_mem[sample_idx] = total_g_pos;
            // Se guarda la goodness positiva total de la ultima epoca para trazas del testbench.

            g_neg_mem[sample_idx] = total_g_neg;
            // Se guarda la goodness negativa total de la ultima epoca para trazas del testbench.

            gap_mem[sample_idx] = total_g_pos - total_g_neg;
            // Se guarda el goodness gap total individual de la ultima epoca.
        }
    }

update_layer2_tile_loop:
    for (int tile = 0; tile < MODEL_LAYER2_TILES; tile++) {
        // Se recorre cada tile de la segunda capa para acumular y aplicar la actualizacion del batch.

        update_accum_t weight_delta[MODEL_PARALLEL_NEURONS][MODEL_LAYER2_INPUT_BITS];
        // Se reserva el acumulador local de dW del tile actual de la capa 2.

        update_accum_t bias_delta[MODEL_PARALLEL_NEURONS];
        // Se reserva el acumulador local de db del tile actual de la capa 2.

#pragma HLS ARRAY_PARTITION variable=weight_delta complete dim=1
        // Se particiona completamente la dimension de lanes del delta de pesos.

#pragma HLS ARRAY_PARTITION variable=bias_delta complete
        // Se particiona completamente el delta de bias del tile.

#pragma HLS BIND_STORAGE variable=weight_delta type=ram_2p impl=bram
        // Se almacena el delta de pesos del tile en BRAM para reducir registros.

zero_layer2_lane_loop:
        for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
            // Se inicializan en paralelo los acumuladores de bias de cada lane de la capa 2.

            bias_delta[lane] = (update_accum_t)0;
            // Se limpia el acumulador de bias del lane actual.

zero_layer2_input_loop:
            for (int input_idx = 0; input_idx < MODEL_LAYER2_INPUT_BITS; input_idx++) {
#pragma HLS PIPELINE II=1
                // Se limpia el acumulador de pesos con pipeline para no inflar la latencia.

                weight_delta[lane][input_idx] = (update_accum_t)0;
                // Se limpia el acumulador de la conexion actual del tile.
            }
        }

accumulate_layer2_batch_loop:
        for (int batch_offset = 0; batch_offset < batch_size; batch_offset++) {
            // Se recorre el batch y se acumulan las correlaciones ponderadas de la capa 2.

            accumulate_dense_tile_update<MODEL_LAYER2_INPUT_BITS, MODEL_LAYER2_NEURONS>(
                batch_h1_pos[batch_offset],
                batch_h1_neg[batch_offset],
                batch_h2_pos[batch_offset],
                batch_h2_neg[batch_offset],
                batch_pos_scale_l2[batch_offset],
                batch_neg_scale_l2[batch_offset],
                weight_delta,
                bias_delta,
                tile
            );
            // Se aplica la regla local del notebook sobre la segunda capa.
        }

        apply_dense_tile_update<MODEL_LAYER2_INPUT_BITS, MODEL_LAYER2_NEURONS>(
            weights_l2,
            biases_l2,
            weight_delta,
            bias_delta,
            batch_lr,
            tile
        );
        // Se escribe la actualizacion acumulada sobre la segunda capa.
    }
}

// ============================================================
// Inferencia multiclase por argmax de goodness
// ============================================================
static label_idx_t ff_predict_label(
    pixels_t pixels,
    latent_t weights_l1[MODEL_LAYER1_NEURONS][MODEL_LAYER1_INPUT_BITS],
    bias_t biases_l1[MODEL_LAYER1_NEURONS],
    latent_t weights_l2[MODEL_LAYER2_NEURONS][MODEL_LAYER2_INPUT_BITS],
    bias_t biases_l2[MODEL_LAYER2_NEURONS]
) {
#pragma HLS inline off
    // Se evita inline para mantener una unica instancia del bloque de inferencia multicapas.

    label_idx_t best_label = 0;
    // Se inicializa la clase ganadora provisional en cero.

    goodness_t best_score = (goodness_t)0;
    // Se inicializa el mejor score total con cero.

    activation_t h1_local[MODEL_LAYER1_NEURONS];
    // Se reserva un buffer temporal de activaciones para la primera capa.

    activation_t h2_local[MODEL_LAYER2_NEURONS];
    // Se reserva un buffer temporal de activaciones para la segunda capa.

#pragma HLS ARRAY_PARTITION variable=h1_local cyclic factor=MODEL_PARALLEL_NEURONS dim=1
    // Se particiona el buffer temporal de la capa 1 para reutilizar el paralelismo de 8 lanes.

#pragma HLS ARRAY_PARTITION variable=h2_local cyclic factor=MODEL_PARALLEL_NEURONS dim=1
    // Se particiona el buffer temporal de la capa 2 para reutilizar el paralelismo de 8 lanes.

candidate_loop:
    for (int candidate = 0; candidate < NUM_CLASSES; candidate++) {
        // Se prueban una por una las 10 hipotesis de etiqueta posibles.

        label_oh_t candidate_onehot = encode_onehot((label_idx_t)candidate);
        // Se construye el label one-hot de la hipotesis actual.

        ff_input_t candidate_input = build_ff_input(candidate_onehot, pixels);
        // Se construye la entrada completa con label incrustado y misma imagen.

        goodness_t g1_score = 0;
        // Se reserva la goodness local de la primera capa para la hipotesis actual.

        goodness_t g2_score = 0;
        // Se reserva la goodness local de la segunda capa para la hipotesis actual.

        ff_layer1_forward(candidate_input, weights_l1, biases_l1, h1_local, g1_score);
        // Se mide la goodness de la primera capa para la hipotesis actual.

        ff_layer_forward_dense<MODEL_LAYER2_INPUT_BITS, MODEL_LAYER2_NEURONS, MODEL_LAYER2_TILES>(
            h1_local,
            weights_l2,
            biases_l2,
            h2_local,
            g2_score
        );
        // Se mide la goodness de la segunda capa usando la activacion de la primera.

        goodness_t candidate_score = g1_score + g2_score;
        // Se forma el score total tal como en el notebook: g_total = g1 + g2.

        if ((candidate == 0) || (candidate_score > best_score)) {
            best_score = candidate_score;
            // Se actualiza el mejor score cuando la hipotesis actual supera a la previa.

            best_label = (label_idx_t)candidate;
            // Se actualiza la etiqueta ganadora provisional.
        }
    }

    return best_label;
    // Se retorna la clase que produce la mayor goodness total del modelo.
}

// ============================================================
// Copia de una capa densa a memoria externa
// ============================================================
template <int INPUT_DIM, int OUTPUT_DIM>
static int snapshot_dense_layer(
    latent_t weights[OUTPUT_DIM][INPUT_DIM],
    bias_t biases[OUTPUT_DIM],
    latent_t *weight_mem_out,
    bias_t *bias_mem_out,
    int weight_base,
    int bias_base
) {
#pragma HLS inline off
    // Se evita inline para mantener una sola rutina de volcado para cualquier capa.

snapshot_dense_weight_loop:
    for (int neuron = 0; neuron < OUTPUT_DIM; neuron++) {
        // Se recorre cada neurona del modelo entrenado en la capa actual.

snapshot_dense_input_loop:
        for (int input_idx = 0; input_idx < INPUT_DIM; input_idx++) {
#pragma HLS PIPELINE II=1
            // Se pipelinea el volcado lineal del snapshot a memoria externa.

            int flat_index = weight_base + neuron * INPUT_DIM + input_idx;
            // Se linealiza el indice bidimensional para el buffer externo global.

            weight_mem_out[flat_index] = weights[neuron][input_idx];
            // Se copia el peso latente al snapshot externo.
        }

        bias_mem_out[bias_base + neuron] = biases[neuron];
        // Se copia el bias latente al snapshot externo respetando el offset de la capa.
    }

    return weight_base + (OUTPUT_DIM * INPUT_DIM);
    // Se retorna el siguiente offset disponible dentro del snapshot global de pesos.
}

// ============================================================
// Copia del modelo multicapa a memoria externa
// ============================================================
static void snapshot_model(
    latent_t weights_l1[MODEL_LAYER1_NEURONS][MODEL_LAYER1_INPUT_BITS],
    bias_t biases_l1[MODEL_LAYER1_NEURONS],
    latent_t weights_l2[MODEL_LAYER2_NEURONS][MODEL_LAYER2_INPUT_BITS],
    bias_t biases_l2[MODEL_LAYER2_NEURONS],
    latent_t *weight_mem_out,
    bias_t *bias_mem_out
) {
#pragma HLS inline off
    // Se evita inline para mantener una unica rutina de volcado del estado entrenado multicapa.

    int next_weight_base = snapshot_dense_layer<MODEL_LAYER1_INPUT_BITS, MODEL_LAYER1_NEURONS>(
        weights_l1,
        biases_l1,
        weight_mem_out,
        bias_mem_out,
        0,
        0
    );
    // Se vuelca primero la primera capa al inicio del snapshot global.

    snapshot_dense_layer<MODEL_LAYER2_INPUT_BITS, MODEL_LAYER2_NEURONS>(
        weights_l2,
        biases_l2,
        weight_mem_out,
        bias_mem_out,
        next_weight_base,
        MODEL_LAYER1_NEURONS
    );
    // Se vuelca despues la segunda capa justo a continuacion de la primera.
}

// ============================================================
// Top heredado de Etapa B
// ============================================================
void forward_fw_top(
    const word_t *in_mem,
    word_t *pos_mem,
    word_t *neg_mem,
    label_idx_t *true_label_mem,
    label_idx_t *neg_label_mem,
    int n_samples,
    uint16_t seed
) {
#pragma HLS INTERFACE m_axi port=in_mem offset=slave bundle=gmem0
    // Se expone la memoria de entrada del dataset congelado.

#pragma HLS INTERFACE m_axi port=pos_mem offset=slave bundle=gmem1
    // Se expone la memoria de salida de muestras positivas.

#pragma HLS INTERFACE m_axi port=neg_mem offset=slave bundle=gmem2
    // Se expone la memoria de salida de muestras negativas.

#pragma HLS INTERFACE m_axi port=true_label_mem offset=slave bundle=gmem3
    // Se expone la memoria de salida de labels verdaderos.

#pragma HLS INTERFACE m_axi port=neg_label_mem offset=slave bundle=gmem4
    // Se expone la memoria de salida de labels negativos.

#pragma HLS INTERFACE s_axilite port=in_mem bundle=control
    // Se expone el puntero de entrada por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=pos_mem bundle=control
    // Se expone el puntero de salida positiva por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=neg_mem bundle=control
    // Se expone el puntero de salida negativa por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=true_label_mem bundle=control
    // Se expone el puntero de labels verdaderos por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=neg_label_mem bundle=control
    // Se expone el puntero de labels negativos por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=n_samples bundle=control
    // Se expone la cantidad de muestras a procesar.

#pragma HLS INTERFACE s_axilite port=seed bundle=control
    // Se expone la semilla del LFSR por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=return bundle=control
    // Se declara el puerto de retorno obligatorio del top HLS.

    int effective_samples = (n_samples < 0) ? 0 : n_samples;
    // Se sanea la cantidad de muestras para evitar lazos invalidos.

    lfsr_t prng_state = (seed == 0) ? (lfsr_t)0xACE1 : (lfsr_t)seed;
    // Se inicializa el generador pseudoaleatorio de etiquetas negativas.

pair_prepare_loop:
    for (int sample_idx = 0; sample_idx < effective_samples; sample_idx++) {
#pragma HLS PIPELINE II=1
        // Se pipelinea la preparacion heredada de pares FF.

        raw_sample_t input_sample = load_sample_from_words(in_mem, sample_idx);
        // Se carga la muestra fisica actual desde memoria externa.

        label_oh_t input_label_onehot;
        // Se reserva el label original de la muestra.

        pixels_t input_pixels;
        // Se reserva la imagen binaria de la muestra.

        padding_t input_padding;
        // Se reserva el padding fisico de la muestra.

        unpack_sample(input_sample, input_label_onehot, input_pixels, input_padding);
        // Se desempaqueta la muestra original.

        label_idx_t true_label_idx = decode_onehot(input_label_onehot);
        // Se decodifica la clase verdadera de la muestra.

        label_idx_t neg_label_idx = generate_negative_label(true_label_idx, prng_state);
        // Se genera una clase incorrecta excluyente para la muestra.

        label_oh_t neg_label_onehot = encode_onehot(neg_label_idx);
        // Se codifica la clase negativa en formato one-hot.

        raw_sample_t positive_sample = pack_sample(input_label_onehot, input_pixels, input_padding);
        // Se reconstruye la muestra positiva preservando imagen y padding.

        raw_sample_t negative_sample = pack_sample(neg_label_onehot, input_pixels, input_padding);
        // Se reconstruye la muestra negativa cambiando solo la etiqueta.

        store_sample_to_words(pos_mem, sample_idx, positive_sample);
        // Se almacena la muestra positiva en memoria externa.

        store_sample_to_words(neg_mem, sample_idx, negative_sample);
        // Se almacena la muestra negativa en memoria externa.

        true_label_mem[sample_idx] = true_label_idx;
        // Se guarda la clase verdadera para validacion en testbench.

        neg_label_mem[sample_idx] = neg_label_idx;
        // Se guarda la clase negativa generada para validacion en testbench.
    }
}

// ============================================================
// Top entrenable de Etapa C
// ============================================================
void ff_train_top(
    const word_t *in_mem,
    label_idx_t *true_label_mem,
    label_idx_t *pred_label_mem,
    latent_t *weight_mem_out,
    bias_t *bias_mem_out,
    goodness_t *g_pos_mem,
    goodness_t *g_neg_mem,
    goodness_t *gap_mem,
    loss_t *epoch_loss_pos_mem,
    loss_t *epoch_loss_neg_mem,
    goodness_t *epoch_g_pos_mem,
    goodness_t *epoch_g_neg_mem,
    goodness_t *epoch_gap_mem,
    ap_uint<32> *correct_count_mem,
    int n_samples,
    int n_train_samples,
    int n_epochs,
    uint16_t seed,
    bool reset_model
) {
#pragma HLS INTERFACE m_axi port=in_mem offset=slave bundle=gmem0
    // Se expone la memoria de entrada del dataset FF congelado.

#pragma HLS INTERFACE m_axi port=true_label_mem offset=slave bundle=gmem1
    // Se expone la memoria de salida de labels verdaderos para inferencia.

#pragma HLS INTERFACE m_axi port=pred_label_mem offset=slave bundle=gmem2
    // Se expone la memoria de salida de predicciones multiclase.

#pragma HLS INTERFACE m_axi port=weight_mem_out offset=slave bundle=gmem3
    // Se expone la memoria de salida del snapshot de pesos.

#pragma HLS INTERFACE m_axi port=bias_mem_out offset=slave bundle=gmem4
    // Se expone la memoria de salida del snapshot de bias.

#pragma HLS INTERFACE m_axi port=g_pos_mem offset=slave bundle=gmem5
    // Se expone la memoria de salida de goodness positiva por muestra.

#pragma HLS INTERFACE m_axi port=g_neg_mem offset=slave bundle=gmem6
    // Se expone la memoria de salida de goodness negativa por muestra.

#pragma HLS INTERFACE m_axi port=gap_mem offset=slave bundle=gmem7
    // Se expone la memoria de salida del goodness gap por muestra.

#pragma HLS INTERFACE m_axi port=epoch_loss_pos_mem offset=slave bundle=gmem8
    // Se expone la memoria de salida del historial de perdida positiva.

#pragma HLS INTERFACE m_axi port=epoch_loss_neg_mem offset=slave bundle=gmem9
    // Se expone la memoria de salida del historial de perdida negativa.

#pragma HLS INTERFACE m_axi port=epoch_g_pos_mem offset=slave bundle=gmem10
    // Se expone la memoria de salida del historial de goodness positiva.

#pragma HLS INTERFACE m_axi port=epoch_g_neg_mem offset=slave bundle=gmem11
    // Se expone la memoria de salida del historial de goodness negativa.

#pragma HLS INTERFACE m_axi port=epoch_gap_mem offset=slave bundle=gmem12
    // Se expone la memoria de salida del historial de goodness gap.

#pragma HLS INTERFACE m_axi port=correct_count_mem offset=slave bundle=gmem13
    // Se expone la memoria de salida del contador de aciertos.

#pragma HLS INTERFACE s_axilite port=in_mem bundle=control
    // Se expone el puntero de entrada por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=true_label_mem bundle=control
    // Se expone el puntero de labels verdaderos por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=pred_label_mem bundle=control
    // Se expone el puntero de predicciones por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=weight_mem_out bundle=control
    // Se expone el puntero del snapshot de pesos por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=bias_mem_out bundle=control
    // Se expone el puntero del snapshot de bias por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=g_pos_mem bundle=control
    // Se expone el puntero de goodness positiva por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=g_neg_mem bundle=control
    // Se expone el puntero de goodness negativa por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=gap_mem bundle=control
    // Se expone el puntero de goodness gap por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=epoch_loss_pos_mem bundle=control
    // Se expone el puntero del historial de perdida positiva por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=epoch_loss_neg_mem bundle=control
    // Se expone el puntero del historial de perdida negativa por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=epoch_g_pos_mem bundle=control
    // Se expone el puntero del historial de goodness positiva por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=epoch_g_neg_mem bundle=control
    // Se expone el puntero del historial de goodness negativa por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=epoch_gap_mem bundle=control
    // Se expone el puntero del historial de goodness gap por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=correct_count_mem bundle=control
    // Se expone el puntero del contador de aciertos por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=n_samples bundle=control
    // Se expone la cantidad de muestras a inferir.

#pragma HLS INTERFACE s_axilite port=n_train_samples bundle=control
    // Se expone la cantidad de muestras a entrenar.

#pragma HLS INTERFACE s_axilite port=n_epochs bundle=control
    // Se expone la cantidad de epocas de entrenamiento.

#pragma HLS INTERFACE s_axilite port=seed bundle=control
    // Se expone la semilla de entrenamiento por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=reset_model bundle=control
    // Se expone la orden de reset del estado persistente del modelo.

#pragma HLS INTERFACE s_axilite port=return bundle=control
    // Se declara el puerto de retorno obligatorio del top HLS.

    static latent_t model_weights_l1[MODEL_LAYER1_NEURONS][MODEL_LAYER1_INPUT_BITS];
    // Se declara la memoria persistente de pesos latentes de la primera capa.

    static bias_t model_biases_l1[MODEL_LAYER1_NEURONS];
    // Se declara la memoria persistente de bias latentes de la primera capa.

    static latent_t model_weights_l2[MODEL_LAYER2_NEURONS][MODEL_LAYER2_INPUT_BITS];
    // Se declara la memoria persistente de pesos latentes de la segunda capa.

    static bias_t model_biases_l2[MODEL_LAYER2_NEURONS];
    // Se declara la memoria persistente de bias latentes de la segunda capa.

    static bool model_initialized = false;
    // Se declara la bandera persistente de inicializacion del modelo.

    // No se fija aqui un bind_storage rigido a BRAM para los pesos del modelo multicapa.
    // La arquitectura 794 -> 512 -> 256 es mucho mayor que la version monolayer y deja abierta
    // una futura migracion del estado latente a DDR o a otra estrategia de almacenamiento.

#pragma HLS ARRAY_PARTITION variable=model_weights_l1 cyclic factor=MODEL_PARALLEL_NEURONS dim=1
    // Se particiona la dimension de neuronas de la primera capa para sostener 8 lanes en paralelo.

#pragma HLS ARRAY_PARTITION variable=model_biases_l1 cyclic factor=MODEL_PARALLEL_NEURONS dim=1
    // Se particiona el vector de bias de la primera capa con la misma granularidad de lanes.

#pragma HLS ARRAY_PARTITION variable=model_weights_l2 cyclic factor=MODEL_PARALLEL_NEURONS dim=1
    // Se particiona la dimension de neuronas de la segunda capa para sostener 8 lanes en paralelo.

#pragma HLS ARRAY_PARTITION variable=model_biases_l2 cyclic factor=MODEL_PARALLEL_NEURONS dim=1
    // Se particiona el vector de bias de la segunda capa con la misma granularidad de lanes.

    int effective_eval_samples = (n_samples < 0) ? 0 : n_samples;
    // Se sanea la cantidad de muestras de inferencia.

    int effective_train_samples = (n_train_samples < 0) ? 0 : n_train_samples;
    // Se sanea la cantidad de muestras de entrenamiento.

    int effective_epochs = (n_epochs < 0) ? 0 : n_epochs;
    // Se sanea la cantidad de epocas solicitadas.

    if (effective_epochs > TRAIN_MAX_EPOCHS) {
        effective_epochs = TRAIN_MAX_EPOCHS;
        // Se limita la cantidad de epocas al tamano del historial hardware.
    }

    lfsr_t model_seed_state = (seed == 0) ? (lfsr_t)0x1D2B : (lfsr_t)seed;
    // Se crea la semilla base usada para inicializacion del modelo.

    if ((reset_model == true) || (model_initialized == false)) {
        initialize_dense_layer<MODEL_LAYER1_INPUT_BITS, MODEL_LAYER1_NEURONS>(
            model_weights_l1,
            model_biases_l1,
            model_seed_state
        );
        // Se inicializa la primera capa cuando se solicita reset o aun no existe estado valido.

        initialize_dense_layer<MODEL_LAYER2_INPUT_BITS, MODEL_LAYER2_NEURONS>(
            model_weights_l2,
            model_biases_l2,
            model_seed_state
        );
        // Se inicializa la segunda capa usando la misma semilla reproducible.

        model_initialized = true;
        // Se marca el modelo persistente como correctamente inicializado.
    }

    if (effective_train_samples > 0) {
        // Solo se limpian metricas cuando realmente se va a ejecutar una fase de entrenamiento.

clear_epoch_history_loop:
        for (int epoch = 0; epoch < TRAIN_MAX_EPOCHS; epoch++) {
#pragma HLS PIPELINE II=1
            // Se limpian los buffers de historial de forma lineal y sintetizable.

            epoch_loss_pos_mem[epoch] = (loss_t)0;
            // Se limpia el historial de perdida positiva.

            epoch_loss_neg_mem[epoch] = (loss_t)0;
            // Se limpia el historial de perdida negativa.

            epoch_g_pos_mem[epoch] = (goodness_t)0;
            // Se limpia el historial de goodness positiva.

            epoch_g_neg_mem[epoch] = (goodness_t)0;
            // Se limpia el historial de goodness negativa.

            epoch_gap_mem[epoch] = (goodness_t)0;
            // Se limpia el historial de goodness gap.
        }

clear_sample_metrics_loop:
        for (int sample_idx = 0; sample_idx < effective_train_samples; sample_idx++) {
#pragma HLS PIPELINE II=1
            // Se limpian las trazas por muestra de la fase de entrenamiento.

            g_pos_mem[sample_idx] = (goodness_t)0;
            // Se limpia la goodness positiva por muestra.

            g_neg_mem[sample_idx] = (goodness_t)0;
            // Se limpia la goodness negativa por muestra.

            gap_mem[sample_idx] = (goodness_t)0;
            // Se limpia el goodness gap por muestra.
        }

        lfsr_t train_state = (seed == 0) ? (lfsr_t)0xACE1 : (lfsr_t)seed;
        // Se inicializa el estado pseudoaleatorio usado para etiquetas negativas del entrenamiento.

training_epoch_loop:
        for (int epoch = 0; epoch < effective_epochs; epoch++) {
            // Se recorre la cantidad efectiva de epocas solicitadas.

            stat_accum_t epoch_loss_pos_sum = 0;
            // Se inicializa el acumulador de perdida positiva de la epoca.

            stat_accum_t epoch_loss_neg_sum = 0;
            // Se inicializa el acumulador de perdida negativa de la epoca.

            stat_accum_t epoch_g_pos_sum = 0;
            // Se inicializa el acumulador de goodness positiva de la epoca.

            stat_accum_t epoch_g_neg_sum = 0;
            // Se inicializa el acumulador de goodness negativa de la epoca.

batch_loop:
            for (int batch_start = 0; batch_start < effective_train_samples; batch_start += MODEL_BATCH_SIZE) {
                // Se recorre el dataset de entrenamiento en bloques pequenos y sintetizables.

                int remaining = effective_train_samples - batch_start;
                // Se calcula cuantas muestras quedan por procesar en esta epoca.

                int batch_size = (remaining > MODEL_BATCH_SIZE) ? MODEL_BATCH_SIZE : remaining;
                // Se ajusta el tamano real del batch para el ultimo bloque parcial.

                bool capture_sample_metrics = (epoch == (effective_epochs - 1));
                // Se decide si esta epoca debe dejar trazas por muestra para el testbench.

                train_batch_local(
                    in_mem,
                    batch_start,
                    batch_size,
                    model_weights_l1,
                    model_biases_l1,
                    model_weights_l2,
                    model_biases_l2,
                    train_state,
                    capture_sample_metrics,
                    g_pos_mem,
                    g_neg_mem,
                    gap_mem,
                    epoch_loss_pos_sum,
                    epoch_loss_neg_sum,
                    epoch_g_pos_sum,
                    epoch_g_neg_sum
                );
                // Se procesa el batch completo con la regla de entrenamiento alineada al notebook.
            }

            epoch_loss_pos_mem[epoch] = compute_mean_loss(epoch_loss_pos_sum, effective_train_samples * MODEL_NUM_LAYERS);
            // Se escribe la perdida positiva media por capa de la epoca actual.

            epoch_loss_neg_mem[epoch] = compute_mean_loss(epoch_loss_neg_sum, effective_train_samples * MODEL_NUM_LAYERS);
            // Se escribe la perdida negativa media por capa de la epoca actual.

            epoch_g_pos_mem[epoch] = compute_mean_goodness(epoch_g_pos_sum, effective_train_samples);
            // Se escribe la goodness positiva media total g1 + g2 de la epoca actual.

            epoch_g_neg_mem[epoch] = compute_mean_goodness(epoch_g_neg_sum, effective_train_samples);
            // Se escribe la goodness negativa media total g1 + g2 de la epoca actual.

            epoch_gap_mem[epoch] = epoch_g_pos_mem[epoch] - epoch_g_neg_mem[epoch];
            // Se escribe el goodness gap medio de la epoca actual.
        }
    }

    ap_uint<32> correct_count = 0;
    // Se inicializa el contador de aciertos de la fase de inferencia.

inference_loop:
    for (int sample_idx = 0; sample_idx < effective_eval_samples; sample_idx++) {
        // Se recorre el subset actual de inferencia con el modelo ya entrenado.

        raw_sample_t input_sample = load_sample_from_words(in_mem, sample_idx);
        // Se carga la muestra actual desde memoria externa.

        label_oh_t true_onehot;
        // Se reserva el label verdadero de la muestra.

        pixels_t pixels;
        // Se reserva la imagen binaria de la muestra.

        padding_t padding;
        // Se reserva el padding fisico aunque aqui no se use.

        unpack_sample(input_sample, true_onehot, pixels, padding);
        // Se desempaqueta la muestra para recuperar su semantica original.

        label_idx_t true_label_idx = decode_onehot(true_onehot);
        // Se decodifica la clase verdadera de la muestra.

        label_idx_t pred_label_idx = ff_predict_label(
            pixels,
            model_weights_l1,
            model_biases_l1,
            model_weights_l2,
            model_biases_l2
        );
        // Se ejecuta la inferencia multiclase probando las 10 etiquetas con g_total = g1 + g2.

        true_label_mem[sample_idx] = true_label_idx;
        // Se guarda la clase verdadera para el testbench.

        pred_label_mem[sample_idx] = pred_label_idx;
        // Se guarda la clase predicha para el testbench.

        if (pred_label_idx == true_label_idx) {
            correct_count++;
            // Se incrementa el contador cuando la prediccion coincide con la etiqueta real.
        }
    }

    correct_count_mem[0] = correct_count;
    // Se expone el total de aciertos de la fase de inferencia actual.

    snapshot_model(
        model_weights_l1,
        model_biases_l1,
        model_weights_l2,
        model_biases_l2,
        weight_mem_out,
        bias_mem_out
    );
    // Se vuelca el estado final del modelo multicapa a memoria externa para inspeccion.
}
