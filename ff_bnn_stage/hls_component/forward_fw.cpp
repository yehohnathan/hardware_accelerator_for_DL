#include "forward_fw.hpp"   // Se incluye el header principal con tipos, constantes y prototipos.

// ============================================================
// Utilidades pequeñas de saturación y valor absoluto
// ============================================================
static latent_t clip_latent(latent_t value) {
    if (value > LATENT_WEIGHT_CLIP) {
        return LATENT_WEIGHT_CLIP;
        // Se satura el valor positivo que excede el clip máximo permitido.
    }

    if (value < -LATENT_WEIGHT_CLIP) {
        return -LATENT_WEIGHT_CLIP;
        // Se satura el valor negativo que excede el clip mínimo permitido.
    }

    return value;
    // Se retorna el valor original cuando ya está dentro del rango permitido.
}

static bias_t clip_bias(bias_t value) {
    if (value > LATENT_BIAS_CLIP) {
        return LATENT_BIAS_CLIP;
        // Se satura el bias positivo al máximo permitido.
    }

    if (value < -LATENT_BIAS_CLIP) {
        return -LATENT_BIAS_CLIP;
        // Se satura el bias negativo al mínimo permitido.
    }

    return value;
    // Se retorna el bias cuando ya está dentro del rango válido.
}

static activation_t relu_local(preact_t value) {
    if (value > 0) {
        return (activation_t)value;
        // Si el pre-activado es positivo, se conserva sin cambio.
    }

    return (activation_t)0;
    // Si el pre-activado es negativo o cero, la activación se anula.
}

static signal_t normalize_signal(goodness_t value) {
    goodness_t limited_value = value;
    // Se crea una copia local para aplicar la saturación previa a la normalización.

    if (limited_value < 0) {
        limited_value = 0;
        // Se fuerza a cero cualquier valor negativo para mantener una magnitud de error válida.
    }

    if (limited_value > (goodness_t)SIGNAL_CLAMP_HW) {
        limited_value = (goodness_t)SIGNAL_CLAMP_HW;
        // Se limita la señal para evitar actualizaciones excesivas y abaratar el hardware.
    }

    return (signal_t)(limited_value / (goodness_t)SIGNAL_CLAMP_HW);
    // Se retorna la señal normalizada al rango aproximado [0, 1].
}

static signal_t normalize_activation(activation_t value) {
    activation_t limited_value = value;
    // Se crea una copia local para aplicar saturación antes de dividir.

    if (limited_value < 0) {
        limited_value = 0;
        // Se fuerza a cero cualquier activación negativa por consistencia con ReLU.
    }

    if (limited_value > ACTIVATION_CLAMP_HW) {
        limited_value = ACTIVATION_CLAMP_HW;
        // Se limita la activación para que la ganancia local permanezca acotada.
    }

    return (signal_t)(limited_value / ACTIVATION_CLAMP_HW);
    // Se retorna una versión reescalada de la activación al rango aproximado [0, 1].
}

static gain_t compute_local_gain(activation_t activation_value, signal_t sample_signal) {
    signal_t normalized_activation = normalize_activation(activation_value);
    // Se normaliza la activación de la neurona para obtener una magnitud acotada.

    gain_t gain_value = (gain_t)(LEARNING_RATE_HW * normalized_activation * sample_signal);
    // Se calcula la ganancia local como producto de tasa de aprendizaje, activación y señal FF.

    return gain_value;
    // Se retorna la magnitud final usada para ajustar pesos y bias.
}

// ============================================================
// LFSR de 16 bits
// ============================================================
lfsr_t lfsr16_next(lfsr_t state) {
    bool new_bit = state[0] ^ state[2] ^ state[3] ^ state[5];
    // Se calcula el nuevo bit de realimentación usando taps sencillos y de bajo costo lógico.

    lfsr_t next_state = (state >> 1);
    // Se desplaza el registro una posición a la derecha para avanzar el estado pseudoaleatorio.

    next_state[15] = new_bit;
    // El nuevo bit calculado se inserta en la posición más significativa del LFSR.

    return next_state;
    // Se retorna el nuevo estado pseudoaleatorio listo para el siguiente uso.
}

// ============================================================
// Validación de one-hot
// ============================================================
bool is_valid_onehot(label_oh_t label_onehot) {
    ap_uint<4> count_ones = 0;
    // Se usa un contador pequeño porque solo es necesario contar hasta 10 bits.

onehot_count_loop:
    for (int i = 0; i < NUM_CLASSES; i++) {
#pragma HLS UNROLL
        // Se desenrolla completamente porque el lazo recorre una longitud fija muy pequeña.

        if (label_onehot[i] == 1) {
            count_ones++;
            // Se incrementa el contador cada vez que se detecta un bit activo.
        }
    }

    return (count_ones == 1);
    // Un one-hot válido debe contener exactamente un único bit en uno.
}

// ============================================================
// one-hot -> índice
// ============================================================
label_idx_t decode_onehot(label_oh_t label_onehot) {
    label_idx_t label_idx = 0;
    // Se inicializa el índice en cero como valor por defecto ante un vector inválido.

onehot_decode_loop:
    for (int i = 0; i < NUM_CLASSES; i++) {
#pragma HLS UNROLL
        // Se desenrolla por completo porque el número de clases es fijo y pequeño.

        if (label_onehot[i] == 1) {
            label_idx = (label_idx_t)i;
            // Cuando el bit i está activo, la clase correspondiente es el índice i.
        }
    }

    return label_idx;
    // Se retorna el índice detectado a partir del vector one-hot.
}

// ============================================================
// índice -> one-hot
// ============================================================
label_oh_t encode_onehot(label_idx_t label_idx) {
    label_oh_t label_onehot = 0;
    // Se limpia completamente el vector de salida antes de activar un único bit.

    label_onehot[label_idx] = 1;
    // Se activa únicamente la posición correspondiente al índice de clase solicitado.

    return label_onehot;
    // Se retorna el vector one-hot resultante.
}

// ============================================================
// Generación de etiqueta negativa excluyente
// ============================================================
label_idx_t generate_negative_label(label_idx_t true_label, lfsr_t &state) {
    state = lfsr16_next(state);
    // Se avanza el estado del generador pseudoaleatorio antes de producir la nueva clase.

    ap_uint<4> offset = (state.range(3, 0) % 9) + 1;
    // Se genera un desplazamiento entre 1 y 9 para garantizar que nunca se repita la clase original.

    ap_uint<5> temp = true_label + offset;
    // Se suma el desplazamiento a la clase verdadera en un contenedor que admite el acarreo.

    label_idx_t negative_label = (temp >= 10) ? (label_idx_t)(temp - 10) : (label_idx_t)temp;
    // Se aplica un módulo 10 manual para mantener el resultado dentro del rango de clases válidas.

    return negative_label;
    // Se retorna una etiqueta incorrecta y distinta de la verdadera.
}

// ============================================================
// Carga de una muestra desde 25 words de 32 bits
// ============================================================
raw_sample_t load_sample_from_words(const word_t *mem, int sample_idx) {
    raw_sample_t sample = 0;
    // Se inicializa el contenedor completo de 800 bits en cero antes de insertar words.

    int base = sample_idx * WORDS_PER_SAMPLE;
    // Se calcula el índice lineal donde inicia la muestra seleccionada dentro de memoria externa.

load_words_loop:
    for (int w = 0; w < WORDS_PER_SAMPLE; w++) {
#pragma HLS UNROLL
        // Se desenrolla completamente porque siempre se leen exactamente 25 words por muestra.

        sample.range((w + 1) * WORD_BITS - 1, w * WORD_BITS) = mem[base + w];
        // Se coloca la palabra w dentro del rango correspondiente del vector total de 800 bits.
    }

    return sample;
    // Se retorna la muestra ya reconstruida como un único vector ancho.
}

// ============================================================
// Almacenamiento de una muestra en 25 words de 32 bits
// ============================================================
void store_sample_to_words(word_t *mem, int sample_idx, raw_sample_t sample) {
    int base = sample_idx * WORDS_PER_SAMPLE;
    // Se calcula la posición base en memoria lineal donde se escribirá la muestra objetivo.

store_words_loop:
    for (int w = 0; w < WORDS_PER_SAMPLE; w++) {
#pragma HLS UNROLL
        // Se desenrolla completamente porque la anchura de la muestra es fija en 25 words.

        mem[base + w] = sample.range((w + 1) * WORD_BITS - 1, w * WORD_BITS);
        // Se extrae cada bloque de 32 bits del vector ancho y se escribe en la memoria destino.
    }
}

// ============================================================
// Desempaquetado de la muestra física de 800 bits
// ============================================================
void unpack_sample(raw_sample_t sample, label_oh_t &label_onehot, pixels_t &pixels, padding_t &padding) {
    label_onehot = sample.range(9, 0);
    // Se extraen los 10 bits menos significativos que contienen la etiqueta one-hot.

    pixels = sample.range(793, 10);
    // Se extraen los 784 bits centrales que contienen la imagen binaria.

    padding = sample.range(799, 794);
    // Se extraen los 6 bits más significativos de padding físico.
}

// ============================================================
// Empaquetado de la muestra física de 800 bits
// ============================================================
raw_sample_t pack_sample(label_oh_t label_onehot, pixels_t pixels, padding_t padding) {
    raw_sample_t sample = 0;
    // Se limpia completamente el contenedor físico antes de insertar los campos útiles.

    sample.range(9, 0) = label_onehot;
    // Se inserta el label one-hot en los 10 bits menos significativos.

    sample.range(793, 10) = pixels;
    // Se inserta la imagen binaria en la zona central del vector físico.

    sample.range(799, 794) = padding;
    // Se inserta el padding original en los bits más significativos para preservar el layout.

    return sample;
    // Se retorna la muestra completa ya reempaquetada.
}

// ============================================================
// Construcción y desempaquetado de la entrada lógica FF de 794 bits
// ============================================================
ff_input_t build_ff_input(label_oh_t label_onehot, pixels_t pixels) {
    ff_input_t ff_input = 0;
    // Se inicializa la entrada lógica útil en cero antes de asignar sus campos.

    ff_input.range(LABEL_BITS - 1, 0) = label_onehot;
    // Se coloca la etiqueta one-hot en la parte baja del vector lógico.

    ff_input.range(MODEL_INPUT_BITS - 1, LABEL_BITS) = pixels;
    // Se coloca la imagen binaria a continuación del label respetando el layout del proyecto.

    return ff_input;
    // Se retorna la entrada lógica útil de 794 bits lista para el forward de la capa.
}

void unpack_ff_input(ff_input_t ff_input, label_oh_t &label_onehot, pixels_t &pixels) {
    label_onehot = ff_input.range(LABEL_BITS - 1, 0);
    // Se recupera la etiqueta one-hot almacenada en la parte baja del vector lógico.

    pixels = ff_input.range(MODEL_INPUT_BITS - 1, LABEL_BITS);
    // Se recuperan los 784 bits correspondientes a la imagen binaria.
}

// ============================================================
// Inicialización del modelo latente entrenable
// ============================================================
static void initialize_model(latent_t weights[MODEL_NEURONS][MODEL_INPUT_BITS], bias_t biases[MODEL_NEURONS], lfsr_t &state) {
#pragma HLS INLINE off
    // Se evita inline total para no replicar toda la inicialización cada vez que se invoque desde el top.

init_neuron_loop:
    for (int neuron = 0; neuron < MODEL_NEURONS; neuron++) {
        // Se recorre cada neurona de la capa entrenable para inicializar su bias y todos sus pesos.

        biases[neuron] = 0;
        // Cada bias parte en cero para no introducir desplazamientos iniciales arbitrarios.

init_weight_loop:
        for (int input_idx = 0; input_idx < MODEL_INPUT_BITS; input_idx++) {
#pragma HLS PIPELINE II=1
            // Se pipelinea la inicialización por peso para abaratar tiempo de arranque sin consumir registros masivos.

            state = lfsr16_next(state);
            // Se avanza el LFSR para obtener un nuevo patrón pseudoaleatorio barato en hardware.

            if (state[0] == 1) {
                weights[neuron][input_idx] = LATENT_INIT_MAG;
                // Cuando el bit pseudoaleatorio vale 1, el peso latente inicia con signo positivo.
            } else {
                weights[neuron][input_idx] = -LATENT_INIT_MAG;
                // Cuando el bit pseudoaleatorio vale 0, el peso latente inicia con signo negativo.
            }
        }
    }
}

// ============================================================
// Forward local de una sola capa FF con 8 neuronas en paralelo
// ============================================================
static void ff_layer_forward(
    ff_input_t ff_input,
    latent_t weights[MODEL_NEURONS][MODEL_INPUT_BITS],
    bias_t biases[MODEL_NEURONS],
    activation_t activations[MODEL_NEURONS],
    goodness_t &goodness
) {
#pragma HLS INLINE off
    // Se evita inline total para mantener una sola instancia clara del forward local dentro del diseño.

    goodness_t sum_sq = 0;
    // Se inicializa el acumulador de suma de cuadrados usado para calcular la goodness de la capa.

tile_forward_loop:
    for (int tile = 0; tile < MODEL_NEURON_TILES; tile++) {
        // Se recorre cada tile de 8 neuronas para explotar el paralelismo fijado por la arquitectura base.

        preact_t z_lane[MODEL_PARALLEL_NEURONS];
        // Se crea un pequeño arreglo local para acumular el pre-activado de cada lane en el tile actual.

#pragma HLS ARRAY_PARTITION variable=z_lane complete
        // Se particiona completamente el arreglo local de acumuladores para permitir 8 operaciones en paralelo.

lane_init_loop:
        for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
            // Se desenrolla totalmente porque cada lane representa una neurona físicamente paralela dentro del tile.

            int neuron = tile * MODEL_PARALLEL_NEURONS + lane;
            // Se calcula el índice absoluto de la neurona asociada al lane actual.

            z_lane[lane] = (preact_t)biases[neuron];
            // Cada acumulador inicia con el bias latente de la neurona correspondiente.
        }

input_forward_loop:
        for (int input_idx = 0; input_idx < MODEL_INPUT_BITS; input_idx++) {
#pragma HLS PIPELINE II=1
            // Se pipelinea el recorrido sobre las 794 entradas para sostener throughput razonable sin desplegar todo el producto interno.

            ap_uint<1> x_bit = ff_input[input_idx];
            // Se extrae el bit de entrada actual para decidir si esta dimensión contribuye o no al pre-activado.

            if (x_bit == 1) {
                // Solo las entradas activas contribuyen porque el dataset ya está binarizado como 0/1 en hardware.

lane_acc_loop:
                for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
                    // Se desenrolla completamente para actualizar las 8 neuronas del tile en paralelo.

                    int neuron = tile * MODEL_PARALLEL_NEURONS + lane;
                    // Se recalcula el índice absoluto de la neurona asociada al lane.

                    if (weights[neuron][input_idx] >= 0) {
                        z_lane[lane] = z_lane[lane] + (preact_t)1;
                        // Si el peso latente tiene signo positivo, el bit activo aporta +1 al pre-activado.
                    } else {
                        z_lane[lane] = z_lane[lane] - (preact_t)1;
                        // Si el peso latente tiene signo negativo, el bit activo aporta -1 al pre-activado.
                    }
                }
            }
        }

lane_output_loop:
        for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
            // Se desenrolla completamente para producir activaciones y cuadrados de las 8 neuronas en paralelo.

            int neuron = tile * MODEL_PARALLEL_NEURONS + lane;
            // Se calcula el índice absoluto de la neurona cuya salida se registrará.

            activation_t activation_value = relu_local(z_lane[lane]);
            // Se aplica la activación ReLU local recomendada para la versión FF del proyecto.

            activations[neuron] = activation_value;
            // Se almacena la activación local para usarla más adelante en actualización o depuración.

            sum_sq = sum_sq + (goodness_t)(activation_value * activation_value);
            // Se acumula el cuadrado de la activación para construir la goodness de la capa.
        }
    }

    goodness = (goodness_t)(sum_sq / (goodness_t)MODEL_NEURONS);
    // Se obtiene la goodness final como media de cuadrados de las activaciones locales.
}

// ============================================================
// Cálculo de la señal local de aprendizaje FF por tramos
// ============================================================
static void ff_compute_training_signal(goodness_t g_pos, goodness_t g_neg, signal_t &e_pos, signal_t &e_neg) {
#pragma HLS INLINE
    // Se fuerza inline porque esta operación es muy pequeña y depende solo de dos goodness y un umbral fijo.

    goodness_t pos_error = 0;
    // Se crea una variable para la corrección positiva asociada a muestras reales.

    goodness_t neg_error = 0;
    // Se crea una variable para la corrección negativa asociada a muestras incorrectamente etiquetadas.

    if (g_pos < GOODNESS_THRESHOLD_HW) {
        pos_error = GOODNESS_THRESHOLD_HW - g_pos;
        // Cuando la goodness positiva queda por debajo del umbral, se genera presión para aumentarla.
    }

    if (g_neg > GOODNESS_THRESHOLD_HW) {
        neg_error = g_neg - GOODNESS_THRESHOLD_HW;
        // Cuando la goodness negativa supera el umbral, se genera presión para reducirla.
    }

    e_pos = normalize_signal(pos_error);
    // Se normaliza la señal positiva resultante para usarla como factor de actualización acotado.

    e_neg = normalize_signal(neg_error);
    // Se normaliza la señal negativa resultante para usarla como factor de actualización acotado.
}

// ============================================================
// Actualización local de pesos y bias latentes
// ============================================================
static void ff_update_model(
    ff_input_t x_pos,
    ff_input_t x_neg,
    activation_t h_pos[MODEL_NEURONS],
    activation_t h_neg[MODEL_NEURONS],
    signal_t e_pos,
    signal_t e_neg,
    latent_t weights[MODEL_NEURONS][MODEL_INPUT_BITS],
    bias_t biases[MODEL_NEURONS]
) {
#pragma HLS INLINE off
    // Se evita inline total para no duplicar el bloque de actualización dentro del top por cada llamada.

tile_update_loop:
    for (int tile = 0; tile < MODEL_NEURON_TILES; tile++) {
        // Se recorre cada tile de 8 neuronas para respetar el paralelismo objetivo de la arquitectura base.

        gain_t pos_gain_lane[MODEL_PARALLEL_NEURONS];
        // Se almacenan las ganancias positivas locales de las 8 neuronas del tile actual.

        gain_t neg_gain_lane[MODEL_PARALLEL_NEURONS];
        // Se almacenan las ganancias negativas locales de las 8 neuronas del tile actual.

#pragma HLS ARRAY_PARTITION variable=pos_gain_lane complete
        // Se particiona por completo el arreglo local positivo para usar todas las lanes en paralelo.

#pragma HLS ARRAY_PARTITION variable=neg_gain_lane complete
        // Se particiona por completo el arreglo local negativo para usar todas las lanes en paralelo.

lane_gain_loop:
        for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
            // Se desenrolla totalmente porque cada lane corresponde a una neurona física paralela del tile.

            int neuron = tile * MODEL_PARALLEL_NEURONS + lane;
            // Se calcula el índice absoluto de la neurona asociada al lane actual.

            pos_gain_lane[lane] = compute_local_gain(h_pos[neuron], e_pos);
            // Se calcula la ganancia de refuerzo positiva a partir de activación local y señal FF positiva.

            neg_gain_lane[lane] = compute_local_gain(h_neg[neuron], e_neg);
            // Se calcula la ganancia de supresión negativa a partir de activación local y señal FF negativa.
        }

input_update_loop:
        for (int input_idx = 0; input_idx < MODEL_INPUT_BITS; input_idx++) {
#pragma HLS PIPELINE II=1
            // Se pipelinea la actualización peso por peso para equilibrar latencia y uso de recursos en HLS.

            ap_uint<1> pos_bit = x_pos[input_idx];
            // Se extrae el bit activo o inactivo de la entrada positiva para esta dimensión.

            ap_uint<1> neg_bit = x_neg[input_idx];
            // Se extrae el bit activo o inactivo de la entrada negativa para esta dimensión.

lane_weight_update_loop:
            for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
                // Se desenrolla por completo para actualizar las 8 neuronas del tile en paralelo.

                int neuron = tile * MODEL_PARALLEL_NEURONS + lane;
                // Se calcula el índice absoluto de la neurona cuyo peso se ajustará.

                latent_t new_weight = weights[neuron][input_idx];
                // Se copia localmente el peso latente actual para ajustarlo antes de reescribirlo.

                if (pos_bit == 1) {
                    new_weight = clip_latent(new_weight + (latent_t)pos_gain_lane[lane]);
                    // Si el bit positivo está activo, el peso recibe refuerzo hebbiano positivo acotado.
                }

                if (neg_bit == 1) {
                    new_weight = clip_latent(new_weight - (latent_t)neg_gain_lane[lane]);
                    // Si el bit negativo está activo, el peso recibe supresión hebbiana negativa acotada.
                }

                weights[neuron][input_idx] = new_weight;
                // Se escribe el peso latente actualizado de vuelta en la memoria del modelo.
            }
        }

lane_bias_update_loop:
        for (int lane = 0; lane < MODEL_PARALLEL_NEURONS; lane++) {
#pragma HLS UNROLL
            // Se desenrolla completamente porque solo se actualizan 8 bias por tile.

            int neuron = tile * MODEL_PARALLEL_NEURONS + lane;
            // Se calcula el índice absoluto de la neurona cuyo bias se ajustará.

            bias_t new_bias = biases[neuron];
            // Se copia localmente el bias actual antes de modificarlo.

            new_bias = clip_bias(new_bias + (bias_t)pos_gain_lane[lane] - (bias_t)neg_gain_lane[lane]);
            // El bias se incrementa con evidencia positiva y se reduce con evidencia negativa de forma local.

            biases[neuron] = new_bias;
            // Se escribe el bias latente actualizado de vuelta en el estado entrenable del modelo.
        }
    }
}

// ============================================================
// Inferencia multiclase por argmax de goodness
// ============================================================
static label_idx_t ff_predict_label(
    pixels_t pixels,
    latent_t weights[MODEL_NEURONS][MODEL_INPUT_BITS],
    bias_t biases[MODEL_NEURONS]
) {
#pragma HLS INLINE off
    // Se evita inline total para concentrar el bloque de inferencia multiclase en una sola instancia reutilizable.

    label_idx_t best_label = 0;
    // Se inicializa la mejor clase en cero antes de evaluar todas las posibilidades.

    goodness_t best_goodness = (goodness_t)-1;
    // Se inicializa el mejor puntaje con un valor menor que cualquier goodness válida.

    activation_t activations_local[MODEL_NEURONS];
    // Se reserva un arreglo local para almacenar las activaciones temporales de cada hipótesis de clase.

#pragma HLS ARRAY_PARTITION variable=activations_local cyclic factor=MODEL_PARALLEL_NEURONS dim=1
    // Se particiona cíclicamente el buffer de activaciones para facilitar el uso paralelo por tiles.

label_hypothesis_loop:
    for (int candidate = 0; candidate < NUM_CLASSES; candidate++) {
        // Se prueban una por una las diez etiquetas posibles para la imagen dada.

        label_oh_t candidate_onehot = encode_onehot((label_idx_t)candidate);
        // Se construye la hipótesis de etiqueta actual en formato one-hot.

        ff_input_t candidate_input = build_ff_input(candidate_onehot, pixels);
        // Se construye la entrada FF lógica para esa etiqueta candidata y la misma imagen.

        goodness_t candidate_goodness = 0;
        // Se crea una variable local para la goodness obtenida con la hipótesis actual.

        ff_layer_forward(candidate_input, weights, biases, activations_local, candidate_goodness);
        // Se ejecuta el forward local de la capa entrenada para medir la goodness de la hipótesis actual.

        if ((candidate == 0) || (candidate_goodness > best_goodness)) {
            best_goodness = candidate_goodness;
            // Si la goodness actual supera a la mejor previa, se actualiza el mejor puntaje observado.

            best_label = (label_idx_t)candidate;
            // También se actualiza la clase ganadora provisional del proceso de argmax.
        }
    }

    return best_label;
    // Se retorna la clase cuya hipótesis produjo la mayor goodness total de la capa.
}

// ============================================================
// Copia del modelo a memoria externa para inspección en testbench
// ============================================================
static void snapshot_model(
    latent_t weights[MODEL_NEURONS][MODEL_INPUT_BITS],
    bias_t biases[MODEL_NEURONS],
    latent_t *weight_mem_out,
    bias_t *bias_mem_out
) {
#pragma HLS INLINE off
    // Se evita inline total para mantener una sola rutina clara de extracción del estado entrenado.

snapshot_weight_loop:
    for (int neuron = 0; neuron < MODEL_NEURONS; neuron++) {
        // Se recorre cada neurona para volcar sus pesos y bias a memoria externa de depuración.

snapshot_input_loop:
        for (int input_idx = 0; input_idx < MODEL_INPUT_BITS; input_idx++) {
#pragma HLS PIPELINE II=1
            // Se pipelinea el recorrido lineal del snapshot para hacerlo eficiente sin desplegar memoria masiva.

            int flat_index = neuron * MODEL_INPUT_BITS + input_idx;
            // Se linealiza el índice bidimensional del peso para escribirlo en un buffer unidimensional.

            weight_mem_out[flat_index] = weights[neuron][input_idx];
            // Se copia el peso latente actual al buffer externo de snapshot.
        }

        bias_mem_out[neuron] = biases[neuron];
        // Se copia el bias latente de la neurona actual al buffer externo de snapshot.
    }
}

// ============================================================
// Top sintetizable heredado de la Etapa B
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
#pragma HLS INTERFACE m_axi     port=in_mem         offset=slave bundle=gmem0
    // Se declara un puerto AXI maestro de lectura para el dataset de entrada ya congelado.

#pragma HLS INTERFACE m_axi     port=pos_mem        offset=slave bundle=gmem1
    // Se declara un puerto AXI maestro de escritura para las muestras positivas reconstruidas.

#pragma HLS INTERFACE m_axi     port=neg_mem        offset=slave bundle=gmem2
    // Se declara un puerto AXI maestro de escritura para las muestras negativas reconstruidas.

#pragma HLS INTERFACE m_axi     port=true_label_mem offset=slave bundle=gmem3
    // Se declara un puerto AXI maestro de escritura para los labels verdaderos decodificados.

#pragma HLS INTERFACE m_axi     port=neg_label_mem  offset=slave bundle=gmem4
    // Se declara un puerto AXI maestro de escritura para los labels negativos generados.

#pragma HLS INTERFACE s_axilite port=in_mem         bundle=control
    // Se expone el puntero de entrada por AXI-Lite para control desde host o testbench.

#pragma HLS INTERFACE s_axilite port=pos_mem        bundle=control
    // Se expone el puntero de salida positiva por AXI-Lite para control.

#pragma HLS INTERFACE s_axilite port=neg_mem        bundle=control
    // Se expone el puntero de salida negativa por AXI-Lite para control.

#pragma HLS INTERFACE s_axilite port=true_label_mem bundle=control
    // Se expone el puntero de labels verdaderos por AXI-Lite para control.

#pragma HLS INTERFACE s_axilite port=neg_label_mem  bundle=control
    // Se expone el puntero de labels negativos por AXI-Lite para control.

#pragma HLS INTERFACE s_axilite port=n_samples      bundle=control
    // Se expone el número de muestras a procesar como registro de control.

#pragma HLS INTERFACE s_axilite port=seed           bundle=control
    // Se expone la semilla inicial del generador pseudoaleatorio como registro de control.

#pragma HLS INTERFACE s_axilite port=return         bundle=control
    // Se declara el puerto de retorno AXI-Lite obligatorio para una función top de HLS.

    lfsr_t prng_state = (seed == 0) ? (lfsr_t)0xACE1 : (lfsr_t)seed;
    // Se inicializa el estado pseudoaleatorio con la semilla dada o con una fija si se recibe cero.

pair_prepare_loop:
    for (int sample_idx = 0; sample_idx < n_samples; sample_idx++) {
#pragma HLS PIPELINE II=1
        // Se pipelinea el procesamiento por muestra para maximizar throughput en esta etapa ya validada.

        raw_sample_t input_sample = load_sample_from_words(in_mem, sample_idx);
        // Se lee la muestra física actual desde memoria externa y se reconstruye en un solo vector de 800 bits.

        label_oh_t input_label_onehot;
        // Se reserva el contenedor para el label original extraído de la muestra.

        pixels_t input_pixels;
        // Se reserva el contenedor para la imagen binaria extraída de la muestra.

        padding_t input_padding;
        // Se reserva el contenedor para el padding físico asociado a la muestra.

        unpack_sample(input_sample, input_label_onehot, input_pixels, input_padding);
        // Se separan el label, la imagen y el padding sin modificar el dataset congelado.

        label_idx_t true_label_idx = decode_onehot(input_label_onehot);
        // Se decodifica el label verdadero a índice entero para depuración y control.

        label_idx_t neg_label_idx = generate_negative_label(true_label_idx, prng_state);
        // Se genera una etiqueta negativa excluyente usando el LFSR ya validado en la Etapa B.

        label_oh_t neg_label_onehot = encode_onehot(neg_label_idx);
        // Se convierte la etiqueta negativa al mismo formato one-hot del dataset original.

        raw_sample_t positive_sample = pack_sample(input_label_onehot, input_pixels, input_padding);
        // Se construye la muestra positiva manteniendo intacta la misma imagen y el mismo padding.

        raw_sample_t negative_sample = pack_sample(neg_label_onehot, input_pixels, input_padding);
        // Se construye la muestra negativa cambiando únicamente la etiqueta y preservando el resto.

        store_sample_to_words(pos_mem, sample_idx, positive_sample);
        // Se escribe la muestra positiva en la memoria externa de salida positiva.

        store_sample_to_words(neg_mem, sample_idx, negative_sample);
        // Se escribe la muestra negativa en la memoria externa de salida negativa.

        true_label_mem[sample_idx] = true_label_idx;
        // Se guarda el índice de la clase verdadera para uso posterior en testbench.

        neg_label_mem[sample_idx] = neg_label_idx;
        // Se guarda el índice de la clase negativa generada para uso posterior en testbench.
    }
}

// ============================================================
// Top sintetizable nuevo de la Etapa C
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
    ap_uint<32> *correct_count_mem,
    int n_samples,
    int n_train_samples,
    int n_epochs,
    uint16_t seed,
    bool reset_model
) {
#pragma HLS INTERFACE m_axi     port=in_mem            offset=slave bundle=gmem0
    // Se declara el puerto AXI maestro de lectura para el dataset FF congelado en memoria externa.

#pragma HLS INTERFACE m_axi     port=true_label_mem    offset=slave bundle=gmem1
    // Se declara el puerto AXI maestro de escritura para labels verdaderos usados en métricas.

#pragma HLS INTERFACE m_axi     port=pred_label_mem    offset=slave bundle=gmem2
    // Se declara el puerto AXI maestro de escritura para predicciones multiclase posteriores al entrenamiento.

#pragma HLS INTERFACE m_axi     port=weight_mem_out    offset=slave bundle=gmem3
    // Se declara el puerto AXI maestro de escritura para el snapshot final de pesos latentes.

#pragma HLS INTERFACE m_axi     port=bias_mem_out      offset=slave bundle=gmem4
    // Se declara el puerto AXI maestro de escritura para el snapshot final de bias latentes.

#pragma HLS INTERFACE m_axi     port=g_pos_mem         offset=slave bundle=gmem5
    // Se declara el puerto AXI maestro de escritura para goodness positiva por muestra en la última época.

#pragma HLS INTERFACE m_axi     port=g_neg_mem         offset=slave bundle=gmem6
    // Se declara el puerto AXI maestro de escritura para goodness negativa por muestra en la última época.

#pragma HLS INTERFACE m_axi     port=gap_mem           offset=slave bundle=gmem7
    // Se declara el puerto AXI maestro de escritura para el goodness gap por muestra en la última época.

#pragma HLS INTERFACE m_axi     port=correct_count_mem offset=slave bundle=gmem8
    // Se declara el puerto AXI maestro de escritura para el conteo de aciertos tras la inferencia.

#pragma HLS INTERFACE s_axilite port=in_mem            bundle=control
    // Se expone el puntero de entrada a través de AXI-Lite para control de alto nivel.

#pragma HLS INTERFACE s_axilite port=true_label_mem    bundle=control
    // Se expone el puntero de salida de labels verdaderos por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=pred_label_mem    bundle=control
    // Se expone el puntero de salida de predicciones por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=weight_mem_out    bundle=control
    // Se expone el puntero de salida del snapshot de pesos por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=bias_mem_out      bundle=control
    // Se expone el puntero de salida del snapshot de bias por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=g_pos_mem         bundle=control
    // Se expone el puntero de salida de goodness positiva por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=g_neg_mem         bundle=control
    // Se expone el puntero de salida de goodness negativa por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=gap_mem           bundle=control
    // Se expone el puntero de salida de goodness gap por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=correct_count_mem bundle=control
    // Se expone el puntero del contador de aciertos por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=n_samples         bundle=control
    // Se expone la cantidad de muestras a inferir y reportar por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=n_train_samples   bundle=control
    // Se expone la cantidad de muestras usadas para entrenamiento por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=n_epochs          bundle=control
    // Se expone la cantidad de épocas de entrenamiento local por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=seed             bundle=control
    // Se expone la semilla inicial del pseudoaleatorio por AXI-Lite.

#pragma HLS INTERFACE s_axilite port=reset_model      bundle=control
    // Se expone la orden de reinicialización del modelo por AXI-Lite para controlar el estado entrenable persistente.

#pragma HLS INTERFACE s_axilite port=return           bundle=control
    // Se declara el puerto de retorno AXI-Lite obligatorio para una función top sintetizable.

    static latent_t model_weights[MODEL_NEURONS][MODEL_INPUT_BITS];
    // Se declara la memoria interna persistente de pesos latentes que se materializará preferentemente en BRAM.

    static bias_t model_biases[MODEL_NEURONS];
    // Se declara la memoria interna persistente de bias latentes del modelo entrenable.

    static bool model_initialized = false;
    // Se declara un flag persistente que indica si el estado del modelo ya fue inicializado al menos una vez.

#pragma HLS ARRAY_PARTITION variable=model_weights cyclic factor=MODEL_PARALLEL_NEURONS dim=1
    // Se particiona cíclicamente la dimensión de neuronas para que 8 lanes puedan leerse y actualizarse en paralelo.

#pragma HLS ARRAY_PARTITION variable=model_biases cyclic factor=MODEL_PARALLEL_NEURONS dim=1
    // Se particiona cíclicamente el vector de bias por la misma razón de paralelismo por tile.

#pragma HLS BIND_STORAGE variable=model_weights type=ram_2p impl=bram
    // Se sugiere mapear la matriz de pesos a BRAM de doble puerto para balancear capacidad y accesos concurrentes.

#pragma HLS BIND_STORAGE variable=model_biases type=ram_2p impl=bram
    // Se sugiere mapear el vector de bias a BRAM para mantener el estado entrenable dentro del FPGA.

    int effective_samples = n_samples;
    // Se crea una copia local de la cantidad de muestras para aplicar saneamiento de límites.

    int effective_train_samples = n_train_samples;
    // Se crea una copia local de la cantidad de muestras de entrenamiento para aplicar saneamiento de límites.

    if (effective_samples < 0) {
        effective_samples = 0;
        // Se corrige cualquier valor negativo de muestras totales a cero para evitar lazos inválidos.
    }

    if (effective_train_samples < 0) {
        effective_train_samples = 0;
        // Se corrige cualquier valor negativo de muestras de entrenamiento a cero.
    }

    if (effective_train_samples > effective_samples) {
        effective_train_samples = effective_samples;
        // Se fuerza que el conjunto de entrenamiento no supere la cantidad total de muestras a evaluar.
    }

    if (n_epochs < 0) {
        n_epochs = 0;
        // Se corrige una cantidad negativa de épocas a cero para evitar lazos inválidos.
    }

    lfsr_t model_seed_state = (seed == 0) ? (lfsr_t)0x1D2B : (lfsr_t)seed;
    // Se inicializa el estado pseudoaleatorio base usado para inicialización del modelo y etiquetas negativas.

    if ((reset_model == true) || (model_initialized == false)) {
        initialize_model(model_weights, model_biases, model_seed_state);
        // Si se solicita reset o el modelo nunca se inicializó, se regeneran pesos y bias latentes desde cero.

        model_initialized = true;
        // Se marca el estado del modelo como correctamente inicializado para llamadas futuras.
    }

    lfsr_t train_state = (seed == 0) ? (lfsr_t)0xACE1 : (lfsr_t)seed;
    // Se inicializa el LFSR específico del entrenamiento para crear etiquetas negativas reproducibles.

    activation_t h_pos[MODEL_NEURONS];
    // Se reserva el buffer local de activaciones positivas de la única capa entrenable.

    activation_t h_neg[MODEL_NEURONS];
    // Se reserva el buffer local de activaciones negativas de la única capa entrenable.

#pragma HLS ARRAY_PARTITION variable=h_pos cyclic factor=MODEL_PARALLEL_NEURONS dim=1
    // Se particiona cíclicamente el buffer de activaciones positivas para acompañar el paralelismo de 8 neuronas.

#pragma HLS ARRAY_PARTITION variable=h_neg cyclic factor=MODEL_PARALLEL_NEURONS dim=1
    // Se particiona cíclicamente el buffer de activaciones negativas por la misma razón arquitectónica.

training_epoch_loop:
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        // Se recorren las épocas locales de entrenamiento FF definidas por el control superior.

training_sample_loop:
        for (int sample_idx = 0; sample_idx < effective_train_samples; sample_idx++) {
            // Se recorren las muestras de entrenamiento seleccionadas para esta llamada al kernel.

            raw_sample_t input_sample = load_sample_from_words(in_mem, sample_idx);
            // Se lee la muestra física actual desde memoria externa.

            label_oh_t true_onehot;
            // Se reserva un contenedor local para la etiqueta verdadera de la muestra.

            pixels_t pixels;
            // Se reserva un contenedor local para la imagen binaria de la muestra.

            padding_t padding;
            // Se reserva un contenedor local para el padding físico aunque no participe en el forward de la capa.

            unpack_sample(input_sample, true_onehot, pixels, padding);
            // Se desempaqueta completamente la muestra para recuperar su semántica original.

            label_idx_t true_label_idx = decode_onehot(true_onehot);
            // Se decodifica el label verdadero a índice entero para formar el par FF supervisado.

            label_idx_t neg_label_idx = generate_negative_label(true_label_idx, train_state);
            // Se genera una etiqueta negativa distinta a la verdadera con el LFSR local del entrenamiento.

            label_oh_t neg_onehot = encode_onehot(neg_label_idx);
            // Se reconstruye el label negativo en formato one-hot para la entrada incorrecta.

            ff_input_t x_pos = build_ff_input(true_onehot, pixels);
            // Se construye la entrada positiva usando la imagen real y su etiqueta correcta.

            ff_input_t x_neg = build_ff_input(neg_onehot, pixels);
            // Se construye la entrada negativa usando la misma imagen y una etiqueta incorrecta.

            goodness_t g_pos = 0;
            // Se reserva la goodness positiva local de la muestra actual.

            goodness_t g_neg = 0;
            // Se reserva la goodness negativa local de la muestra actual.

            ff_layer_forward(x_pos, model_weights, model_biases, h_pos, g_pos);
            // Se ejecuta el forward local para la muestra positiva con pesos binarizados por signo latente.

            ff_layer_forward(x_neg, model_weights, model_biases, h_neg, g_neg);
            // Se ejecuta el forward local para la muestra negativa con la misma imagen y etiqueta incorrecta.

            signal_t e_pos = 0;
            // Se reserva la señal de aprendizaje asociada al ejemplo positivo.

            signal_t e_neg = 0;
            // Se reserva la señal de aprendizaje asociada al ejemplo negativo.

            ff_compute_training_signal(g_pos, g_neg, e_pos, e_neg);
            // Se calcula la señal local FF por tramos usando threshold, g_pos y g_neg.

            ff_update_model(x_pos, x_neg, h_pos, h_neg, e_pos, e_neg, model_weights, model_biases);
            // Se actualizan pesos y bias latentes mediante una regla local coherente con Forward-Forward.

            if (epoch == (n_epochs - 1)) {
                g_pos_mem[sample_idx] = g_pos;
                // En la última época se registra la goodness positiva observada para esta muestra.

                g_neg_mem[sample_idx] = g_neg;
                // En la última época se registra la goodness negativa observada para esta muestra.

                gap_mem[sample_idx] = g_pos - g_neg;
                // En la última época se registra el goodness gap, útil para inspección y experimentación.
            }
        }
    }

    ap_uint<32> correct_count = 0;
    // Se inicializa el contador de aciertos que medirá la capacidad de clasificación del modelo entrenado.

inference_loop:
    for (int sample_idx = 0; sample_idx < effective_samples; sample_idx++) {
        // Se recorre cada muestra a inferir usando el modelo ya entrenado dentro del FPGA.

        raw_sample_t input_sample = load_sample_from_words(in_mem, sample_idx);
        // Se vuelve a leer la muestra original desde memoria externa para inferencia multiclase.

        label_oh_t true_onehot;
        // Se reserva un contenedor local para el label real de la muestra evaluada.

        pixels_t pixels;
        // Se reserva un contenedor local para la imagen binaria de la muestra evaluada.

        padding_t padding;
        // Se reserva un contenedor local para el padding físico aunque aquí no se use directamente.

        unpack_sample(input_sample, true_onehot, pixels, padding);
        // Se extraen el label real, la imagen y el padding desde el dataset congelado.

        label_idx_t true_label_idx = decode_onehot(true_onehot);
        // Se decodifica la etiqueta verdadera para comparar luego con la predicción.

        label_idx_t pred_label_idx = ff_predict_label(pixels, model_weights, model_biases);
        // Se realiza inferencia multiclase probando las 10 etiquetas y tomando la de mayor goodness.

        true_label_mem[sample_idx] = true_label_idx;
        // Se almacena la etiqueta verdadera para evaluación posterior en el testbench.

        pred_label_mem[sample_idx] = pred_label_idx;
        // Se almacena la etiqueta predicha por el acelerador entrenado.

        if (pred_label_idx == true_label_idx) {
            correct_count++;
            // Se incrementa el contador de aciertos cuando la predicción coincide con la clase real.
        }
    }

    correct_count_mem[0] = correct_count;
    // Se escribe el total de aciertos en la primera posición del buffer de salida de métricas.

    snapshot_model(model_weights, model_biases, weight_mem_out, bias_mem_out);
    // Se vuelca el estado entrenado del modelo a memoria externa para inspección desde el testbench.
}
