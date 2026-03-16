#include "forward_fw.hpp"   // Se incluye el header principal con tipos, constantes y prototipos.

// ============================================================
// LFSR de 16 bits
// ============================================================

lfsr_t lfsr16_next(lfsr_t state) {
#pragma HLS INLINE
    // Se marca la función como INLINE para que HLS la integre dentro del datapath y no cree lógica separada.

    bool new_bit = state[0] ^ state[2] ^ state[3] ^ state[5];
    // Se calcula el nuevo bit realimentado usando taps sencillos del LFSR.

    lfsr_t next_state = (state >> 1);
    // Se desplaza el registro una posición hacia la derecha.

    next_state[15] = new_bit;
    // El nuevo bit se inserta en la posición más significativa.

    return next_state;
    // Se retorna el nuevo estado pseudoaleatorio.
}

// ============================================================
// Validación de one-hot
// ============================================================

bool is_valid_onehot(label_oh_t label_onehot) {
#pragma HLS INLINE
    // Se fuerza inline para mantener la lógica simple dentro del kernel.

    ap_uint<4> count_ones = 0;
    // Se usa un contador pequeño para saber cuántos bits en 1 hay.

    for (int i = 0; i < NUM_CLASSES; i++) {
#pragma HLS UNROLL
        // Se desenrolla completamente porque solo hay 10 iteraciones fijas.

        if (label_onehot[i] == 1) {
            count_ones++;
            // Se incrementa el contador por cada bit encendido.
        }
    }

    return (count_ones == 1);
    // Un label one-hot válido debe tener exactamente un bit activo.
}

// ============================================================
// one-hot -> índice
// ============================================================

label_idx_t decode_onehot(label_oh_t label_onehot) {
#pragma HLS INLINE
    // Esta función también se integra inline.

    label_idx_t label_idx = 0;
    // Se inicializa el índice en 0 como valor por defecto.

    for (int i = 0; i < NUM_CLASSES; i++) {
#pragma HLS UNROLL
        // Se desenrolla por ser un lazo pequeño y fijo.

        if (label_onehot[i] == 1) {
            label_idx = (label_idx_t)i;
            // Si el bit i está activo, entonces la clase correspondiente es i.
        }
    }

    return label_idx;
    // Se retorna el índice detectado.
}

// ============================================================
// índice -> one-hot
// ============================================================

label_oh_t encode_onehot(label_idx_t label_idx) {
#pragma HLS INLINE
    // Se fuerza inline para que sea lógica combinacional simple.

    label_oh_t label_onehot = 0;
    // Primero se limpia todo el vector.

    label_onehot[label_idx] = 1;
    // Se activa únicamente el bit correspondiente al índice dado.

    return label_onehot;
    // Se retorna el vector one-hot resultante.
}

// ============================================================
// Generación de etiqueta negativa excluyente
// ============================================================

label_idx_t generate_negative_label(label_idx_t true_label, lfsr_t &state) {
#pragma HLS INLINE
    // La función se integra inline para que forme parte del pipeline principal.

    state = lfsr16_next(state);
    // Se avanza el estado del generador pseudoaleatorio.

    ap_uint<4> offset = (state.range(3, 0) % 9) + 1;
    // Se genera un desplazamiento entre 1 y 9.
    // Nunca puede ser 0, así se evita repetir la clase original.

    ap_uint<5> temp = true_label + offset;
    // Se suma el desplazamiento a la etiqueta verdadera.

    label_idx_t negative_label = (temp >= 10) ? (label_idx_t)(temp - 10) : (label_idx_t)temp;
    // Se aplica módulo 10 manualmente.
    // Esto garantiza que el resultado siempre esté en [0, 9].

    return negative_label;
    // Se retorna una clase incorrecta y distinta de la original.
}

// ============================================================
// Carga de una muestra desde 25 words de 32 bits
// ============================================================

raw_sample_t load_sample_from_words(const word_t *mem, int sample_idx) {
#pragma HLS INLINE
    // Se integra inline para no crear sobrecosto de llamada.

    raw_sample_t sample = 0;
    // Se inicializa el contenedor de 800 bits en cero.

    int base = sample_idx * WORDS_PER_SAMPLE;
    // Se calcula el índice base donde empieza la muestra sample_idx en memoria lineal.

    for (int w = 0; w < WORDS_PER_SAMPLE; w++) {
#pragma HLS UNROLL
        // Se desenrolla el lazo porque siempre son 25 words exactas.

        sample.range((w + 1) * WORD_BITS - 1, w * WORD_BITS) = mem[base + w];
        // La palabra w se coloca en su rango correspondiente dentro de los 800 bits.
        // word 0 -> bits [31:0]
        // word 1 -> bits [63:32]
        // ...
        // word 24 -> bits [799:768]
    }

    return sample;
    // Se retorna la muestra ya armada como vector de 800 bits.
}

// ============================================================
// Almacenamiento de una muestra en 25 words de 32 bits
// ============================================================

void store_sample_to_words(word_t *mem, int sample_idx, raw_sample_t sample) {
#pragma HLS INLINE
    // Se fuerza inline para simplificar el datapath.

    int base = sample_idx * WORDS_PER_SAMPLE;
    // Se calcula la posición base donde se escribirá la muestra sample_idx.

    for (int w = 0; w < WORDS_PER_SAMPLE; w++) {
#pragma HLS UNROLL
        // Se desenrolla completamente porque el número de words es fijo.

        mem[base + w] = sample.range((w + 1) * WORD_BITS - 1, w * WORD_BITS);
        // Se extrae cada bloque de 32 bits del vector completo y se escribe en memoria.
    }
}

// ============================================================
// Desempaquetado de la muestra
// Layout confirmado:
//   label_onehot = bits [9:0]
//   pixels       = bits [793:10]
//   padding      = bits [799:794]
// ============================================================

void unpack_sample(
    raw_sample_t sample,
    label_oh_t &label_onehot,
    pixels_t &pixels,
    padding_t &padding
) {
#pragma HLS INLINE
    // La función se integra inline porque es una mera selección de rangos.

    label_onehot = sample.range(9, 0);
    // Se extraen los 10 bits menos significativos para el label one-hot.

    pixels = sample.range(793, 10);
    // Se extraen los 784 bits centrales correspondientes a la imagen binaria.

    padding = sample.range(799, 794);
    // Se extraen los 6 bits más significativos de padding.
}

// ============================================================
// Empaquetado de la muestra
// ============================================================

raw_sample_t pack_sample(
    label_oh_t label_onehot,
    pixels_t pixels,
    padding_t padding
) {
#pragma HLS INLINE
    // Se fuerza inline porque es una operación fija y pequeña.

    raw_sample_t sample = 0;
    // Se inicializa el contenedor completo en cero.

    sample.range(9, 0) = label_onehot;
    // Se coloca el label one-hot en los bits [9:0].

    sample.range(793, 10) = pixels;
    // Se coloca la imagen binaria en los bits [793:10].

    sample.range(799, 794) = padding;
    // Se colocan los 6 bits de padding en la parte más significativa.

    return sample;
    // Se retorna la muestra completa ya reempaquetada.
}

// ============================================================
// Top function sintetizable
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
    // Puerto AXI maestro de lectura para la memoria de entrada.

#pragma HLS INTERFACE m_axi     port=pos_mem        offset=slave bundle=gmem1
    // Puerto AXI maestro de escritura para las muestras positivas.

#pragma HLS INTERFACE m_axi     port=neg_mem        offset=slave bundle=gmem2
    // Puerto AXI maestro de escritura para las muestras negativas.

#pragma HLS INTERFACE m_axi     port=true_label_mem offset=slave bundle=gmem3
    // Puerto AXI maestro de escritura para guardar labels verdaderos decodificados.

#pragma HLS INTERFACE m_axi     port=neg_label_mem  offset=slave bundle=gmem4
    // Puerto AXI maestro de escritura para guardar labels negativos decodificados.

#pragma HLS INTERFACE s_axilite port=in_mem         bundle=control
    // Puerto de control AXI-Lite asociado al puntero de entrada.

#pragma HLS INTERFACE s_axilite port=pos_mem        bundle=control
    // Puerto de control AXI-Lite asociado al puntero de salida positiva.

#pragma HLS INTERFACE s_axilite port=neg_mem        bundle=control
    // Puerto de control AXI-Lite asociado al puntero de salida negativa.

#pragma HLS INTERFACE s_axilite port=true_label_mem bundle=control
    // Puerto de control AXI-Lite para el buffer de labels verdaderos.

#pragma HLS INTERFACE s_axilite port=neg_label_mem  bundle=control
    // Puerto de control AXI-Lite para el buffer de labels negativos.

#pragma HLS INTERFACE s_axilite port=n_samples      bundle=control
    // Puerto de control AXI-Lite para la cantidad de muestras a procesar.

#pragma HLS INTERFACE s_axilite port=seed           bundle=control
    // Puerto de control AXI-Lite para la semilla inicial del LFSR.

#pragma HLS INTERFACE s_axilite port=return         bundle=control
    // Puerto AXI-Lite de retorno obligatorio para el top function.

    lfsr_t prng_state = (seed == 0) ? (lfsr_t)0xACE1 : (lfsr_t)seed;
    // Se inicializa el estado pseudoaleatorio.
    // Si la semilla vale 0, se usa una semilla fija no nula para evitar atascar el LFSR.

sample_loop:
    for (int i = 0; i < n_samples; i++) {
#pragma HLS PIPELINE II=1
        // Se pipelinea el procesamiento por muestra para mejorar throughput.

        raw_sample_t input_sample = load_sample_from_words(in_mem, i);
        // Se lee la muestra i desde memoria externa y se arma en 800 bits.

        label_oh_t input_label_onehot;
        // Variable para almacenar el one-hot original.

        pixels_t input_pixels;
        // Variable para almacenar la imagen binaria.

        padding_t input_padding;
        // Variable para almacenar el padding.

        unpack_sample(input_sample, input_label_onehot, input_pixels, input_padding);
        // Se separan los campos de la muestra de entrada.

        label_idx_t true_label_idx = decode_onehot(input_label_onehot);
        // Se decodifica el one-hot original a índice entero.

        label_idx_t neg_label_idx = generate_negative_label(true_label_idx, prng_state);
        // Se genera una etiqueta negativa garantizada distinta de la verdadera.

        label_oh_t neg_label_onehot = encode_onehot(neg_label_idx);
        // Se convierte la etiqueta negativa a formato one-hot.

        raw_sample_t positive_sample = pack_sample(input_label_onehot, input_pixels, input_padding);
        // La muestra positiva conserva el label original, los mismos pixels y el mismo padding.

        raw_sample_t negative_sample = pack_sample(neg_label_onehot, input_pixels, input_padding);
        // La muestra negativa solo cambia el label; pixels y padding se conservan.

        store_sample_to_words(pos_mem, i, positive_sample);
        // Se escribe la muestra positiva en la memoria de salida correspondiente.

        store_sample_to_words(neg_mem, i, negative_sample);
        // Se escribe la muestra negativa en la memoria de salida correspondiente.

        true_label_mem[i] = true_label_idx;
        // Se guarda el índice del label verdadero para depuración o verificación posterior.

        neg_label_mem[i] = neg_label_idx;
        // Se guarda el índice del label negativo generado.
    }
}