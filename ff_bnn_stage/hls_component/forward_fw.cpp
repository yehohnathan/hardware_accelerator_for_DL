// ============================================================================
// forward_fw.cpp
// ----------------------------------------------------------------------------
// Implementación del kernel principal y de todas las funciones auxiliares.
// Todo se mantiene explícito y legible para facilitar depuración.
// ============================================================================

#include "forward_fw.hpp"

// ============================================================================
// Convierte 25 palabras de 32 bits en una muestra lógica de 800 bits.
// ----------------------------------------------------------------------------
// Convención asumida:
// - la palabra 0 ocupa los bits [31:0]
// - la palabra 1 ocupa los bits [63:32]
// - ...
// - la palabra 24 ocupa los bits [799:768]
//
// Si al depurar notas que el orden real del archivo binario está invertido,
// solo debes modificar esta función y la función sample_to_words().
// ============================================================================
sample_t words_to_sample(const word_t in_words[WORDS_PER_SAMPLE]) {
    // Variable donde se almacenará la muestra completa de 800 bits.
    sample_t sample = 0;

    // Recorre las 25 palabras de 32 bits.
    for (int i = 0; i < WORDS_PER_SAMPLE; i++) {
        // Permite que el compilador expanda completamente este bucle.
#pragma HLS UNROLL

        // Calcula el bit menos significativo del segmento actual.
        const int lo = i * WORD_BITS;

        // Calcula el bit más significativo del segmento actual.
        const int hi = lo + WORD_BITS - 1;

        // Copia la palabra i dentro del rango [hi:lo] de la muestra.
        sample.range(hi, lo) = in_words[i];
    }

    // Retorna la muestra de 800 bits.
    return sample;
}

// ============================================================================
// Convierte una muestra lógica de 800 bits a 25 palabras de 32 bits.
// ----------------------------------------------------------------------------
// Usa la misma convención de orden que words_to_sample().
// ============================================================================
void sample_to_words(sample_t sample, word_t out_words[WORDS_PER_SAMPLE]) {
    // Recorre las 25 palabras destino.
    for (int i = 0; i < WORDS_PER_SAMPLE; i++) {
        // Permite expansión total del bucle para simplificar hardware.
#pragma HLS UNROLL

        // Calcula el bit menos significativo del segmento actual.
        const int lo = i * WORD_BITS;

        // Calcula el bit más significativo del segmento actual.
        const int hi = lo + WORD_BITS - 1;

        // Extrae el rango [hi:lo] y lo guarda como palabra de 32 bits.
        out_words[i] = sample.range(hi, lo);
    }
}

// ============================================================================
// Extrae el label one-hot desde la muestra.
// ----------------------------------------------------------------------------
// Layout asumido:
// - bits [799:790] = label one-hot
// ============================================================================
label_oh_t extract_label_onehot(sample_t sample) {
    // Extrae los 10 bits superiores de la muestra.
    return sample.range(SAMPLE_BITS - 1, SAMPLE_BITS - LABEL_BITS);
}

// ============================================================================
// Extrae los pixeles desde la muestra.
// ----------------------------------------------------------------------------
// Layout asumido:
// - bits [789:6] = 784 bits de pixeles
// ============================================================================
pixels_t extract_pixels(sample_t sample) {
    // Bit superior de los pixeles, justo debajo del label.
    const int hi = SAMPLE_BITS - LABEL_BITS - 1;

    // Bit inferior de los pixeles, justo encima del padding.
    const int lo = PADDING_BITS;

    // Extrae el rango de pixeles.
    return sample.range(hi, lo);
}

// ============================================================================
// Extrae el padding desde la muestra.
// ----------------------------------------------------------------------------
// Layout asumido:
// - bits [5:0] = padding
// ============================================================================
padding_t extract_padding(sample_t sample) {
    // Extrae los 6 bits menos significativos.
    return sample.range(PADDING_BITS - 1, 0);
}

// ============================================================================
// Construye una muestra desde label, pixeles y padding.
// ----------------------------------------------------------------------------
// Layout resultante:
// - bits [799:790] = label
// - bits [789:6]   = pixeles
// - bits [5:0]     = padding
// ============================================================================
sample_t build_sample(label_oh_t label, pixels_t pixels, padding_t padding) {
    // Crea la muestra de salida inicializada en cero.
    sample_t sample = 0;

    // Inserta el padding en los bits bajos.
    sample.range(PADDING_BITS - 1, 0) = padding;

    // Inserta los pixeles encima del padding.
    sample.range(SAMPLE_BITS - LABEL_BITS - 1, PADDING_BITS) = pixels;

    // Inserta el label en los bits más altos.
    sample.range(SAMPLE_BITS - 1, SAMPLE_BITS - LABEL_BITS) = label;

    // Retorna la muestra armada.
    return sample;
}

// ============================================================================
// Decodifica un label one-hot a índice.
// ----------------------------------------------------------------------------
// Ejemplo:
// 0000000010 -> 1
//
// Si hubiese más de un bit en 1, se conserva el último detectado.
// En un dataset correcto solo debe existir un bit en 1.
// ============================================================================
label_idx_t decode_onehot(label_oh_t label) {
    // Índice por defecto.
    label_idx_t idx = 0;

    // Recorre los 10 bits del one-hot.
    for (int i = 0; i < NUM_CLASSES; i++) {
        // Pide al compilador expandir completamente este bucle.
#pragma HLS UNROLL

        // Si el bit i está en 1, entonces el índice es i.
        if (label[i] == 1) {
            idx = (label_idx_t)i;
        }
    }

    // Retorna el índice decodificado.
    return idx;
}

// ============================================================================
// Codifica un índice [0..9] a one-hot de 10 bits.
// ----------------------------------------------------------------------------
// Ejemplo:
// 1 -> 0000000010
// ============================================================================
label_oh_t encode_onehot(label_idx_t idx) {
    // Inicializa el vector one-hot en cero.
    label_oh_t label = 0;

    // Activa únicamente el bit correspondiente al índice.
    label[idx] = 1;

    // Retorna el label one-hot.
    return label;
}

// ============================================================================
// LFSR de 16 bits.
// ----------------------------------------------------------------------------
// Polinomio simple para pseudoaleatoriedad barata en hardware.
// ============================================================================
lfsr_t lfsr16_next(lfsr_t state) {
    // Calcula el nuevo bit usando taps del registro actual.
    bool new_bit = state[0] ^ state[2] ^ state[3] ^ state[5];

    // Desplaza el registro hacia la derecha.
    lfsr_t next = (state >> 1);

    // Inserta el nuevo bit en la posición más alta.
    next[15] = new_bit;

    // Retorna el siguiente estado.
    return next;
}

// ============================================================================
// Genera una etiqueta incorrecta distinta de la verdadera.
// ----------------------------------------------------------------------------
// Estrategia:
// - genera un offset entre 1 y 9
// - calcula (true_label + offset) mod 10
//
// Esto garantiza matemáticamente que wrong_label != true_label.
// ============================================================================
label_idx_t gen_wrong_label_excluding_true(label_idx_t true_label, lfsr_t &state) {
    // Avanza el estado del generador pseudoaleatorio.
    state = lfsr16_next(state);

    // Toma 4 bits del LFSR, calcula módulo 9 y suma 1.
    // Resultado garantizado en el rango [1..9].
    ap_uint<4> offset = (state.range(3, 0) % 9) + 1;

    // Suma temporal con 5 bits para evitar desbordamiento.
    ap_uint<5> tmp = true_label + offset;

    // Aplica módulo 10 manualmente para mantenerlo simple.
    label_idx_t wrong_label = (tmp >= 10) ? (label_idx_t)(tmp - 10)
                                          : (label_idx_t)tmp;

    // Retorna una etiqueta que nunca coincide con la verdadera.
    return wrong_label;
}

// ============================================================================
// Función top del kernel.
// ----------------------------------------------------------------------------
// Por cada muestra:
// 1. la lee desde memoria
// 2. la desempaqueta
// 3. preserva el positivo
// 4. genera un label negativo distinto
// 5. reempaqueta y escribe ambas salidas
// ============================================================================
void forward_fw(
    const word_t *in_mem,
    word_t *pos_mem,
    word_t *neg_mem,
    label_idx_t *neg_labels,
    int n_samples,
    uint16_t seed
) {
    // Interfaz AXI master para memoria de entrada.
#pragma HLS INTERFACE m_axi port=in_mem offset=slave bundle=gmem0

    // Interfaz AXI master para memoria de salida positiva.
#pragma HLS INTERFACE m_axi port=pos_mem offset=slave bundle=gmem1

    // Interfaz AXI master para memoria de salida negativa.
#pragma HLS INTERFACE m_axi port=neg_mem offset=slave bundle=gmem2

    // Interfaz AXI master para labels negativos de depuración.
#pragma HLS INTERFACE m_axi port=neg_labels offset=slave bundle=gmem3

    // Interfaces AXI-Lite de control.
#pragma HLS INTERFACE s_axilite port=in_mem bundle=control
#pragma HLS INTERFACE s_axilite port=pos_mem bundle=control
#pragma HLS INTERFACE s_axilite port=neg_mem bundle=control
#pragma HLS INTERFACE s_axilite port=neg_labels bundle=control
#pragma HLS INTERFACE s_axilite port=n_samples bundle=control
#pragma HLS INTERFACE s_axilite port=seed bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Estado del LFSR.
    // Si seed es cero, se fuerza un valor no nulo.
    lfsr_t lfsr_state = (seed == 0) ? (lfsr_t)0xACE1 : (lfsr_t)seed;

    // Buffer local para leer una muestra completa de 25 palabras.
    word_t in_words[WORDS_PER_SAMPLE];

    // Buffer local para la salida positiva.
    word_t pos_words[WORDS_PER_SAMPLE];

    // Buffer local para la salida negativa.
    word_t neg_words[WORDS_PER_SAMPLE];

    // Loop principal por muestra.
    sample_loop:
    for (int s = 0; s < n_samples; s++) {
        // Pipelining por muestra.
#pragma HLS PIPELINE II=1

        // --------------------------------------------------------------------
        // 1. Leer 25 palabras desde memoria externa.
        // --------------------------------------------------------------------
        read_loop:
        for (int w = 0; w < WORDS_PER_SAMPLE; w++) {
#pragma HLS UNROLL
            in_words[w] = in_mem[s * WORDS_PER_SAMPLE + w];
        }

        // --------------------------------------------------------------------
        // 2. Convertir las 25 palabras a una muestra lógica de 800 bits.
        // --------------------------------------------------------------------
        sample_t in_sample = words_to_sample(in_words);

        // --------------------------------------------------------------------
        // 3. Desempaquetar componentes.
        // --------------------------------------------------------------------
        label_oh_t true_label_oh = extract_label_onehot(in_sample);
        pixels_t pixels = extract_pixels(in_sample);
        padding_t padding = extract_padding(in_sample);

        // --------------------------------------------------------------------
        // 4. Decodificar el label verdadero.
        // --------------------------------------------------------------------
        label_idx_t true_label_idx = decode_onehot(true_label_oh);

        // --------------------------------------------------------------------
        // 5. Generar etiqueta incorrecta distinta de la real.
        // --------------------------------------------------------------------
        label_idx_t wrong_label_idx = gen_wrong_label_excluding_true(true_label_idx, lfsr_state);

        // --------------------------------------------------------------------
        // 6. Codificar la etiqueta incorrecta a one-hot.
        // --------------------------------------------------------------------
        label_oh_t wrong_label_oh = encode_onehot(wrong_label_idx);

        // --------------------------------------------------------------------
        // 7. Construir muestra positiva.
        //    Conserva exactamente label, pixeles y padding originales.
        // --------------------------------------------------------------------
        sample_t pos_sample = build_sample(true_label_oh, pixels, padding);

        // --------------------------------------------------------------------
        // 8. Construir muestra negativa.
        //    Conserva pixeles y padding, reemplaza únicamente el label.
        // --------------------------------------------------------------------
        sample_t neg_sample = build_sample(wrong_label_oh, pixels, padding);

        // --------------------------------------------------------------------
        // 9. Convertir ambas muestras a 25 palabras de 32 bits.
        // --------------------------------------------------------------------
        sample_to_words(pos_sample, pos_words);
        sample_to_words(neg_sample, neg_words);

        // --------------------------------------------------------------------
        // 10. Escribir resultados en memoria externa.
        // --------------------------------------------------------------------
        write_loop:
        for (int w = 0; w < WORDS_PER_SAMPLE; w++) {
#pragma HLS UNROLL
            pos_mem[s * WORDS_PER_SAMPLE + w] = pos_words[w];
            neg_mem[s * WORDS_PER_SAMPLE + w] = neg_words[w];
        }

        // --------------------------------------------------------------------
        // 11. Guardar el índice de la etiqueta negativa para depuración.
        // --------------------------------------------------------------------
        neg_labels[s] = wrong_label_idx;
    }
}