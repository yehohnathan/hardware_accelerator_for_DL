#ifndef FORWARD_FW_HPP
#define FORWARD_FW_HPP

#include <ap_int.h>   // Se incluyen enteros arbitrarios de HLS, necesarios para manejar 800 bits, 784 bits, etc.
#include <stdint.h>   // Se incluyen tipos estándar como uint16_t para la semilla del LFSR.

// ============================================================
// Constantes del formato de muestra
// ============================================================

static const int NUM_CLASSES      = 10;   // MNIST tiene 10 clases: 0,1,2,3,4,5,6,7,8,9.
static const int WORD_BITS        = 32;   // Cada palabra del binario tiene 32 bits.
static const int WORDS_PER_SAMPLE = 25;   // Cada muestra ocupa 25 palabras.
static const int TOTAL_BITS       = 800;  // 25 * 32 = 800 bits totales por muestra.

static const int LABEL_BITS       = 10;   // El label one-hot usa 10 bits.
static const int PIXEL_BITS       = 784;  // La imagen binaria de MNIST ocupa 28x28 = 784 bits.
static const int PADDING_BITS     = 6;    // Sobran 6 bits de padding: 800 - (10 + 784) = 6.

// ============================================================
// Tipos base del diseño
// ============================================================

typedef ap_uint<WORD_BITS>  word_t;         // Tipo para cada word individual de 32 bits.
typedef ap_uint<TOTAL_BITS> raw_sample_t;   // Tipo para la muestra completa de 800 bits.
typedef ap_uint<LABEL_BITS> label_oh_t;     // Tipo para el label en formato one-hot de 10 bits.
typedef ap_uint<PIXEL_BITS> pixels_t;       // Tipo para la imagen binaria de 784 bits.
typedef ap_uint<PADDING_BITS> padding_t;    // Tipo para el padding de 6 bits.
typedef ap_uint<4> label_idx_t;             // 4 bits bastan para representar índices de 0 a 9.
typedef ap_uint<16> lfsr_t;                 // Estado del generador pseudoaleatorio LFSR de 16 bits.

// ============================================================
// Prototipos de funciones auxiliares sintetizables
// ============================================================

// Avanza el estado del LFSR una vez.
lfsr_t lfsr16_next(lfsr_t state);

// Verifica si un label one-hot tiene exactamente un bit encendido.
bool is_valid_onehot(label_oh_t label_onehot);

// Convierte un label one-hot a índice entero.
// Ejemplo: [0,1,0,0,0,0,0,0,0,0] -> 1
label_idx_t decode_onehot(label_oh_t label_onehot);

// Convierte un índice de clase a one-hot.
// Ejemplo: 1 -> [0,1,0,0,0,0,0,0,0,0]
label_oh_t encode_onehot(label_idx_t label_idx);

// Genera una etiqueta incorrecta distinta de la verdadera.
label_idx_t generate_negative_label(label_idx_t true_label, lfsr_t &state);

// Carga una muestra completa desde memoria lineal (25 words de 32 bits) y la arma como vector de 800 bits.
raw_sample_t load_sample_from_words(const word_t *mem, int sample_idx);

// Guarda una muestra completa de 800 bits en memoria lineal (25 words de 32 bits).
void store_sample_to_words(word_t *mem, int sample_idx, raw_sample_t sample);

// Extrae label, pixels y padding desde la muestra empaquetada.
void unpack_sample(
    raw_sample_t sample,
    label_oh_t &label_onehot,
    pixels_t &pixels,
    padding_t &padding
);

// Empaqueta label, pixels y padding en una sola muestra de 800 bits.
raw_sample_t pack_sample(
    label_oh_t label_onehot,
    pixels_t pixels,
    padding_t padding
);

// ============================================================
// Top function sintetizable para Vitis HLS
// ============================================================

// Esta función:
// 1. Lee muestras desde memoria.
// 2. Conserva una versión positiva idéntica.
// 3. Genera una versión negativa cambiando solo el label.
// 4. Guarda true_label y neg_label como índices para debug.
void forward_fw_top(
    const word_t *in_mem,
    word_t *pos_mem,
    word_t *neg_mem,
    label_idx_t *true_label_mem,
    label_idx_t *neg_label_mem,
    int n_samples,
    uint16_t seed
);

#endif