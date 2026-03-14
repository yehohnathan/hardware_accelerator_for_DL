#ifndef FORWARD_FW_HPP
#define FORWARD_FW_HPP

// ============================================================================
// forward_fw.hpp
// ----------------------------------------------------------------------------
// Este archivo define:
// - constantes globales del formato de muestra
// - tipos de datos usados en HLS
// - prototipos de funciones auxiliares
// - prototipo de la función top del kernel
// ============================================================================

// Biblioteca de enteros arbitrarios de Xilinx/Vitis HLS.
#include <ap_int.h>

// Biblioteca estándar para tipos enteros fijos como uint32_t, uint16_t.
#include <stdint.h>

// ============================================================================
// Constantes del formato de entrada
// ============================================================================

// Cantidad de clases de MNIST.
static const int NUM_CLASSES = 10;

// Cantidad de pixeles de una imagen MNIST 28x28.
static const int PIXEL_BITS = 784;

// Cantidad de bits del label one-hot.
static const int LABEL_BITS = 10;

// Cantidad de bits de padding.
// 10 + 784 + 6 = 800 bits = 25 palabras de 32 bits.
static const int PADDING_BITS = 6;

// Cantidad total de bits por muestra.
static const int SAMPLE_BITS = 800;

// Cantidad de palabras de 32 bits por muestra.
static const int WORDS_PER_SAMPLE = 25;

// Ancho de cada palabra de memoria.
static const int WORD_BITS = 32;

// ============================================================================
// Tipos HLS
// ============================================================================

// Tipo de una palabra de memoria de 32 bits.
typedef ap_uint<32> word_t;

// Tipo para representar una muestra completa de 800 bits.
typedef ap_uint<SAMPLE_BITS> sample_t;

// Tipo para representar los 784 bits de pixeles.
typedef ap_uint<PIXEL_BITS> pixels_t;

// Tipo para representar el label one-hot de 10 bits.
typedef ap_uint<LABEL_BITS> label_oh_t;

// Tipo para representar el padding de 6 bits.
typedef ap_uint<PADDING_BITS> padding_t;

// Tipo para índice de etiqueta [0..9].
typedef ap_uint<4> label_idx_t;

// Tipo para el estado del generador pseudoaleatorio LFSR.
typedef ap_uint<16> lfsr_t;

// ============================================================================
// Funciones auxiliares de empaquetado / desempaquetado
// ============================================================================

// Convierte 25 palabras de 32 bits en una muestra lógica de 800 bits.
sample_t words_to_sample(const word_t in_words[WORDS_PER_SAMPLE]);

// Convierte una muestra lógica de 800 bits a 25 palabras de 32 bits.
void sample_to_words(sample_t sample, word_t out_words[WORDS_PER_SAMPLE]);

// Extrae el label one-hot desde la muestra.
label_oh_t extract_label_onehot(sample_t sample);

// Extrae los pixeles desde la muestra.
pixels_t extract_pixels(sample_t sample);

// Extrae el padding desde la muestra.
padding_t extract_padding(sample_t sample);

// Construye una muestra desde label, pixeles y padding.
sample_t build_sample(label_oh_t label, pixels_t pixels, padding_t padding);

// Decodifica one-hot a índice.
// Ejemplo: [0,1,0,0,0,0,0,0,0,0] -> 1
label_idx_t decode_onehot(label_oh_t label);

// Codifica índice a one-hot.
// Ejemplo: 1 -> [0,1,0,0,0,0,0,0,0,0]
label_oh_t encode_onehot(label_idx_t idx);

// Avanza el LFSR de 16 bits.
lfsr_t lfsr16_next(lfsr_t state);

// Genera una etiqueta incorrecta distinta de la verdadera.
label_idx_t gen_wrong_label_excluding_true(label_idx_t true_label, lfsr_t &state);

// ============================================================================
// Función top del kernel
// ----------------------------------------------------------------------------
// in_mem     : memoria de entrada con muestras empaquetadas
// pos_mem    : memoria de salida para positivos
// neg_mem    : memoria de salida para negativos
// neg_labels : buffer opcional para depuración del label negativo generado
// n_samples  : cantidad de muestras a procesar
// seed       : semilla inicial del LFSR
// ============================================================================
void forward_fw(
    const word_t *in_mem,
    word_t *pos_mem,
    word_t *neg_mem,
    label_idx_t *neg_labels,
    int n_samples,
    uint16_t seed
);

#endif