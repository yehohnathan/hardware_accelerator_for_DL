#ifndef DEBUG_UTILS_HPP
#define DEBUG_UTILS_HPP

// ============================================================================
// debug_utils.hpp
// ----------------------------------------------------------------------------
// Utilidades para imprimir y verificar el contenido de las muestras en el
// testbench. Estas funciones NO forman parte del kernel HLS.
// ============================================================================

#include "forward_fw.hpp"

// Biblioteca de C++ para salida por consola.
#include <iostream>

// Biblioteca para formateo.
#include <iomanip>

// Biblioteca para cadenas.
#include <string>

// Imprime un label one-hot de 10 bits.
void print_label_onehot(label_oh_t label, const std::string &name);

// Imprime un índice de etiqueta.
void print_label_index(label_idx_t idx, const std::string &name);

// Imprime el padding de 6 bits.
void print_padding_bits(padding_t padding, const std::string &name);

// Imprime una cantidad limitada de pixeles para inspección.
void print_pixels_preview(pixels_t pixels, const std::string &name, int count);

// Imprime las 25 palabras de una muestra.
void print_words_25x32(const word_t words[WORDS_PER_SAMPLE], const std::string &name);

// Verifica si dos bloques de 25 palabras son iguales.
bool compare_words_25x32(const word_t a[WORDS_PER_SAMPLE], const word_t b[WORDS_PER_SAMPLE]);

#endif