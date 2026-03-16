#ifndef DEBUG_UTILS_HPP
#define DEBUG_UTILS_HPP

#include <vector>         // Se usa std::vector para almacenar el contenido del binario en el testbench.
#include <string>         // Se usa std::string para nombres y mensajes de debug.
#include "forward_fw.hpp" // Se reutilizan los tipos y constantes definidos para el kernel.

// ============================================================
// Utilidades de depuración para C simulation / testbench
// ============================================================

// Lee un archivo binario completo como words de 32 bits.
bool read_binary_file_words(const std::string &file_path, std::vector<word_t> &buffer);

// Extrae una muestra concreta (25 words) desde un vector lineal.
void extract_sample_words(
    const std::vector<word_t> &buffer,
    int sample_idx,
    word_t sample_words[WORDS_PER_SAMPLE]
);

// Imprime las 25 palabras de una muestra en hexadecimal.
void print_sample_words(
    const word_t sample_words[WORDS_PER_SAMPLE],
    const std::string &title
);

// Imprime el label one-hot como vector de 10 posiciones.
void print_label_onehot(label_oh_t label_onehot);

// Imprime el padding de 6 bits.
void print_padding_bits(padding_t padding);

// Imprime un resumen de pixels: cantidad de unos y una ventana inicial de bits.
void print_pixels_summary(pixels_t pixels, int preview_count = 64);

// Imprime toda la información desempaquetada de una muestra.
void print_unpacked_sample(
    raw_sample_t sample,
    const std::string &title
);

// Compara si dos muestras tienen exactamente los mismos pixels.
bool same_pixels(raw_sample_t a, raw_sample_t b);

// Compara si dos muestras tienen exactamente el mismo padding.
bool same_padding(raw_sample_t a, raw_sample_t b);

// Compara si dos muestras son idénticas bit a bit.
bool same_raw_sample(raw_sample_t a, raw_sample_t b);

// Imprime un separador visual en consola.
void print_separator(const std::string &title);

// Mostrar un rango de muestras del binario
void print_samples_range(
    const std::vector<word_t> &buffer,
    int start_sample,
    int num_samples
);

#endif