#ifndef DEBUG_UTILS_HPP
#define DEBUG_UTILS_HPP

#include <vector>         // Se incluye std::vector para los buffers del testbench y las utilidades de inspección.
#include <string>         // Se incluye std::string para títulos y mensajes de depuración.
#include "forward_fw.hpp" // Se reutilizan tipos, constantes y prototipos del kernel principal.

// ============================================================
// Utilidades de depuración para C simulation y testbench
// ============================================================
// Se declara la lectura completa del binario como words de 32 bits.
bool read_binary_file_words(const std::string &file_path,
							std::vector<word_t> &buffer);

// Se declara la extracción de una muestra concreta desde un buffer lineal.
void extract_sample_words(const std::vector<word_t> &buffer,
						  int sample_idx, word_t sample_words[WORDS_PER_SAMPLE]);

// Se declara la impresión hexadecimal de una muestra cruda.
void print_sample_words(const word_t sample_words[WORDS_PER_SAMPLE],
						const std::string &title);

// Se declara la impresión amigable de un label one-hot.
void print_label_onehot(label_oh_t label_onehot);

// Se declara la impresión amigable del padding físico.
void print_padding_bits(padding_t padding);

// Se declara la impresión resumida de la imagen binaria.
void print_pixels_summary(pixels_t pixels, int preview_count = 64);

// Se declara la impresión completa de una muestra desempaquetada.
void print_unpacked_sample(raw_sample_t sample, const std::string &title);

// Se declara la comparación de pixels entre dos muestras.
bool same_pixels(raw_sample_t a, raw_sample_t b);

// Se declara la comparación de padding entre dos muestras.
bool same_padding(raw_sample_t a, raw_sample_t b);

// Se declara la comparación bit a bit de dos muestras completas.
bool same_raw_sample(raw_sample_t a, raw_sample_t b);

// Se declara la impresión de un separador visual en consola.
void print_separator(const std::string &title);

// Se declara la inspección de un rango de muestras desempaquetadas.
void print_samples_range(const std::vector<word_t> &buffer,
						 int start_sample, int num_samples);

// Se declara la impresión de métricas de entrenamiento FF.
void print_training_preview(const std::vector<goodness_t> &g_pos,
							const std::vector<goodness_t> &g_neg,
							const std::vector<goodness_t> &gap,
							int preview_count);

// Se declara la impresión de una vista rápida de predicciones y clases reales.
void print_prediction_preview(const std::vector<label_idx_t> &true_labels,
							  const std::vector<label_idx_t> &pred_labels,
							  int preview_count);

// Se declara la impresión resumida del estado entrenado del modelo.
void print_model_overview(const std::vector<latent_t> &weights, 
						  const std::vector<bias_t> &biases,
						  int preview_weights);

#endif
