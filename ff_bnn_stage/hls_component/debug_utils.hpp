#ifndef DEBUG_UTILS_HPP
#define DEBUG_UTILS_HPP

#include <string>         // Se incluye std::string para titulos y mensajes de depuracion.
#include <vector>         // Se incluye std::vector para buffers del testbench.

#include "forward_fw.hpp" // Se reutilizan tipos, constantes y formatos del kernel principal.

// ============================================================
// Utilidades de depuracion para C simulation y testbench
// ============================================================
bool read_binary_file_words(const std::string &file_path, std::vector<word_t> &buffer);  // Se declara la lectura completa del binario.

void extract_sample_words(const std::vector<word_t> &buffer, int sample_idx, word_t sample_words[WORDS_PER_SAMPLE]);  // Se declara la extraccion de una muestra concreta.

void print_sample_words(const word_t sample_words[WORDS_PER_SAMPLE], const std::string &title);  // Se declara la impresion hexadecimal de una muestra.

void print_label_onehot(label_oh_t label_onehot);  // Se declara la impresion amigable del label one-hot.

void print_padding_bits(padding_t padding);  // Se declara la impresion amigable del padding.

void print_pixels_summary(pixels_t pixels, int preview_count = 64);  // Se declara la impresion resumida de la imagen binaria.

void print_unpacked_sample(raw_sample_t sample, const std::string &title);  // Se declara la impresion completa de una muestra desempaquetada.

bool same_pixels(raw_sample_t a, raw_sample_t b);  // Se declara la comparacion de pixeles entre dos muestras.

bool same_padding(raw_sample_t a, raw_sample_t b);  // Se declara la comparacion de padding entre dos muestras.

bool same_raw_sample(raw_sample_t a, raw_sample_t b);  // Se declara la comparacion bit a bit de dos muestras.

void print_separator(const std::string &title);  // Se declara la impresion de un separador visual.

void print_samples_range(const std::vector<word_t> &buffer, int start_sample, int num_samples);  // Se declara la inspeccion de un rango de muestras.

void print_training_preview(
    const std::vector<goodness_t> &g_pos,
    const std::vector<goodness_t> &g_neg,
    const std::vector<goodness_t> &gap,
    int preview_count
);  // Se declara la impresion de goodness y gap por muestra.

void print_epoch_history(
    const std::vector<loss_t> &epoch_loss_pos,
    const std::vector<loss_t> &epoch_loss_neg,
    const std::vector<goodness_t> &epoch_g_pos,
    const std::vector<goodness_t> &epoch_g_neg,
    const std::vector<goodness_t> &epoch_gap,
    int epochs_to_print
);  // Se declara la impresion del historial por epoca.

void print_epoch_terminal_update(
    int epoch_idx,
    int total_epochs,
    loss_t epoch_loss_pos,
    loss_t epoch_loss_neg,
    goodness_t epoch_g_pos,
    goodness_t epoch_g_neg,
    goodness_t epoch_gap,
    double val_accuracy,
    double elapsed_sec,
    bool has_validation
);  // Se declara la impresion incremental por epoca con formato inspirado en el notebook.

void print_prediction_preview(
    const std::vector<label_idx_t> &true_labels,
    const std::vector<label_idx_t> &pred_labels,
    int preview_count,
    int start_sample = 0
);  // Se declara la impresion de verdad vs prediccion.

void print_model_overview(
    const std::vector<latent_t> &weights,
    const std::vector<bias_t> &biases,
    int preview_weights
);  // Se declara la impresion resumida del estado entrenado del modelo.

void print_epoch_model_delta(
    int epoch_idx,
    const std::vector<latent_t> &previous_weights,
    const std::vector<latent_t> &current_weights,
    const std::vector<bias_t> &previous_biases,
    const std::vector<bias_t> &current_biases
);  // Se declara la impresion del cambio real del modelo entre dos snapshots consecutivos.

void print_epoch_freeze_justification(
    const std::vector<loss_t> &epoch_loss_pos,
    const std::vector<loss_t> &epoch_loss_neg,
    const std::vector<goodness_t> &epoch_g_pos,
    const std::vector<goodness_t> &epoch_g_neg,
    const std::vector<goodness_t> &epoch_gap,
    const std::vector<latent_t> &weights,
    const std::vector<bias_t> &biases,
    int epochs_to_check
);  // Se declara la impresion de una justificacion cuando las metricas quedan congeladas entre epocas.

#endif
