#ifndef DEBUG_UTILS_HPP
#define DEBUG_UTILS_HPP

#include <string>
#include <vector>

#include "forward_fw.hpp"

/**
 * @brief Describe la cabecera host del binario generado por Python.
 *
 * @note Esta estructura solo vive en el testbench host. El kernel HLS sigue
 * recibiendo exclusivamente el payload de words, sin la cabecera.
 */
struct ff_binary_header_t {
    uint32_t magic;
    uint32_t version;
    uint32_t header_words;
    uint32_t sample_count;
    uint32_t image_width;
    uint32_t image_height;
    uint32_t source_width;
    uint32_t source_height;
    uint32_t pixel_bits;
    uint32_t num_classes;
    uint32_t label_bits;
    uint32_t useful_bits;
    uint32_t words_per_sample;
    uint32_t total_bits;
    uint32_t padding_bits;
    uint32_t resize_applied;
    uint32_t num_pixels;
    uint32_t payload_words;
    std::string suffix;
    std::string mode_name;
};

/**
 * @brief Lee un binario con cabecera y devuelve solo el payload de muestras.
 *
 * @param file_path Indica la ruta del binario que se desea abrir.
 * @param header Retorna la cabecera reconstruida a partir del archivo.
 * @param payload_words Retorna el payload lineal listo para el kernel HLS.
 *
 * @return Retorna true cuando la lectura y la cabecera son validas.
 */
bool read_binary_dataset(
    const std::string &file_path,
    ff_binary_header_t &header,
    std::vector<word_t> &payload_words
);

/**
 * @brief Comprueba si la cabecera del binario coincide con la configuracion.
 *
 * @param header Contiene la cabecera reconstruida desde el archivo.
 *
 * @return Retorna true cuando la cabecera coincide con la configuracion
 * compilada del kernel y del testbench.
 */
bool validate_binary_header_against_build(const ff_binary_header_t &header);

/**
 * @brief Imprime el resumen de la cabecera y del payload del binario cargado.
 *
 * @param file_path Indica la ruta del binario cargado.
 * @param header Contiene la cabecera reconstruida desde el archivo.
 * @param payload_words Contiene el payload ya separado de la cabecera.
 */
void print_binary_header_summary(
    const std::string &file_path,
    const ff_binary_header_t &header,
    const std::vector<word_t> &payload_words
);

/**
 * @brief Extrae las words de una muestra concreta desde el payload lineal.
 *
 * @param buffer Contiene el payload completo del dataset ya cargado.
 * @param sample_idx Indica la muestra que se desea extraer.
 * @param sample_words Retorna las words correspondientes a la muestra.
 */
void extract_sample_words(
    const std::vector<word_t> &buffer,
    int sample_idx,
    word_t sample_words[WORDS_PER_SAMPLE]
);

/**
 * @brief Imprime en hexadecimal las words crudas de una muestra.
 *
 * @param sample_words Contiene las words de la muestra seleccionada.
 * @param title Indica el titulo usado para la impresion.
 */
void print_sample_words(
    const word_t sample_words[WORDS_PER_SAMPLE],
    const std::string &title
);

/**
 * @brief Imprime la etiqueta one-hot de una muestra de forma legible.
 *
 * @param label_onehot Contiene la etiqueta en formato one-hot.
 */
void print_label_onehot(label_oh_t label_onehot);

/**
 * @brief Imprime el padding fisico de una muestra.
 *
 * @param padding Contiene el padding fisico almacenado en la muestra.
 */
void print_padding_bits(padding_t padding);

/**
 * @brief Resume e imprime la imagen contenida dentro de pixels_t.
 *
 * @param pixels Contiene el bloque de pixeles empaquetados de la muestra.
 * @param preview_count Indica cuantos pixeles se imprimen como maximo.
 */
void print_pixels_summary(pixels_t pixels, int preview_count = -1);

/**
 * @brief Imprime una muestra completa tras desempaquetarla.
 *
 * @param sample Contiene la muestra fisica empaquetada.
 * @param title Indica el titulo usado para la impresion.
 */
void print_unpacked_sample(raw_sample_t sample, const std::string &title);

/**
 * @brief Compara si dos muestras comparten exactamente los mismos pixeles.
 *
 * @param a Contiene la primera muestra fisica.
 * @param b Contiene la segunda muestra fisica.
 *
 * @return Retorna true cuando ambas muestras comparten los mismos pixeles.
 */
bool same_pixels(raw_sample_t a, raw_sample_t b);

/**
 * @brief Compara si dos muestras comparten exactamente el mismo padding.
 *
 * @param a Contiene la primera muestra fisica.
 * @param b Contiene la segunda muestra fisica.
 *
 * @return Retorna true cuando ambas muestras comparten el mismo padding.
 */
bool same_padding(raw_sample_t a, raw_sample_t b);

/**
 * @brief Compara si dos muestras fisicas son iguales bit a bit.
 *
 * @param a Contiene la primera muestra fisica.
 * @param b Contiene la segunda muestra fisica.
 *
 * @return Retorna true cuando ambas muestras son identicas.
 */
bool same_raw_sample(raw_sample_t a, raw_sample_t b);

/**
 * @brief Imprime un separador visual reutilizable dentro del testbench.
 *
 * @param title Indica el titulo de la seccion que se desea abrir.
 */
void print_separator(const std::string &title);

/**
 * @brief Imprime un rango de muestras ya desempaquetadas.
 *
 * @param buffer Contiene el payload completo del dataset ya cargado.
 * @param start_sample Indica el primer indice de muestra a inspeccionar.
 * @param num_samples Indica cuantas muestras consecutivas se imprimen.
 */
void print_samples_range(
    const std::vector<word_t> &buffer,
    int start_sample,
    int num_samples
);

/**
 * @brief Imprime una vista previa de las metricas por muestra.
 *
 * @param g_pos Contiene la goodness positiva por muestra.
 * @param g_neg Contiene la goodness negativa por muestra.
 * @param gap Contiene la separacion positiva-negativa por muestra.
 * @param preview_count Indica cuantas muestras se desean mostrar.
 */
void print_training_preview(
    const std::vector<goodness_t> &g_pos,
    const std::vector<goodness_t> &g_neg,
    const std::vector<goodness_t> &gap,
    int preview_count
);

/**
 * @brief Imprime el historial medio por epoca del entrenamiento FF.
 *
 * @param epoch_loss_pos Contiene la perdida positiva por epoca.
 * @param epoch_loss_neg Contiene la perdida negativa por epoca.
 * @param epoch_g_pos Contiene la goodness positiva por epoca.
 * @param epoch_g_neg Contiene la goodness negativa por epoca.
 * @param epoch_gap Contiene el gap positivo-negativo por epoca.
 * @param epochs_to_print Indica cuantas epocas se desean mostrar.
 */
void print_epoch_history(
    const std::vector<loss_t> &epoch_loss_pos,
    const std::vector<loss_t> &epoch_loss_neg,
    const std::vector<goodness_t> &epoch_g_pos,
    const std::vector<goodness_t> &epoch_g_neg,
    const std::vector<goodness_t> &epoch_gap,
    int epochs_to_print
);

/**
 * @brief Imprime la actualizacion incremental de una epoca de entrenamiento.
 *
 * @param epoch_idx Indica el indice de la epoca actual.
 * @param total_epochs Indica la cantidad total de epocas del experimento.
 * @param epoch_loss_pos Contiene la perdida positiva de la epoca.
 * @param epoch_loss_neg Contiene la perdida negativa de la epoca.
 * @param epoch_g_pos Contiene la goodness positiva de la epoca.
 * @param epoch_g_neg Contiene la goodness negativa de la epoca.
 * @param epoch_gap Contiene el gap de la epoca.
 * @param val_accuracy Indica la accuracy de evaluacion de la epoca.
 * @param elapsed_sec Indica la duracion total de la epoca en segundos.
 * @param has_validation Indica si existe fase de evaluacion asociada.
 */
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
);

/**
 * @brief Imprime una vista previa de etiquetas reales y predichas.
 *
 * @param true_labels Contiene las etiquetas verdaderas del subconjunto.
 * @param pred_labels Contiene las predicciones del subconjunto.
 * @param preview_count Indica cuantas muestras se desean mostrar.
 * @param start_sample Indica el offset absoluto del subconjunto evaluado.
 */
void print_prediction_preview(
    const std::vector<label_idx_t> &true_labels,
    const std::vector<label_idx_t> &pred_labels,
    int preview_count,
    int start_sample = 0
);

/**
 * @brief Imprime un resumen del snapshot final del modelo entrenado.
 *
 * @param weights Contiene los pesos latentes del modelo.
 * @param biases Contiene los bias latentes del modelo.
 * @param preview_weights Indica cuantas posiciones se muestran por capa.
 */
void print_model_overview(
    const std::vector<latent_t> &weights,
    const std::vector<bias_t> &biases,
    int preview_weights
);

/**
 * @brief Resume el cambio del modelo entre dos snapshots consecutivos.
 *
 * @param epoch_idx Indica la epoca asociada al cambio de snapshot.
 * @param previous_weights Contiene el snapshot previo de pesos.
 * @param current_weights Contiene el snapshot actual de pesos.
 * @param previous_biases Contiene el snapshot previo de bias.
 * @param current_biases Contiene el snapshot actual de bias.
 */
void print_epoch_model_delta(
    int epoch_idx,
    const std::vector<latent_t> &previous_weights,
    const std::vector<latent_t> &current_weights,
    const std::vector<bias_t> &previous_biases,
    const std::vector<bias_t> &current_biases
);

/**
 * @brief Explica por que el entrenamiento se congelo entre epocas.
 *
 * @param epoch_loss_pos Contiene la perdida positiva por epoca.
 * @param epoch_loss_neg Contiene la perdida negativa por epoca.
 * @param epoch_g_pos Contiene la goodness positiva por epoca.
 * @param epoch_g_neg Contiene la goodness negativa por epoca.
 * @param epoch_gap Contiene el gap positivo-negativo por epoca.
 * @param weights Contiene el snapshot final de pesos.
 * @param biases Contiene el snapshot final de bias.
 * @param epochs_to_check Indica cuantas epocas se revisan.
 */
void print_epoch_freeze_justification(
    const std::vector<loss_t> &epoch_loss_pos,
    const std::vector<loss_t> &epoch_loss_neg,
    const std::vector<goodness_t> &epoch_g_pos,
    const std::vector<goodness_t> &epoch_g_neg,
    const std::vector<goodness_t> &epoch_gap,
    const std::vector<latent_t> &weights,
    const std::vector<bias_t> &biases,
    int epochs_to_check
);

#endif
