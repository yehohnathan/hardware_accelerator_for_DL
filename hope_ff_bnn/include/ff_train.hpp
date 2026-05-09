
/**
 * @file ff_train.hpp
 * @brief Interfaz de entrenamiento Forward-Forward y top function HLS.
 *
 * Las funciones declaradas aquí ejecutan entrenamiento online por muestra. La
 * ruta evita backpropagation global: cada capa calcula su goodness positiva y
 * negativa y actualiza pesos/bias localmente. `ff_train_kernel` es la frontera
 * sintetizable hacia memoria externa y control AXI-Lite.
 */
#ifndef HOPE_FF_TRAIN_HPP
#define HOPE_FF_TRAIN_HPP

#include "ff_layer.hpp"

/**
 * @brief Entrena el modelo con una muestra MNIST.
 *
 * @param model Modelo actualizado in-place.
 * @param sample_words Muestra empaquetada copiada a memoria local.
 * @param rng_state Estado usado para generar la etiqueta negativa.
 * @param metrics Goodness y etiquetas usadas por la muestra.
 * @param pos_update_events Contador acumulado de updates positivos.
 * @param neg_update_events Contador acumulado de updates negativos.
 *
 * @sideeffect Modifica pesos/bias de todas las capas activas y avanza el LFSR.
 */
void ff_train_one_sample(
    ff_model_t &model,
    const ff_word_t sample_words[FF_WORDS_PER_SAMPLE],
    uint32_t &rng_state,
    ff_sample_metrics_t &metrics,
    uint32_t &pos_update_events,
    uint32_t &neg_update_events
);

/**
 * @brief Ejecuta una época online sobre `n_samples`.
 *
 * @param model Modelo entrenado in-place.
 * @param dataset_words Payload contiguo del dataset, sin cabecera.
 * @param n_samples Cantidad de muestras a recorrer.
 * @param seed Semilla que inicializa el LFSR de etiquetas negativas.
 * @param initial_model Copia usada para contar cambios.
 * @param epoch_metrics Resumen de goodness y modificaciones.
 */
void ff_train_epoch(
    ff_model_t &model,
    const ff_word_t *dataset_words,
    int n_samples,
    uint32_t seed,
    const ff_model_t &initial_model,
    ff_epoch_metrics_t &epoch_metrics
);

/**
 * @brief Top sintetizable de entrenamiento FF-BNN.
 *
 * @param dataset_words Entrada en memoria externa con muestras empaquetadas.
 * @param weights_out Salida plana de pesos latentes entrenados.
 * @param biases_out Salida plana de bias entrenados.
 * @param epoch_g_pos_out Goodness positiva promedio por época.
 * @param epoch_g_neg_out Goodness negativa promedio por época.
 * @param epoch_gap_out Separación positiva-negativa por época.
 * @param changed_count_out Conteo final de pesos y bias modificados.
 * @param n_samples Muestras usadas en entrenamiento.
 * @param n_epochs Épocas online, acotadas por `FF_MAX_EPOCHS`.
 * @param seed Semilla reproducible de inicialización/etiquetas negativas.
 * @param reset_model Fuerza reinicio del modelo estático antes de entrenar.
 *
 * @note Sintetizable. Los buffers `m_axi` conectan con memoria externa; el
 * modelo estático puede inferirse como BRAM/registros según tamaño y
 * particiones. Separar bundles puede permitir concurrencia de accesos, sujeto
 * a la plataforma y al interconnect generado por Vitis.
 */
extern "C" void ff_train_kernel(
    const ff_word_t *dataset_words,
    ff_latent_t *weights_out,
    ff_bias_t *biases_out,
    ff_goodness_t *epoch_g_pos_out,
    ff_goodness_t *epoch_g_neg_out,
    ff_goodness_t *epoch_gap_out,
    uint32_t *changed_count_out,
    int n_samples,
    int n_epochs,
    uint32_t seed,
    int reset_model
);

#endif
