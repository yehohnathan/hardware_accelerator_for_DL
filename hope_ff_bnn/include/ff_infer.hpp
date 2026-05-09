
/**
 * @file ff_infer.hpp
 * @brief Interfaz de inferencia multiclase para la BNN Forward-Forward.
 *
 * La inferencia no usa softmax. Para una imagen MNIST se construyen diez
 * entradas, cada una con una etiqueta one-hot distinta, y se elige la clase con
 * mayor goodness promedio. El top HLS carga pesos entrenados y produce
 * predicciones desde memoria externa.
 */
#ifndef HOPE_FF_INFER_HPP
#define HOPE_FF_INFER_HPP

#include "ff_layer.hpp"

/**
 * @brief Predice una muestra probando las diez etiquetas posibles.
 *
 * @param model Modelo BNN ya entrenado.
 * @param sample_words Muestra empaquetada.
 * @param class_goodness Salida con goodness de cada clase.
 * @return Etiqueta con mayor goodness.
 */
uint8_t ff_predict_sample(
    const ff_model_t &model,
    const ff_word_t sample_words[FF_WORDS_PER_SAMPLE],
    ff_goodness_t class_goodness[FF_NUM_CLASSES]
);

/**
 * @brief Evalúa un bloque de muestras y acumula accuracy.
 *
 * @param model Modelo BNN a evaluar.
 * @param dataset_words Payload contiguo sin cabecera.
 * @param n_samples Cantidad de muestras de evaluación.
 * @param eval_metrics Contadores y primera predicción observada.
 */
void ff_evaluate_dataset(
    const ff_model_t &model,
    const ff_word_t *dataset_words,
    int n_samples,
    ff_eval_metrics_t &eval_metrics
);

/**
 * @brief Top sintetizable de inferencia FF-BNN.
 *
 * @param dataset_words Entrada en memoria externa con muestras empaquetadas.
 * @param weights_in Pesos latentes planos producidos por entrenamiento.
 * @param biases_in Bias planos producidos por entrenamiento.
 * @param predictions_out Predicciones por muestra.
 * @param correct_count_out Conteo de aciertos frente a etiquetas reales.
 * @param n_samples Cantidad de muestras a inferir.
 *
 * @note Sintetizable. Consume el modelo desde memoria externa, lo reconstruye
 * en arreglos estáticos y ejecuta inferencia multiclase por goodness.
 */
extern "C" void ff_infer_kernel(
    const ff_word_t *dataset_words,
    const ff_latent_t *weights_in,
    const ff_bias_t *biases_in,
    uint8_t *predictions_out,
    uint32_t *correct_count_out,
    int n_samples
);

#endif
