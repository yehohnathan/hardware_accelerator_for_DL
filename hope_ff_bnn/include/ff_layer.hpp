
/**
 * @file ff_layer.hpp
 * @brief Operaciones de capa para una BNN entrenada con Forward-Forward.
 *
 * Este archivo declara el núcleo algorítmico que sí debe mantenerse compatible
 * con HLS: inicialización determinista, construcción de características,
 * forward con pesos binarios derivados del signo y actualización local por
 * goodness. Las funciones evitan memoria dinámica y trabajan con arreglos de
 * tamaño máximo configurado.
 */
#ifndef HOPE_FF_LAYER_HPP
#define HOPE_FF_LAYER_HPP

#include "ff_dataset.hpp"
#include "ff_types.hpp"

/**
 * @brief Avanza el generador LFSR determinista usado por el modelo.
 *
 * @param state Estado actual del registro de desplazamiento.
 * @return Nuevo estado pseudoaleatorio no criptográfico.
 *
 * @note Sintetizable. Se usa para inicializar pesos latentes y escoger
 * etiquetas negativas reproducibles sin depender de librerías de host.
 */
uint32_t ff_lfsr_next(uint32_t state);

/**
 * @brief Genera una etiqueta negativa distinta de la etiqueta real.
 *
 * @param true_label Etiqueta correcta de la muestra MNIST.
 * @param rng_state Estado LFSR actualizado por referencia.
 * @return Etiqueta incorrecta usada para el ejemplo negativo supervisado.
 *
 * Esta función implementa la separación positiva/negativa del algoritmo
 * Forward-Forward aplicado a clasificación.
 */
uint8_t ff_make_negative_label(
    uint8_t true_label,
    uint32_t &rng_state
);

/**
 * @brief Inicializa pesos latentes y bias del modelo BNN.
 *
 * @param model Modelo que será escrito.
 * @param seed Semilla determinista para reproducir simulaciones.
 *
 * @sideeffect Sobrescribe pesos y bias. Las zonas no activas por la
 * arquitectura se limpian para evitar que basura afecte exportación o métricas.
 */
void ff_init_model(
    ff_model_t &model,
    uint32_t seed
);

/** @brief Devuelve la dimensión de entrada activa para una capa configurada. */
int ff_layer_input_dim(int layer_idx);
/** @brief Devuelve el número de neuronas activas de una capa configurada. */
int ff_layer_neurons(int layer_idx);

/**
 * @brief Construye el vector `[one-hot | pixeles]` para una etiqueta candidata.
 *
 * @param sample_words Muestra MNIST empaquetada.
 * @param label_index Etiqueta que se incrusta como hipótesis.
 * @param features Vector de salida usado por la primera capa.
 *
 * Se llama con la etiqueta real para ejemplos positivos y con una etiqueta
 * falsa para ejemplos negativos; en inferencia se llama para las diez clases.
 */
void ff_build_features(
    const ff_word_t sample_words[FF_WORDS_PER_SAMPLE],
    uint8_t label_index,
    ff_feature_t features[FF_MAX_LAYER_INPUT_DIM]
);

/**
 * @brief Ejecuta el forward de una capa binaria y calcula su goodness.
 *
 * @param model Pesos latentes y bias del modelo.
 * @param layer_idx Capa a evaluar.
 * @param features Entrada de la capa.
 * @param activations Activaciones ReLU saturadas de salida.
 * @param goodness Promedio de activaciones al cuadrado.
 *
 * @note Sintetizable. El parámetro `FF_PARALLEL_NEURONS` controla cuántas
 * neuronas se procesan por tile, afectando throughput y uso de LUT/FF/BRAM.
 */
void ff_forward_layer(
    const ff_model_t &model,
    int layer_idx,
    const ff_feature_t features[FF_MAX_LAYER_INPUT_DIM],
    ff_activation_t activations[FF_MAX_LAYER_NEURONS],
    ff_goodness_t &goodness
);

/**
 * @brief Aplica la regla local Forward-Forward sobre una capa.
 *
 * @param model Modelo modificado in-place.
 * @param layer_idx Capa entrenada.
 * @param pos_features Entrada del ejemplo positivo.
 * @param neg_features Entrada del ejemplo negativo.
 * @param pos_acts Activaciones producidas por el ejemplo positivo.
 * @param neg_acts Activaciones producidas por el ejemplo negativo.
 * @param g_pos Goodness positiva de la capa.
 * @param g_neg Goodness negativa de la capa.
 * @param pos_update_events Contador de neuronas positivas actualizadas.
 * @param neg_update_events Contador de neuronas negativas actualizadas.
 *
 * @sideeffect Cambia pesos latentes y bias de la capa. La actualización es
 * local: no propaga gradientes globales ni requiere almacenar activaciones de
 * toda la red.
 */
void ff_update_layer_local(
    ff_model_t &model,
    int layer_idx,
    const ff_feature_t pos_features[FF_MAX_LAYER_INPUT_DIM],
    const ff_feature_t neg_features[FF_MAX_LAYER_INPUT_DIM],
    const ff_activation_t pos_acts[FF_MAX_LAYER_NEURONS],
    const ff_activation_t neg_acts[FF_MAX_LAYER_NEURONS],
    ff_goodness_t g_pos,
    ff_goodness_t g_neg,
    uint32_t &pos_update_events,
    uint32_t &neg_update_events
);

/**
 * @brief Cuenta pesos latentes modificados en regiones activas del modelo.
 *
 * @return Número de pesos que cambiaron desde una instantánea anterior.
 */
uint32_t ff_count_changed_weights(
    const ff_model_t &before,
    const ff_model_t &after
);

/**
 * @brief Cuenta bias modificados en regiones activas del modelo.
 *
 * @return Número de bias que cambiaron desde una instantánea anterior.
 */
uint32_t ff_count_changed_biases(
    const ff_model_t &before,
    const ff_model_t &after
);

/**
 * @brief Exporta el modelo a buffers planos compatibles con memoria externa.
 *
 * @sideeffect Escribe `weights_out` y `biases_out`. Esta forma es útil para
 * comunicación host-kernel y para encadenar entrenamiento con inferencia.
 */
void ff_store_model_flat(
    const ff_model_t &model,
    ff_latent_t *weights_out,
    ff_bias_t *biases_out
);

/**
 * @brief Restaura el modelo desde buffers planos de memoria externa.
 *
 * @sideeffect Sobrescribe el arreglo estático interno del modelo.
 */
void ff_load_model_flat(
    ff_model_t &model,
    const ff_latent_t *weights_in,
    const ff_bias_t *biases_in
);

#endif
