
/**
 * @file ff_types.hpp
 * @brief Tipos de datos compartidos entre kernels HLS, capas y testbenches.
 *
 * Este archivo pertenece a la interfaz común del proyecto. Usa tipos enteros de
 * ancho explícito y arreglos C de tamaño fijo para que las mismas estructuras
 * sean válidas en simulación C++ y en síntesis HLS. No contiene STL ni memoria
 * dinámica en las estructuras que cruzan hacia el núcleo sintetizable.
 */
#ifndef HOPE_FF_TYPES_HPP
#define HOPE_FF_TYPES_HPP

#include "ff_config.hpp"

#include <cstdint>

/** Palabra física del dataset empaquetado y alineado a 32 bits. */
typedef uint32_t ff_word_t;
/** Valor de entrada de capa tras insertar etiqueta y escalar pixeles. */
typedef int16_t ff_feature_t;
/** Peso entrenable latente; su signo produce el peso binario de la BNN. */
typedef int16_t ff_latent_t;
/** Bias entrenable por neurona. */
typedef int16_t ff_bias_t;
/** Acumulador de productos signo-por-característica y bias. */
typedef int32_t ff_accum_t;
/** Activación ReLU saturada que puede alimentar la capa siguiente. */
typedef int32_t ff_activation_t;
/** Goodness local basada en promedio de activaciones al cuadrado. */
typedef int32_t ff_goodness_t;
/** Acumulador ancho para promediar muchas muestras sin overflow temprano. */
typedef int64_t ff_metric_accum_t;

/**
 * @brief Cabecera portable del dataset MNIST preprocesado.
 *
 * El testbench valida estos campos contra `ff_config.hpp` antes de entrenar. La
 * cabecera no se usa dentro del kernel sintetizable; el kernel recibe solo el
 * payload ya cargado en memoria externa.
 */
struct ff_dataset_header_t {
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
};

/**
 * @brief Modelo BNN con pesos latentes y bias en arreglos estáticos.
 *
 * Los pesos se reservan con la dimensión máxima activa de la build para que HLS
 * pueda inferir memoria estática. En hardware, estos arreglos pueden mapearse a
 * BRAM o registros según tamaño, particiones y directivas aplicadas.
 */
struct ff_model_t {
    ff_latent_t weights[FF_HIDDEN_LAYERS]
                       [FF_MAX_LAYER_NEURONS]
                       [FF_MAX_LAYER_INPUT_DIM];
    ff_bias_t biases[FF_HIDDEN_LAYERS][FF_MAX_LAYER_NEURONS];
};

/**
 * @brief Métricas producidas por una muestra durante entrenamiento FF.
 *
 * Guarda goodness positiva/negativa y las etiquetas usadas para permitir
 * auditoría del ejemplo supervisado que actualizó la capa.
 */
struct ff_sample_metrics_t {
    ff_goodness_t g_pos;
    ff_goodness_t g_neg;
    ff_goodness_t gap;
    uint8_t true_label;
    uint8_t negative_label;
};

/**
 * @brief Resumen de una época online de entrenamiento.
 *
 * Los contadores de eventos ayudan a verificar que la regla local realmente
 * activó cambios en pesos o bias.
 */
struct ff_epoch_metrics_t {
    ff_goodness_t avg_g_pos;
    ff_goodness_t avg_g_neg;
    ff_goodness_t avg_gap;
    uint32_t pos_update_events;
    uint32_t neg_update_events;
    uint32_t changed_weights;
    uint32_t changed_biases;
};

/**
 * @brief Métricas compactas de evaluación multiclase.
 *
 * Además de accuracy, conserva la primera predicción para depurar rápidamente
 * el comportamiento de inferencia sin imprimir todas las muestras.
 */
struct ff_eval_metrics_t {
    uint32_t correct;
    uint32_t total;
    uint8_t first_true_label;
    uint8_t first_pred_label;
    ff_goodness_t first_best_goodness;
};

#endif
