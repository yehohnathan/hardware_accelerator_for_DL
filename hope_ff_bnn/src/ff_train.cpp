
/**
 * @file ff_train.cpp
 * @brief Entrenamiento online Forward-Forward y kernel HLS de training.
 *
 * Este archivo implementa la ruta de entrenamiento de la BNN: para cada muestra
 * se crea un ejemplo positivo con la etiqueta real y un ejemplo negativo con
 * una etiqueta incorrecta. Cada capa compara goodness positiva y negativa,
 * actualiza localmente pesos/bias y pasa activaciones a la capa siguiente.
 */
#include "ff_train.hpp"

/**
 * @brief Entrena una muestra mediante pares positivo/negativo.
 *
 * @param model Modelo actualizado in-place.
 * @param sample_words Muestra MNIST empaquetada en buffer local.
 * @param rng_state Estado LFSR usado para seleccionar etiqueta negativa.
 * @param metrics Goodness y etiquetas observadas en esta muestra.
 * @param pos_update_events Contador acumulado de refuerzos positivos.
 * @param neg_update_events Contador acumulado de penalizaciones negativas.
 *
 * @sideeffect Modifica pesos y bias de cada capa activa. También avanza
 * `rng_state`, por lo que la secuencia de etiquetas negativas depende del orden
 * de muestras.
 */
void ff_train_one_sample(
    ff_model_t &model,
    const ff_word_t sample_words[FF_WORDS_PER_SAMPLE],
    uint32_t &rng_state,
    ff_sample_metrics_t &metrics,
    uint32_t &pos_update_events,
    uint32_t &neg_update_events
) {
    ff_feature_t pos_features[FF_MAX_LAYER_INPUT_DIM];
    ff_feature_t neg_features[FF_MAX_LAYER_INPUT_DIM];
    ff_activation_t pos_acts[FF_MAX_LAYER_NEURONS];
    ff_activation_t neg_acts[FF_MAX_LAYER_NEURONS];
/* Las activaciones se particionan por neuronas paralelas para que el forward y
 * el update puedan leer varios lanes del tile sin serializar accesos.
 */
#pragma HLS ARRAY_PARTITION variable=pos_acts cyclic factor=FF_PARALLEL_NEURONS
#pragma HLS ARRAY_PARTITION variable=neg_acts cyclic factor=FF_PARALLEL_NEURONS

    uint8_t true_label = ff_decode_label_from_words(sample_words);
    uint8_t negative_label = ff_make_negative_label(true_label, rng_state);

/* El ejemplo positivo combina la imagen con su etiqueta real; FF debe aumentar
 * su goodness por encima del umbral.
 */
    ff_build_features(sample_words, true_label, pos_features);

/* El ejemplo negativo usa la misma imagen con una etiqueta incorrecta; FF debe
 * reducir su goodness para separar clases.
 */
    ff_build_features(sample_words, negative_label, neg_features);

    metrics.true_label = true_label;
    metrics.negative_label = negative_label;
    metrics.g_pos = 0;
    metrics.g_neg = 0;

train_layer_loop:
    for (int layer = 0; layer < FF_HIDDEN_LAYERS; layer++) {
        ff_goodness_t layer_g_pos = 0;
        ff_goodness_t layer_g_neg = 0;

        ff_forward_layer(model, layer, pos_features, pos_acts, layer_g_pos);
        ff_forward_layer(model, layer, neg_features, neg_acts, layer_g_neg);

/* La actualización usa solo información local de la capa: entradas, activaciones
 * y goodness. No existe backpropagation global.
 */
        ff_update_layer_local(
            model,
            layer,
            pos_features,
            neg_features,
            pos_acts,
            neg_acts,
            layer_g_pos,
            layer_g_neg,
            pos_update_events,
            neg_update_events
        );

        metrics.g_pos += layer_g_pos;
        metrics.g_neg += layer_g_neg;

        int neurons = ff_layer_neurons(layer);
copy_next_layer_input_loop:
        for (int i = 0; i < FF_MAX_LAYER_INPUT_DIM; i++) {
/* Las activaciones de la capa actual se convierten en características de la
 * siguiente. El pipeline reduce el coste de copiar vectores intermedios.
 */
#pragma HLS PIPELINE II=1
            pos_features[i] = (i < neurons) ? (ff_feature_t)pos_acts[i] : 0;
            neg_features[i] = (i < neurons) ? (ff_feature_t)neg_acts[i] : 0;
        }
    }

    metrics.g_pos = metrics.g_pos / FF_HIDDEN_LAYERS;
    metrics.g_neg = metrics.g_neg / FF_HIDDEN_LAYERS;
    metrics.gap = metrics.g_pos - metrics.g_neg;
}

/**
 * @brief Ejecuta una época de entrenamiento online.
 *
 * @param model Modelo entrenado in-place.
 * @param dataset_words Payload MNIST sin cabecera.
 * @param n_samples Número de muestras recorridas.
 * @param seed Semilla para etiquetas negativas de esta época.
 * @param initial_model Copia usada para medir cambios acumulados.
 * @param epoch_metrics Promedios y contadores de salida.
 *
 * El entrenamiento online reduce memoria frente a mini-batches grandes, una
 * decisión razonable para una primera implementación HLS en Artix-7.
 */
void ff_train_epoch(
    ff_model_t &model,
    const ff_word_t *dataset_words,
    int n_samples,
    uint32_t seed,
    const ff_model_t &initial_model,
    ff_epoch_metrics_t &epoch_metrics
) {
    ff_metric_accum_t sum_pos = 0;
    ff_metric_accum_t sum_neg = 0;
    uint32_t pos_events = 0;
    uint32_t neg_events = 0;
    uint32_t rng_state = (seed == 0u) ? 0xACE1u : seed;

train_sample_loop:
    for (int sample_idx = 0; sample_idx < n_samples; sample_idx++) {
        ff_word_t sample_words[FF_WORDS_PER_SAMPLE];
        ff_sample_metrics_t sample_metrics;
/* Particionar completamente la muestra local permite que los decodificadores de
 * bits accedan a diferentes palabras sin un único puerto de memoria local.
 */
#pragma HLS ARRAY_PARTITION variable=sample_words complete

        ff_copy_sample_words(dataset_words, sample_idx, sample_words);
        ff_train_one_sample(
            model,
            sample_words,
            rng_state,
            sample_metrics,
            pos_events,
            neg_events
        );

        sum_pos += sample_metrics.g_pos;
        sum_neg += sample_metrics.g_neg;
    }

    if (n_samples > 0) {
        epoch_metrics.avg_g_pos = (ff_goodness_t)(sum_pos / n_samples);
        epoch_metrics.avg_g_neg = (ff_goodness_t)(sum_neg / n_samples);
    } else {
        epoch_metrics.avg_g_pos = 0;
        epoch_metrics.avg_g_neg = 0;
    }

    epoch_metrics.avg_gap = epoch_metrics.avg_g_pos - epoch_metrics.avg_g_neg;
    epoch_metrics.pos_update_events = pos_events;
    epoch_metrics.neg_update_events = neg_events;
    epoch_metrics.changed_weights = ff_count_changed_weights(initial_model, model);
    epoch_metrics.changed_biases = ff_count_changed_biases(initial_model, model);
}

/**
 * @brief Kernel HLS de entrenamiento Forward-Forward.
 *
 * @param dataset_words Entrada `m_axi` con muestras empaquetadas.
 * @param weights_out Salida `m_axi` con pesos latentes entrenados.
 * @param biases_out Salida `m_axi` con bias entrenados.
 * @param epoch_g_pos_out Salida `m_axi` con goodness positiva por época.
 * @param epoch_g_neg_out Salida `m_axi` con goodness negativa por época.
 * @param epoch_gap_out Salida `m_axi` con separación de goodness.
 * @param changed_count_out Salida `m_axi` con cambios de pesos y bias.
 * @param n_samples Muestras usadas por época.
 * @param n_epochs Épocas solicitadas, acotadas por `FF_MAX_EPOCHS`.
 * @param seed Semilla reproducible.
 * @param reset_model Reinicia el modelo estático cuando es distinto de cero.
 *
 * @sideeffect Conserva `model` como estado estático entre llamadas si no se
 * reinicia. Esto permite continuar entrenamiento, pero el testbench usa reset
 * para obtener corridas reproducibles.
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
) {
/* Cada arreglo grande usa un bundle AXI independiente para que Vitis pueda
 * planificar canales de memoria separados entre dataset, modelo y métricas.
 * En una plataforma pequeña puede no haber memoria física separada, pero la
 * separación evita dependencias artificiales en la interfaz.
 */
#pragma HLS INTERFACE m_axi port=dataset_words offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=weights_out offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=biases_out offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=epoch_g_pos_out offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=epoch_g_neg_out offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=epoch_gap_out offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=changed_count_out offset=slave bundle=gmem6
/* AXI-Lite transporta direcciones de buffers y escalares de control. Es la
 * interfaz típica para lanzar kernels desde software host o desde csim.
 */
#pragma HLS INTERFACE s_axilite port=dataset_words bundle=control
#pragma HLS INTERFACE s_axilite port=weights_out bundle=control
#pragma HLS INTERFACE s_axilite port=biases_out bundle=control
#pragma HLS INTERFACE s_axilite port=epoch_g_pos_out bundle=control
#pragma HLS INTERFACE s_axilite port=epoch_g_neg_out bundle=control
#pragma HLS INTERFACE s_axilite port=epoch_gap_out bundle=control
#pragma HLS INTERFACE s_axilite port=changed_count_out bundle=control
#pragma HLS INTERFACE s_axilite port=n_samples bundle=control
#pragma HLS INTERFACE s_axilite port=n_epochs bundle=control
#pragma HLS INTERFACE s_axilite port=seed bundle=control
#pragma HLS INTERFACE s_axilite port=reset_model bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static ff_model_t model;
/* Particionar por dimensión de neurona habilita accesos paralelos a lanes del
 * tile. El coste esperado es más bancos/registros y mayor presión de recursos.
 */
#pragma HLS ARRAY_PARTITION variable=model.weights cyclic factor=FF_PARALLEL_NEURONS dim=2
#pragma HLS ARRAY_PARTITION variable=model.biases cyclic factor=FF_PARALLEL_NEURONS dim=2

    ff_model_t initial_model;
/* La copia inicial también se particiona para que los conteos de cambios usen
 * el mismo patrón paralelo que el modelo entrenado.
 */
#pragma HLS ARRAY_PARTITION variable=initial_model.weights cyclic factor=FF_PARALLEL_NEURONS dim=2
#pragma HLS ARRAY_PARTITION variable=initial_model.biases cyclic factor=FF_PARALLEL_NEURONS dim=2

    int effective_samples = n_samples;
    if (effective_samples < 0) {
        effective_samples = 0;
    }

    int effective_epochs = n_epochs;
    if (effective_epochs < 0) {
        effective_epochs = 0;
    }
    if (effective_epochs > FF_MAX_EPOCHS) {
        effective_epochs = FF_MAX_EPOCHS;
    }

    if (reset_model != 0) {
        ff_init_model(model, seed);
    }

    initial_model = model;

kernel_epoch_loop:
    for (int epoch = 0; epoch < effective_epochs; epoch++) {
        ff_epoch_metrics_t metrics;
        ff_train_epoch(
            model,
            dataset_words,
            effective_samples,
            seed + (uint32_t)(epoch * 17 + 1),
            initial_model,
            metrics
        );

        epoch_g_pos_out[epoch] = metrics.avg_g_pos;
        epoch_g_neg_out[epoch] = metrics.avg_g_neg;
        epoch_gap_out[epoch] = metrics.avg_gap;
    }

    changed_count_out[0] = ff_count_changed_weights(initial_model, model);
    changed_count_out[1] = ff_count_changed_biases(initial_model, model);
    ff_store_model_flat(model, weights_out, biases_out);
}
