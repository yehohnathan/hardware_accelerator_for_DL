
/**
 * @file ff_infer.cpp
 * @brief Inferencia multiclase por goodness para la BNN Forward-Forward.
 *
 * En lugar de una capa softmax, la inferencia FF prueba la misma imagen con
 * cada etiqueta one-hot posible. La clase predicha es la que produce mayor
 * goodness promedio al atravesar las capas entrenadas.
 */
#include "ff_infer.hpp"

/**
 * @brief Predice una muestra evaluando las diez hipótesis de etiqueta.
 *
 * @param model Modelo BNN entrenado.
 * @param sample_words Muestra MNIST empaquetada.
 * @param class_goodness Arreglo de salida con goodness por clase.
 * @return Etiqueta con mayor goodness.
 *
 * @sideeffect Escribe `class_goodness`. No modifica pesos ni bias.
 */
uint8_t ff_predict_sample(
    const ff_model_t &model,
    const ff_word_t sample_words[FF_WORDS_PER_SAMPLE],
    ff_goodness_t class_goodness[FF_NUM_CLASSES]
) {
    uint8_t best_label = 0;
    ff_goodness_t best_goodness = -2147483647;

predict_class_loop:
    for (int label = 0; label < FF_NUM_CLASSES; label++) {
        ff_feature_t features[FF_MAX_LAYER_INPUT_DIM];
        ff_activation_t activations[FF_MAX_LAYER_NEURONS];
        ff_goodness_t goodness = 0;
/* Las activaciones se particionan por lanes para coincidir con el paralelismo
 * de `ff_forward_layer` y evitar un cuello de botella al copiar a la siguiente
 * capa.
 */
#pragma HLS ARRAY_PARTITION variable=activations cyclic factor=FF_PARALLEL_NEURONS

/* Cada iteración cambia solo el prefijo one-hot. Los pixeles son idénticos, lo
 * que implementa la regla de clasificación propia de Forward-Forward.
 */
        ff_build_features(sample_words, (uint8_t)label, features);
predict_layer_loop:
        for (int layer = 0; layer < FF_HIDDEN_LAYERS; layer++) {
            ff_goodness_t layer_goodness = 0;
            ff_forward_layer(model, layer, features, activations, layer_goodness);
            goodness += layer_goodness;

            int neurons = ff_layer_neurons(layer);
copy_predict_next_loop:
            for (int i = 0; i < FF_MAX_LAYER_INPUT_DIM; i++) {
/* Copiar activaciones a features mantiene una interfaz uniforme entre capas y
 * permite usar la misma función forward para todas.
 */
#pragma HLS PIPELINE II=1
                features[i] = (i < neurons) ? (ff_feature_t)activations[i] : 0;
            }
        }
        goodness = goodness / FF_HIDDEN_LAYERS;
        class_goodness[label] = goodness;

        if (label == 0 || goodness > best_goodness) {
            best_goodness = goodness;
            best_label = (uint8_t)label;
        }
    }

    return best_label;
}

/**
 * @brief Evalúa accuracy sobre un bloque de muestras.
 *
 * @param model Modelo BNN a evaluar.
 * @param dataset_words Payload de evaluación sin cabecera.
 * @param n_samples Cantidad de muestras.
 * @param eval_metrics Métricas de salida.
 *
 * @sideeffect Actualiza contadores de `eval_metrics`; no cambia el modelo.
 */
void ff_evaluate_dataset(
    const ff_model_t &model,
    const ff_word_t *dataset_words,
    int n_samples,
    ff_eval_metrics_t &eval_metrics
) {
    eval_metrics.correct = 0;
    eval_metrics.total = (n_samples > 0) ? (uint32_t)n_samples : 0u;
    eval_metrics.first_true_label = 0;
    eval_metrics.first_pred_label = 0;
    eval_metrics.first_best_goodness = 0;

eval_sample_loop:
    for (int sample_idx = 0; sample_idx < n_samples; sample_idx++) {
        ff_word_t sample_words[FF_WORDS_PER_SAMPLE];
        ff_goodness_t scores[FF_NUM_CLASSES];
/* La muestra y los scores se particionan porque ambos son pequeños y se acceden
 * varias veces durante la predicción multiclase.
 */
#pragma HLS ARRAY_PARTITION variable=sample_words complete
#pragma HLS ARRAY_PARTITION variable=scores complete

        ff_copy_sample_words(dataset_words, sample_idx, sample_words);
        uint8_t true_label = ff_decode_label_from_words(sample_words);
        uint8_t pred_label = ff_predict_sample(model, sample_words, scores);

        if (pred_label == true_label) {
            eval_metrics.correct++;
        }

        if (sample_idx == 0) {
            eval_metrics.first_true_label = true_label;
            eval_metrics.first_pred_label = pred_label;
            eval_metrics.first_best_goodness = scores[pred_label];
        }
    }
}

/**
 * @brief Kernel HLS de inferencia.
 *
 * @param dataset_words Entrada `m_axi` con muestras empaquetadas.
 * @param weights_in Entrada `m_axi` con pesos latentes planos.
 * @param biases_in Entrada `m_axi` con bias planos.
 * @param predictions_out Salida `m_axi` con una predicción por muestra.
 * @param correct_count_out Salida `m_axi` con el número total de aciertos.
 * @param n_samples Número de muestras a procesar.
 *
 * @note Sintetizable. Carga el modelo plano a arreglos locales y luego evalúa
 * muestras. Separar bundles de entrada/salida permite mayor concurrencia de
 * memoria si la plataforma lo soporta.
 */
extern "C" void ff_infer_kernel(
    const ff_word_t *dataset_words,
    const ff_latent_t *weights_in,
    const ff_bias_t *biases_in,
    uint8_t *predictions_out,
    uint32_t *correct_count_out,
    int n_samples
) {
/* Interfaces AXI master: dataset, modelo y resultados viajan por memoria
 * externa, lo que permite reutilizar el kernel sin recompilar para cada buffer.
 */
#pragma HLS INTERFACE m_axi port=dataset_words offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=weights_in offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=biases_in offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=predictions_out offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=correct_count_out offset=slave bundle=gmem4
/* AXI-Lite controla direcciones y escalares de ejecución. */
#pragma HLS INTERFACE s_axilite port=dataset_words bundle=control
#pragma HLS INTERFACE s_axilite port=weights_in bundle=control
#pragma HLS INTERFACE s_axilite port=biases_in bundle=control
#pragma HLS INTERFACE s_axilite port=predictions_out bundle=control
#pragma HLS INTERFACE s_axilite port=correct_count_out bundle=control
#pragma HLS INTERFACE s_axilite port=n_samples bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    ff_model_t model;
/* La partición por neurona permite alimentar en paralelo los lanes definidos por
 * `FF_PARALLEL_NEURONS`, con el coste de más bancos o registros locales.
 */
#pragma HLS ARRAY_PARTITION variable=model.weights cyclic factor=FF_PARALLEL_NEURONS dim=2
#pragma HLS ARRAY_PARTITION variable=model.biases cyclic factor=FF_PARALLEL_NEURONS dim=2

    ff_load_model_flat(model, weights_in, biases_in);

    uint32_t correct = 0;
    int effective_samples = n_samples;
    if (effective_samples < 0) {
        effective_samples = 0;
    }

infer_kernel_sample_loop:
    for (int sample_idx = 0; sample_idx < effective_samples; sample_idx++) {
        ff_word_t sample_words[FF_WORDS_PER_SAMPLE];
        ff_goodness_t scores[FF_NUM_CLASSES];
/* La inferencia de una muestra reutiliza sus palabras y los diez scores varias
 * veces; la partición evita serializar accesos locales pequeños.
 */
#pragma HLS ARRAY_PARTITION variable=sample_words complete
#pragma HLS ARRAY_PARTITION variable=scores complete

        ff_copy_sample_words(dataset_words, sample_idx, sample_words);
        uint8_t true_label = ff_decode_label_from_words(sample_words);
        uint8_t pred_label = ff_predict_sample(model, sample_words, scores);
        predictions_out[sample_idx] = pred_label;

        if (pred_label == true_label) {
            correct++;
        }
    }

    correct_count_out[0] = correct;
}
