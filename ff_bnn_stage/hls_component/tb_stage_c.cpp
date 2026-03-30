#include "tb_stage_c.hpp"   // Se incluye el header propio del experimento modular de la Etapa C.
#include "debug_utils.hpp"  // Se incluyen las utilidades de inspección y visualización del testbench.

#include <iostream>          // Se incluye iostream para imprimir métricas y resúmenes en consola.
#include <vector>            // Se incluye vector para buffers de snapshots, métricas y predicciones.

// ============================================================
// Ejecución modular de la Etapa C
// ============================================================
void run_stage_c_experiment(
    const std::vector<word_t> &input_words,
    int total_samples,
    uint16_t seed,
    const stage_c_experiment_cfg_t &cfg
) {
    int train_samples = total_samples;
    // Se inicializa la cantidad de muestras de entrenamiento con el total disponible por defecto.

    if (cfg.train_samples_limit > 0 && train_samples > cfg.train_samples_limit) {
        train_samples = cfg.train_samples_limit;
        // Si se definió un límite positivo, se recorta el entrenamiento para mantener la corrida controlada.
    }

    int eval_samples = train_samples;
    // En esta versión modular se mantiene la evaluación sobre el mismo subconjunto usado para entrenamiento.

    int epochs = cfg.epochs;
    // Se copia la cantidad de épocas definida por la configuración del experimento.

    if (epochs < 0) {
        epochs = 0;
        // Se sanea un valor inválido de épocas para evitar lazos negativos dentro del flujo experimental.
    }

    std::vector<label_idx_t> train_true_labels(eval_samples, 0);
    // Se reserva el buffer de etiquetas verdaderas que devolverá el top entrenable durante la inferencia.

    std::vector<label_idx_t> pred_labels(eval_samples, 0);
    // Se reserva el buffer de etiquetas predichas que devolverá el top entrenable.

    std::vector<latent_t> weight_snapshot(MODEL_WEIGHT_COUNT);
    // Se reserva el buffer externo donde el top copiará el snapshot final de pesos latentes.

    std::vector<bias_t> bias_snapshot(MODEL_NEURONS);
    // Se reserva el buffer externo donde el top copiará el snapshot final de bias latentes.

    std::vector<goodness_t> g_pos(train_samples, 0);
    // Se reserva el buffer de goodness positiva registrada en la última época de entrenamiento.

    std::vector<goodness_t> g_neg(train_samples, 0);
    // Se reserva el buffer de goodness negativa registrada en la última época de entrenamiento.

    std::vector<goodness_t> gap(train_samples, 0);
    // Se reserva el buffer de goodness gap registrada en la última época de entrenamiento.

    std::vector<ap_uint<32> > correct_count(1, 0);
    // Se reserva un buffer de una sola posición para recibir el conteo final de aciertos de inferencia.

    ff_train_top(
        input_words.data(),
        train_true_labels.data(),
        pred_labels.data(),
        weight_snapshot.data(),
        bias_snapshot.data(),
        g_pos.data(),
        g_neg.data(),
        gap.data(),
        correct_count.data(),
        eval_samples,
        train_samples,
        epochs,
        seed,
        true
    );
    // Se ejecuta el top de la Etapa C que entrena el modelo localmente e infiere sobre el subconjunto seleccionado.

    print_separator("RESULTADOS DE ENTRENAMIENTO DE ETAPA C");
    // Se abre una sección para mostrar las métricas principales del entrenamiento hardware.

    std::cout << "train_samples   = " << train_samples << std::endl;
    // Se imprime la cantidad real de muestras usadas para entrenamiento local.

    std::cout << "eval_samples    = " << eval_samples << std::endl;
    // Se imprime la cantidad real de muestras usadas para inferencia posterior.

    std::cout << "epochs          = " << epochs << std::endl;
    // Se imprime la cantidad de épocas ejecutadas por el kernel entrenable.

    std::cout << "correct_count   = " << (unsigned int)correct_count[0] << std::endl;
    // Se imprime el número total de predicciones correctas obtenidas tras el entrenamiento.

    double train_accuracy = 0.0;
    // Se reserva una variable escalar para mostrar la exactitud de forma explícita y legible.

    if (eval_samples > 0) {
        train_accuracy = ((double)((unsigned int)correct_count[0]) / (double)eval_samples);
        // Se calcula la exactitud solo cuando el subconjunto evaluado tiene tamaño positivo.
    }

    std::cout << "train_accuracy  = " << train_accuracy << std::endl;
    // Se imprime la exactitud obtenida sobre el subconjunto usado en esta corrida modular.

    print_training_preview(g_pos, g_neg, gap, 12);
    // Se muestran algunas goodness positivas, negativas y gaps de la última época para inspección rápida.

    print_prediction_preview(train_true_labels, pred_labels, 20);
    // Se muestran varias parejas verdad-predicción para verificar visualmente el comportamiento del modelo.

    print_model_overview(weight_snapshot, bias_snapshot, 24);
    // Se resume el estado aprendido de pesos y bias después del entrenamiento local hardware.

    if (cfg.inspect_new_samples == true) {
        if (cfg.inspect_start_sample >= 0 && cfg.inspect_num_samples > 0 && cfg.inspect_start_sample < total_samples) {
            print_separator("NUEVOS SAMPLES PARA TEST");
            // Si se habilitó la inspección y el rango es válido, se abre una sección adicional de depuración visual.

            print_samples_range(input_words, cfg.inspect_start_sample, cfg.inspect_num_samples);
            // Se imprime el rango solicitado para observar muestras alejadas del subconjunto principal.
        }
    }
}
