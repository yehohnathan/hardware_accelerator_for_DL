#include "tb_stage_c.hpp"   // Se incluye el header propio del experimento modular de la Etapa C.
#include "debug_utils.hpp"  // Se incluyen las utilidades de inspeccion y visualizacion del testbench.

#include <chrono>           // Se incluye chrono para medir el tiempo de cada epoca en terminal.
#include <iostream>         // Se incluye iostream para imprimir metricas y resumenes.
#include <vector>           // Se incluye vector para buffers temporales y snapshots.

// ============================================================
// Ejecucion modular de la Etapa C
// ============================================================
void run_stage_c_experiment(
    const std::vector<word_t> &input_words,
    int total_samples,
    uint16_t seed,
    const stage_c_experiment_cfg_t &cfg
) {
    int train_samples = total_samples;
    // Se inicializa el tamano de entrenamiento con el total disponible.

    if ((cfg.train_samples_limit > 0) && (train_samples > cfg.train_samples_limit)) {
        train_samples = cfg.train_samples_limit;
        // Se recorta el subset de entrenamiento cuando el usuario fijo un limite positivo.
    }

    if (train_samples < 0) {
        train_samples = 0;
        // Se corrige cualquier valor invalido a cero para evitar lazos negativos.
    }

    int epochs = cfg.epochs;
    // Se copia la cantidad de epocas solicitadas.

    if (epochs < 0) {
        epochs = 0;
        // Se corrige un valor invalido de epocas.
    }

    if (epochs > TRAIN_MAX_EPOCHS) {
        epochs = TRAIN_MAX_EPOCHS;
        // Se limita la cantidad de epocas al tamano maximo del historial hardware.
    }

    int eval_samples = total_samples;
    // Se inicializa el tamano de evaluacion con el total disponible.

    if ((cfg.eval_samples_limit > 0) && (eval_samples > cfg.eval_samples_limit)) {
        eval_samples = cfg.eval_samples_limit;
        // Se recorta el subset de evaluacion cuando el usuario fijo un limite positivo.
    }

    if (eval_samples < 0) {
        eval_samples = 0;
        // Se corrige cualquier valor invalido a cero.
    }

    int eval_start = total_samples - eval_samples;
    // Se intenta colocar el subset de evaluacion al final del dataset para evitar solapamiento.

    bool eval_reuses_train = false;
    // Se crea una bandera para informar si la evaluacion reutiliza muestras del entrenamiento.

    if (eval_start < train_samples) {
        eval_start = 0;
        // Si el hold-out no cabe al final, se usa el inicio del dataset como fallback.

        eval_reuses_train = true;
        // Se marca que la evaluacion y el entrenamiento comparten muestras.
    }

    if (eval_samples == 0) {
        eval_start = 0;
        // Si no hay evaluacion, el indice inicial se fuerza a cero solo por consistencia visual.
    }

    int train_batches = 0;
    // Se reserva el numero de mini-batches efectivos que se ejecutaran por epoca.

    if (train_samples > 0) {
        train_batches = (train_samples + MODEL_BATCH_SIZE - 1) / MODEL_BATCH_SIZE;
        // Se calcula cuantas tandas reales de entrenamiento requiere el subset actual.
    }

    int eval_hypotheses = eval_samples * NUM_CLASSES;
    // Se estima cuantas hipotesis multiclase se probaran por pasada completa de evaluacion.

    std::vector<label_idx_t> dummy_true_labels(1, 0);
    // Se reserva un buffer minimo para llamadas sin fase de inferencia.

    std::vector<label_idx_t> dummy_pred_labels(1, 0);
    // Se reserva un buffer minimo para llamadas sin fase de inferencia.

    std::vector<latent_t> weight_snapshot(MODEL_WEIGHT_COUNT);
    // Se reserva el snapshot externo de pesos latentes del modelo entrenado.

    std::vector<bias_t> bias_snapshot(MODEL_BIAS_COUNT);
    // Se reserva el snapshot externo de bias latentes del modelo entrenado.

    std::vector<latent_t> previous_weight_snapshot(MODEL_WEIGHT_COUNT);
    // Se reserva el snapshot previo de pesos para medir si cada epoca realmente cambio el modelo.

    std::vector<bias_t> previous_bias_snapshot(MODEL_BIAS_COUNT);
    // Se reserva el snapshot previo de bias para medir si cada epoca realmente cambio el modelo.

    std::vector<goodness_t> g_pos(train_samples, 0);
    // Se reserva el buffer de goodness positiva por muestra de la ultima epoca.

    std::vector<goodness_t> g_neg(train_samples, 0);
    // Se reserva el buffer de goodness negativa por muestra de la ultima epoca.

    std::vector<goodness_t> gap(train_samples, 0);
    // Se reserva el buffer de goodness gap por muestra de la ultima epoca.

    std::vector<loss_t> epoch_loss_pos(TRAIN_MAX_EPOCHS, 0);
    // Se reserva el historial acumulado de perdida positiva por epoca.

    std::vector<loss_t> epoch_loss_neg(TRAIN_MAX_EPOCHS, 0);
    // Se reserva el historial acumulado de perdida negativa por epoca.

    std::vector<goodness_t> epoch_g_pos(TRAIN_MAX_EPOCHS, 0);
    // Se reserva el historial acumulado de goodness positiva por epoca.

    std::vector<goodness_t> epoch_g_neg(TRAIN_MAX_EPOCHS, 0);
    // Se reserva el historial acumulado de goodness negativa por epoca.

    std::vector<goodness_t> epoch_gap(TRAIN_MAX_EPOCHS, 0);
    // Se reserva el historial acumulado de goodness gap por epoca.

    std::vector<loss_t> epoch_loss_pos_current(TRAIN_MAX_EPOCHS, 0);
    // Se reserva el buffer temporal de perdida positiva de la llamada HLS actual.

    std::vector<loss_t> epoch_loss_neg_current(TRAIN_MAX_EPOCHS, 0);
    // Se reserva el buffer temporal de perdida negativa de la llamada HLS actual.

    std::vector<goodness_t> epoch_g_pos_current(TRAIN_MAX_EPOCHS, 0);
    // Se reserva el buffer temporal de goodness positiva de la llamada HLS actual.

    std::vector<goodness_t> epoch_g_neg_current(TRAIN_MAX_EPOCHS, 0);
    // Se reserva el buffer temporal de goodness negativa de la llamada HLS actual.

    std::vector<goodness_t> epoch_gap_current(TRAIN_MAX_EPOCHS, 0);
    // Se reserva el buffer temporal de goodness gap de la llamada HLS actual.

    std::vector<ap_uint<32> > correct_count(1, 0);
    // Se reserva un buffer de una posicion para el contador de aciertos de inferencia.

    std::vector<label_idx_t> eval_true_labels(eval_samples, 0);
    // Se reserva el buffer de etiquetas verdaderas del subset de evaluacion.

    std::vector<label_idx_t> eval_pred_labels(eval_samples, 0);
    // Se reserva el buffer de predicciones del subset de evaluacion.

    print_separator("CONFIGURACION DE ETAPA C");
    // Se abre una seccion con la configuracion efectiva del experimento.

    std::cout << "train_start       = 0" << std::endl;
    // Se imprime el inicio del subset de entrenamiento.

    std::cout << "train_samples     = " << train_samples << std::endl;
    // Se imprime la cantidad real de muestras de entrenamiento.

    std::cout << "eval_start        = " << eval_start << std::endl;
    // Se imprime el inicio del subset de evaluacion.

    std::cout << "eval_samples      = " << eval_samples << std::endl;
    // Se imprime la cantidad real de muestras de evaluacion.

    std::cout << "eval_reuses_train = " << (eval_reuses_train ? "true" : "false") << std::endl;
    // Se informa si la evaluacion comparte muestras con el entrenamiento.

    std::cout << "epochs            = " << epochs << std::endl;
    // Se imprime la cantidad de epocas ejecutadas.

    std::cout << "train_batches     = " << train_batches << std::endl;
    // Se imprime la cantidad de mini-batches por epoca derivada del subset real.

    std::cout << "architecture      = "
              << MODEL_INPUT_BITS
              << " -> "
              << MODEL_LAYER1_NEURONS
              << " -> "
              << MODEL_LAYER2_NEURONS
              << std::endl;
    // Se imprime la arquitectura multicapa efectiva del perfil HLS activo.

    std::cout << "score_rule        = g1 + g2" << std::endl;
    // Se recuerda que la inferencia usa la goodness total como suma de ambas capas.

    std::cout << "batch_size        = " << MODEL_BATCH_SIZE << std::endl;
    // Se imprime el tamano de mini-batch sintetizable fijado en hardware.

    std::cout << "eval_hypotheses   = " << eval_hypotheses << std::endl;
    // Se imprime el costo multiclase basico de la fase de evaluacion para entender el tiempo de csim.

    std::cout << "label_scale       = " << LABEL_SCALE_HW.to_double() << std::endl;
    // Se imprime la escala usada para incrustar la etiqueta en la entrada.

    std::cout << "threshold         = " << GOODNESS_THRESHOLD_HW.to_double() << std::endl;
    // Se imprime el threshold FF usado por el kernel.

    std::cout << "learning_rate     = " << LEARNING_RATE_HW.to_double() << std::endl;
    // Se imprime la tasa de aprendizaje fija usada por el kernel.

    ff_train_top(
        input_words.data(),
        dummy_true_labels.data(),
        dummy_pred_labels.data(),
        previous_weight_snapshot.data(),
        previous_bias_snapshot.data(),
        g_pos.data(),
        g_neg.data(),
        gap.data(),
        epoch_loss_pos_current.data(),
        epoch_loss_neg_current.data(),
        epoch_g_pos_current.data(),
        epoch_g_neg_current.data(),
        epoch_gap_current.data(),
        correct_count.data(),
        0,
        0,
        0,
        seed,
        true
    );
    // Se captura el snapshot inicial del modelo reseteado para comparar la primera epoca contra el estado de arranque.

    if ((train_samples > 0) && (epochs > 0)) {
        print_separator("ACTUALIZACION POR EPOCA");
        // Se abre la seccion incremental solo cuando realmente habra entrenamiento.

epoch_train_loop:
        for (int epoch_idx = 0; epoch_idx < epochs; epoch_idx++) {
            // Se ejecuta una epoca por llamada para poder imprimir progreso real en terminal.

            std::chrono::steady_clock::time_point epoch_start = std::chrono::steady_clock::now();
            // Se captura el instante inicial de la epoca para reportar su duracion.

            ff_train_top(
                input_words.data(),
                dummy_true_labels.data(),
                dummy_pred_labels.data(),
                weight_snapshot.data(),
                bias_snapshot.data(),
                g_pos.data(),
                g_neg.data(),
                gap.data(),
                epoch_loss_pos_current.data(),
                epoch_loss_neg_current.data(),
                epoch_g_pos_current.data(),
                epoch_g_neg_current.data(),
                epoch_gap_current.data(),
                correct_count.data(),
                0,
                train_samples,
                1,
                seed,
                (epoch_idx == 0)
            );
            // Se ejecuta exactamente una epoca de entrenamiento reutilizando el estado persistente del modelo.

            epoch_loss_pos[epoch_idx] = epoch_loss_pos_current[0];
            // Se copia la perdida positiva de la epoca actual al historial global.

            epoch_loss_neg[epoch_idx] = epoch_loss_neg_current[0];
            // Se copia la perdida negativa de la epoca actual al historial global.

            epoch_g_pos[epoch_idx] = epoch_g_pos_current[0];
            // Se copia la goodness positiva de la epoca actual al historial global.

            epoch_g_neg[epoch_idx] = epoch_g_neg_current[0];
            // Se copia la goodness negativa de la epoca actual al historial global.

            epoch_gap[epoch_idx] = epoch_gap_current[0];
            // Se copia el goodness gap de la epoca actual al historial global.

            double val_accuracy = 0.0;
            // Se reserva la accuracy de validacion de la epoca actual.

            if (eval_samples > 0) {
                const word_t *eval_ptr = input_words.data() + (eval_start * WORDS_PER_SAMPLE);
                // Se calcula el puntero al inicio real del subset de evaluacion dentro del buffer lineal.

                ff_train_top(
                    eval_ptr,
                    eval_true_labels.data(),
                    eval_pred_labels.data(),
                    weight_snapshot.data(),
                    bias_snapshot.data(),
                    g_pos.data(),
                    g_neg.data(),
                    gap.data(),
                    epoch_loss_pos_current.data(),
                    epoch_loss_neg_current.data(),
                    epoch_g_pos_current.data(),
                    epoch_g_neg_current.data(),
                    epoch_gap_current.data(),
                    correct_count.data(),
                    eval_samples,
                    0,
                    0,
                    seed,
                    false
                );
                // Se ejecuta la inferencia hold-out de la epoca actual reutilizando el modelo ya actualizado.

                val_accuracy = ((double)((unsigned int)correct_count[0]) / (double)eval_samples);
                // Se calcula la accuracy de validacion de la epoca actual.
            }

            std::chrono::steady_clock::time_point epoch_end = std::chrono::steady_clock::now();
            // Se captura el instante final de la epoca tras entrenamiento y evaluacion.

            double elapsed_sec =
                std::chrono::duration_cast<std::chrono::duration<double> >(epoch_end - epoch_start).count();
            // Se convierte el tiempo transcurrido de la epoca a segundos para imprimirlo en terminal.

            print_epoch_terminal_update(
                epoch_idx + 1,
                epochs,
                epoch_loss_pos[epoch_idx],
                epoch_loss_neg[epoch_idx],
                epoch_g_pos[epoch_idx],
                epoch_g_neg[epoch_idx],
                epoch_gap[epoch_idx],
                val_accuracy,
                elapsed_sec,
                (eval_samples > 0)
            );
            // Se imprime la actualizacion visible por terminal de la epoca actual.

            print_epoch_model_delta(
                epoch_idx + 1,
                previous_weight_snapshot,
                weight_snapshot,
                previous_bias_snapshot,
                bias_snapshot
            );
            // Se imprime si el snapshot del modelo realmente cambio respecto a la epoca anterior.

            previous_weight_snapshot = weight_snapshot;
            // Se actualiza el snapshot previo de pesos para comparar la siguiente epoca.

            previous_bias_snapshot = bias_snapshot;
            // Se actualiza el snapshot previo de bias para comparar la siguiente epoca.
        }
    } else if (eval_samples > 0) {
        const word_t *eval_ptr = input_words.data() + (eval_start * WORDS_PER_SAMPLE);
        // Se calcula el puntero al inicio real del subset de evaluacion dentro del buffer lineal.

        ff_train_top(
            eval_ptr,
            eval_true_labels.data(),
            eval_pred_labels.data(),
            weight_snapshot.data(),
            bias_snapshot.data(),
            g_pos.data(),
            g_neg.data(),
            gap.data(),
            epoch_loss_pos_current.data(),
            epoch_loss_neg_current.data(),
            epoch_g_pos_current.data(),
            epoch_g_neg_current.data(),
            epoch_gap_current.data(),
            correct_count.data(),
            eval_samples,
            0,
            0,
            seed,
            true
        );
        // Si no hay entrenamiento, se ejecuta una inferencia pura sobre el modelo reseteado por consistencia.
    }

    print_separator("RESULTADOS DE ENTRENAMIENTO DE ETAPA C");
    // Se abre la seccion de resultados principales de la etapa entrenable.

    std::cout << "holdout_correct_count = " << (unsigned int)correct_count[0] << std::endl;
    // Se imprime el numero total de aciertos sobre el subset de evaluacion.

    double holdout_accuracy = 0.0;
    // Se reserva una variable escalar para la exactitud de evaluacion.

    if (eval_samples > 0) {
        holdout_accuracy = ((double)((unsigned int)correct_count[0]) / (double)eval_samples);
        // Se calcula la exactitud solo cuando existe al menos una muestra de evaluacion.
    }

    std::cout << "holdout_accuracy     = " << holdout_accuracy << std::endl;
    // Se imprime la exactitud de evaluacion posterior al entrenamiento.

    print_epoch_history(
        epoch_loss_pos,
        epoch_loss_neg,
        epoch_g_pos,
        epoch_g_neg,
        epoch_gap,
        epochs
    );
    // Se imprime el historial completo por epoca para verificar si aparece separacion FF.

    print_training_preview(g_pos, g_neg, gap, 12);
    // Se muestran algunas metricas por muestra de la ultima epoca de entrenamiento.

    print_prediction_preview(eval_true_labels, eval_pred_labels, 20, eval_start);
    // Se muestran varias parejas verdad-prediccion del subset de evaluacion.

    print_model_overview(weight_snapshot, bias_snapshot, 24);
    // Se resume el estado final de pesos y bias despues del entrenamiento local.

    print_epoch_freeze_justification(
        epoch_loss_pos,
        epoch_loss_neg,
        epoch_g_pos,
        epoch_g_neg,
        epoch_gap,
        weight_snapshot,
        bias_snapshot,
        epochs
    );
    // Se imprime una justificacion explicita cuando el historial queda congelado entre epocas.

    if (cfg.inspect_new_samples == true) {
        if ((cfg.inspect_start_sample >= 0) && (cfg.inspect_num_samples > 0) && (cfg.inspect_start_sample < total_samples)) {
            print_separator("NUEVOS SAMPLES PARA TEST");
            // Si se habilito la inspeccion adicional, se abre una seccion dedicada al nuevo rango.

            print_samples_range(input_words, cfg.inspect_start_sample, cfg.inspect_num_samples);
            // Se imprime el rango solicitado del dataset congelado.
        }
    }
}
