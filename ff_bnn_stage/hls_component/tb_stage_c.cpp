#include "tb_stage_c.hpp"

#include <chrono>
#include <iostream>
#include <vector>

/**
 * @brief Ejecuta entrenamiento, evaluacion y resumen de la Etapa C.
 *
 * @param dataset_header Contiene la metadata del binario ya cargado.
 * @param input_words Contiene el payload lineal del dataset sin cabecera.
 * @param total_samples Indica cuantas muestras contiene el payload.
 * @param seed Indica la semilla reproducible del experimento.
 * @param cfg Contiene la configuracion host del experimento modular.
 */
void run_stage_c_experiment(
    const ff_binary_header_t &dataset_header,
    const std::vector<word_t> &input_words,
    int total_samples,
    uint16_t seed,
    const stage_c_experiment_cfg_t &cfg
) {
    int train_samples = total_samples;
    // Se inicializa el tamano de entrenamiento con el total disponible.

    if ((cfg.train_samples_limit > 0) &&
        (train_samples > cfg.train_samples_limit)) {
        train_samples = cfg.train_samples_limit;
        // Se recorta el subset de entrenamiento cuando existe un limite host.
    }

    if (train_samples < 0) {
        train_samples = 0;
        // Se corrige cualquier valor invalido a cero.
    }

    int epochs = cfg.epochs;
    // Se copia la cantidad de epocas solicitadas por el experimento.

    if (epochs < 0) {
        epochs = 0;
        // Se corrige un valor invalido de epocas.
    }

    if (epochs > TRAIN_MAX_EPOCHS) {
        epochs = TRAIN_MAX_EPOCHS;
        // Se limita la cantidad de epocas al tamano del historial hardware.
    }

    int eval_samples = total_samples;
    // Se inicializa el tamano de evaluacion con el total disponible.

    if ((cfg.eval_samples_limit > 0) &&
        (eval_samples > cfg.eval_samples_limit)) {
        eval_samples = cfg.eval_samples_limit;
        // Se recorta el subset de evaluacion cuando existe un limite host.
    }

    if (eval_samples < 0) {
        eval_samples = 0;
        // Se corrige cualquier valor invalido a cero.
    }

    int eval_start = total_samples - eval_samples;
    // Se intenta colocar el subset de evaluacion al final del dataset.

    bool eval_reuses_train = false;
    // Se reserva una bandera para indicar solapamiento entre subsets.

    if (eval_start < train_samples) {
        eval_start = 0;
        eval_reuses_train = true;
        // Se marca el solapamiento cuando el hold-out no cabe al final.
    }

    if (eval_samples == 0) {
        eval_start = 0;
        // Se fija el offset a cero cuando no existe subset de evaluacion.
    }

    int train_batches = 0;
    // Se reserva la cantidad de mini-batches efectivos.

    if (train_samples > 0) {
        train_batches = (train_samples + MODEL_BATCH_SIZE - 1)
                      / MODEL_BATCH_SIZE;
        // Se calcula cuantas tandas reales requiere el subset actual.
    }

    int eval_hypotheses = eval_samples * NUM_CLASSES;
    // Se calcula cuantas hipotesis multiclase se evaluaran por pasada.

    std::vector<label_idx_t> dummy_true_labels(1, 0);
    std::vector<label_idx_t> dummy_pred_labels(1, 0);
    // Se reservan buffers minimos para llamadas sin fase de inferencia.

    std::vector<latent_t> weight_snapshot(MODEL_WEIGHT_COUNT);
    std::vector<bias_t> bias_snapshot(MODEL_BIAS_COUNT);
    std::vector<latent_t> previous_weight_snapshot(MODEL_WEIGHT_COUNT);
    std::vector<bias_t> previous_bias_snapshot(MODEL_BIAS_COUNT);
    // Se reservan los snapshots usados para seguir el estado del modelo.

    std::vector<goodness_t> g_pos(train_samples, 0);
    std::vector<goodness_t> g_neg(train_samples, 0);
    std::vector<goodness_t> gap(train_samples, 0);
    // Se reservan los buffers de goodness por muestra.

    std::vector<loss_t> epoch_loss_pos(TRAIN_MAX_EPOCHS, 0);
    std::vector<loss_t> epoch_loss_neg(TRAIN_MAX_EPOCHS, 0);
    std::vector<goodness_t> epoch_g_pos(TRAIN_MAX_EPOCHS, 0);
    std::vector<goodness_t> epoch_g_neg(TRAIN_MAX_EPOCHS, 0);
    std::vector<goodness_t> epoch_gap(TRAIN_MAX_EPOCHS, 0);
    // Se reservan los historiales globales por epoca.

    std::vector<loss_t> epoch_loss_pos_current(TRAIN_MAX_EPOCHS, 0);
    std::vector<loss_t> epoch_loss_neg_current(TRAIN_MAX_EPOCHS, 0);
    std::vector<goodness_t> epoch_g_pos_current(TRAIN_MAX_EPOCHS, 0);
    std::vector<goodness_t> epoch_g_neg_current(TRAIN_MAX_EPOCHS, 0);
    std::vector<goodness_t> epoch_gap_current(TRAIN_MAX_EPOCHS, 0);
    // Se reservan los buffers temporales de la llamada HLS actual.

    std::vector<ap_uint<32> > correct_count(1, 0);
    // Se reserva un buffer de una posicion para el contador de aciertos.

    std::vector<label_idx_t> eval_true_labels(eval_samples, 0);
    std::vector<label_idx_t> eval_pred_labels(eval_samples, 0);
    // Se reservan los buffers de verdad y prediccion para el hold-out.

    print_separator("CONFIGURACION DE ETAPA C");
    // Se abre una seccion con la configuracion efectiva del experimento.

    std::cout << "train_start       = 0" << std::endl;
    std::cout << "train_samples     = " << train_samples << std::endl;
    std::cout << "eval_start        = " << eval_start << std::endl;
    std::cout << "eval_samples      = " << eval_samples << std::endl;
    std::cout << "eval_reuses_train = "
              << (eval_reuses_train ? "true" : "false") << std::endl;
    std::cout << "epochs            = " << epochs << std::endl;
    std::cout << "train_batches     = " << train_batches << std::endl;
    std::cout << "input_mode        = " << dataset_header.mode_name << std::endl;
    std::cout << "image_resolution  = "
              << dataset_header.image_width << "x"
              << dataset_header.image_height << std::endl;
    std::cout << "logical_pixels    = " << NUM_LOGICAL_PIXELS << std::endl;
    std::cout << "pixel_bits        = " << INPUT_BITS_PER_PIXEL << std::endl;
    std::cout << "storage_bits      = " << INPUT_STORAGE_BITS << std::endl;
    std::cout << "words_per_sample  = " << WORDS_PER_SAMPLE << std::endl;
    std::cout << "padding_bits      = " << PADDING_BITS << std::endl;
    std::cout << "architecture      = "
              << MODEL_INPUT_BITS << " -> "
              << MODEL_LAYER1_NEURONS << " -> "
              << MODEL_LAYER2_NEURONS << std::endl;
    std::cout << "score_rule        = g1 + g2" << std::endl;
    std::cout << "batch_size        = " << MODEL_BATCH_SIZE << std::endl;
    std::cout << "eval_hypotheses   = " << eval_hypotheses << std::endl;
    std::cout << "label_scale       = " << LABEL_SCALE_HW.to_double() << std::endl;
    std::cout << "pixel_scale       = " << PIXEL_SCALE_HW.to_double() << std::endl;
    std::cout << "threshold         = "
              << GOODNESS_THRESHOLD_HW.to_double() << std::endl;
    std::cout << "learning_rate     = "
              << LEARNING_RATE_HW.to_double() << std::endl;
    // Se imprime la configuracion derivada del binario y de la build HLS.

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
    // Se captura el snapshot inicial del modelo antes de entrenar.

    if ((train_samples > 0) && (epochs > 0)) {
        print_separator("ACTUALIZACION POR EPOCA");
        // Se abre la seccion incremental solo cuando habra entrenamiento real.

epoch_train_loop:
        for (int epoch_idx = 0; epoch_idx < epochs; epoch_idx++) {
            std::chrono::steady_clock::time_point epoch_start =
                std::chrono::steady_clock::now();
            // Se captura el instante inicial de la epoca actual.

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
            // Se ejecuta exactamente una epoca de entrenamiento por llamada.

            epoch_loss_pos[epoch_idx] = epoch_loss_pos_current[0];
            epoch_loss_neg[epoch_idx] = epoch_loss_neg_current[0];
            epoch_g_pos[epoch_idx] = epoch_g_pos_current[0];
            epoch_g_neg[epoch_idx] = epoch_g_neg_current[0];
            epoch_gap[epoch_idx] = epoch_gap_current[0];
            // Se vuelcan las metricas de la epoca actual al historial global.

            double val_accuracy = 0.0;
            // Se reserva la accuracy de validacion de la epoca actual.

            if (eval_samples > 0) {
                const word_t *eval_ptr =
                    input_words.data() + (eval_start * WORDS_PER_SAMPLE);
                // Se calcula el puntero al inicio del subset de evaluacion.

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
                // Se ejecuta la inferencia hold-out de la epoca actual.

                val_accuracy =
                    ((double)((unsigned int)correct_count[0]))
                    / (double)eval_samples;
                // Se calcula la accuracy del hold-out actual.
            }

            std::chrono::steady_clock::time_point epoch_end =
                std::chrono::steady_clock::now();
            // Se captura el instante final de la epoca actual.

            double elapsed_sec =
                std::chrono::duration_cast<std::chrono::duration<double> >(
                    epoch_end - epoch_start
                ).count();
            // Se convierte la duracion de la epoca a segundos.

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
            // Se imprime la actualizacion visible de la epoca actual.

            print_epoch_model_delta(
                epoch_idx + 1,
                previous_weight_snapshot,
                weight_snapshot,
                previous_bias_snapshot,
                bias_snapshot
            );
            // Se resume si la epoca altero realmente el modelo latente.

            previous_weight_snapshot = weight_snapshot;
            previous_bias_snapshot = bias_snapshot;
            // Se actualizan los snapshots previos para la siguiente epoca.
        }
    } else if (eval_samples > 0) {
        const word_t *eval_ptr =
            input_words.data() + (eval_start * WORDS_PER_SAMPLE);
        // Se calcula el puntero al inicio del subset de evaluacion.

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
        // Si no hay entrenamiento, se ejecuta una inferencia pura por consistencia.
    }

    print_separator("RESULTADOS DE ENTRENAMIENTO DE ETAPA C");
    // Se abre la seccion de resultados principales de la etapa entrenable.

    std::cout << "holdout_correct_count = "
              << (unsigned int)correct_count[0] << std::endl;
    // Se imprime la cantidad total de aciertos del hold-out.

    double holdout_accuracy = 0.0;
    // Se reserva la exactitud final del subset de evaluacion.

    if (eval_samples > 0) {
        holdout_accuracy =
            ((double)((unsigned int)correct_count[0])) / (double)eval_samples;
        // Se calcula la exactitud final cuando existe hold-out.
    }

    std::cout << "holdout_accuracy     = " << holdout_accuracy << std::endl;
    // Se imprime la exactitud final de evaluacion.

    print_epoch_history(
        epoch_loss_pos,
        epoch_loss_neg,
        epoch_g_pos,
        epoch_g_neg,
        epoch_gap,
        epochs
    );
    // Se imprime el historial completo por epoca.

    print_training_preview(g_pos, g_neg, gap, 12);
    // Se muestran metricas por muestra de la ultima epoca entrenada.

    print_prediction_preview(eval_true_labels, eval_pred_labels, 20, eval_start);
    // Se muestran varias parejas verdad-prediccion del hold-out.

    print_model_overview(weight_snapshot, bias_snapshot, 24);
    // Se resume el estado final de pesos y bias tras el entrenamiento.

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
    // Se explica el congelamiento si el historial quedo estatico.

    if (cfg.inspect_new_samples == true) {
        if ((cfg.inspect_start_sample >= 0) &&
            (cfg.inspect_num_samples > 0) &&
            (cfg.inspect_start_sample < total_samples)) {
            print_separator("NUEVOS SAMPLES PARA TEST");
            // Se abre una seccion dedicada al rango adicional pedido por host.

            print_samples_range(
                input_words,
                cfg.inspect_start_sample,
                cfg.inspect_num_samples
            );
            // Se imprime el rango solicitado del dataset cargado.
        }
    }
}
