#ifndef TB_STAGE_C_HPP
#define TB_STAGE_C_HPP

#include <stdint.h>
#include <vector>

#include "debug_utils.hpp"
#include "forward_fw.hpp"

/**
 * @brief Reune la configuracion host del experimento modular de Etapa C.
 */
struct stage_c_experiment_cfg_t {
    int train_samples_limit;
    // Se define el maximo de muestras usadas en entrenamiento.

    int eval_samples_limit;
    // Se define el maximo de muestras usadas en evaluacion hold-out.

    int epochs;
    // Se define la cantidad de epocas del entrenamiento FF.

    bool inspect_new_samples;
    // Se define si se imprime un rango adicional del dataset al final.

    int inspect_start_sample;
    // Se define el indice inicial del rango adicional a inspeccionar.

    int inspect_num_samples;
    // Se define cuantas muestras consecutivas se imprimen en esa inspeccion.
};

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
);

#endif
