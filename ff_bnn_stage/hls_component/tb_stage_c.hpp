#ifndef TB_STAGE_C_HPP
#define TB_STAGE_C_HPP

#include <stdint.h>       // Se incluye uint16_t para la semilla reproducible del experimento.
#include <vector>         // Se incluye std::vector para transportar el dataset ya cargado.

#include "forward_fw.hpp" // Se reutilizan tipos y constantes del nucleo HLS.

// ============================================================
// Configuracion del experimento modular de Etapa C
// ============================================================
struct stage_c_experiment_cfg_t {
    int train_samples_limit;
    // Se define el maximo de muestras usadas en la fase de entrenamiento.

    int eval_samples_limit;
    // Se define el maximo de muestras usadas en la fase de evaluacion hold-out.

    int epochs;
    // Se define la cantidad de epocas del entrenamiento FF.

    bool inspect_new_samples;
    // Se define si se imprime un rango adicional del dataset al final.

    int inspect_start_sample;
    // Se define el indice inicial del rango adicional a inspeccionar.

    int inspect_num_samples;
    // Se define cuantas muestras consecutivas se imprimen en esa inspeccion.
};

// ============================================================
// Ejecucion modular de la Etapa C
// ============================================================
void run_stage_c_experiment(
    const std::vector<word_t> &input_words,
    int total_samples,
    uint16_t seed,
    const stage_c_experiment_cfg_t &cfg
);  // Se declara la rutina que entrena, evalua y resume la capa FF entrenable.

#endif
