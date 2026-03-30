#ifndef TB_STAGE_C_HPP
#define TB_STAGE_C_HPP

#include <vector>         // Se incluye std::vector para transportar el dataset ya leído en memoria.
#include <stdint.h>       // Se incluye uint16_t para recibir una semilla reproducible.
#include "forward_fw.hpp" // Se reutilizan tipos y constantes del núcleo HLS.

// ============================================================
// Configuración del experimento modular de la Etapa C
// ============================================================
struct stage_c_experiment_cfg_t {
    int train_samples_limit;
    // Se define el máximo de muestras de entrenamiento permitidas para la corrida actual.

    int epochs;
    // Se define la cantidad de épocas que ejecutará el kernel entrenable.

    bool inspect_new_samples;
    // Se define si al final se inspeccionará un rango adicional del dataset para depuración visual.

    int inspect_start_sample;
    // Se define el índice inicial del rango adicional a inspeccionar.

    int inspect_num_samples;
    // Se define cuántas muestras consecutivas se imprimirán en esa inspección adicional.
};

// ============================================================
// Ejecución modular de la Etapa C
// ============================================================
// Se declara la rutina que entrena, evalúa y resume la primera capa FF entrenable.
void run_stage_c_experiment(
    const std::vector<word_t> &input_words,
    int total_samples,
    uint16_t seed,
    const stage_c_experiment_cfg_t &cfg
);

#endif
