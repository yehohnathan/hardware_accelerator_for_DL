#ifndef TB_STAGE_B_HPP
#define TB_STAGE_B_HPP

#include <vector>         // Se incluye std::vector para recibir el buffer lineal del binario ya leído.
#include <stdint.h>       // Se incluye uint16_t para la semilla reproducible de la Etapa B.
#include "forward_fw.hpp" // Se reutilizan tipos y funciones del kernel principal.

// ============================================================
// Validación modular de la Etapa B
// ============================================================
// Se declara la rutina que verifica la preparación de pares FF sin mezclarla con la Etapa C.
bool run_stage_b_validation(
    const std::vector<word_t> &input_words,
    int total_samples,
    uint16_t seed
);

#endif
