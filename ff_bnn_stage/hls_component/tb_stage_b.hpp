#ifndef TB_STAGE_B_HPP
#define TB_STAGE_B_HPP

#include <stdint.h>
#include <vector>

#include "debug_utils.hpp"
#include "forward_fw.hpp"

/**
 * @brief Ejecuta la validacion modular de la Etapa B.
 *
 * @param dataset_header Contiene la metadata del binario ya cargado.
 * @param input_words Contiene el payload lineal del dataset sin cabecera.
 * @param total_samples Indica cuantas muestras contiene el payload.
 * @param seed Indica la semilla reproducible usada por la Etapa B.
 *
 * @return Retorna true cuando la preparacion de pares positivos y negativos
 * mantiene la consistencia esperada.
 */
bool run_stage_b_validation(
    const ff_binary_header_t &dataset_header,
    const std::vector<word_t> &input_words,
    int total_samples,
    uint16_t seed
);

#endif
