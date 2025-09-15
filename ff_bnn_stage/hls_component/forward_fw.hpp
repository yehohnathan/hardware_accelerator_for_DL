// forward_fw.hpp
#ifndef FORWARD_FW_HPP
#define FORWARD_FW_HPP

#include <cstdint>

// Metadatos básicos (rellenos por el testbench leyendo meta.json).
struct Meta {
    // Dimensión total del vector (imagen + tokens one-hot).
    uint32_t input_dim = 0;   // p.ej., 784+10 = 794

    // Nº de clases (dimensión del token one-hot).
    uint32_t token_dim = 10;  // p.ej., 10

    // Tamaño de imagen (para debug/ASCII en TB si quisieras).
    uint32_t rows = 0, cols = 0;

    // "prefix" o "suffix" (posición de los tokens).
    char where_tokens[8] = "suffix";

    // Configuración de empaquetado.
    char dtype[16] = "bitpacked"; // "bitpacked" o "float32"
    uint32_t word_bits = 0;       // 32/64 si bitpacked
    uint32_t bytes_per_row = 0;   // stride por muestra en binario

    // Conteos
    uint32_t n_pos = 0, n_neg = 0;
    uint32_t batch_size = 0;
};

// Ejemplo de top sintetizable (placeholder): suma los bytes de cada fila.
void forward_fw_top(const uint8_t* inputs,
                    const uint8_t* labels,
                    uint8_t* outputs,
                    uint32_t feat_dim,
                    uint32_t batch);

#endif // FORWARD_FW_HPP
