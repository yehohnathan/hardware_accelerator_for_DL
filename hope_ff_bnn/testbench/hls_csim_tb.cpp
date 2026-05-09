
/**
 * @file hls_csim_tb.cpp
 * @brief Testbench mínimo para validar los top functions con Vitis HLS csim.
 *
 * Este archivo es código de host/testbench, no sintetizable. Carga el dataset,
 * reserva buffers planos que imitan memoria externa AXI, ejecuta
 * `ff_train_kernel` y luego `ff_infer_kernel`. Su objetivo es comprobar que la
 * configuración seleccionada por Makefile puede pasar por la ruta de C
 * simulation usada en Vivado/Vitis Unified IDE 2024.2.
 */
#include "ff_dataset.hpp"
#include "ff_infer.hpp"
#include "ff_train.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Lee un entero positivo desde una variable de entorno.
 *
 * @param name Nombre de la variable, por ejemplo `FF_TRAIN_SAMPLES`.
 * @param default_value Valor usado si la variable falta o no es positiva.
 * @return Entero validado para controlar csim.
 *
 * @note Solo host/testbench. Permite que `make csim-run` configure muestras y
 * épocas sin cambiar el archivo HLS config.
 */
static int read_env_int(const char *name, int default_value) {
    const char *value = std::getenv(name);
    if (value == 0 || value[0] == '\0') {
        return default_value;
    }

    int parsed = std::atoi(value);
    return (parsed > 0) ? parsed : default_value;
}

/**
 * @brief Lee una semilla de 32 bits desde el entorno.
 *
 * @param name Nombre de la variable de entorno.
 * @param default_value Semilla por defecto para reproducibilidad.
 * @return Semilla validada.
 */
static uint32_t read_env_u32(const char *name, uint32_t default_value) {
    const char *value = std::getenv(name);
    if (value == 0 || value[0] == '\0') {
        return default_value;
    }

    unsigned long parsed = std::strtoul(value, 0, 0);
    return (parsed == 0ul) ? default_value : (uint32_t)parsed;
}

/**
 * @brief Ejecuta la prueba C++ de compatibilidad HLS.
 *
 * @return 0 si entrenamiento e inferencia completan y el modelo cambia.
 *
 * El testbench usa `std::vector` e I/O de consola, por lo que no es
 * sintetizable. Los datos en esos vectores se pasan a los top functions como
 * punteros, simulando buffers de memoria externa.
 */
int main() {
    std::string dataset_path = ff_default_dataset_path();
    ff_dataset_header_t header;
    std::vector<ff_word_t> payload_words;
    std::string error;

    if (!ff_read_dataset(dataset_path, header, payload_words, error)) {
        std::cerr << "HLS_CSIM_ERROR dataset: " << error << '\n';
        return 1;
    }

    int total_samples = (int)header.sample_count;
    int train_samples = std::min(read_env_int("FF_TRAIN_SAMPLES", 16),
                                 total_samples);
    int eval_samples = std::min(read_env_int("FF_EVAL_SAMPLES", 8),
                                total_samples);
    int epochs = read_env_int("FF_EPOCHS", 1);
    uint32_t seed = read_env_u32("FF_SEED", 0x1234u);

    std::vector<ff_latent_t> weights((size_t)FF_MODEL_WEIGHT_COUNT, 0);
    std::vector<ff_bias_t> biases((size_t)FF_MODEL_BIAS_COUNT, 0);
    std::vector<ff_goodness_t> epoch_g_pos((size_t)FF_MAX_EPOCHS, 0);
    std::vector<ff_goodness_t> epoch_g_neg((size_t)FF_MAX_EPOCHS, 0);
    std::vector<ff_goodness_t> epoch_gap((size_t)FF_MAX_EPOCHS, 0);
    std::vector<uint32_t> changed(2u, 0u);
    std::vector<uint8_t> predictions((size_t)eval_samples, 0u);
    std::vector<uint32_t> correct(1u, 0u);

    std::cout << "HLS_CSIM_DATASET=" << dataset_path << '\n';
    std::cout << "HLS_CSIM_PART_COMPAT=Artix7_Nexys_A7_100T\n";
    std::cout << "===== CONFIGURACION DE RED =====\n";
    std::cout << "Dataset: " << dataset_path << '\n';
    std::cout << "Input dim: " << FF_INPUT_DIM << '\n';
    std::cout << "Hidden layers: " << FF_HIDDEN_LAYERS << '\n';
    std::cout << "Hidden values: ";
    for (int layer = 0; layer < FF_HIDDEN_LAYERS; layer++) {
        if (layer != 0) {
            std::cout << " -> ";
        }
        std::cout << FF_LAYER_NEURONS[layer];
    }
    std::cout << '\n';
    std::cout << "Parallel neurons: " << FF_PARALLEL_NEURONS << '\n';
    std::cout << "Architecture: " << FF_INPUT_DIM;
    for (int layer = 0; layer < FF_HIDDEN_LAYERS; layer++) {
        std::cout << " -> " << FF_LAYER_NEURONS[layer];
    }
    std::cout << "\n================================\n";
    std::cout << "HLS_CSIM_MODEL=" << FF_INPUT_DIM;
    for (int layer = 0; layer < FF_HIDDEN_LAYERS; layer++) {
        std::cout << "->" << FF_LAYER_NEURONS[layer];
    }
    std::cout << " parallel=" << FF_PARALLEL_NEURONS << '\n';
    std::cout << "HLS_CSIM_TRAIN_SAMPLES=" << train_samples << '\n';
    std::cout << "HLS_CSIM_EVAL_SAMPLES=" << eval_samples << '\n';
    std::cout << "HLS_CSIM_EPOCHS=" << epochs << '\n';

/* Primer kernel bajo prueba: entrena con reset para que la simulación sea
 * determinista y exporta el modelo a buffers planos.
 */
    ff_train_kernel(
        payload_words.data(),
        weights.data(),
        biases.data(),
        epoch_g_pos.data(),
        epoch_g_neg.data(),
        epoch_gap.data(),
        changed.data(),
        train_samples,
        epochs,
        seed,
        1
    );

/* Segundo kernel bajo prueba: recarga los buffers exportados y mide inferencia
 * sobre una porción pequeña del dataset.
 */
    ff_infer_kernel(
        payload_words.data(),
        weights.data(),
        biases.data(),
        predictions.data(),
        correct.data(),
        eval_samples
    );

    std::cout << "HLS_CSIM_EPOCH0_G_POS=" << epoch_g_pos[0] << '\n';
    std::cout << "HLS_CSIM_EPOCH0_G_NEG=" << epoch_g_neg[0] << '\n';
    std::cout << "HLS_CSIM_EPOCH0_GAP=" << epoch_gap[0] << '\n';
    std::cout << "HLS_CSIM_CHANGED_WEIGHTS=" << changed[0] << '\n';
    std::cout << "HLS_CSIM_CHANGED_BIASES=" << changed[1] << '\n';
    std::cout << "HLS_CSIM_CORRECT=" << correct[0] << '/' << eval_samples
              << '\n';

/* Si no cambia ningún parámetro entrenable, la csim no demuestra aprendizaje y
 * debe fallar para detectar problemas de configuración o umbral.
 */
    if ((changed[0] == 0u) && (changed[1] == 0u)) {
        std::cerr << "HLS_CSIM_ERROR no model state changed\n";
        return 2;
    }

    std::cout << "HLS_CSIM_TB_OK\n";
    return 0;
}
