#include "tb_stage_b.hpp"
#include "tb_stage_c.hpp"
#include "debug_utils.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/**
 * @brief Construye el nombre canonico del binario esperado por la build.
 *
 * @return Retorna el nombre de archivo asociado a la resolucion y al modo
 * configurados en forward_fw.hpp.
 */
static std::string build_default_input_name() {
    std::ostringstream oss;
    // Se construye el nombre del binario a partir de la build HLS activa.

    oss << "mnist_"
        << IMAGE_WIDTH << "x" << IMAGE_HEIGHT
        << "_" << INPUT_BITS_PER_PIXEL << "b_packed.bin";
    // Se genera el nombre de archivo esperado por el dataset actual.

    return oss.str();
    // Se retorna el nombre canonico del binario esperado.
}

/**
 * @brief Localiza el binario de entrada segun la build y el entorno.
 *
 * @return Retorna la ruta del binario que se intentara abrir.
 *
 * @note La variable de entorno FF_INPUT_BIN_PATH tiene prioridad absoluta.
 * @note Si no se define un override, se intenta abrir el nombre canonico
 * generado a partir de IMAGE_WIDTH, IMAGE_HEIGHT e INPUT_BITS_PER_PIXEL.
 */
static std::string resolve_input_path() {
    const char *env_input_path = std::getenv("FF_INPUT_BIN_PATH");
    // Se consulta primero un override explicito desde variables de entorno.

    if (env_input_path != 0) {
        std::ifstream env_file(env_input_path, std::ios::binary);
        // Se comprueba si la ruta dada por entorno realmente existe.

        if (env_file.is_open()) {
            env_file.close();
            return std::string(env_input_path);
            // Se retorna el override cuando la ruta es valida.
        }
    }

    std::string input_name = build_default_input_name();
    // Se construye el nombre canonico del binario esperado por la build actual.

    const char *candidate_dirs[] = {
        "D:/TFG/hardware_accelerator_for_DL/ff_bnn_stage/mnist/data/processed/",
        "ff_bnn_stage/mnist/data/processed/",
        "../mnist/data/processed/",
        "data/processed/",
    };
    // Se definen varias rutas candidatas para reutilizar el testbench.

    for (int i = 0; i < 4; i++) {
        std::string candidate = std::string(candidate_dirs[i]) + input_name;
        std::ifstream file(candidate.c_str(), std::ios::binary);
        // Se intenta abrir cada ruta candidata para verificar si existe.

        if (file.is_open()) {
            file.close();
            return candidate;
            // Se retorna la primera ruta valida encontrada.
        }
    }

    return std::string(candidate_dirs[0]) + input_name;
    // Se retorna la ruta principal para que un fallo posterior sea explicito.
}

/**
 * @brief Lee un override entero desde el entorno o usa el valor por defecto.
 *
 * @param env_name Indica el nombre de la variable de entorno consultada.
 * @param default_value Indica el valor por defecto del experimento.
 *
 * @return Retorna el override leido o el valor por defecto si no es valido.
 */
static int read_env_int_or_default(
    const char *env_name,
    int default_value
) {
    const char *env_value = std::getenv(env_name);
    // Se consulta la variable de entorno solicitada por el experimento.

    if (env_value == 0) {
        return default_value;
        // Se conserva el valor por defecto si la variable no existe.
    }

    int parsed_value = std::atoi(env_value);
    // Se convierte el texto de la variable a entero de forma simple.

    if (parsed_value <= 0) {
        return default_value;
        // Se conserva el valor por defecto si el override es invalido.
    }

    return parsed_value;
    // Se retorna el override valido para el experimento actual.
}

/**
 * @brief Ejecuta el testbench modular completo del acelerador FF.
 *
 * @return Retorna 0 cuando la simulacion termina correctamente.
 *
 * @note Las variables de entorno FF_TRAIN_SAMPLES, FF_EVAL_SAMPLES y
 * FF_EPOCHS permiten ajustar el experimento sin recompilar.
 */
int main() {
    std::string file_path = resolve_input_path();
    // Se determina la ruta efectiva del binario de entrada.

    ff_binary_header_t dataset_header;
    std::vector<word_t> input_words;
    // Se reservan la cabecera y el payload del binario cargado.

    bool read_ok = read_binary_dataset(file_path, dataset_header, input_words);
    // Se intenta leer el archivo completo y separar cabecera y payload.

    if (!read_ok) {
        std::cerr << "ERROR: no fue posible leer el binario de entrada." << std::endl;
        return 1;
        // Se detiene el testbench si la lectura del binario falla.
    }

    bool layout_ok = validate_binary_header_against_build(dataset_header);
    // Se comprueba si la build HLS coincide con la metadata del binario.

    if (!layout_ok) {
        return 1;
        // Se detiene el testbench cuando el binario no coincide con la build.
    }

    if (input_words.empty()) {
        std::cerr << "ERROR: el payload del archivo esta vacio." << std::endl;
        return 1;
        // Se verifica que el payload no sea vacio antes de seguir.
    }

    int total_samples = (int)dataset_header.sample_count;
    // Se recupera la cantidad de muestras desde la cabecera ya validada.

    print_binary_header_summary(file_path, dataset_header, input_words);
    // Se imprime el resumen completo del archivo cargado.

    uint16_t seed = 0x1234;
    // Se fija una semilla determinista para reproducir resultados.

    bool stage_b_ok = run_stage_b_validation(
        dataset_header,
        input_words,
        total_samples,
        seed
    );
    // Se ejecuta la validacion modular de la Etapa B.

    print_separator("RESUMEN GLOBAL DE MODULOS");
    // Se imprime una pequena seccion de estado global.

    std::cout << "stage_b_ok     = "
              << (stage_b_ok ? "true" : "false") << std::endl;
    // Se reporta el veredicto general retornado por la Etapa B.

    stage_c_experiment_cfg_t cfg;
    // Se crea la estructura de configuracion de la Etapa C.

    cfg.train_samples_limit = read_env_int_or_default("FF_TRAIN_SAMPLES", 5000);
    cfg.eval_samples_limit = read_env_int_or_default("FF_EVAL_SAMPLES", 128);
    cfg.epochs = read_env_int_or_default("FF_EPOCHS", 30);
    cfg.inspect_new_samples = false;
    cfg.inspect_start_sample = 50;
    cfg.inspect_num_samples = 5;
    // Se fija una configuracion host simple y reutilizable para el experimento.

    run_stage_c_experiment(
        dataset_header,
        input_words,
        total_samples,
        seed,
        cfg
    );
    // Se ejecuta el experimento modular de la Etapa C.

    print_separator("FIN DEL TESTBENCH");
    // Se imprime la seccion final del testbench.

    return 0;
    // Se finaliza la simulacion con codigo de exito.
}
