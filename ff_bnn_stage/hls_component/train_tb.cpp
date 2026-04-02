#include "tb_stage_b.hpp"   // Se incluye el módulo de validación independiente de la Etapa B.
#include "tb_stage_c.hpp"   // Se incluye el módulo de entrenamiento y evaluación de la Etapa C.
#include "debug_utils.hpp"  // Se incluyen utilidades comunes para lectura del binario y separación visual.

#include <iostream>          // Se incluye iostream para mensajes de estado del testbench.
#include <vector>            // Se incluye vector para almacenar el binario cargado en memoria.
#include <string>            // Se incluye string para manejar rutas del archivo de entrada.
#include <fstream>           // Se incluye fstream para verificar rutas candidatas del binario.
#include <cstdlib>           // Se incluye cstdlib para leer overrides opcionales desde variables de entorno.

// ============================================================
// Utilidad simple para localizar el binario de entrada
// ============================================================
static std::string resolve_input_path() {
    const char *candidate_paths[] = {
        "D:/TFG/hardware_accelerator_for_DL/ff_bnn_stage/mnist/data/processed/mnist_ff_input_packed.bin",
        "data/processed/mnist_ff_input_packed.bin",
        "/mnt/data/mnist_ff_input_packed.bin"
    };
    // Se define un conjunto pequeño de rutas candidatas para reutilizar el mismo testbench en distintos entornos.

resolve_path_loop:
    for (int i = 0; i < 3; i++) {
        std::ifstream file(candidate_paths[i], std::ios::binary);
        // Se intenta abrir cada ruta candidata solo para verificar si existe realmente en el entorno actual.

        if (file.is_open()) {
            file.close();
            // Si la ruta existe, se cierra de inmediato porque aquí solo interesa validar disponibilidad.

            return std::string(candidate_paths[i]);
            // Se retorna la primera ruta válida encontrada.
        }
    }

    return std::string(candidate_paths[0]);
    // Si ninguna ruta existe, se retorna la ruta principal para que el error posterior sea explícito.
}

// ============================================================
// Utilidades simples para overrides desde entorno
// ============================================================
static int read_env_int_or_default(const char *env_name, int default_value) {
    const char *env_value = std::getenv(env_name);
    // Se consulta la variable de entorno solicitada para permitir barridos sin editar el testbench.

    if (env_value == 0) {
        return default_value;
        // Si la variable no existe, se conserva el valor por defecto del experimento.
    }

    int parsed_value = std::atoi(env_value);
    // Se convierte el texto a entero usando una rutina simple y portable para el testbench.

    if (parsed_value <= 0) {
        return default_value;
        // Si el override es invalido o no positivo, se conserva el valor por defecto.
    }

    return parsed_value;
    // Se retorna el override valido para el experimento actual.
}

// ============================================================
// Función principal del testbench modular
// ============================================================
int main() {
    std::string file_path = resolve_input_path();
    // Se determina la ruta efectiva del binario de entrada según el entorno actual de simulación.

    std::vector<word_t> input_words;
    // Se reserva el buffer lineal donde se almacenarán todas las words leídas del archivo binario.

    bool read_ok = read_binary_file_words(file_path, input_words);
    // Se intenta leer el archivo completo usando la utilidad de depuración compartida.

    if (!read_ok) {
        std::cerr << "ERROR: no fue posible leer el binario de entrada." << std::endl;
        // Si la lectura falla, se reporta el problema y no se intenta seguir con el testbench.

        return 1;
        // Se finaliza la ejecución con código de error.
    }

    if (input_words.size() == 0) {
        std::cerr << "ERROR: el archivo está vacío." << std::endl;
        // Se verifica explícitamente que el binario no sea vacío antes de invocar cualquier módulo.

        return 1;
        // Se finaliza la ejecución con código de error.
    }

    if ((input_words.size() % WORDS_PER_SAMPLE) != 0) {
        std::cerr << "ERROR: la cantidad total de words no es múltiplo de " << WORDS_PER_SAMPLE << "." << std::endl;
        // Se verifica que el archivo contenga un número entero de muestras completas con 25 words cada una.

        return 1;
        // Se finaliza la ejecución con código de error.
    }

    int total_samples = (int)(input_words.size() / WORDS_PER_SAMPLE);
    // Se calcula cuántas muestras completas existen realmente en el archivo binario cargado.

    print_separator("INFORMACION GENERAL DEL BINARIO");
    // Se abre la primera sección del testbench con información general del dataset congelado.

    std::cout << "file_path      = " << file_path << std::endl;
    // Se imprime la ruta efectiva usada para la lectura del binario.

    std::cout << "total_words    = " << input_words.size() << std::endl;
    // Se imprime la cantidad total de words de 32 bits leídas del archivo.

    std::cout << "words/sample   = " << WORDS_PER_SAMPLE << std::endl;
    // Se recuerda que cada muestra ocupa exactamente 25 words.

    std::cout << "total_samples  = " << total_samples << std::endl;
    // Se imprime la cantidad total de muestras detectadas en el binario.

    uint16_t seed = 0x1234;
    // Se fija una semilla determinista para reproducir resultados entre simulaciones consecutivas.

    bool stage_b_ok = run_stage_b_validation(input_words, total_samples, seed);
    // Se ejecuta la validación modular de la Etapa B como un bloque independiente del entrenamiento FF.

    print_separator("RESUMEN GLOBAL DE MODULOS");
    // Se imprime una pequeña sección de estado global antes de pasar a la siguiente etapa.

    std::cout << "stage_b_ok     = " << (stage_b_ok ? "true" : "false") << std::endl;
    // Se reporta el veredicto general retornado por el módulo de validación de la Etapa B.

    stage_c_experiment_cfg_t cfg;
    // Se crea la estructura de configuración del experimento modular de la Etapa C.

    cfg.train_samples_limit = read_env_int_or_default("FF_TRAIN_SAMPLES", 5000);
    // Se fija un default de 1024 muestras porque fue suficiente para comparar configuraciones con menor ruido.

    cfg.eval_samples_limit = read_env_int_or_default("FF_EVAL_SAMPLES", 128);
    // Se fija un hold-out de 128 muestras porque permite medir mejor la accuracy sin disparar demasiado el tiempo.

    cfg.epochs = read_env_int_or_default("FF_EPOCHS", 30);
    // Se fijan 3 epocas por defecto para detectar rapidamente si el modelo ya se esta moviendo.

    cfg.inspect_new_samples = false;
    // Se habilita la inspección visual de algunas muestras alejadas al final del experimento.

    cfg.inspect_start_sample = 50;
    // Se fija el inicio del rango adicional a inspeccionar cuando el dataset sea suficientemente grande.

    cfg.inspect_num_samples = 5;
    // Se define cuántas muestras consecutivas se imprimirán en esa inspección adicional.

    run_stage_c_experiment(input_words, total_samples, seed, cfg);
    // Se ejecuta el módulo independiente de la Etapa C encargado de entrenar, evaluar y resumir el modelo FF.

    print_separator("FIN DEL TESTBENCH");
    // Se imprime la sección final del testbench indicando que la ejecución modular completa terminó.

    return 0;
    // Se finaliza la simulación con código de éxito.
}
