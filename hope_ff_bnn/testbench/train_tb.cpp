
/**
 * @file train_tb.cpp
 * @brief Testbench principal de entrenamiento Forward-Forward en C++.
 *
 * Este archivo es código de host/testbench. No es sintetizable porque usa STL,
 * sistema de archivos, parsing de argumentos y generación de reportes. Su papel
 * es ejecutar el flujo académico completo: cargar MNIST, validar arquitectura,
 * entrenar online, evaluar accuracy, demostrar cambios en pesos/bias y guardar
 * `metrics.csv` y `summary.md` para análisis posterior.
 */
#include "ff_dataset.hpp"
#include "ff_infer.hpp"
#include "ff_train.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/**
 * @brief Opciones configurables de una corrida de simulación.
 *
 * La estructura refleja los argumentos que `make run` pasa al testbench. Los
 * campos de arquitectura se validan contra macros compiladas para evitar que el
 * log declare una red distinta de la que realmente se ejecutó.
 */
struct run_options_t {
    std::string dataset_path;
    std::string run_name = "manual";
    std::string output_dir;
    std::string group_name = "custom";
    int train_samples = 1024;
    int eval_samples = 512;
    int epochs = 5;
    int eval_every_epoch = 1;
    int hidden_layers = FF_HIDDEN_LAYERS;
    std::string hidden_values;
    int parallel_neurons = FF_PARALLEL_NEURONS;
    uint32_t seed = 0x1234u;
};

/**
 * @brief Fila de métricas emitida por época.
 *
 * Combina goodness, eventos de actualización, cambios de estado y accuracy para
 * que los scripts puedan comparar datasets y arquitecturas.
 */
struct epoch_row_t {
    int epoch;
    double accuracy;
    ff_goodness_t avg_g_pos;
    ff_goodness_t avg_g_neg;
    ff_goodness_t avg_gap;
    uint32_t pos_events;
    uint32_t neg_events;
    uint32_t changed_weights;
    uint32_t changed_biases;
    uint32_t sign_changes;
    double elapsed_sec;
};

/**
 * @brief Lee un entero positivo desde el entorno.
 *
 * @param name Variable usada por Make o por ejecución manual.
 * @param default_value Valor por defecto si la variable no existe.
 * @return Entero positivo o el valor por defecto.
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
 * @brief Lee una semilla entera desde el entorno.
 *
 * @param name Variable de entorno.
 * @param default_value Semilla por defecto.
 * @return Semilla validada para inicializar el modelo y etiquetas negativas.
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
 * @brief Compara un argumento CLI con una opción esperada.
 *
 * @param arg Token recibido en `argv`.
 * @param expected Nombre de opción esperado.
 * @return `true` si ambos textos coinciden.
 */
static bool arg_eq(const char *arg, const char *expected) {
    return std::string(arg) == std::string(expected);
}

/**
 * @brief Imprime la interfaz de línea de comandos del testbench.
 *
 * @param program_name Nombre del ejecutable.
 */
static void print_usage(const char *program_name) {
    std::cout
        << "Uso: " << program_name << " [opciones]\n"
        << "  --dataset <ruta_bin>\n"
        << "  --train-samples <N>\n"
        << "  --eval-samples <N>\n"
        << "  --epochs <N>\n"
        << "  --eval-every-epoch <0|1>\n"
        << "  --run-name <nombre>\n"
        << "  --group <nombre>\n"
        << "  --output-dir <carpeta>\n"
        << "  --hidden-layers <N>\n"
        << "  --hidden-values \"N0 N1 ...\"\n"
        << "  --parallel-neurons <N>\n"
        << "  --seed <entero>\n";
}

/**
 * @brief Convierte la arquitectura compilada a lista de neuronas ocultas.
 *
 * @return Texto compatible con `--hidden-values`, por ejemplo `"32 8"`.
 */
static std::string compiled_hidden_values_string() {
    std::ostringstream out;
    for (int layer = 0; layer < FF_HIDDEN_LAYERS; layer++) {
        if (layer != 0) {
            out << ' ';
        }
        out << FF_LAYER_NEURONS[layer];
    }
    return out.str();
}

/**
 * @brief Construye la arquitectura completa como texto.
 *
 * @return Cadena como `410 -> 32 -> 8`, usada en consola y reportes.
 */
static std::string architecture_string() {
    std::ostringstream out;
    out << FF_INPUT_DIM;
    for (int layer = 0; layer < FF_HIDDEN_LAYERS; layer++) {
        out << " -> " << FF_LAYER_NEURONS[layer];
    }
    return out.str();
}

/**
 * @brief Construye solo la parte oculta de la arquitectura.
 *
 * @return Cadena como `32 -> 8`.
 */
static std::string hidden_arrow_string() {
    std::ostringstream out;
    for (int layer = 0; layer < FF_HIDDEN_LAYERS; layer++) {
        if (layer != 0) {
            out << " -> ";
        }
        out << FF_LAYER_NEURONS[layer];
    }
    return out.str();
}

/**
 * @brief Parsea la lista de neuronas solicitada por Make.
 *
 * @param text Texto separado por espacios.
 * @return Vector de enteros con una entrada por capa oculta.
 */
static std::vector<int> parse_hidden_values_list(const std::string &text) {
    std::vector<int> values;
    std::istringstream in(text);
    int value = 0;
    while (in >> value) {
        values.push_back(value);
    }
    return values;
}

/**
 * @brief Genera una etiqueta compacta para carpetas de resultados.
 *
 * @return Texto como `L2_32_8_P8`.
 */
static std::string architecture_tag() {
    std::ostringstream out;
    out << 'L' << FF_HIDDEN_LAYERS;
    for (int layer = 0; layer < FF_HIDDEN_LAYERS; layer++) {
        out << '_' << FF_LAYER_NEURONS[layer];
    }
    out << "_P" << FF_PARALLEL_NEURONS;
    return out.str();
}

/**
 * @brief Parsea y valida todos los argumentos de la corrida.
 *
 * @param argc Cantidad de argumentos.
 * @param argv Arreglo de argumentos.
 * @return Opciones finales normalizadas.
 *
 * @sideeffect Puede finalizar con `std::exit(2)` si la arquitectura solicitada
 * no coincide con las macros compiladas. Esta validación protege la trazabilidad
 * de experimentos generados por Makefile.
 */
static run_options_t parse_args(int argc, char **argv) {
    run_options_t opt;

    opt.dataset_path = ff_default_dataset_path();
    opt.train_samples = read_env_int("FF_TRAIN_SAMPLES", opt.train_samples);
    opt.eval_samples = read_env_int("FF_EVAL_SAMPLES", opt.eval_samples);
    opt.epochs = read_env_int("FF_EPOCHS", opt.epochs);
    opt.eval_every_epoch =
        read_env_int("FF_EVAL_EVERY_EPOCH", opt.eval_every_epoch);
    opt.seed = read_env_u32("FF_SEED", opt.seed);
    opt.hidden_values = compiled_hidden_values_string();

    for (int i = 1; i < argc; i++) {
        if (arg_eq(argv[i], "--help") || arg_eq(argv[i], "-h")) {
            print_usage(argv[0]);
            std::exit(0);
        }

        if ((i + 1) >= argc) {
            std::cerr << "ERROR: falta valor para argumento " << argv[i] << '\n';
            std::exit(2);
        }

        const char *value = argv[i + 1];
        if (arg_eq(argv[i], "--dataset")) {
            opt.dataset_path = value;
        } else if (arg_eq(argv[i], "--train-samples")) {
            opt.train_samples = std::atoi(value);
        } else if (arg_eq(argv[i], "--eval-samples")) {
            opt.eval_samples = std::atoi(value);
        } else if (arg_eq(argv[i], "--epochs")) {
            opt.epochs = std::atoi(value);
        } else if (arg_eq(argv[i], "--eval-every-epoch")) {
            opt.eval_every_epoch = std::atoi(value);
        } else if (arg_eq(argv[i], "--run-name")) {
            opt.run_name = value;
        } else if (arg_eq(argv[i], "--group")) {
            opt.group_name = value;
        } else if (arg_eq(argv[i], "--output-dir")) {
            opt.output_dir = value;
        } else if (arg_eq(argv[i], "--hidden-layers")) {
            opt.hidden_layers = std::atoi(value);
        } else if (arg_eq(argv[i], "--hidden-values")) {
            opt.hidden_values = value;
        } else if (arg_eq(argv[i], "--parallel-neurons")) {
            opt.parallel_neurons = std::atoi(value);
        } else if (arg_eq(argv[i], "--seed")) {
            opt.seed = (uint32_t)std::strtoul(value, 0, 0);
        } else {
            std::cerr << "ERROR: argumento no reconocido: " << argv[i] << '\n';
            print_usage(argv[0]);
            std::exit(2);
        }
        i++;
    }

    if (opt.train_samples < 0) {
        opt.train_samples = 0;
    }
    if (opt.eval_samples < 0) {
        opt.eval_samples = 0;
    }
    if (opt.epochs < 0) {
        opt.epochs = 0;
    }
    if (opt.eval_every_epoch != 0) {
        opt.eval_every_epoch = 1;
    }

    std::vector<int> parsed_hidden = parse_hidden_values_list(opt.hidden_values);
/* La arquitectura se recibe por CLI para que el log sea explícito, pero el
 * ejecutable ya fue compilado con macros fijas. Estos chequeos detectan
 * desalineaciones entre comando, binario y dataset.
 */
    if (opt.hidden_layers != (int)parsed_hidden.size()) {
        std::cerr << "ERROR: HIDDEN_LAYERS no coincide con HIDDEN_VALUES.\n";
        std::exit(2);
    }
    if (opt.hidden_layers != FF_HIDDEN_LAYERS) {
        std::cerr << "ERROR: --hidden-layers no coincide con macros compiladas.\n";
        std::exit(2);
    }
    if (opt.parallel_neurons <= 0) {
        std::cerr << "ERROR: PARALLEL_NEURONS debe ser mayor que cero.\n";
        std::exit(2);
    }
    if (opt.parallel_neurons != FF_PARALLEL_NEURONS) {
        std::cerr << "ERROR: --parallel-neurons no coincide con macros compiladas.\n";
        std::exit(2);
    }
    for (int layer = 0; layer < opt.hidden_layers; layer++) {
        if (parsed_hidden[(size_t)layer] <= 0) {
            std::cerr << "ERROR: cada valor de HIDDEN_VALUES debe ser positivo.\n";
            std::exit(2);
        }
        if (parsed_hidden[(size_t)layer] != FF_LAYER_NEURONS[layer]) {
            std::cerr << "ERROR: --hidden-values no coincide con macros compiladas.\n";
            std::exit(2);
        }
    }

    return opt;
}

/** @brief Extrae el stem del archivo de dataset para reportes. */
static std::string dataset_stem_from_path(const std::string &path) {
    return std::filesystem::path(path).stem().string();
}

/**
 * @brief Normaliza el nombre de dataset quitando el sufijo `_packed`.
 *
 * @param path Ruta del binario.
 * @return Nombre lógico usado en `summary.md`.
 */
static std::string dataset_stem_without_packed(const std::string &path) {
    std::string stem = dataset_stem_from_path(path);
    const std::string suffix = "_packed";
    if (stem.size() >= suffix.size() &&
        stem.substr(stem.size() - suffix.size()) == suffix) {
        stem.resize(stem.size() - suffix.size());
    }
    return stem;
}

/**
 * @brief Escribe las métricas de entrenamiento/evaluación en CSV.
 *
 * @param output_dir Directorio de resultados de la corrida.
 * @param rows Filas por época evaluada.
 *
 * @sideeffect Crea o sobrescribe `metrics.csv`.
 */
static void write_metrics_csv(
    const std::string &output_dir,
    const std::vector<epoch_row_t> &rows
) {
    std::ofstream out(output_dir + "/metrics.csv");
    out << "epoch,accuracy,avg_g_pos,avg_g_neg,avg_gap,"
        << "pos_events,neg_events,changed_weights,changed_biases,"
        << "sign_changes,elapsed_sec\n";

    for (size_t i = 0; i < rows.size(); i++) {
        out << rows[i].epoch << ','
            << std::fixed << std::setprecision(6) << rows[i].accuracy << ','
            << rows[i].avg_g_pos << ','
            << rows[i].avg_g_neg << ','
            << rows[i].avg_gap << ','
            << rows[i].pos_events << ','
            << rows[i].neg_events << ','
            << rows[i].changed_weights << ','
            << rows[i].changed_biases << ','
            << rows[i].sign_changes << ','
            << std::fixed << std::setprecision(6) << rows[i].elapsed_sec
            << '\n';
    }
}

/**
 * @brief Escribe el resumen Markdown consumido por los scripts de análisis.
 *
 * @param output_dir Directorio del experimento.
 * @param opt Opciones efectivas de la corrida.
 * @param header Cabecera del dataset validado.
 * @param dataset_path Ruta original del binario.
 * @param final_row Métricas finales.
 * @param final_changed_weights Pesos latentes modificados.
 * @param final_changed_biases Bias modificados.
 * @param final_sign_changes Cambios de signo binario.
 * @param status Estado final `OK` o `FAIL`.
 *
 * @sideeffect Crea o sobrescribe `summary.md`, base de las tablas globales.
 */
static void write_summary_md(
    const std::string &output_dir,
    const run_options_t &opt,
    const ff_dataset_header_t &header,
    const std::string &dataset_path,
    const epoch_row_t &final_row,
    uint32_t final_changed_weights,
    uint32_t final_changed_biases,
    uint32_t final_sign_changes,
    const std::string &status
) {
    std::ofstream out(output_dir + "/summary.md");
    std::string stem = dataset_stem_without_packed(dataset_path);

    out << "# Experiment summary\n\n";
    out << "run_name: " << opt.run_name << '\n';
    out << "dataset: " << stem << '\n';
    out << "dataset_path: " << dataset_path << '\n';
    out << "resolution: " << header.image_width << "x"
        << header.image_height << '\n';
    out << "bits: " << header.pixel_bits << "b\n";
    out << "group: " << opt.group_name << '\n';
    out << "architecture: " << architecture_string() << '\n';
    out << "architecture_tag: " << architecture_tag() << '\n';
    out << "hidden_layers: " << FF_HIDDEN_LAYERS << '\n';
    out << "hidden_values: " << hidden_arrow_string() << '\n';
    out << "parallel_neurons: " << FF_PARALLEL_NEURONS << '\n';
    out << "threshold: " << FF_GOODNESS_THRESHOLD << '\n';
    out << "label_scale: " << FF_LABEL_SCALE << '\n';
    out << "pixel_scale: " << FF_PIXEL_SCALE << '\n';
    out << "train_samples: " << opt.train_samples << '\n';
    out << "eval_samples: " << opt.eval_samples << '\n';
    out << "epochs: " << opt.epochs << '\n';
    out << "eval_every_epoch: " << opt.eval_every_epoch << '\n';
    out << "accuracy: " << std::fixed << std::setprecision(6)
        << final_row.accuracy << '\n';
    out << "avg_g_pos: " << final_row.avg_g_pos << '\n';
    out << "avg_g_neg: " << final_row.avg_g_neg << '\n';
    out << "avg_gap: " << final_row.avg_gap << '\n';
    out << "elapsed_sec: " << std::fixed << std::setprecision(6)
        << final_row.elapsed_sec << '\n';
    out << "changed_weights: " << final_changed_weights << '\n';
    out << "changed_biases: " << final_changed_biases << '\n';
    out << "sign_changes: " << final_sign_changes << '\n';
    out << "status: " << status << '\n';
}

/**
 * @brief Convierte un peso latente al signo usado por la BNN.
 *
 * @param value Peso latente.
 * @return `+1` o `-1`.
 */
static int binary_sign(ff_latent_t value) {
    return (value >= 0) ? 1 : -1;
}

/**
 * @brief Cuenta cambios de signo binario tras el entrenamiento.
 *
 * @param before Modelo inicial.
 * @param after Modelo entrenado.
 * @return Número de pesos cuyo signo cambió.
 *
 * Esta métrica es más estricta que contar cambios latentes: solo registra
 * cambios que alteran el forward binario de la BNN.
 */
static uint32_t count_binary_sign_changes(
    const ff_model_t &before,
    const ff_model_t &after
) {
    uint32_t changed = 0;

    for (int layer = 0; layer < FF_HIDDEN_LAYERS; layer++) {
        int neurons = ff_layer_neurons(layer);
        int input_dim = ff_layer_input_dim(layer);
        for (int neuron = 0; neuron < neurons; neuron++) {
            for (int input_idx = 0; input_idx < input_dim; input_idx++) {
                if (binary_sign(before.weights[layer][neuron][input_idx]) !=
                    binary_sign(after.weights[layer][neuron][input_idx])) {
                    changed++;
                }
            }
        }
    }

    return changed;
}

/**
 * @brief Verifica que el dataset cargado coincide con la build.
 *
 * @param header Cabecera validada por el lector.
 * @return `true` si resolución, bits y clases son las esperadas.
 */
static bool validate_dataset_matches_compile(const ff_dataset_header_t &header) {
    if ((int)header.image_width != FF_IMAGE_WIDTH ||
        (int)header.image_height != FF_IMAGE_HEIGHT ||
        (int)header.pixel_bits != FF_PIXEL_BITS ||
        (int)header.num_classes != FF_NUM_CLASSES) {
        return false;
    }
    return true;
}

/**
 * @brief Imprime la configuración de red al inicio de la corrida.
 *
 * @param dataset_path Ruta del dataset usado.
 */
static void print_network_config(
    const std::string &dataset_path
) {
    std::cout << "===== CONFIGURACION DE RED =====\n";
    std::cout << "Dataset: " << dataset_path << '\n';
    std::cout << "Input dim: " << FF_INPUT_DIM << '\n';
    std::cout << "Hidden layers: " << FF_HIDDEN_LAYERS << '\n';
    std::cout << "Hidden values: " << hidden_arrow_string() << '\n';
    std::cout << "Parallel neurons: " << FF_PARALLEL_NEURONS << '\n';
    std::cout << "Architecture: " << architecture_string() << '\n';
    std::cout << "================================\n";
}

/**
 * @brief Imprime una pequeña muestra de parámetros del modelo.
 *
 * @param prefix Etiqueta textual, por ejemplo `initial` o `trained`.
 * @param model Modelo inspeccionado.
 *
 * Permite demostrar en el log que los parámetros cambiaron después del
 * entrenamiento, criterio de aceptación del flujo.
 */
static void print_model_probe(
    const char *prefix,
    const ff_model_t &model
) {
    std::cout << prefix << " layer0.bias[0]=" << model.biases[0][0]
              << " layer0.weights[0][0..7]=";
    for (int i = 0; i < 8 && i < ff_layer_input_dim(0); i++) {
        std::cout << model.weights[0][0][i];
        if (i != 7 && i != ff_layer_input_dim(0) - 1) {
            std::cout << ',';
        }
    }
    std::cout << '\n';
}

/**
 * @brief Imprime los scores de clase para la primera muestra de evaluación.
 *
 * @param model Modelo usado para inferencia.
 * @param eval_words Puntero al bloque de evaluación.
 *
 * Esta salida ayuda a explicar la inferencia FF: la predicción sale de comparar
 * goodness de las diez etiquetas, no de una capa softmax.
 */
static void print_scores_for_first_eval(
    const ff_model_t &model,
    const ff_word_t *eval_words
) {
    ff_word_t sample_words[FF_WORDS_PER_SAMPLE];
    ff_goodness_t scores[FF_NUM_CLASSES];

    ff_copy_sample_words(eval_words, 0, sample_words);
    uint8_t true_label = ff_decode_label_from_words(sample_words);
    uint8_t pred = ff_predict_sample(model, sample_words, scores);

    std::cout << "first_eval_true=" << (int)true_label
              << " first_eval_pred=" << (int)pred << " scores=";
    for (int c = 0; c < FF_NUM_CLASSES; c++) {
        std::cout << scores[c];
        if (c != FF_NUM_CLASSES - 1) {
            std::cout << ',';
        }
    }
    std::cout << '\n';
}

/**
 * @brief Ejecuta la simulación completa de entrenamiento y evaluación.
 *
 * @param argc Cantidad de argumentos CLI.
 * @param argv Argumentos CLI generados por Make o escritos manualmente.
 * @return Código de salida: 0 si el entrenamiento cambia parámetros y escribe
 * resultados, distinto de 0 si falla dataset, configuración o aprendizaje.
 *
 * @note Solo host/testbench. Usa STL, archivos, consola y reloj de pared; no se
 * sintetiza. Las funciones internas de entrenamiento/inferencia sí comparten la
 * lógica preparada para HLS.
 */
int main(int argc, char **argv) {
    auto time_start = std::chrono::steady_clock::now();
    run_options_t opt = parse_args(argc, argv);

    std::string dataset_path = opt.dataset_path;
    ff_dataset_header_t header;
    std::vector<ff_word_t> payload_words;
    std::string error;

    if (!ff_read_dataset(dataset_path, header, payload_words, error)) {
        std::cerr << "ERROR dataset: " << error << '\n';
        return 1;
    }

    ff_print_header_summary(dataset_path, header);
/* Esta validación duplicada deja el error visible en el testbench, incluso si
 * el lector ya verificó la cabecera. Ayuda a depurar comandos Make mal armados.
 */
    if (!validate_dataset_matches_compile(header)) {
        std::cerr << "ERROR: dataset no coincide con macros compiladas: "
                  << "header=" << header.image_width << "x"
                  << header.image_height << "_" << header.pixel_bits
                  << "b, build=" << FF_IMAGE_WIDTH << "x"
                  << FF_IMAGE_HEIGHT << "_" << FF_PIXEL_BITS << "b\n";
        return 2;
    }
    print_network_config(dataset_path);

    int total_samples = (int)header.sample_count;
    int train_samples = std::min(opt.train_samples, total_samples);
    int eval_samples = opt.eval_samples;
    int epochs = opt.epochs;
    uint32_t seed = opt.seed;

    int eval_offset = train_samples;
/* La evaluación empieza después del tramo de entrenamiento para evitar medir
 * exactamente las mismas muestras cuando hay datos suficientes.
 */
    if (eval_offset >= total_samples) {
        eval_offset = 0;
    }
    if (eval_offset + eval_samples > total_samples) {
        eval_samples = total_samples - eval_offset;
    }
    if (epochs > FF_MAX_EPOCHS) {
        epochs = FF_MAX_EPOCHS;
    }
    opt.train_samples = train_samples;
    opt.eval_samples = eval_samples;
    opt.epochs = epochs;

    if (opt.output_dir.empty()) {
        opt.output_dir = std::string("results/") +
            dataset_stem_without_packed(dataset_path) + "/" +
            opt.group_name + "/" + architecture_tag();
    }
    std::filesystem::create_directories(opt.output_dir);

    std::cout << "run_name              : " << opt.run_name << '\n';
    std::cout << "group                 : " << opt.group_name << '\n';
    std::cout << "output_dir            : " << opt.output_dir << '\n';
    std::cout << "model                : " << architecture_string()
              << " BNN FF, parallel_neurons=" << FF_PARALLEL_NEURONS << '\n';
    std::cout << "threshold            : " << FF_GOODNESS_THRESHOLD << '\n';
    std::cout << "label_scale          : " << FF_LABEL_SCALE << '\n';
    std::cout << "train_samples        : " << train_samples << '\n';
    std::cout << "eval_samples         : " << eval_samples
              << " offset=" << eval_offset << '\n';
    std::cout << "epochs               : " << epochs << '\n';
    std::cout << "eval_every_epoch     : " << opt.eval_every_epoch << '\n';
    std::cout << "seed                 : 0x" << std::hex << seed
              << std::dec << "\n\n";

    ff_model_t model;
    ff_init_model(model, seed);
    ff_model_t initial_model = model;

    print_model_probe("initial", model);

    const ff_word_t *train_words = payload_words.data();
    const ff_word_t *eval_words =
        payload_words.data() + (eval_offset * FF_WORDS_PER_SAMPLE);

    ff_eval_metrics_t baseline_eval;
/* La accuracy inicial sirve como referencia de modelo no entrenado antes de
 * aplicar la regla Forward-Forward.
 */
    ff_evaluate_dataset(model, eval_words, eval_samples, baseline_eval);
    double baseline_acc = (baseline_eval.total == 0u) ? 0.0 :
        (100.0 * (double)baseline_eval.correct / (double)baseline_eval.total);
    std::cout << "baseline_accuracy    : " << std::fixed << std::setprecision(2)
              << baseline_acc << "% (" << baseline_eval.correct << '/'
              << baseline_eval.total << ")\n";
    print_scores_for_first_eval(model, eval_words);

    std::cout << "\nepoch,g_pos_avg,g_neg_avg,gap_avg,pos_events,neg_events,"
              << "changed_weights,changed_biases,sign_changes,accuracy,first_true,"
              << "first_pred\n";

    ff_epoch_metrics_t last_epoch_metrics;
    std::vector<epoch_row_t> metric_rows;
    epoch_row_t final_row;
    final_row.epoch = 0;
    final_row.accuracy = 0.0;
    final_row.avg_g_pos = 0;
    final_row.avg_g_neg = 0;
    final_row.avg_gap = 0;
    final_row.pos_events = 0;
    final_row.neg_events = 0;
    final_row.changed_weights = 0;
    final_row.changed_biases = 0;
    final_row.sign_changes = 0;
    final_row.elapsed_sec = 0.0;

    for (int epoch = 0; epoch < epochs; epoch++) {
/* La semilla varía por época para que la secuencia de etiquetas negativas no
 * sea idéntica en todas las pasadas.
 */
        ff_train_epoch(
            model,
            train_words,
            train_samples,
            seed + (uint32_t)(epoch * 97 + 1),
            initial_model,
            last_epoch_metrics
        );

        bool should_eval = (opt.eval_every_epoch != 0) || (epoch == (epochs - 1));
        ff_eval_metrics_t eval_metrics;
        eval_metrics.correct = 0;
        eval_metrics.total = 0;
        eval_metrics.first_true_label = 0;
        eval_metrics.first_pred_label = 0;
        eval_metrics.first_best_goodness = 0;

        if (should_eval) {
            ff_evaluate_dataset(model, eval_words, eval_samples, eval_metrics);
        }

        double accuracy = (!should_eval || eval_metrics.total == 0u) ? -1.0 :
            (100.0 * (double)eval_metrics.correct / (double)eval_metrics.total);
        uint32_t sign_changes = count_binary_sign_changes(initial_model, model);

        auto time_now = std::chrono::steady_clock::now();
        double elapsed_sec =
            std::chrono::duration<double>(time_now - time_start).count();

        epoch_row_t row;
        row.epoch = epoch + 1;
        row.accuracy = (accuracy < 0.0) ? 0.0 : (accuracy / 100.0);
        row.avg_g_pos = last_epoch_metrics.avg_g_pos;
        row.avg_g_neg = last_epoch_metrics.avg_g_neg;
        row.avg_gap = last_epoch_metrics.avg_gap;
        row.pos_events = last_epoch_metrics.pos_update_events;
        row.neg_events = last_epoch_metrics.neg_update_events;
        row.changed_weights = last_epoch_metrics.changed_weights;
        row.changed_biases = last_epoch_metrics.changed_biases;
        row.sign_changes = sign_changes;
        row.elapsed_sec = elapsed_sec;

        if (should_eval || opt.eval_every_epoch != 0) {
            metric_rows.push_back(row);
        }
        if (should_eval) {
            final_row = row;
        }

        std::cout << (epoch + 1) << ','
                  << last_epoch_metrics.avg_g_pos << ','
                  << last_epoch_metrics.avg_g_neg << ','
                  << last_epoch_metrics.avg_gap << ','
                  << last_epoch_metrics.pos_update_events << ','
                  << last_epoch_metrics.neg_update_events << ','
                  << last_epoch_metrics.changed_weights << ','
                  << last_epoch_metrics.changed_biases << ','
                  << sign_changes << ','
                  << std::fixed << std::setprecision(2) << accuracy << ','
                  << (int)eval_metrics.first_true_label << ','
                  << (int)eval_metrics.first_pred_label << '\n';
    }

    std::cout << '\n';
    print_model_probe("trained", model);
    print_scores_for_first_eval(model, eval_words);

    uint32_t final_changed_weights =
        ff_count_changed_weights(initial_model, model);
    uint32_t final_changed_biases =
        ff_count_changed_biases(initial_model, model);
    uint32_t final_sign_changes =
        count_binary_sign_changes(initial_model, model);

    std::cout << "final_changed_weights: " << final_changed_weights << '\n';
    std::cout << "final_changed_biases : " << final_changed_biases << '\n';
    std::cout << "final_sign_changes   : " << final_sign_changes << '\n';

    if (final_changed_weights == 0u && final_changed_biases == 0u) {
/* Si nada cambió, la corrida no demuestra entrenamiento. Se escriben archivos
 * de fallo para que los resúmenes globales lo reflejen.
 */
        std::cerr << "ERROR: el entrenamiento no modifico pesos ni bias.\n";
        if (metric_rows.empty()) {
            metric_rows.push_back(final_row);
        }
        write_metrics_csv(opt.output_dir, metric_rows);
        write_summary_md(
            opt.output_dir,
            opt,
            header,
            dataset_path,
            final_row,
            final_changed_weights,
            final_changed_biases,
            final_sign_changes,
            "FAIL"
        );
        return 2;
    }

    if (metric_rows.empty()) {
        metric_rows.push_back(final_row);
    }
    write_metrics_csv(opt.output_dir, metric_rows);
    write_summary_md(
        opt.output_dir,
        opt,
        header,
        dataset_path,
        final_row,
        final_changed_weights,
        final_changed_biases,
        final_sign_changes,
        "OK"
    );

    std::cout << "metrics_csv          : " << opt.output_dir << "/metrics.csv\n";
    std::cout << "summary_md           : " << opt.output_dir << "/summary.md\n";
    std::cout << "TRAIN_TB_OK\n";
    return 0;
}
