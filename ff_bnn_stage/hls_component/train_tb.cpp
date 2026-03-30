#include "forward_fw.hpp"   // Se incluye el kernel y todas sus funciones sintetizables y tipos de modelo.
#include "debug_utils.hpp"  // Se incluyen las utilidades de inspección y depuración para C simulation.

#include <iostream>          // Se incluye iostream para mostrar resultados del testbench.
#include <vector>            // Se incluye vector para manejar buffers de entrada y salida de forma práctica en simulación.
#include <string>            // Se incluye string para manejar rutas de archivo y mensajes.
#include <fstream>           // Se incluye fstream para comprobar rutas candidatas del binario de entrada.

// ============================================================
// Utilidad simple para localizar el binario de entrada
// ============================================================
static std::string resolve_input_path() {
    const char *candidate_paths[] = {
        "D:/TFG/hardware_accelerator_for_DL/ff_bnn_stage/mnist/data/processed/mnist_ff_input_packed.bin",
        "data/processed/mnist_ff_input_packed.bin",
        "/mnt/data/mnist_ff_input_packed.bin"
    };
    // Se define un pequeño conjunto de rutas candidatas para facilitar la reutilización del testbench en distintos entornos.

resolve_path_loop:
    for (int i = 0; i < 3; i++) {
        std::ifstream file(candidate_paths[i], std::ios::binary);
        // Se intenta abrir cada ruta candidata en modo binario para verificar su existencia.

        if (file.is_open()) {
            file.close();
            // Si la ruta existe, se cierra el archivo inmediatamente porque solo interesa comprobar disponibilidad.

            return std::string(candidate_paths[i]);
            // Se retorna la primera ruta válida encontrada.
        }
    }

    return std::string(candidate_paths[0]);
    // Si ninguna ruta existe, se retorna la ruta principal original para que el mensaje de error sea explícito.
}

// ============================================================
// Función principal del testbench
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
        // Se verifica explícitamente que el binario no sea vacío antes de invocar el kernel.

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

    // ============================================================
    // Etapa B: validación de la preparación de pares FF heredada
    // ============================================================
    std::vector<word_t> pos_words(input_words.size(), 0);
    // Se reserva el buffer de salida para las muestras positivas reconstruidas por el top heredado.

    std::vector<word_t> neg_words(input_words.size(), 0);
    // Se reserva el buffer de salida para las muestras negativas reconstruidas por el top heredado.

    std::vector<label_idx_t> pair_true_labels(total_samples, 0);
    // Se reserva el buffer de etiquetas verdaderas generado por la preparación de pares FF.

    std::vector<label_idx_t> pair_neg_labels(total_samples, 0);
    // Se reserva el buffer de etiquetas negativas generado por la preparación de pares FF.

    uint16_t seed = 0x1234;
    // Se fija una semilla determinista para reproducir resultados entre simulaciones consecutivas.

    forward_fw_top(
        input_words.data(),
        pos_words.data(),
        neg_words.data(),
        pair_true_labels.data(),
        pair_neg_labels.data(),
        total_samples,
        seed
    );
    // Se ejecuta el top heredado para comprobar que la funcionalidad de Etapa B sigue intacta.

    word_t first_input_words[WORDS_PER_SAMPLE];
    // Se reserva un arreglo temporal para copiar la primera muestra original y verla en consola.

    extract_sample_words(input_words, 0, first_input_words);
    // Se extraen las 25 words de la primera muestra del binario original.

    print_separator("PRIMERA MUESTRA ORIGINAL EN HEX");
    // Se abre una sección para inspeccionar la primera muestra cruda en hexadecimal.

    print_sample_words(first_input_words, "Contenido de la primera muestra:");
    // Se imprimen las 25 words de la primera muestra original.

    raw_sample_t first_input_sample = load_sample_from_words(input_words.data(), 0);
    // Se reconstruye la primera muestra original como vector de 800 bits.

    raw_sample_t first_pos_sample = load_sample_from_words(pos_words.data(), 0);
    // Se reconstruye la primera muestra positiva generada por el top heredado.

    raw_sample_t first_neg_sample = load_sample_from_words(neg_words.data(), 0);
    // Se reconstruye la primera muestra negativa generada por el top heredado.

    print_unpacked_sample(first_input_sample, "DESEMPAQUETADO DE LA PRIMERA MUESTRA ORIGINAL");
    // Se imprime el contenido desempaquetado de la primera muestra original.

    print_unpacked_sample(first_pos_sample, "DESEMPAQUETADO DE LA PRIMERA MUESTRA POSITIVA");
    // Se imprime el contenido desempaquetado de la primera muestra positiva.

    print_unpacked_sample(first_neg_sample, "DESEMPAQUETADO DE LA PRIMERA MUESTRA NEGATIVA");
    // Se imprime el contenido desempaquetado de la primera muestra negativa.

    print_separator("VERIFICACIONES DE CONSISTENCIA DE ETAPA B");
    // Se abre la sección de verificaciones automáticas para la preparación heredada de pares FF.

    bool positive_equals_input = same_raw_sample(first_input_sample, first_pos_sample);
    // Se comprueba que la salida positiva sea idéntica bit a bit a la entrada original.

    bool negative_same_pixels = same_pixels(first_input_sample, first_neg_sample);
    // Se comprueba que la salida negativa conserve exactamente la misma imagen.

    bool negative_same_padding = same_padding(first_input_sample, first_neg_sample);
    // Se comprueba que la salida negativa conserve exactamente el mismo padding físico.

    label_oh_t input_label_onehot;
    // Se reserva el contenedor del label original desempaquetado de la primera muestra.

    pixels_t input_pixels;
    // Se reserva el contenedor de la imagen original desempaquetada de la primera muestra.

    padding_t input_padding;
    // Se reserva el contenedor del padding original desempaquetado de la primera muestra.

    unpack_sample(first_input_sample, input_label_onehot, input_pixels, input_padding);
    // Se desempaqueta la primera muestra original para recuperar su clase verdadera.

    label_idx_t input_label_idx = decode_onehot(input_label_onehot);
    // Se decodifica la clase verdadera de la primera muestra a formato índice.

    std::cout << "positive_equals_input = " << (positive_equals_input ? "true" : "false") << std::endl;
    // Se imprime si la muestra positiva coincide exactamente con la original.

    std::cout << "negative_same_pixels  = " << (negative_same_pixels ? "true" : "false") << std::endl;
    // Se imprime si la muestra negativa mantiene la imagen sin cambios.

    std::cout << "negative_same_padding = " << (negative_same_padding ? "true" : "false") << std::endl;
    // Se imprime si la muestra negativa mantiene el padding sin cambios.

    std::cout << "true_label(first)     = " << (unsigned int)input_label_idx << std::endl;
    // Se imprime la clase verdadera de la primera muestra.

    std::cout << "neg_label(first)      = " << (unsigned int)pair_neg_labels[0] << std::endl;
    // Se imprime la clase negativa asignada a la primera muestra.

    std::cout << "neg_differs_from_true = " << ((pair_neg_labels[0] != input_label_idx) ? "true" : "false") << std::endl;
    // Se verifica explícitamente que la etiqueta negativa no coincida con la verdadera.

    int collisions = 0;
    // Se inicializa el contador de colisiones de etiquetas negativas con etiquetas verdaderas.

collision_count_loop:
    for (int i = 0; i < total_samples; i++) {
        if (pair_true_labels[i] == pair_neg_labels[i]) {
            collisions++;
            // Se incrementa el contador si alguna etiqueta negativa coincidiera con la verdadera.
        }
    }

    std::cout << "total_collisions      = " << collisions << std::endl;
    // Se imprime el número total de colisiones encontradas en la preparación de pares.

    // ============================================================
    // Etapa C: entrenamiento local de la primera capa FF en hardware
    // ============================================================
    int train_samples = total_samples;
    // Se inicializa la cantidad de muestras de entrenamiento con el total disponible por defecto.

    if (train_samples > 4096) {
        train_samples = 4096;
        // Para C simulation se limita el tamaño inicial a 512 muestras para mantener tiempos razonables.
    }

    int eval_samples = train_samples;
    // En esta versión mínima funcional se evalúa sobre el mismo subconjunto usado para entrenamiento.

    int epochs = 20;
    // Se fijan 5 épocas como punto de partida pragmático para demostrar aprendizaje funcional en hardware.

    std::vector<label_idx_t> train_true_labels(eval_samples, 0);
    // Se reserva el buffer de etiquetas verdaderas que devolverá el top entrenable durante la inferencia.

    std::vector<label_idx_t> pred_labels(eval_samples, 0);
    // Se reserva el buffer de etiquetas predichas que devolverá el top entrenable.

    std::vector<latent_t> weight_snapshot(MODEL_WEIGHT_COUNT);
    // Se reserva el buffer externo donde el top copiará el snapshot final de pesos latentes.

    std::vector<bias_t> bias_snapshot(MODEL_NEURONS);
    // Se reserva el buffer externo donde el top copiará el snapshot final de bias latentes.

    std::vector<goodness_t> g_pos(train_samples, 0);
    // Se reserva el buffer de goodness positiva registrada en la última época de entrenamiento.

    std::vector<goodness_t> g_neg(train_samples, 0);
    // Se reserva el buffer de goodness negativa registrada en la última época de entrenamiento.

    std::vector<goodness_t> gap(train_samples, 0);
    // Se reserva el buffer de goodness gap registrada en la última época de entrenamiento.

    std::vector<ap_uint<32> > correct_count(1, 0);
    // Se reserva un buffer de una sola posición para recibir el conteo final de aciertos de inferencia.

    ff_train_top(
        input_words.data(),
        train_true_labels.data(),
        pred_labels.data(),
        weight_snapshot.data(),
        bias_snapshot.data(),
        g_pos.data(),
        g_neg.data(),
        gap.data(),
        correct_count.data(),
        eval_samples,
        train_samples,
        epochs,
        seed,
        true
    );
    // Se ejecuta el nuevo top de la Etapa C que entrena el modelo localmente e infiere sobre el subconjunto seleccionado.

    print_separator("RESULTADOS DE ENTRENAMIENTO DE ETAPA C");
    // Se abre una sección para mostrar las métricas principales del entrenamiento hardware.

    std::cout << "train_samples   = " << train_samples << std::endl;
    // Se imprime la cantidad real de muestras usadas para entrenamiento local.

    std::cout << "eval_samples    = " << eval_samples << std::endl;
    // Se imprime la cantidad real de muestras usadas para inferencia posterior.

    std::cout << "epochs          = " << epochs << std::endl;
    // Se imprime la cantidad de épocas ejecutadas por el kernel entrenable.

    std::cout << "correct_count   = " << (unsigned int)correct_count[0] << std::endl;
    // Se imprime el número total de predicciones correctas obtenidas tras el entrenamiento.

    std::cout << "train_accuracy  = " << ((double)((unsigned int)correct_count[0]) / (double)eval_samples) << std::endl;
    // Se imprime la exactitud obtenida sobre el subconjunto usado en esta versión mínima funcional.

    print_training_preview(g_pos, g_neg, gap, 12);
    // Se muestran algunas goodness positivas, negativas y gaps de la última época para inspección rápida.

    print_prediction_preview(train_true_labels, pred_labels, 20);
    // Se muestran varias parejas verdad-predicción para verificar visualmente el comportamiento del modelo.

    print_model_overview(weight_snapshot, bias_snapshot, 24);
    // Se resume el estado aprendido de pesos y bias después del entrenamiento local hardware.

    if (total_samples > 10002) {
        print_separator("NUEVOS SAMPLES PARA TEST");
        // Si el archivo es suficientemente grande, se abre una sección para inspeccionar muestras más alejadas.

        print_samples_range(input_words, 10000, 3);
        // Se imprimen tres muestras a partir de la posición 10000 para revisar variedad del binario.
    }

    print_separator("FIN DEL TESTBENCH");
    // Se imprime la sección final del testbench indicando que toda la ejecución terminó.

    return 0;
    // Se finaliza la simulación con código de éxito.
}
