#include "forward_fw.hpp"   // Se incluye el kernel y todas sus funciones sintetizables.
#include "debug_utils.hpp"  // Se incluyen las utilidades de depuración para la simulación.

#include <iostream>         // Se usa para imprimir resultados del testbench.
#include <vector>           // Se usa para almacenar el contenido completo del archivo binario.
#include <string>           // Se usa para manejar la ruta del archivo.

// ============================================================
// Testbench principal
// ============================================================

int main() {
    std::string file_path = "D:/TFG/hardware_accelerator_for_DL/ff_bnn_stage/mnist/data/processed/mnist_ff_input_packed.bin";
    // Ruta del archivo binario de entrada.
    // Cambia esta cadena si el archivo está en otra carpeta.

    std::vector<word_t> input_words;
    // Buffer donde se almacenarán todas las words del archivo.

    bool read_ok = read_binary_file_words(file_path, input_words);
    // Se intenta leer el archivo completo.

    if (!read_ok) {
        std::cerr << "ERROR: no fue posible leer el binario de entrada." << std::endl;
        // Se reporta el error si la lectura falla.

        return 1;
        // Se termina con código de error.
    }

    if (input_words.size() == 0) {
        std::cerr << "ERROR: el archivo está vacío." << std::endl;
        // Se verifica que el archivo no esté vacío.

        return 1;
        // Se termina con código de error.
    }

    if ((input_words.size() % WORDS_PER_SAMPLE) != 0) {
        std::cerr << "ERROR: la cantidad total de words no es múltiplo de "
                  << WORDS_PER_SAMPLE << "." << std::endl;
        // Se verifica que el archivo contenga un número entero de muestras.

        return 1;
        // Se termina con código de error.
    }

    int n_samples = (int)(input_words.size() / WORDS_PER_SAMPLE);
    // Se calcula cuántas muestras completas hay en el archivo.

    print_separator("INFORMACION GENERAL DEL BINARIO");
    // Se imprime el encabezado de la primera sección.

    std::cout << "total_words   = " << input_words.size() << std::endl;
    // Se imprime la cantidad total de words.

    std::cout << "words/sample  = " << WORDS_PER_SAMPLE << std::endl;
    // Se recuerda que cada muestra tiene 25 words.

    std::cout << "total_samples = " << n_samples << std::endl;
    // Se imprime la cantidad total de muestras detectadas.

    std::vector<word_t> pos_words(input_words.size(), 0);
    // Buffer de salida para las muestras positivas.

    std::vector<word_t> neg_words(input_words.size(), 0);
    // Buffer de salida para las muestras negativas.

    std::vector<label_idx_t> true_labels(n_samples, 0);
    // Buffer para almacenar los labels verdaderos decodificados.

    std::vector<label_idx_t> neg_labels(n_samples, 0);
    // Buffer para almacenar los labels negativos generados.

    uint16_t seed = 0x1234;
    // Semilla inicial del LFSR usada en la prueba.

    forward_fw_top(
        input_words.data(),
        pos_words.data(),
        neg_words.data(),
        true_labels.data(),
        neg_labels.data(),
        n_samples,
        seed
    );
    // Se llama al top function del kernel para procesar todas las muestras.

    word_t first_input_words[WORDS_PER_SAMPLE];
    // Arreglo temporal para guardar la primera muestra original.

    extract_sample_words(input_words, 0, first_input_words);
    // Se extraen las 25 words de la primera muestra del archivo original.

    print_separator("PRIMERA MUESTRA ORIGINAL EN HEX");
    // Se abre una sección para inspeccionar la primera muestra cruda.

    print_sample_words(first_input_words, "Contenido de la primera muestra:");
    // Se imprimen sus 25 words en hexadecimal.


    raw_sample_t first_input_sample = load_sample_from_words(input_words.data(), 0);
    // Se arma la primera muestra original como vector de 800 bits.

    raw_sample_t first_pos_sample = load_sample_from_words(pos_words.data(), 0);
    // Se arma la primera muestra positiva generada por el kernel.

    raw_sample_t first_neg_sample = load_sample_from_words(neg_words.data(), 0);
    // Se arma la primera muestra negativa generada por el kernel.

    print_unpacked_sample(first_input_sample, "DESEMPAQUETADO DE LA PRIMERA MUESTRA ORIGINAL");
    // Se imprime el contenido desempaquetado de la primera muestra original.

    print_unpacked_sample(first_pos_sample, "DESEMPAQUETADO DE LA PRIMERA MUESTRA POSITIVA");
    // Se imprime el contenido desempaquetado de la primera muestra positiva.

    print_unpacked_sample(first_neg_sample, "DESEMPAQUETADO DE LA PRIMERA MUESTRA NEGATIVA");
    // Se imprime el contenido desempaquetado de la primera muestra negativa.

    print_separator("VERIFICACIONES DE CONSISTENCIA");
    // Se abre la sección de verificaciones automáticas.

    bool positive_equals_input = same_raw_sample(first_input_sample, first_pos_sample);
    // Se comprueba que la salida positiva sea idéntica a la entrada.

    bool negative_same_pixels = same_pixels(first_input_sample, first_neg_sample);
    // Se comprueba que la salida negativa conserve exactamente la misma imagen.

    bool negative_same_padding = same_padding(first_input_sample, first_neg_sample);
    // Se comprueba que la salida negativa conserve exactamente el mismo padding.

    label_oh_t input_label_onehot;
    // Variable para el label one-hot original.

    pixels_t input_pixels;
    // Variable para los pixels originales.

    padding_t input_padding;
    // Variable para el padding original.

    unpack_sample(first_input_sample, input_label_onehot, input_pixels, input_padding);
    // Se desempaqueta la muestra original.

    label_idx_t input_label_idx = decode_onehot(input_label_onehot);
    // Se decodifica el label original.

    std::cout << "positive_equals_input = " << (positive_equals_input ? "true" : "false") << std::endl;
    // Se imprime si la positiva coincide bit a bit con la entrada.

    std::cout << "negative_same_pixels  = " << (negative_same_pixels ? "true" : "false") << std::endl;
    // Se imprime si la negativa conserva la imagen.

    std::cout << "negative_same_padding = " << (negative_same_padding ? "true" : "false") << std::endl;
    // Se imprime si la negativa conserva el padding.

    std::cout << "true_label(first)     = " << (unsigned int)input_label_idx << std::endl;
    // Se imprime la clase verdadera de la primera muestra.

    std::cout << "neg_label(first)      = " << (unsigned int)neg_labels[0] << std::endl;
    // Se imprime la clase negativa asignada a la primera muestra.

    std::cout << "neg_differs_from_true = "
              << ((neg_labels[0] != input_label_idx) ? "true" : "false")
              << std::endl;
    // Se verifica explícitamente que la etiqueta negativa no sea la misma que la original.

    print_separator("VALIDACION GLOBAL DE ETIQUETAS NEGATIVAS");
    // Se abre una sección para validar todas las muestras, no solo la primera.

    int collisions = 0;
    // Contador de casos donde la etiqueta negativa coincida con la verdadera.
    // Idealmente debe quedar en 0.

    for (int i = 0; i < n_samples; i++) {
        if (true_labels[i] == neg_labels[i]) {
            collisions++;
            // Se incrementa si alguna etiqueta negativa coincide con la verdadera.
        }
    }

    std::cout << "total_collisions = " << collisions << std::endl;
    // Se imprime el número total de colisiones.

    if (collisions == 0) {
        std::cout << "OK: ninguna etiqueta negativa coincide con su etiqueta verdadera." << std::endl;
        // Mensaje de éxito esperado.
    } else {
        std::cout << "ERROR: existen colisiones entre etiquetas verdaderas y negativas." << std::endl;
        // Mensaje de error si hubiera coincidencias.
    }

    print_separator("NUEVOS SAMPLES PARA TEST");
    print_samples_range(input_words, 10000, 3);
    // Se abre la sección de verificación de samples.

    print_separator("FIN DEL TESTBENCH");
    // Se imprime una sección final.

    return 0;
    // El programa termina exitosamente.
}