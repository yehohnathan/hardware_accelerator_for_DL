#include "tb_stage_b.hpp"   // Se incluye el header propio de la validación modular de la Etapa B.
#include "debug_utils.hpp"  // Se incluyen utilidades de inspección y comparación del testbench.

#include <iostream>          // Se incluye iostream para mostrar el resultado de las verificaciones.
#include <vector>            // Se incluye vector para manejar buffers temporales de palabras y labels.

// ============================================================
// Validación modular de la Etapa B
// ============================================================
bool run_stage_b_validation(
    const std::vector<word_t> &input_words,
    int total_samples,
    uint16_t seed
) {
    std::vector<word_t> pos_words(input_words.size(), 0);
    // Se reserva el buffer de salida para las muestras positivas reconstruidas por el top heredado.

    std::vector<word_t> neg_words(input_words.size(), 0);
    // Se reserva el buffer de salida para las muestras negativas reconstruidas por el top heredado.

    std::vector<label_idx_t> pair_true_labels(total_samples, 0);
    // Se reserva el buffer de etiquetas verdaderas generado por la preparación de pares FF.

    std::vector<label_idx_t> pair_neg_labels(total_samples, 0);
    // Se reserva el buffer de etiquetas negativas generado por la preparación de pares FF.

    forward_fw_top(
        input_words.data(),
        pos_words.data(),
        neg_words.data(),
        pair_true_labels.data(),
        pair_neg_labels.data(),
        total_samples,
        seed
    );
    // Se ejecuta el top heredado para validar que la construcción de pares FF siga intacta.

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

    bool stage_b_ok = positive_equals_input
                   && negative_same_pixels
                   && negative_same_padding
                   && (pair_neg_labels[0] != input_label_idx)
                   && (collisions == 0);
    // Se resume el estado global de la Etapa B con base en las verificaciones principales.

    std::cout << "stage_b_validation_ok = " << (stage_b_ok ? "true" : "false") << std::endl;
    // Se imprime un veredicto global para facilitar la lectura rápida del módulo.

    return stage_b_ok;
    // Se retorna el resultado global de la validación modular de la Etapa B.
}
