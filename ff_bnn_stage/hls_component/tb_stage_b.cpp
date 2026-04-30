#include "tb_stage_b.hpp"

#include <iostream>
#include <vector>

/**
 * @brief Ejecuta la validacion modular de la Etapa B.
 *
 * @param dataset_header Contiene la metadata del binario ya cargado.
 * @param input_words Contiene el payload lineal del dataset sin cabecera.
 * @param total_samples Indica cuantas muestras contiene el payload.
 * @param seed Indica la semilla reproducible usada por la Etapa B.
 *
 * @return Retorna true cuando la Etapa B conserva pixeles y padding y
 * genera etiquetas negativas validas.
 */
bool run_stage_b_validation(
    const ff_binary_header_t &dataset_header,
    const std::vector<word_t> &input_words,
    int total_samples,
    uint16_t seed
) {
    std::vector<word_t> pos_words(input_words.size(), 0);
    // Se reserva el buffer de salida para las muestras positivas.

    std::vector<word_t> neg_words(input_words.size(), 0);
    // Se reserva el buffer de salida para las muestras negativas.

    std::vector<label_idx_t> pair_true_labels(total_samples, 0);
    // Se reserva el buffer de etiquetas verdaderas de la Etapa B.

    std::vector<label_idx_t> pair_neg_labels(total_samples, 0);
    // Se reserva el buffer de etiquetas negativas de la Etapa B.

    forward_fw_top(
        input_words.data(),
        pos_words.data(),
        neg_words.data(),
        pair_true_labels.data(),
        pair_neg_labels.data(),
        total_samples,
        seed
    );
    // Se ejecuta el top heredado para validar la preparacion de pares FF.

    print_separator("CONFIGURACION DE ETAPA B");
    // Se abre una seccion con la metadata usada durante la validacion.

    std::cout << "input_mode        = " << dataset_header.mode_name << std::endl;
    std::cout << "image_resolution  = "
              << dataset_header.image_width << "x"
              << dataset_header.image_height << std::endl;
    std::cout << "sample_count      = " << dataset_header.sample_count << std::endl;
    std::cout << "words_per_sample  = "
              << dataset_header.words_per_sample << std::endl;
    // Se imprime la configuracion del dataset asociada a la validacion.

    word_t first_input_words[WORDS_PER_SAMPLE];
    // Se reserva un arreglo temporal para copiar la primera muestra original.

    extract_sample_words(input_words, 0, first_input_words);
    // Se extraen las words de la primera muestra del payload original.

    print_separator("PRIMERA MUESTRA ORIGINAL EN HEX");
    // Se abre una seccion para inspeccionar la primera muestra cruda.

    print_sample_words(first_input_words, "Contenido de la primera muestra:");
    // Se imprimen las words de la primera muestra original.

    raw_sample_t first_input_sample = load_sample_from_words(
        input_words.data(),
        0
    );
    // Se reconstruye la primera muestra original como vector fisico completo.

    raw_sample_t first_pos_sample = load_sample_from_words(pos_words.data(), 0);
    // Se reconstruye la primera muestra positiva generada por la Etapa B.

    raw_sample_t first_neg_sample = load_sample_from_words(neg_words.data(), 0);
    // Se reconstruye la primera muestra negativa generada por la Etapa B.

    print_unpacked_sample(
        first_input_sample,
        "DESEMPAQUETADO DE LA PRIMERA MUESTRA ORIGINAL"
    );
    // Se imprime la primera muestra original ya desempaquetada.

    print_unpacked_sample(
        first_pos_sample,
        "DESEMPAQUETADO DE LA PRIMERA MUESTRA POSITIVA"
    );
    // Se imprime la primera muestra positiva ya desempaquetada.

    print_unpacked_sample(
        first_neg_sample,
        "DESEMPAQUETADO DE LA PRIMERA MUESTRA NEGATIVA"
    );
    // Se imprime la primera muestra negativa ya desempaquetada.

    print_separator("VERIFICACIONES DE CONSISTENCIA DE ETAPA B");
    // Se abre la seccion de verificaciones automaticas.

    bool positive_equals_input = same_raw_sample(
        first_input_sample,
        first_pos_sample
    );
    // Se comprueba que la salida positiva coincida bit a bit con la entrada.

    bool negative_same_pixels = same_pixels(
        first_input_sample,
        first_neg_sample
    );
    // Se comprueba que la salida negativa conserve exactamente la imagen.

    bool negative_same_padding = same_padding(
        first_input_sample,
        first_neg_sample
    );
    // Se comprueba que la salida negativa conserve el mismo padding fisico.

    label_oh_t input_label_onehot;
    pixels_t input_pixels;
    padding_t input_padding;
    // Se reservan los contenedores usados para desempaquetar la entrada.

    unpack_sample(
        first_input_sample,
        input_label_onehot,
        input_pixels,
        input_padding
    );
    // Se desempaqueta la primera muestra original para recuperar su clase.

    label_idx_t input_label_idx = decode_onehot(input_label_onehot);
    // Se decodifica la clase verdadera de la primera muestra.

    std::cout << "positive_equals_input = "
              << (positive_equals_input ? "true" : "false") << std::endl;
    std::cout << "negative_same_pixels  = "
              << (negative_same_pixels ? "true" : "false") << std::endl;
    std::cout << "negative_same_padding = "
              << (negative_same_padding ? "true" : "false") << std::endl;
    std::cout << "true_label(first)     = "
              << (unsigned int)input_label_idx << std::endl;
    std::cout << "neg_label(first)      = "
              << (unsigned int)pair_neg_labels[0] << std::endl;
    std::cout << "neg_differs_from_true = "
              << ((pair_neg_labels[0] != input_label_idx) ? "true" : "false")
              << std::endl;
    // Se imprimen las verificaciones principales de la primera muestra.

    int collisions = 0;
    // Se inicializa el contador de colisiones entre etiqueta real y negativa.

collision_count_loop:
    for (int i = 0; i < total_samples; i++) {
        if (pair_true_labels[i] == pair_neg_labels[i]) {
            collisions++;
            // Se cuenta cada colision encontrada en el dataset completo.
        }
    }

    std::cout << "total_collisions      = " << collisions << std::endl;
    // Se imprime el numero total de colisiones detectadas.

    bool stage_b_ok = positive_equals_input
                   && negative_same_pixels
                   && negative_same_padding
                   && (pair_neg_labels[0] != input_label_idx)
                   && (collisions == 0);
    // Se resume el estado global de la validacion modular.

    std::cout << "stage_b_validation_ok = "
              << (stage_b_ok ? "true" : "false") << std::endl;
    // Se imprime un veredicto global facil de leer.

    return stage_b_ok;
    // Se retorna el resultado global de la validacion de Etapa B.
}
