// ============================================================================
// train_tb.cpp
// ----------------------------------------------------------------------------
// Testbench inicial para validar:
//
// 1. empaquetado y desempaquetado
// 2. posición de label, pixeles y padding
// 3. preservación del positivo
// 4. reemplazo correcto del label negativo
// 5. garantía de que el negativo nunca sea igual al positivo
// ============================================================================

#include "forward_fw.hpp"
#include "debug_utils.hpp"

// Biblioteca para salida estándar.
#include <iostream>

// Biblioteca para archivo binario futuro.
#include <fstream>

// ============================================================================
// Función auxiliar para crear una muestra sintética controlada.
// ----------------------------------------------------------------------------
// Crea:
// - label = clase 1
// - pixeles con patrón alternado
// - padding fijo
// ============================================================================
static sample_t create_test_sample() {
    // Define la etiqueta verdadera como 1.
    label_idx_t label_idx = 1;

    // Codifica el label a one-hot.
    label_oh_t label_oh = encode_onehot(label_idx);

    // Inicializa los pixeles en cero.
    pixels_t pixels = 0;

    // Crea un patrón alternado 1010... para depuración visual.
    for (int i = 0; i < PIXEL_BITS; i++) {
        if ((i % 2) == 0) {
            pixels[i] = 1;
        } else {
            pixels[i] = 0;
        }
    }

    // Define un padding conocido.
    padding_t padding = 0;

    // Activa algunos bits del padding para probar extracción correcta.
    padding[0] = 1;
    padding[2] = 1;
    padding[4] = 1;

    // Construye y retorna la muestra completa.
    return build_sample(label_oh, pixels, padding);
}

// ============================================================================
// main del testbench
// ============================================================================
int main() {
    // Cantidad de muestras a probar.
    const int N_SAMPLES = 1;

    // Buffers de memoria tipo "RAM" para el testbench.
    word_t in_mem[N_SAMPLES * WORDS_PER_SAMPLE];
    word_t pos_mem[N_SAMPLES * WORDS_PER_SAMPLE];
    word_t neg_mem[N_SAMPLES * WORDS_PER_SAMPLE];
    label_idx_t neg_labels[N_SAMPLES];

    // ------------------------------------------------------------------------
    // 1. Crear una muestra sintética.
    // ------------------------------------------------------------------------
    sample_t test_sample = create_test_sample();

    // Buffer temporal de 25 palabras para la entrada.
    word_t temp_in_words[WORDS_PER_SAMPLE];

    // Convierte la muestra sintética a formato memoria.
    sample_to_words(test_sample, temp_in_words);

    // Copia las 25 palabras al buffer de entrada.
    for (int i = 0; i < WORDS_PER_SAMPLE; i++) {
        in_mem[i] = temp_in_words[i];
    }

    // ------------------------------------------------------------------------
    // 2. Mostrar la muestra de entrada.
    // ------------------------------------------------------------------------
    std::cout << "================ INPUT SAMPLE ================" << std::endl;
    print_words_25x32(temp_in_words, "Input words");

    // Extrae componentes para verificar.
    label_oh_t in_label = extract_label_onehot(test_sample);
    pixels_t in_pixels = extract_pixels(test_sample);
    padding_t in_padding = extract_padding(test_sample);
    label_idx_t in_label_idx = decode_onehot(in_label);

    // Imprime campos desempaquetados.
    print_label_onehot(in_label, "Input label one-hot");
    print_label_index(in_label_idx, "Input label index");
    print_padding_bits(in_padding, "Input padding");
    print_pixels_preview(in_pixels, "Input pixels", 32);

    // ------------------------------------------------------------------------
    // 3. Ejecutar el kernel.
    // ------------------------------------------------------------------------
    forward_fw(
        in_mem,
        pos_mem,
        neg_mem,
        neg_labels,
        N_SAMPLES,
        0x1234
    );

    // ------------------------------------------------------------------------
    // 4. Reconstruir salida positiva y negativa como muestras lógicas.
    // ------------------------------------------------------------------------
    word_t pos_words[WORDS_PER_SAMPLE];
    word_t neg_words[WORDS_PER_SAMPLE];

    // Copia la salida positiva localmente.
    for (int i = 0; i < WORDS_PER_SAMPLE; i++) {
        pos_words[i] = pos_mem[i];
    }

    // Copia la salida negativa localmente.
    for (int i = 0; i < WORDS_PER_SAMPLE; i++) {
        neg_words[i] = neg_mem[i];
    }

    // Convierte los arreglos a muestras lógicas.
    sample_t pos_sample = words_to_sample(pos_words);
    sample_t neg_sample = words_to_sample(neg_words);

    // ------------------------------------------------------------------------
    // 5. Extraer campos del positivo.
    // ------------------------------------------------------------------------
    label_oh_t pos_label = extract_label_onehot(pos_sample);
    pixels_t pos_pixels = extract_pixels(pos_sample);
    padding_t pos_padding = extract_padding(pos_sample);
    label_idx_t pos_label_idx = decode_onehot(pos_label);

    // ------------------------------------------------------------------------
    // 6. Extraer campos del negativo.
    // ------------------------------------------------------------------------
    label_oh_t neg_label = extract_label_onehot(neg_sample);
    pixels_t neg_pixels = extract_pixels(neg_sample);
    padding_t neg_padding = extract_padding(neg_sample);
    label_idx_t neg_label_idx = decode_onehot(neg_label);

    // ------------------------------------------------------------------------
    // 7. Imprimir salida positiva.
    // ------------------------------------------------------------------------
    std::cout << "\n================ POSITIVE SAMPLE ================" << std::endl;
    print_words_25x32(pos_words, "Positive words");
    print_label_onehot(pos_label, "Positive label one-hot");
    print_label_index(pos_label_idx, "Positive label index");
    print_padding_bits(pos_padding, "Positive padding");
    print_pixels_preview(pos_pixels, "Positive pixels", 32);

    // ------------------------------------------------------------------------
    // 8. Imprimir salida negativa.
    // ------------------------------------------------------------------------
    std::cout << "\n================ NEGATIVE SAMPLE ================" << std::endl;
    print_words_25x32(neg_words, "Negative words");
    print_label_onehot(neg_label, "Negative label one-hot");
    print_label_index(neg_label_idx, "Negative label index");
    print_padding_bits(neg_padding, "Negative padding");
    print_pixels_preview(neg_pixels, "Negative pixels", 32);

    // ------------------------------------------------------------------------
    // 9. Verificaciones automáticas.
    // ------------------------------------------------------------------------
    bool ok = true;

    // El positivo debe conservar el mismo label.
    if (pos_label != in_label) {
        std::cout << "[ERROR] El label positivo no coincide con la entrada." << std::endl;
        ok = false;
    }

    // El positivo debe conservar los mismos pixeles.
    if (pos_pixels != in_pixels) {
        std::cout << "[ERROR] Los pixeles positivos no coinciden con la entrada." << std::endl;
        ok = false;
    }

    // El positivo debe conservar el mismo padding.
    if (pos_padding != in_padding) {
        std::cout << "[ERROR] El padding positivo no coincide con la entrada." << std::endl;
        ok = false;
    }

    // El negativo debe conservar los mismos pixeles.
    if (neg_pixels != in_pixels) {
        std::cout << "[ERROR] Los pixeles negativos no coinciden con la entrada." << std::endl;
        ok = false;
    }

    // El negativo debe conservar el mismo padding.
    if (neg_padding != in_padding) {
        std::cout << "[ERROR] El padding negativo no coincide con la entrada." << std::endl;
        ok = false;
    }

    // El negativo NO puede tener el mismo label.
    if (neg_label_idx == in_label_idx) {
        std::cout << "[ERROR] La etiqueta negativa coincide con la verdadera." << std::endl;
        ok = false;
    }

    // El label negativo generado por el kernel debe coincidir con el buffer de debug.
    if (neg_label_idx != neg_labels[0]) {
        std::cout << "[ERROR] neg_labels[0] no coincide con el label negativo reconstruido." << std::endl;
        ok = false;
    }

    // ------------------------------------------------------------------------
    // 10. Resultado final.
    // ------------------------------------------------------------------------
    if (ok) {
        std::cout << "\n[OK] Testbench completado correctamente." << std::endl;
        return 0;
    } else {
        std::cout << "\n[FAIL] El testbench detectó errores." << std::endl;
        return 1;
    }
}