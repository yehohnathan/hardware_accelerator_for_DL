#include "debug_utils.hpp" // Se incluye el header con los prototipos de utilidades de debug.

#include <iostream>        // Se usa para imprimir en consola.
#include <fstream>         // Se usa para leer el archivo binario.
#include <iomanip>         // Se usa para imprimir en hexadecimal con formato.
#include <stdint.h>        // Se usan tipos estándar como uint32_t.

// ============================================================
// Lectura del archivo binario
// ============================================================
bool read_binary_file_words(const std::string &file_path, std::vector<word_t> &buffer) {
    std::ifstream file(file_path.c_str(), std::ios::binary);
    // Se abre el archivo en modo binario.

    if (!file.is_open()) {
        std::cerr << "ERROR: no se pudo abrir el archivo: " << file_path << std::endl;
        // Se reporta fallo de apertura.

        return false;
        // Se retorna false si no se pudo abrir.
    }

    buffer.clear();
    // Se limpia el vector por si ya tenía datos.

    while (true) {
        uint32_t temp_word = 0;
        // Variable temporal de 32 bits para leer una palabra del archivo.

        file.read(reinterpret_cast<char*>(&temp_word), sizeof(uint32_t));
        // Se leen 4 bytes del archivo.

        if (!file) {
            break;
            // Si la lectura falla o llega al final, se sale del lazo.
        }

        buffer.push_back((word_t)temp_word);
        // Se agrega la palabra al vector como tipo word_t.
    }

    file.close();
    // Se cierra el archivo.

    return true;
    // Se indica que la lectura fue exitosa.
}

// ============================================================
// Extracción de una muestra de 25 words
// ============================================================
void extract_sample_words(
    const std::vector<word_t> &buffer,
    int sample_idx,
    word_t sample_words[WORDS_PER_SAMPLE]
) {
    int base = sample_idx * WORDS_PER_SAMPLE;
    // Se calcula el índice base de la muestra deseada.

    for (int i = 0; i < WORDS_PER_SAMPLE; i++) {
        sample_words[i] = buffer[base + i];
        // Se copian las 25 palabras de la muestra hacia el arreglo local.
    }
}

// ============================================================
// Impresión de las 25 palabras de la muestra
// ============================================================
void print_sample_words(
    const word_t sample_words[WORDS_PER_SAMPLE],
    const std::string &title
) {
    std::cout << title << std::endl;
    // Se imprime el título de la sección.

    for (int i = 0; i < WORDS_PER_SAMPLE; i++) {
        std::cout << "word[" << std::setw(2) << i << "] = 0x"
                  << std::hex << std::setw(8) << std::setfill('0')
                  << (unsigned int)sample_words[i]
                  << std::dec << std::setfill(' ')
                  << std::endl;
        // Se imprime cada word en hexadecimal de 8 dígitos.
    }
}

// ============================================================
// Impresión del label one-hot
// ============================================================
void print_label_onehot(label_oh_t label_onehot) {
    std::cout << "label_onehot = [";
    // Se abre el vector visual.

    for (int i = 0; i < NUM_CLASSES; i++) {
        std::cout << (unsigned int)label_onehot[i];
        // Se imprime el bit i del one-hot.

        if (i != NUM_CLASSES - 1) {
            std::cout << ", ";
            // Se agrega coma entre elementos.
        }
    }

    std::cout << "]" << std::endl;
    // Se cierra la impresión del vector.
}

// ============================================================
// Impresión del padding
// ============================================================
void print_padding_bits(padding_t padding) {
    std::cout << "padding = [";
    // Se abre la representación del padding.

    for (int i = PADDING_BITS - 1; i >= 0; i--) {
        std::cout << (unsigned int)padding[i];
        // Se imprime cada bit del padding desde MSB a LSB.
    }

    std::cout << "]" << std::endl;
    // Se cierra la impresión del padding.
}

// ============================================================
// Resumen de pixels
// ============================================================
void print_pixels_summary(pixels_t pixels, int preview_count) {
    int ones_count = 0;
    // Se usará para contar cuántos píxeles valen 1.

    for (int i = 0; i < PIXEL_BITS; i++) {
        if (pixels[i] == 1) {
            ones_count++;
            // Se contabiliza cada pixel activo.
        }
    }

    std::cout << "pixels ones_count = " << ones_count << std::endl;
    // Se imprime el total de bits en 1 dentro de la imagen.

    std::cout << "pixels preview [0.." << (preview_count - 1) << "] = \n";
    // Se anuncia la ventana de preview.

    for (int i = 0; i < preview_count; i++) {
        if (i % 28 == 0){std::cout << "\n";}

        std::cout << (unsigned int)pixels[i];
        // Se imprimen los primeros preview_count bits tal como están indexados.
    }

    std::cout << std::endl;
    // Fin del preview.
}

// ============================================================
// Impresión completa de muestra desempaquetada
// ============================================================
void print_unpacked_sample(
    raw_sample_t sample,
    const std::string &title
) {
    label_oh_t label_onehot;
    // Variable para guardar el label extraído.

    pixels_t pixels;
    // Variable para guardar los píxeles extraídos.

    padding_t padding;
    // Variable para guardar el padding extraído.

    unpack_sample(sample, label_onehot, pixels, padding);
    // Se desempaqueta la muestra usando la misma lógica del kernel.

    print_separator(title);
    // Se imprime un separador visual.

    print_label_onehot(label_onehot);
    // Se imprime el one-hot.

    std::cout << "decoded_label = " << (unsigned int)decode_onehot(label_onehot) << std::endl;
    // Se imprime el índice de clase decodificado.

    std::cout << "onehot_valid = " << (is_valid_onehot(label_onehot) ? "true" : "false") << std::endl;
    // Se reporta si el one-hot tiene exactamente un bit encendido.

    print_padding_bits(padding);
    // Se imprime el padding.

    print_pixels_summary(pixels, 784);
}

// ============================================================
// Comparadores de apoyo
// ============================================================
bool same_pixels(raw_sample_t a, raw_sample_t b) {
    pixels_t pixels_a;
    // Pixels de la muestra A.

    pixels_t pixels_b;
    // Pixels de la muestra B.

    label_oh_t label_dummy_a;
    // Variable dummy para ignorar el label en A.

    label_oh_t label_dummy_b;
    // Variable dummy para ignorar el label en B.

    padding_t padding_dummy_a;
    // Variable dummy para ignorar padding en A.

    padding_t padding_dummy_b;
    // Variable dummy para ignorar padding en B.

    unpack_sample(a, label_dummy_a, pixels_a, padding_dummy_a);
    // Se extraen los pixels de A.

    unpack_sample(b, label_dummy_b, pixels_b, padding_dummy_b);
    // Se extraen los pixels de B.

    return (pixels_a == pixels_b);
    // Se retorna true solo si ambos vectores de pixels son idénticos.
}

bool same_padding(raw_sample_t a, raw_sample_t b) {
    pixels_t pixels_dummy_a;
    // Variable dummy para ignorar pixels en A.

    pixels_t pixels_dummy_b;
    // Variable dummy para ignorar pixels en B.

    label_oh_t label_dummy_a;
    // Variable dummy para ignorar label en A.

    label_oh_t label_dummy_b;
    // Variable dummy para ignorar label en B.

    padding_t padding_a;
    // Padding extraído de A.

    padding_t padding_b;
    // Padding extraído de B.

    unpack_sample(a, label_dummy_a, pixels_dummy_a, padding_a);
    // Se extrae el padding de A.

    unpack_sample(b, label_dummy_b, pixels_dummy_b, padding_b);
    // Se extrae el padding de B.

    return (padding_a == padding_b);
    // Se retorna true si ambos paddings son idénticos.
}

bool same_raw_sample(raw_sample_t a, raw_sample_t b) {
    return (a == b);
    // Se compara la igualdad bit a bit de toda la muestra de 800 bits.
}

// ============================================================
// Separador visual
// ============================================================
void print_separator(const std::string &title) {
    std::cout << std::endl;
    // Línea en blanco inicial.

    std::cout << "============================================================" << std::endl;
    // Línea superior del separador.

    std::cout << title << std::endl;
    // Título de la sección.

    std::cout << "============================================================" << std::endl;
    // Línea inferior del separador.
}

// ============================================================
// Mostrar un rango de muestras desempaquetadas
// ============================================================
void print_samples_range(
    const std::vector<word_t> &buffer,
    int start_sample,
    int num_samples
) {
    // ============================================================
    // Verificación básica de rango
    // ============================================================

    int total_samples = buffer.size() / WORDS_PER_SAMPLE;
    // Se calcula cuántas muestras existen realmente en el buffer.

    if (start_sample < 0) {
        std::cout << "ERROR: start_sample no puede ser negativo." << std::endl;
        return;
    }

    if (start_sample >= total_samples) {
        std::cout << "ERROR: start_sample fuera de rango." << std::endl;
        return;
    }

    if (num_samples <= 0) {
        std::cout << "ERROR: num_samples debe ser mayor que 0." << std::endl;
        return;
    }

    int end_sample = start_sample + num_samples;
    // Se calcula el índice final del rango solicitado.

    if (end_sample > total_samples) {
        end_sample = total_samples;
        // Se recorta el rango si excede el total disponible.
    }

    print_separator("INSPECCION DE RANGO DE MUESTRAS");
    // Se imprime encabezado visual.

    std::cout << "start_sample = " << start_sample << std::endl;
    std::cout << "num_samples  = " << num_samples << std::endl;
    std::cout << "range        = [" << start_sample << " : " << end_sample << ")" << std::endl;

    // ============================================================
    // Iteración sobre las muestras solicitadas
    // ============================================================

    for (int s = start_sample; s < end_sample; s++) {

        print_separator("MUESTRA " + std::to_string(s));
        // Se imprime un separador con el índice de la muestra.

        word_t sample_words[WORDS_PER_SAMPLE];
        // Arreglo temporal para contener las 25 words de la muestra.

        extract_sample_words(buffer, s, sample_words);
        // Se extraen las 25 words correspondientes a la muestra s.

        print_sample_words(sample_words, "words crudas:");
        // Se imprimen las 25 words en hexadecimal.

        raw_sample_t sample = load_sample_from_words(buffer.data(), s);
        // Se reconstruye la muestra completa de 800 bits.

        print_unpacked_sample(sample, "desempaquetado:");
        // Se muestra el contenido interpretado: label, pixels, padding.
    }
}