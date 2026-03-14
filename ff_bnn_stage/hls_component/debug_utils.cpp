// ============================================================================
// debug_utils.cpp
// ----------------------------------------------------------------------------
// Implementación de utilidades de depuración para el testbench.
// ============================================================================

#include "debug_utils.hpp"

// ============================================================================
// Imprime el label one-hot de 10 bits.
// ============================================================================
void print_label_onehot(label_oh_t label, const std::string &name) {
    // Imprime el nombre del campo.
    std::cout << name << " = [";

    // Recorre los 10 bits en orden ascendente de índice.
    for (int i = 0; i < NUM_CLASSES; i++) {
        // Imprime el bit actual.
        std::cout << (unsigned)label[i];

        // Agrega separador si no es el último elemento.
        if (i != NUM_CLASSES - 1) {
            std::cout << ", ";
        }
    }

    // Cierra el vector.
    std::cout << "]" << std::endl;
}

// ============================================================================
// Imprime el índice de etiqueta.
// ============================================================================
void print_label_index(label_idx_t idx, const std::string &name) {
    // Convierte explícitamente a entero sin signo para impresión limpia.
    std::cout << name << " = " << (unsigned)idx << std::endl;
}

// ============================================================================
// Imprime los 6 bits de padding.
// ============================================================================
void print_padding_bits(padding_t padding, const std::string &name) {
    // Imprime el nombre del campo.
    std::cout << name << " = [";

    // Recorre cada bit del padding.
    for (int i = 0; i < PADDING_BITS; i++) {
        // Imprime el bit actual.
        std::cout << (unsigned)padding[i];

        // Agrega separador si no es el último.
        if (i != PADDING_BITS - 1) {
            std::cout << ", ";
        }
    }

    // Cierra el vector.
    std::cout << "]" << std::endl;
}

// ============================================================================
// Imprime una vista previa de los primeros "count" pixeles.
// ============================================================================
void print_pixels_preview(pixels_t pixels, const std::string &name, int count) {
    // Imprime encabezado.
    std::cout << name << " preview = [";

    // Limita count a PIXEL_BITS por seguridad.
    int limit = (count > PIXEL_BITS) ? PIXEL_BITS : count;

    // Recorre los pixeles solicitados.
    for (int i = 0; i < limit; i++) {
        // Imprime el bit actual.
        std::cout << (unsigned)pixels[i];

        // Agrega separador si no es el último.
        if (i != limit - 1) {
            std::cout << ", ";
        }
    }

    // Cierra la lista.
    std::cout << "]" << std::endl;
}

// ============================================================================
// Imprime las 25 palabras de 32 bits en hexadecimal.
// ============================================================================
void print_words_25x32(const word_t words[WORDS_PER_SAMPLE], const std::string &name) {
    // Imprime encabezado.
    std::cout << name << ":" << std::endl;

    // Recorre las 25 palabras.
    for (int i = 0; i < WORDS_PER_SAMPLE; i++) {
        // Imprime índice de palabra.
        std::cout << "  word[" << std::setw(2) << i << "] = 0x";

        // Imprime la palabra en hexadecimal de 8 dígitos.
        std::cout << std::hex << std::setw(8) << std::setfill('0')
                  << (uint32_t)words[i]
                  << std::dec << std::setfill(' ')
                  << std::endl;
    }
}

// ============================================================================
// Compara dos arreglos de 25 palabras.
// ============================================================================
bool compare_words_25x32(const word_t a[WORDS_PER_SAMPLE], const word_t b[WORDS_PER_SAMPLE]) {
    // Recorre las 25 palabras.
    for (int i = 0; i < WORDS_PER_SAMPLE; i++) {
        // Si alguna palabra difiere, retorna falso.
        if (a[i] != b[i]) {
            return false;
        }
    }

    // Si todas coinciden, retorna verdadero.
    return true;
}