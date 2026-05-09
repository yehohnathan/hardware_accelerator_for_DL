
/**
 * @file ff_dataset.hpp
 * @brief Interfaz de decodificación del dataset MNIST empaquetado.
 *
 * Las funciones de extracción de bits, etiqueta y pixeles son compatibles con
 * HLS porque trabajan sobre arreglos estáticos y no usan memoria dinámica. Las
 * funciones de lectura de archivo quedan protegidas por `ifndef __SYNTHESIS__`
 * porque pertenecen al host/testbench y usan STL e I/O de C++.
 */
#ifndef HOPE_FF_DATASET_HPP
#define HOPE_FF_DATASET_HPP

#include "ff_types.hpp"

#ifndef __SYNTHESIS__
#include <string>
#include <vector>
#endif

/**
 * @brief Extrae un bit de una muestra con empaquetado LSB-first.
 *
 * @param sample_words Muestra local copiada desde el payload del dataset.
 * @param bit_index Índice absoluto del bit dentro de la muestra empaquetada.
 * @return Valor 0/1 del bit solicitado.
 *
 * @note Sintetizable. La función se inserta inline para que los decodificadores
 * de etiqueta y pixel no introduzcan una jerarquía innecesaria en HLS.
 */
static inline uint8_t ff_get_packed_bit(
    const ff_word_t sample_words[FF_WORDS_PER_SAMPLE],
    int bit_index
) {
/*
 * HLS INLINE reduce la sobrecarga de llamada en un primitivo muy pequeño que se
 * ejecuta repetidamente dentro del desempaquetado de MNIST.
 */
#pragma HLS INLINE
    int word_index = bit_index / FF_WORD_BITS;
    int inner_bit = bit_index % FF_WORD_BITS;
    return (uint8_t)((sample_words[word_index] >> inner_bit) & 1u);
}

/**
 * @brief Decodifica la etiqueta real one-hot de una muestra MNIST.
 *
 * @param sample_words Muestra local en formato `[label_onehot | pixels]`.
 * @return Índice de clase MNIST en el rango 0..9.
 *
 * @note Sintetizable. Se usa para construir el ejemplo positivo y para evaluar
 * si la predicción multiclase coincide con la etiqueta real.
 */
uint8_t ff_decode_label_from_words(
    const ff_word_t sample_words[FF_WORDS_PER_SAMPLE]
);

/**
 * @brief Reconstruye un pixel cuantizado desde el payload empaquetado.
 *
 * @param sample_words Muestra local empaquetada.
 * @param pixel_index Índice del pixel dentro de la imagen ya redimensionada.
 * @return Valor entero del pixel con `FF_PIXEL_BITS` de precisión.
 *
 * @note Sintetizable. El valor devuelto alimenta el vector de características
 * de la primera capa BNN.
 */
uint8_t ff_decode_pixel_from_words(
    const ff_word_t sample_words[FF_WORDS_PER_SAMPLE],
    int pixel_index
);

/**
 * @brief Copia una muestra desde memoria externa hacia un buffer local.
 *
 * @param dataset_words Payload contiguo sin cabecera.
 * @param sample_index Muestra a copiar.
 * @param sample_words Buffer local de salida con `FF_WORDS_PER_SAMPLE`.
 *
 * @note Sintetizable. En el kernel, `dataset_words` representa memoria externa
 * AXI; `sample_words` es memoria local que se puede particionar o acceder con
 * menor complejidad de direccionamiento.
 */
void ff_copy_sample_words(
    const ff_word_t *dataset_words,
    int sample_index,
    ff_word_t sample_words[FF_WORDS_PER_SAMPLE]
);

#ifndef __SYNTHESIS__
/**
 * @brief Resuelve la ruta de dataset usada por simulación C++.
 *
 * @return Ruta configurada por entorno o una ruta conocida del proyecto.
 *
 * @note Solo host/testbench. Usa `std::string`, entorno y sistema de archivos,
 * por lo que no es parte del hardware sintetizable.
 */
std::string ff_default_dataset_path();

/**
 * @brief Lee y valida un archivo `.bin` de MNIST preprocesado.
 *
 * @param path Ruta del archivo binario.
 * @param header Cabecera decodificada como salida.
 * @param payload_words Payload sin cabecera, listo para entrenamiento.
 * @param error Mensaje de error si la lectura o validación falla.
 * @return `true` si el archivo coincide con la build; `false` si no.
 *
 * @note Solo host/testbench. Carga los datos que luego se entregan a las
 * funciones sintetizables mediante punteros.
 */
bool ff_read_dataset(
    const std::string &path,
    ff_dataset_header_t &header,
    std::vector<ff_word_t> &payload_words,
    std::string &error
);

/**
 * @brief Comprueba que la cabecera del dataset coincide con la build.
 *
 * @param header Metadatos leídos desde el archivo.
 * @param error Explicación del primer campo incompatible.
 * @return `true` cuando resolución, bits y layout son compatibles.
 *
 * Evita entrenar con dimensiones incorrectas, una condición que en HLS podría
 * producir accesos fuera de rango o métricas sin significado.
 */
bool ff_validate_header_for_build(
    const ff_dataset_header_t &header,
    std::string &error
);

/**
 * @brief Imprime los metadatos del dataset en el log de simulación.
 *
 * @param path Ruta usada para cargar el dataset.
 * @param header Cabecera previamente validada.
 *
 * @note Solo host/testbench. Facilita trazabilidad académica de cada corrida.
 */
void ff_print_header_summary(
    const std::string &path,
    const ff_dataset_header_t &header
);
#endif

#endif
