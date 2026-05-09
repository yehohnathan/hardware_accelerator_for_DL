
/**
 * @file ff_dataset.cpp
 * @brief Decodificación de MNIST empaquetado y carga de datos para testbench.
 *
 * La parte sintetizable transforma muestras `[one-hot | pixeles]` empaquetadas
 * en palabras de 32 bits hacia etiquetas y valores de pixel. La parte protegida
 * por `__SYNTHESIS__` pertenece al host: abre archivos, valida cabeceras y
 * prepara el payload que se entregará al kernel HLS por memoria externa.
 */
#include "ff_dataset.hpp"

#ifndef __SYNTHESIS__
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#endif

/**
 * @brief Recupera la etiqueta one-hot ubicada al inicio de la muestra.
 *
 * @param sample_words Muestra ya copiada a memoria local.
 * @return Índice de la clase MNIST activa.
 *
 * @sideeffect No modifica memoria. Si un archivo corrupto tuviera más de un bit
 * de etiqueta activo, el índice más alto prevalecería; los datasets generados
 * para este proyecto contienen exactamente una clase activa.
 */
uint8_t ff_decode_label_from_words(
    const ff_word_t sample_words[FF_WORDS_PER_SAMPLE]
) {
/* Se inserta inline porque es una rutina corta usada por entrenamiento e
 * inferencia; eliminar la jerarquía ayuda a HLS a optimizar el camino de datos.
 */
#pragma HLS INLINE
    uint8_t label = 0;

decode_label_loop:
    for (int i = 0; i < FF_NUM_CLASSES; i++) {
/* El one-hot de MNIST tiene 10 bits fijos; desenrollar el bucle permite evaluar
 * todas las clases en paralelo con un coste pequeño de LUT.
 */
#pragma HLS UNROLL
        if (ff_get_packed_bit(sample_words, i) != 0u) {
            label = (uint8_t)i;
        }
    }

    return label;
}

/**
 * @brief Reconstruye un pixel cuantizado desde bits LSB-first.
 *
 * @param sample_words Muestra empaquetada en palabras de 32 bits.
 * @param pixel_index Índice del pixel dentro de la imagen redimensionada.
 * @return Valor entero del pixel con `FF_PIXEL_BITS` bits.
 *
 * El resultado alimenta el vector de características de la primera capa. Para
 * datasets binarios el valor es 0/1; para 4b/6b representa intensidad
 * cuantizada antes de escalarse.
 */
uint8_t ff_decode_pixel_from_words(
    const ff_word_t sample_words[FF_WORDS_PER_SAMPLE],
    int pixel_index
) {
/* Inline evita crear una función separada dentro del bucle que recorre todos
 * los pixeles de MNIST.
 */
#pragma HLS INLINE
    uint8_t value = 0;
    int base_bit = FF_LABEL_BITS + (pixel_index * FF_PIXEL_BITS);

decode_pixel_bit_loop:
    for (int bit = 0; bit < FF_PIXEL_BITS; bit++) {
/* II=1 permite reconstruir un bit por ciclo siempre que la lectura del buffer
 * local no cree conflictos; el bucle está acotado por 1, 4 o 6 bits.
 */
#pragma HLS PIPELINE II=1
        value = (uint8_t)(value |
            (uint8_t)(ff_get_packed_bit(sample_words, base_bit + bit) << bit));
    }

    return value;
}

/**
 * @brief Copia una muestra del payload global hacia memoria local.
 *
 * @param dataset_words Payload contiguo sin cabecera.
 * @param sample_index Índice de muestra dentro del payload.
 * @param sample_words Buffer local de salida.
 *
 * En hardware, `dataset_words` representa memoria externa AXI. La copia local
 * reduce el direccionamiento repetido durante desempaquetado y facilita que HLS
 * optimice los accesos posteriores.
 */
void ff_copy_sample_words(
    const ff_word_t *dataset_words,
    int sample_index,
    ff_word_t sample_words[FF_WORDS_PER_SAMPLE]
) {
/* La copia es pequeña y se usa en rutas críticas; inline evita una jerarquía
 * extra alrededor del movimiento de muestra.
 */
#pragma HLS INLINE
    int base = sample_index * FF_WORDS_PER_SAMPLE;

copy_sample_words_loop:
    for (int w = 0; w < FF_WORDS_PER_SAMPLE; w++) {
/* El pipeline busca transferir una palabra por ciclo desde el payload al buffer
 * local. La latencia total crece con `FF_WORDS_PER_SAMPLE`.
 */
#pragma HLS PIPELINE II=1
        sample_words[w] = dataset_words[base + w];
    }
}

#ifndef __SYNTHESIS__
/**
 * @brief Convierte las 16 palabras de cabecera en una estructura tipada.
 *
 * @param words Cabecera cruda leída desde archivo.
 * @return Cabecera con nombres de campo usados por validación y logs.
 *
 * @note Solo host/testbench. No es sintetizable porque se usa únicamente antes
 * de invocar los kernels.
 */
static ff_dataset_header_t ff_header_from_words(const ff_word_t words[FF_HEADER_WORDS]) {
    ff_dataset_header_t header;
    header.magic = words[0];
    header.version = words[1];
    header.header_words = words[2];
    header.sample_count = words[3];
    header.image_width = words[4];
    header.image_height = words[5];
    header.source_width = words[6];
    header.source_height = words[7];
    header.pixel_bits = words[8];
    header.num_classes = words[9];
    header.label_bits = words[10];
    header.useful_bits = words[11];
    header.words_per_sample = words[12];
    header.total_bits = words[13];
    header.padding_bits = words[14];
    header.resize_applied = words[15];
    return header;
}

/**
 * @brief Selecciona la ruta del dataset para simulación.
 *
 * @return Ruta explícita por `FF_DATASET_BIN` o primera ruta conocida existente.
 *
 * @note Solo host/testbench. Usa variables de entorno, cadenas y búsqueda de
 * archivos; por eso queda fuera de síntesis.
 */
std::string ff_default_dataset_path() {
    const char *env_path = std::getenv("FF_DATASET_BIN");
    if (env_path != 0 && env_path[0] != '\0') {
        return std::string(env_path);
    }

    std::ostringstream name;
    name << "mnist_" << FF_IMAGE_WIDTH << "x" << FF_IMAGE_HEIGHT
         << "_" << FF_PIXEL_BITS << "b_packed.bin";

    const char *candidate_dirs[] = {
        "../mnist/data/processed/",
        "mnist/data/processed/",
        "ff_bnn_stage/mnist/data/processed/",
        "D:/TFG/hardware_accelerator_for_DL/ff_bnn_stage/mnist/data/processed/"
    };

    for (int i = 0; i < 4; i++) {
        std::string candidate = std::string(candidate_dirs[i]) + name.str();
        std::ifstream file(candidate.c_str(), std::ios::binary);
        if (file.good()) {
            return candidate;
        }
    }

    return std::string(candidate_dirs[0]) + name.str();
}

/**
 * @brief Valida que el binario coincida con las macros de compilación.
 *
 * @param header Cabecera decodificada del archivo.
 * @param error Mensaje descriptivo si algún campo es incompatible.
 * @return `true` si el dataset puede alimentar esta build.
 *
 * Esta verificación evita mezclar, por ejemplo, una build 20x20_4b con un
 * dataset 28x28_1b. Sin esa validación, el kernel leería offsets equivocados y
 * las métricas Forward-Forward serían inválidas.
 */
bool ff_validate_header_for_build(
    const ff_dataset_header_t &header,
    std::string &error
) {
    if (header.magic != FF_HEADER_MAGIC) {
        error = "magic de cabecera invalido";
        return false;
    }

    if (header.version != FF_HEADER_VERSION) {
        error = "version de cabecera no soportada";
        return false;
    }

    if (header.header_words != FF_HEADER_WORDS) {
        error = "cantidad de words de cabecera invalida";
        return false;
    }

    if (header.image_width != (uint32_t)FF_IMAGE_WIDTH ||
        header.image_height != (uint32_t)FF_IMAGE_HEIGHT ||
        header.pixel_bits != (uint32_t)FF_PIXEL_BITS) {
        std::ostringstream oss;
        oss << "el binario declara " << header.image_width << "x"
            << header.image_height << "_" << header.pixel_bits
            << "b, pero la build espera " << FF_IMAGE_WIDTH << "x"
            << FF_IMAGE_HEIGHT << "_" << FF_PIXEL_BITS << "b";
        error = oss.str();
        return false;
    }

    if (header.num_classes != (uint32_t)FF_NUM_CLASSES ||
        header.label_bits != (uint32_t)FF_LABEL_BITS) {
        error = "layout de etiqueta one-hot incompatible";
        return false;
    }

    if (header.useful_bits != (uint32_t)FF_USEFUL_BITS ||
        header.words_per_sample != (uint32_t)FF_WORDS_PER_SAMPLE ||
        header.total_bits != (uint32_t)FF_TOTAL_BITS ||
        header.padding_bits != (uint32_t)FF_PADDING_BITS) {
        error = "campos derivados del layout no coinciden con la build";
        return false;
    }

    return true;
}

/**
 * @brief Carga un dataset `.bin` completo para simulación C++.
 *
 * @param path Ruta del binario MNIST preprocesado.
 * @param header Cabecera validada como salida.
 * @param payload_words Palabras de muestras sin cabecera.
 * @param error Mensaje de error en caso de fallo.
 * @return `true` cuando cabecera y payload se leen por completo.
 *
 * @note Solo host/testbench. Usa `std::ifstream` y `std::vector`; el kernel HLS
 * recibe `payload_words.data()` como memoria externa ya preparada.
 */
bool ff_read_dataset(
    const std::string &path,
    ff_dataset_header_t &header,
    std::vector<ff_word_t> &payload_words,
    std::string &error
) {
    std::ifstream file(path.c_str(), std::ios::binary);
    if (!file.good()) {
        error = "no se pudo abrir el binario: " + path;
        return false;
    }

    ff_word_t header_words[FF_HEADER_WORDS];
    file.read(reinterpret_cast<char *>(header_words),
              sizeof(ff_word_t) * FF_HEADER_WORDS);
    if (file.gcount() != (std::streamsize)(sizeof(ff_word_t) * FF_HEADER_WORDS)) {
        error = "el archivo no contiene la cabecera completa de 64 bytes";
        return false;
    }

    header = ff_header_from_words(header_words);
    if (!ff_validate_header_for_build(header, error)) {
        return false;
    }

    uint64_t payload_count =
        (uint64_t)header.sample_count * (uint64_t)header.words_per_sample;
    payload_words.assign((size_t)payload_count, 0u);

    file.read(reinterpret_cast<char *>(payload_words.data()),
              (std::streamsize)(payload_count * sizeof(ff_word_t)));

    if (file.gcount() != (std::streamsize)(payload_count * sizeof(ff_word_t))) {
        error = "el payload leido no coincide con sample_count * words_per_sample";
        return false;
    }

    return true;
}

/**
 * @brief Escribe en consola los metadatos críticos del dataset.
 *
 * @param path Ruta que se abrió.
 * @param header Cabecera ya validada.
 *
 * El log resultante permite justificar en el TFG qué resolución, cuantización y
 * cantidad de muestras se usaron en cada experimento.
 */
void ff_print_header_summary(
    const std::string &path,
    const ff_dataset_header_t &header
) {
    std::cout << "dataset              : " << path << '\n';
    std::cout << "samples              : " << header.sample_count << '\n';
    std::cout << "layout               : " << header.image_width << "x"
              << header.image_height << "_" << header.pixel_bits << "b\n";
    std::cout << "source               : " << header.source_width << "x"
              << header.source_height << '\n';
    std::cout << "useful_bits/sample   : " << header.useful_bits << '\n';
    std::cout << "words_per_sample     : " << header.words_per_sample << '\n';
    std::cout << "padding_bits/sample  : " << header.padding_bits << '\n';
    std::cout << "resize_applied       : " << header.resize_applied << '\n';
}
#endif
