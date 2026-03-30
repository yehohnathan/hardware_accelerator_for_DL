#include "debug_utils.hpp" // Se incluye el header con los prototipos de utilidades de debug.

#include <iostream>         // Se incluye iostream para imprimir en consola durante la simulación.
#include <fstream>          // Se incluye fstream para leer el archivo binario del dataset congelado.
#include <iomanip>          // Se incluye iomanip para alinear y formatear salidas numéricas.
#include <stdint.h>         // Se incluyen tipos estándar como uint32_t para lectura del binario.
#include <limits>           // Se incluye limits para inicializar mínimos y máximos al resumir el modelo.

// ============================================================
// Lectura del archivo binario
// ============================================================

bool read_binary_file_words(const std::string &file_path, std::vector<word_t> &buffer) {
    std::ifstream file(file_path.c_str(), std::ios::binary);
    // Se abre el archivo solicitado en modo binario para leer exactamente las words empaquetadas.

    if (!file.is_open()) {
        std::cerr << "ERROR: no se pudo abrir el archivo: " << file_path << std::endl;
        // Si la apertura falla, se reporta el problema para depuración del testbench.

        return false;
        // Se retorna false para indicar que la lectura no pudo completarse.
    }

    buffer.clear();
    // Se limpia el vector por si ya contenía datos de una lectura previa.

    while (true) {
        uint32_t temp_word = 0;
        // Se reserva una variable temporal de 32 bits para leer una word cruda del archivo.

        file.read(reinterpret_cast<char*>(&temp_word), sizeof(uint32_t));
        // Se leen 4 bytes del archivo y se reinterpretan como una palabra little-endian.

        if (!file) {
            break;
            // Si la lectura falla o llega al final del archivo, se termina el lazo.
        }

        buffer.push_back((word_t)temp_word);
        // La word leída se convierte al tipo del proyecto y se agrega al buffer lineal.
    }

    file.close();
    // Se cierra explícitamente el archivo porque ya no se requieren más accesos.

    return true;
    // Se retorna true para indicar que la lectura fue exitosa.
}

// ============================================================
// Extracción de una muestra de 25 words
// ============================================================
void extract_sample_words(const std::vector<word_t> &buffer, int sample_idx, word_t sample_words[WORDS_PER_SAMPLE]) {
    int base = sample_idx * WORDS_PER_SAMPLE;
    // Se calcula el índice base de la muestra deseada dentro del buffer lineal de words.

extract_sample_loop:
    for (int i = 0; i < WORDS_PER_SAMPLE; i++) {
        sample_words[i] = buffer[base + i];
        // Se copian las 25 words consecutivas de la muestra al arreglo local de inspección.
    }
}

// ============================================================
// Impresión de las 25 words de una muestra
// ============================================================
void print_sample_words(const word_t sample_words[WORDS_PER_SAMPLE], const std::string &title) {
    std::cout << title << std::endl;
    // Se imprime el título de la sección que precede a la muestra cruda.

print_word_loop:
    for (int i = 0; i < WORDS_PER_SAMPLE; i++) {
        std::cout << "word[" << std::setw(2) << i << "] = 0x"
                  << std::hex << std::setw(8) << std::setfill('0')
                  << (unsigned int)sample_words[i]
                  << std::dec << std::setfill(' ')
                  << std::endl;
        // Se imprime cada word en hexadecimal de ocho dígitos para verificar el layout del archivo.
    }
}

// ============================================================
// Impresión del label one-hot
// ============================================================
void print_label_onehot(label_oh_t label_onehot) {
    std::cout << "label_onehot = [";
    // Se abre la representación visual del vector one-hot.

print_onehot_loop:
    for (int i = 0; i < NUM_CLASSES; i++) {
        std::cout << (unsigned int)label_onehot[i];
        // Se imprime el bit correspondiente a la posición i del vector one-hot.

        if (i != NUM_CLASSES - 1) {
            std::cout << ", ";
            // Se agrega una coma entre elementos para mejorar la legibilidad del vector.
        }
    }

    std::cout << "]" << std::endl;
    // Se cierra la impresión del vector y se finaliza la línea.
}

// ============================================================
// Impresión del padding
// ============================================================
void print_padding_bits(padding_t padding) {
    std::cout << "padding = [";
    // Se abre la representación textual del padding físico.

print_padding_loop:
    for (int i = PADDING_BITS - 1; i >= 0; i--) {
        std::cout << (unsigned int)padding[i];
        // Se imprime cada bit del padding desde el más significativo al menos significativo.
    }

    std::cout << "]" << std::endl;
    // Se cierra la impresión del padding y se finaliza la línea.
}

// ============================================================
// Resumen de píxeles
// ============================================================
void print_pixels_summary(pixels_t pixels, int preview_count) {
    int ones_count = 0;
    // Se reserva un contador para conocer cuántos bits activos tiene la imagen binaria.

count_pixels_loop:
    for (int i = 0; i < PIXEL_BITS; i++) {
        if (pixels[i] == 1) {
            ones_count++;
            // Se incrementa el conteo cada vez que se detecta un pixel activo.
        }
    }

    std::cout << "pixels ones_count = " << ones_count << std::endl;
    // Se imprime el número total de unos de la imagen para inspección rápida.

    std::cout << "pixels preview [0.." << (preview_count - 1) << "] = \n";
    // Se anuncia el rango de bits que se mostrará como previsualización del vector binario.

preview_pixels_loop:
    for (int i = 0; i < preview_count; i++) {
        if (i % 28 == 0) {
            std::cout << "\n";
            // Se inserta un salto de línea cada 28 bits para recordar la geometría 28x28 de MNIST.
        }

        std::cout << (unsigned int)pixels[i];
        // Se imprime el bit actual de la vista previa solicitada.
    }

    std::cout << std::endl;
    // Se finaliza la vista previa con un salto de línea final.
}

// ============================================================
// Impresión completa de muestra desempaquetada
// ============================================================
void print_unpacked_sample(raw_sample_t sample, const std::string &title) {
    label_oh_t label_onehot;
    // Se reserva un contenedor local para la etiqueta extraída de la muestra.

    pixels_t pixels;
    // Se reserva un contenedor local para la imagen binaria extraída de la muestra.

    padding_t padding;
    // Se reserva un contenedor local para el padding físico extraído de la muestra.

    unpack_sample(sample, label_onehot, pixels, padding);
    // Se desempaqueta la muestra usando exactamente la misma lógica definida para el kernel.

    print_separator(title);
    // Se imprime un separador visual antes de mostrar el contenido interpretado.

    print_label_onehot(label_onehot);
    // Se imprime la etiqueta one-hot en forma legible.

    std::cout << "decoded_label = " << (unsigned int)decode_onehot(label_onehot) << std::endl;
    // Se imprime el índice decodificado de la clase para facilitar la revisión rápida.

    std::cout << "onehot_valid = " << (is_valid_onehot(label_onehot) ? "true" : "false") << std::endl;
    // Se informa si el one-hot desempaquetado contiene exactamente un bit activo.

    print_padding_bits(padding);
    // Se imprime el padding físico asociado a la muestra.

    print_pixels_summary(pixels, 784);
    // Se imprime un resumen completo de la imagen binaria contenida en la muestra.
}

// ============================================================
// Comparadores de apoyo
// ============================================================
bool same_pixels(raw_sample_t a, raw_sample_t b) {
    pixels_t pixels_a;
    // Se reserva el contenedor local para los pixels de la muestra A.

    pixels_t pixels_b;
    // Se reserva el contenedor local para los pixels de la muestra B.

    label_oh_t label_dummy_a;
    // Se reserva un label dummy porque aquí solo interesa comparar la imagen de A.

    label_oh_t label_dummy_b;
    // Se reserva un label dummy porque aquí solo interesa comparar la imagen de B.

    padding_t padding_dummy_a;
    // Se reserva un padding dummy porque aquí no interesa el padding de A.

    padding_t padding_dummy_b;
    // Se reserva un padding dummy porque aquí no interesa el padding de B.

    unpack_sample(a, label_dummy_a, pixels_a, padding_dummy_a);
    // Se desempaqueta A para extraer únicamente su vector de píxeles.

    unpack_sample(b, label_dummy_b, pixels_b, padding_dummy_b);
    // Se desempaqueta B para extraer únicamente su vector de píxeles.

    return (pixels_a == pixels_b);
    // Se retorna true solo si ambas imágenes binarizadas son idénticas bit a bit.
}

bool same_padding(raw_sample_t a, raw_sample_t b) {
    pixels_t pixels_dummy_a;
    // Se reserva un contenedor dummy para ignorar los pixels de A.

    pixels_t pixels_dummy_b;
    // Se reserva un contenedor dummy para ignorar los pixels de B.

    label_oh_t label_dummy_a;
    // Se reserva un contenedor dummy para ignorar el label de A.

    label_oh_t label_dummy_b;
    // Se reserva un contenedor dummy para ignorar el label de B.

    padding_t padding_a;
    // Se reserva el contenedor local para el padding de A.

    padding_t padding_b;
    // Se reserva el contenedor local para el padding de B.

    unpack_sample(a, label_dummy_a, pixels_dummy_a, padding_a);
    // Se extrae únicamente el padding útil de la muestra A.

    unpack_sample(b, label_dummy_b, pixels_dummy_b, padding_b);
    // Se extrae únicamente el padding útil de la muestra B.

    return (padding_a == padding_b);
    // Se retorna true solo si ambos paddings son idénticos bit a bit.
}

bool same_raw_sample(raw_sample_t a, raw_sample_t b) {
    return (a == b);
    // Se compara la igualdad total de la muestra completa de 800 bits.
}

// ============================================================
// Separador visual
// ============================================================
void print_separator(const std::string &title) {
    std::cout << std::endl;
    // Se deja una línea en blanco inicial para separar visualmente secciones consecutivas.

    std::cout << "============================================================" << std::endl;
    // Se imprime la línea superior del separador.

    std::cout << title << std::endl;
    // Se imprime el título descriptivo de la sección actual.

    std::cout << "============================================================" << std::endl;
    // Se imprime la línea inferior del separador.
}

// ============================================================
// Mostrar un rango de muestras desempaquetadas
// ============================================================
void print_samples_range(const std::vector<word_t> &buffer, int start_sample, int num_samples) {
	
	// ============= Verificación básica de rango =============
	int total_samples = buffer.size() / WORDS_PER_SAMPLE;
    // Se calcula cuántas muestras completas existen dentro del buffer lineal cargado desde disco.

    if (start_sample < 0) {
        std::cout << "ERROR: start_sample no puede ser negativo." << std::endl;
        // Se reporta un error si el índice inicial solicitado es inválido.

        return;
        // Se abandona la función porque no es posible continuar con un rango inválido.
    }

    if (start_sample >= total_samples) {
        std::cout << "ERROR: start_sample fuera de rango." << std::endl;
        // Se reporta un error si el inicio cae fuera del número real de muestras.

        return;
        // Se abandona la función porque no hay muestras válidas que imprimir.
    }

    if (num_samples <= 0) {
        std::cout << "ERROR: num_samples debe ser mayor que 0." << std::endl;
        // Se reporta un error si la cantidad solicitada de muestras no es positiva.

        return;
        // Se abandona la función porque no tiene sentido imprimir un rango vacío o negativo.
    }

    int end_sample = start_sample + num_samples;
    // Se calcula el índice final del rango solicitado antes de aplicar recorte.

    if (end_sample > total_samples) {
        end_sample = total_samples;
        // Si el rango se sale del buffer real, se recorta al último índice válido disponible.
    }

    print_separator("INSPECCION DE RANGO DE MUESTRAS");
    std::cout << "start_sample = " << start_sample << std::endl;
    std::cout << "num_samples  = " << num_samples << std::endl;
    std::cout << "range        = [" << start_sample << " : " << end_sample << ")" << std::endl;

	// ======= Iteración sobre las muestras solicitadas =======
print_range_loop:
    for (int sample_idx = start_sample; sample_idx < end_sample; sample_idx++) {
        print_separator("MUESTRA " + std::to_string(sample_idx));
        // Se abre una subsección independiente para la muestra actual.

        word_t sample_words[WORDS_PER_SAMPLE];
        // Se reserva un arreglo temporal donde se copiarán las 25 words crudas de la muestra.

        extract_sample_words(buffer, sample_idx, sample_words);
        // Se extraen las 25 words correspondientes a la muestra actual.

        print_sample_words(sample_words, "words crudas:");
        // Se imprimen las 25 words en hexadecimal para revisión manual del binario.

        raw_sample_t sample = load_sample_from_words(buffer.data(), sample_idx);
        // Se reconstruye la muestra física completa de 800 bits desde el buffer lineal.

        print_unpacked_sample(sample, "desempaquetado:");
        // Se muestra el contenido interpretado de la muestra actual: label, pixels y padding.
    }
}

// ============================================================
// Vista previa de métricas de entrenamiento
// ============================================================
void print_training_preview(const std::vector<goodness_t> &g_pos, const std::vector<goodness_t> &g_neg, const std::vector<goodness_t> &gap, int preview_count) {
    print_separator("VISTA PREVIA DE METRICAS DE ENTRENAMIENTO FF");
    // Se abre una sección dedicada a las métricas FF almacenadas desde el top entrenable.

    int total = (int)gap.size();
    // Se obtiene la cantidad total de entradas disponibles para imprimir.

    int effective_preview = preview_count;
    // Se crea una copia local del número de elementos solicitados para impresión.

    if (effective_preview > total) {
        effective_preview = total;
        // Si el usuario pide más de lo disponible, se ajusta automáticamente al tamaño real del buffer.
    }

preview_metric_loop:
    for (int i = 0; i < effective_preview; i++) {
        std::cout << "sample[" << i << "]"
                  << " g_pos=" << g_pos[i].to_double()
                  << " g_neg=" << g_neg[i].to_double()
                  << " gap=" << gap[i].to_double()
                  << std::endl;
        // Se imprime para cada muestra la goodness positiva, la negativa y el gap observado en la última época.
    }
}

// ============================================================
// Vista previa de predicciones
// ============================================================
void print_prediction_preview(const std::vector<label_idx_t> &true_labels, const std::vector<label_idx_t> &pred_labels, int preview_count) {
    print_separator("VISTA PREVIA DE PREDICCIONES");
    // Se abre una sección dedicada a comparar etiquetas reales contra predicciones del acelerador.

    int total = (int)true_labels.size();
    // Se obtiene la cantidad total de etiquetas reales disponibles.

    int effective_preview = preview_count;
    // Se crea una copia local del número de ejemplos a imprimir.

    if (effective_preview > total) {
        effective_preview = total;
        // Se recorta la vista previa si supera el tamaño real de los vectores.
    }

preview_prediction_loop:
    for (int i = 0; i < effective_preview; i++) {
        std::cout << "sample[" << i << "]"
                  << " true=" << (unsigned int)true_labels[i]
                  << " pred=" << (unsigned int)pred_labels[i]
                  << " hit=" << ((true_labels[i] == pred_labels[i]) ? "true" : "false")
                  << std::endl;
        // Se imprime una comparación compacta entre clase real y clase predicha para cada ejemplo mostrado.
    }
}

// ============================================================
// Resumen del estado entrenado del modelo
// ============================================================
void print_model_overview(const std::vector<latent_t> &weights, const std::vector<bias_t> &biases, int preview_weights) {
    print_separator("RESUMEN DEL MODELO ENTRENADO");
    // Se abre una sección destinada a inspeccionar el estado final de pesos y bias latentes.

    double min_weight = std::numeric_limits<double>::max();
    // Se inicializa el mínimo observado de pesos en un valor muy grande.

    double max_weight = -std::numeric_limits<double>::max();
    // Se inicializa el máximo observado de pesos en un valor muy pequeño.

    int positive_count = 0;
    // Se inicializa el conteo de pesos con signo positivo.

    int negative_count = 0;
    // Se inicializa el conteo de pesos con signo negativo.

    int zero_count = 0;
    // Se inicializa el conteo de pesos exactamente en cero.

model_weight_summary_loop:
    for (std::size_t i = 0; i < weights.size(); i++) {
        double value = weights[i].to_double();
        // Se convierte el peso actual a double solo para fines de impresión y análisis fuera del kernel.

        if (value < min_weight) {
            min_weight = value;
            // Se actualiza el mínimo si el peso actual es menor que todos los anteriores.
        }

        if (value > max_weight) {
            max_weight = value;
            // Se actualiza el máximo si el peso actual es mayor que todos los anteriores.
        }

        if (value > 0.0) {
            positive_count++;
            // Se cuenta el peso actual como positivo para conocer el balance de signos aprendido.
        } else if (value < 0.0) {
            negative_count++;
            // Se cuenta el peso actual como negativo para conocer el balance de signos aprendido.
        } else {
            zero_count++;
            // Se cuenta el peso actual como cero exacto para detectar posibles estancamientos.
        }
    }

    std::cout << "total_weights = " << weights.size() << std::endl;
    std::cout << "min_weight    = " << min_weight << std::endl;
    std::cout << "max_weight    = " << max_weight << std::endl;

    std::cout << "positive_w    = " << positive_count << std::endl;
    // Se imprime la cantidad de pesos positivos del modelo entrenado.

    std::cout << "negative_w    = " << negative_count << std::endl;
    // Se imprime la cantidad de pesos negativos del modelo entrenado.

    std::cout << "zero_w        = " << zero_count << std::endl;
    // Se imprime la cantidad de pesos exactamente nulos del modelo entrenado.

    std::cout << "bias_count    = " << biases.size() << std::endl;
    // Se imprime la cantidad total de bias latentes presentes en el snapshot.

    std::cout << "first_weights = ";
    // Se abre una lista compacta con las primeras posiciones del vector de pesos para revisión rápida.
preview_weight_loop:
    for (int i = 0; i < preview_weights; i++) {
        if (i < (int)weights.size()) {
            std::cout << weights[i].to_double() << " ";
            // Se imprime el peso de la posición i si existe dentro del tamaño real del snapshot.
        }
    }
    std::cout << std::endl;
    // Se cierra la línea de pesos de vista previa.

    std::cout << "first_biases  = ";
    // Se abre una lista compacta con los primeros bias del modelo entrenado.
preview_bias_loop:
    for (int i = 0; i < (int)biases.size(); i++) {
        if (i < 16) {
            std::cout << biases[i].to_double() << " ";
            // Se imprime un subconjunto inicial de bias para revisar rápidamente su evolución tras el entrenamiento.
        }
    }
    std::cout << std::endl;
    // Se cierra la línea de bias de vista previa.
}
