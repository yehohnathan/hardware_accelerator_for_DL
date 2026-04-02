#include "debug_utils.hpp" // Se incluye el header con los prototipos de utilidades de debug.

#include <fstream>          // Se incluye fstream para leer el binario del dataset congelado.
#include <iomanip>          // Se incluye iomanip para alinear y formatear las salidas.
#include <iostream>         // Se incluye iostream para imprimir durante la simulacion.
#include <limits>           // Se incluye limits para resumir minimos y maximos del modelo.
#include <stdint.h>         // Se incluyen tipos estandar para la lectura de words crudas.

// ============================================================
// Lectura del archivo binario
// ============================================================
bool read_binary_file_words(const std::string &file_path, std::vector<word_t> &buffer) {
    std::ifstream file(file_path.c_str(), std::ios::binary);
    // Se abre el archivo solicitado en modo binario.

    if (!file.is_open()) {
        std::cerr << "ERROR: no se pudo abrir el archivo: " << file_path << std::endl;
        // Se reporta el problema si la ruta no pudo abrirse.

        return false;
        // Se retorna false para detener el testbench.
    }

    buffer.clear();
    // Se limpia el vector por si ya tenia contenido previo.

    while (true) {
        uint32_t temp_word = 0;
        // Se reserva una word temporal de 32 bits.

        file.read(reinterpret_cast<char *>(&temp_word), sizeof(uint32_t));
        // Se leen 4 bytes del archivo y se reinterpretan como una word.

        if (!file) {
            break;
            // Se abandona el lazo al llegar al fin de archivo o ante error de lectura.
        }

        buffer.push_back((word_t)temp_word);
        // Se agrega la word leida al buffer lineal del dataset.
    }

    file.close();
    // Se cierra el archivo porque ya no se requiere mas acceso.

    return true;
    // Se retorna true para indicar una lectura correcta.
}

// ============================================================
// Extraccion de una muestra de 25 words
// ============================================================
void extract_sample_words(const std::vector<word_t> &buffer, int sample_idx, word_t sample_words[WORDS_PER_SAMPLE]) {
    int base = sample_idx * WORDS_PER_SAMPLE;
    // Se calcula la posicion base de la muestra solicitada.

extract_sample_loop:
    for (int i = 0; i < WORDS_PER_SAMPLE; i++) {
        sample_words[i] = buffer[base + i];
        // Se copian las 25 words consecutivas al arreglo local de inspeccion.
    }
}

// ============================================================
// Impresion hexadecimal de una muestra
// ============================================================
void print_sample_words(const word_t sample_words[WORDS_PER_SAMPLE], const std::string &title) {
    std::cout << title << std::endl;
    // Se imprime el titulo descriptivo de la muestra cruda.

print_word_loop:
    for (int i = 0; i < WORDS_PER_SAMPLE; i++) {
        std::cout << "word[" << std::setw(2) << i << "] = 0x"
                  << std::hex << std::setw(8) << std::setfill('0')
                  << (unsigned int)sample_words[i]
                  << std::dec << std::setfill(' ')
                  << std::endl;
        // Se imprime cada word en hexadecimal de ocho digitos.
    }
}

// ============================================================
// Impresion del label one-hot
// ============================================================
void print_label_onehot(label_oh_t label_onehot) {
    std::cout << "label_onehot = [";
    // Se abre la representacion del vector one-hot.

print_onehot_loop:
    for (int i = 0; i < NUM_CLASSES; i++) {
        std::cout << (unsigned int)label_onehot[i];
        // Se imprime el bit correspondiente a la clase actual.

        if (i != NUM_CLASSES - 1) {
            std::cout << ", ";
            // Se agrega una coma entre elementos para facilitar la lectura.
        }
    }

    std::cout << "]" << std::endl;
    // Se cierra la representacion del one-hot.
}

// ============================================================
// Impresion del padding
// ============================================================
void print_padding_bits(padding_t padding) {
    std::cout << "padding = [";
    // Se abre la representacion textual del padding.

print_padding_loop:
    for (int i = PADDING_BITS - 1; i >= 0; i--) {
        std::cout << (unsigned int)padding[i];
        // Se imprime cada bit del padding desde el mas significativo.
    }

    std::cout << "]" << std::endl;
    // Se cierra la representacion del padding.
}

// ============================================================
// Resumen de pixeles
// ============================================================
void print_pixels_summary(pixels_t pixels, int preview_count) {
    int ones_count = 0;
    // Se reserva un contador para medir cuantos pixeles activos tiene la imagen.

count_pixels_loop:
    for (int i = 0; i < PIXEL_BITS; i++) {
        if (pixels[i] == 1) {
            ones_count++;
            // Se incrementa el conteo cuando se detecta un pixel activo.
        }
    }

    std::cout << "pixels ones_count = " << ones_count << std::endl;
    // Se imprime la cantidad de bits activos de la imagen binaria.

    std::cout << "pixels preview [0.." << (preview_count - 1) << "] = \n";
    // Se anuncia el rango de bits que se mostrara a continuacion.

preview_pixels_loop:
    for (int i = 0; i < preview_count; i++) {
        if (i % 28 == 0) {
            std::cout << "\n";
            // Se inserta un salto de linea cada 28 bits para recordar la geometria 28x28.
        }

        std::cout << (unsigned int)pixels[i];
        // Se imprime el bit actual de la vista previa.
    }

    std::cout << std::endl;
    // Se finaliza la salida de la vista previa.
}

// ============================================================
// Impresion completa de una muestra desempaquetada
// ============================================================
void print_unpacked_sample(raw_sample_t sample, const std::string &title) {
    label_oh_t label_onehot;
    // Se reserva el contenedor local para la etiqueta de la muestra.

    pixels_t pixels;
    // Se reserva el contenedor local para la imagen binaria.

    padding_t padding;
    // Se reserva el contenedor local para el padding fisico.

    unpack_sample(sample, label_onehot, pixels, padding);
    // Se desempaqueta la muestra usando exactamente la logica del kernel.

    print_separator(title);
    // Se abre una seccion visual dedicada a la muestra actual.

    print_label_onehot(label_onehot);
    // Se imprime el vector one-hot de la muestra.

    std::cout << "decoded_label = " << (unsigned int)decode_onehot(label_onehot) << std::endl;
    // Se imprime el indice de clase decodificado.

    std::cout << "onehot_valid = " << (is_valid_onehot(label_onehot) ? "true" : "false") << std::endl;
    // Se informa si el one-hot contiene exactamente un bit activo.

    print_padding_bits(padding);
    // Se imprime el padding fisico de la muestra.

    print_pixels_summary(pixels, 784);
    // Se imprime un resumen completo de la imagen binaria.
}

// ============================================================
// Comparadores auxiliares
// ============================================================
bool same_pixels(raw_sample_t a, raw_sample_t b) {
    pixels_t pixels_a;
    // Se reserva el contenedor de pixeles de la muestra A.

    pixels_t pixels_b;
    // Se reserva el contenedor de pixeles de la muestra B.

    label_oh_t label_dummy_a;
    // Se reserva un dummy para ignorar el label de A.

    label_oh_t label_dummy_b;
    // Se reserva un dummy para ignorar el label de B.

    padding_t padding_dummy_a;
    // Se reserva un dummy para ignorar el padding de A.

    padding_t padding_dummy_b;
    // Se reserva un dummy para ignorar el padding de B.

    unpack_sample(a, label_dummy_a, pixels_a, padding_dummy_a);
    // Se extraen los pixeles de la muestra A.

    unpack_sample(b, label_dummy_b, pixels_b, padding_dummy_b);
    // Se extraen los pixeles de la muestra B.

    return (pixels_a == pixels_b);
    // Se retorna true solo si ambas imagenes son iguales bit a bit.
}

bool same_padding(raw_sample_t a, raw_sample_t b) {
    pixels_t pixels_dummy_a;
    // Se reserva un dummy para ignorar los pixeles de A.

    pixels_t pixels_dummy_b;
    // Se reserva un dummy para ignorar los pixeles de B.

    label_oh_t label_dummy_a;
    // Se reserva un dummy para ignorar el label de A.

    label_oh_t label_dummy_b;
    // Se reserva un dummy para ignorar el label de B.

    padding_t padding_a;
    // Se reserva el contenedor del padding de A.

    padding_t padding_b;
    // Se reserva el contenedor del padding de B.

    unpack_sample(a, label_dummy_a, pixels_dummy_a, padding_a);
    // Se extrae el padding de la muestra A.

    unpack_sample(b, label_dummy_b, pixels_dummy_b, padding_b);
    // Se extrae el padding de la muestra B.

    return (padding_a == padding_b);
    // Se retorna true solo si ambos paddings son identicos.
}

bool same_raw_sample(raw_sample_t a, raw_sample_t b) {
    return (a == b);
    // Se compara directamente la igualdad de los 800 bits fisicos.
}

// ============================================================
// Separador visual
// ============================================================
void print_separator(const std::string &title) {
    std::cout << std::endl;
    // Se deja una linea en blanco para separar secciones consecutivas.

    std::cout << "============================================================" << std::endl;
    // Se imprime la linea superior del separador.

    std::cout << title << std::endl;
    // Se imprime el titulo de la seccion actual.

    std::cout << "============================================================" << std::endl;
    // Se imprime la linea inferior del separador.
}

// ============================================================
// Inspeccion de un rango de muestras
// ============================================================
void print_samples_range(const std::vector<word_t> &buffer, int start_sample, int num_samples) {
    int total_samples = (int)(buffer.size() / WORDS_PER_SAMPLE);
    // Se calcula cuantas muestras completas existen dentro del buffer lineal.

    if (start_sample < 0) {
        std::cout << "ERROR: start_sample no puede ser negativo." << std::endl;
        // Se reporta un rango invalido cuando el inicio es negativo.

        return;
        // Se abandona la funcion porque no existe un rango valido que imprimir.
    }

    if (start_sample >= total_samples) {
        std::cout << "ERROR: start_sample fuera de rango." << std::endl;
        // Se reporta un rango invalido cuando el inicio cae fuera del dataset.

        return;
        // Se abandona la funcion porque no existe un rango valido que imprimir.
    }

    if (num_samples <= 0) {
        std::cout << "ERROR: num_samples debe ser mayor que 0." << std::endl;
        // Se reporta un rango invalido cuando la cantidad solicitada no es positiva.

        return;
        // Se abandona la funcion porque no tiene sentido imprimir un rango vacio.
    }

    int end_sample = start_sample + num_samples;
    // Se calcula el indice final antes de aplicar recorte.

    if (end_sample > total_samples) {
        end_sample = total_samples;
        // Se recorta el final si el rango sobrepasa el dataset real.
    }

    print_separator("INSPECCION DE RANGO DE MUESTRAS");
    // Se abre una seccion dedicada a la inspeccion de rango.

    std::cout << "start_sample = " << start_sample << std::endl;
    // Se imprime el inicio efectivo del rango.

    std::cout << "num_samples  = " << num_samples << std::endl;
    // Se imprime la cantidad solicitada de muestras.

    std::cout << "range        = [" << start_sample << " : " << end_sample << ")" << std::endl;
    // Se imprime el rango final realmente inspeccionado.

print_range_loop:
    for (int sample_idx = start_sample; sample_idx < end_sample; sample_idx++) {
        // Se recorre cada muestra del rango solicitado.

        print_separator("MUESTRA " + std::to_string(sample_idx));
        // Se abre una subseccion propia para la muestra actual.

        word_t sample_words[WORDS_PER_SAMPLE];
        // Se reserva el arreglo temporal de las 25 words crudas.

        extract_sample_words(buffer, sample_idx, sample_words);
        // Se extraen las 25 words de la muestra actual.

        print_sample_words(sample_words, "words crudas:");
        // Se imprimen las words crudas de la muestra actual.

        raw_sample_t sample = load_sample_from_words(buffer.data(), sample_idx);
        // Se reconstruye la muestra fisica completa de 800 bits.

        print_unpacked_sample(sample, "desempaquetado:");
        // Se imprime la muestra interpretada como label, pixeles y padding.
    }
}

// ============================================================
// Vista previa de metricas por muestra
// ============================================================
void print_training_preview(
    const std::vector<goodness_t> &g_pos,
    const std::vector<goodness_t> &g_neg,
    const std::vector<goodness_t> &gap,
    int preview_count
) {
    print_separator("VISTA PREVIA DE METRICAS DE ENTRENAMIENTO FF");
    // Se abre una seccion dedicada a goodness y gap por muestra.

    int total = (int)gap.size();
    // Se calcula la cantidad total de registros disponibles.

    int effective_preview = preview_count;
    // Se crea una copia local del numero de elementos a imprimir.

    if (effective_preview > total) {
        effective_preview = total;
        // Se recorta la vista previa si supera el tamano real del buffer.
    }

preview_metric_loop:
    for (int i = 0; i < effective_preview; i++) {
        std::cout << "sample[" << i << "]"
                  << " g_pos=" << g_pos[i].to_double()
                  << " g_neg=" << g_neg[i].to_double()
                  << " gap=" << gap[i].to_double()
                  << std::endl;
        // Se imprime la goodness positiva, negativa y su diferencia para cada ejemplo visible.
    }
}

// ============================================================
// Historial por epoca
// ============================================================
void print_epoch_history(
    const std::vector<loss_t> &epoch_loss_pos,
    const std::vector<loss_t> &epoch_loss_neg,
    const std::vector<goodness_t> &epoch_g_pos,
    const std::vector<goodness_t> &epoch_g_neg,
    const std::vector<goodness_t> &epoch_gap,
    int epochs_to_print
) {
    print_separator("HISTORIAL POR EPOCA");
    // Se abre una seccion dedicada a las metricas medias por epoca.

    int total_epochs = epochs_to_print;
    // Se copia localmente la cantidad de epocas que se desea imprimir.

    if (total_epochs < 0) {
        total_epochs = 0;
        // Se corrige un valor negativo para evitar recorridos invalidos.
    }

    if (total_epochs > (int)epoch_gap.size()) {
        total_epochs = (int)epoch_gap.size();
        // Se recorta el numero de epocas al tamano real del historial.
    }

epoch_history_loop:
    for (int epoch = 0; epoch < total_epochs; epoch++) {
        std::cout << "epoch[" << (epoch + 1) << "]"
                  << " loss_pos=" << epoch_loss_pos[epoch].to_double()
                  << " loss_neg=" << epoch_loss_neg[epoch].to_double()
                  << " g_pos=" << epoch_g_pos[epoch].to_double()
                  << " g_neg=" << epoch_g_neg[epoch].to_double()
                  << " gap=" << epoch_gap[epoch].to_double()
                  << std::endl;
        // Se imprime una linea compacta con todas las metricas medias de la epoca actual.
    }
}

// ============================================================
// Actualizacion incremental por epoca
// ============================================================
void print_epoch_terminal_update(
    int epoch_idx,
    int total_epochs,
    loss_t epoch_loss_pos,
    loss_t epoch_loss_neg,
    goodness_t epoch_g_pos,
    goodness_t epoch_g_neg,
    goodness_t epoch_gap,
    double val_accuracy,
    double elapsed_sec,
    bool has_validation
) {
    print_separator("ACTUALIZACION DE EPOCA");
    // Se abre una seccion compacta para la epoca que acaba de terminar.

    std::cout << "Epoca " << epoch_idx << "/" << total_epochs << std::endl;
    // Se imprime el indice de epoca con el mismo estilo general del notebook.

    if (has_validation == true) {
        std::cout << "Val accuracy: " << val_accuracy << std::endl;
        // Se imprime la accuracy de validacion u hold-out cuando existe evaluacion.
    } else {
        std::cout << "Val accuracy: n/a" << std::endl;
        // Se imprime n/a cuando no existe un subset de validacion habilitado.
    }

    std::cout << "Loss pos: " << epoch_loss_pos.to_double() << std::endl;
    // Se imprime la perdida positiva media de la epoca actual.

    std::cout << "Loss neg: " << epoch_loss_neg.to_double() << std::endl;
    // Se imprime la perdida negativa media de la epoca actual.

    std::cout << "Goodness pos: " << epoch_g_pos.to_double() << std::endl;
    // Se imprime la goodness positiva media total de la epoca actual.

    std::cout << "Goodness neg: " << epoch_g_neg.to_double() << std::endl;
    // Se imprime la goodness negativa media total de la epoca actual.

    std::cout << "Goodness gap: " << epoch_gap.to_double() << std::endl;
    // Se imprime la separacion FF media entre positivos y negativos.

    std::cout << "Tiempo: " << elapsed_sec << " s" << std::endl;
    // Se imprime el tiempo transcurrido de la epoca para seguir el progreso en terminal.
}

// ============================================================
// Vista previa de predicciones
// ============================================================
void print_prediction_preview(
    const std::vector<label_idx_t> &true_labels,
    const std::vector<label_idx_t> &pred_labels,
    int preview_count,
    int start_sample
) {
    print_separator("VISTA PREVIA DE PREDICCIONES");
    // Se abre una seccion dedicada a comparar clases reales y predichas.

    int total = (int)true_labels.size();
    // Se calcula la cantidad total de elementos disponibles.

    int effective_preview = preview_count;
    // Se crea una copia local de la cantidad de elementos a imprimir.

    if (effective_preview > total) {
        effective_preview = total;
        // Se recorta la vista previa al tamano real de los vectores.
    }

preview_prediction_loop:
    for (int i = 0; i < effective_preview; i++) {
        int sample_idx = start_sample + i;
        // Se calcula el indice absoluto del ejemplo mostrado.

        std::cout << "sample[" << sample_idx << "]"
                  << " true=" << (unsigned int)true_labels[i]
                  << " pred=" << (unsigned int)pred_labels[i]
                  << " hit=" << ((true_labels[i] == pred_labels[i]) ? "true" : "false")
                  << std::endl;
        // Se imprime una comparacion compacta entre clase real y clase predicha.
    }
}

// ============================================================
// Resumen del modelo entrenado
// ============================================================
void print_model_overview(const std::vector<latent_t> &weights, const std::vector<bias_t> &biases, int preview_weights) {
    print_separator("RESUMEN DEL MODELO ENTRENADO");
    // Se abre una seccion para inspeccionar el estado final del modelo.

    double min_weight = std::numeric_limits<double>::max();
    // Se inicializa el minimo observado de pesos en un valor muy grande.

    double max_weight = -std::numeric_limits<double>::max();
    // Se inicializa el maximo observado de pesos en un valor muy pequeno.

    double min_bias = std::numeric_limits<double>::max();
    // Se inicializa el minimo observado de bias en un valor muy grande.

    double max_bias = -std::numeric_limits<double>::max();
    // Se inicializa el maximo observado de bias en un valor muy pequeno.

    int positive_count = 0;
    // Se inicializa el conteo de pesos positivos.

    int negative_count = 0;
    // Se inicializa el conteo de pesos negativos.

    int zero_count = 0;
    // Se inicializa el conteo de pesos exactamente en cero.

    int clipped_positive = 0;
    // Se inicializa el conteo de pesos saturados en el clip positivo.

    int clipped_negative = 0;
    // Se inicializa el conteo de pesos saturados en el clip negativo.

model_weight_summary_loop:
    for (std::size_t i = 0; i < weights.size(); i++) {
        double value = weights[i].to_double();
        // Se convierte el peso actual a double solo para fines de impresion.

        if (value < min_weight) {
            min_weight = value;
            // Se actualiza el minimo de pesos.
        }

        if (value > max_weight) {
            max_weight = value;
            // Se actualiza el maximo de pesos.
        }

        if (value > 0.0) {
            positive_count++;
            // Se cuenta el peso actual como positivo.
        } else if (value < 0.0) {
            negative_count++;
            // Se cuenta el peso actual como negativo.
        } else {
            zero_count++;
            // Se cuenta el peso actual como exactamente cero.
        }

        if (value >= LATENT_WEIGHT_CLIP.to_double()) {
            clipped_positive++;
            // Se cuenta un peso saturado en el clip positivo.
        }

        if (value <= -LATENT_WEIGHT_CLIP.to_double()) {
            clipped_negative++;
            // Se cuenta un peso saturado en el clip negativo.
        }
    }

model_bias_summary_loop:
    for (std::size_t i = 0; i < biases.size(); i++) {
        double value = biases[i].to_double();
        // Se convierte el bias actual a double solo para fines de impresion.

        if (value < min_bias) {
            min_bias = value;
            // Se actualiza el minimo de bias.
        }

        if (value > max_bias) {
            max_bias = value;
            // Se actualiza el maximo de bias.
        }
    }

    std::cout << "total_weights   = " << weights.size() << std::endl;
    // Se imprime la cantidad total de pesos del modelo.

    std::cout << "layer1_weights  = " << MODEL_LAYER1_WEIGHT_COUNT << std::endl;
    // Se imprime la cantidad de pesos correspondiente a la primera capa.

    std::cout << "layer2_weights  = " << MODEL_LAYER2_WEIGHT_COUNT << std::endl;
    // Se imprime la cantidad de pesos correspondiente a la segunda capa.

    std::cout << "min_weight      = " << min_weight << std::endl;
    // Se imprime el menor peso observado.

    std::cout << "max_weight      = " << max_weight << std::endl;
    // Se imprime el mayor peso observado.

    std::cout << "positive_w      = " << positive_count << std::endl;
    // Se imprime la cantidad de pesos positivos.

    std::cout << "negative_w      = " << negative_count << std::endl;
    // Se imprime la cantidad de pesos negativos.

    std::cout << "zero_w          = " << zero_count << std::endl;
    // Se imprime la cantidad de pesos exactamente nulos.

    std::cout << "clip_pos_w      = " << clipped_positive << std::endl;
    // Se imprime la cantidad de pesos saturados en el clip positivo.

    std::cout << "clip_neg_w      = " << clipped_negative << std::endl;
    // Se imprime la cantidad de pesos saturados en el clip negativo.

    std::cout << "bias_count      = " << biases.size() << std::endl;
    // Se imprime la cantidad total de bias del modelo.

    std::cout << "layer1_biases   = " << MODEL_LAYER1_NEURONS << std::endl;
    // Se imprime la cantidad de bias de la primera capa.

    std::cout << "layer2_biases   = " << MODEL_LAYER2_NEURONS << std::endl;
    // Se imprime la cantidad de bias de la segunda capa.

    std::cout << "min_bias        = " << min_bias << std::endl;
    // Se imprime el menor bias observado.

    std::cout << "max_bias        = " << max_bias << std::endl;
    // Se imprime el mayor bias observado.

    std::cout << "first_l1_weights = ";
    // Se abre una lista compacta de los primeros pesos de la primera capa.

preview_l1_weight_loop:
    for (int i = 0; i < preview_weights; i++) {
        if (i < MODEL_LAYER1_WEIGHT_COUNT) {
            std::cout << weights[i].to_double() << " ";
            // Se imprime el peso i si cae dentro del rango de la primera capa.
        }
    }

    std::cout << std::endl;
    // Se cierra la linea de pesos de vista previa de la primera capa.

    std::cout << "first_l2_weights = ";
    // Se abre una lista compacta de los primeros pesos de la segunda capa.

preview_l2_weight_loop:
    for (int i = 0; i < preview_weights; i++) {
        int flat_index = MODEL_LAYER1_WEIGHT_COUNT + i;
        // Se calcula el offset del peso i dentro de la segunda capa linealizada.

        if (flat_index < (int)weights.size()) {
            std::cout << weights[flat_index].to_double() << " ";
            // Se imprime el peso i si cae dentro del rango de la segunda capa.
        }
    }

    std::cout << std::endl;
    // Se cierra la linea de pesos de vista previa de la segunda capa.

    std::cout << "first_l1_biases  = ";
    // Se abre una lista compacta de los primeros bias de la primera capa.

preview_l1_bias_loop:
    for (int i = 0; i < MODEL_LAYER1_NEURONS; i++) {
        if (i < 16) {
            std::cout << biases[i].to_double() << " ";
            // Se imprime un subconjunto inicial de bias de la primera capa.
        }
    }

    std::cout << std::endl;
    // Se cierra la linea de bias de vista previa de la primera capa.

    std::cout << "first_l2_biases  = ";
    // Se abre una lista compacta de los primeros bias de la segunda capa.

preview_l2_bias_loop:
    for (int i = 0; i < MODEL_LAYER2_NEURONS; i++) {
        if (i < 16) {
            std::cout << biases[MODEL_LAYER1_NEURONS + i].to_double() << " ";
            // Se imprime un subconjunto inicial de bias de la segunda capa.
        }
    }

    std::cout << std::endl;
    // Se cierra la linea de bias de vista previa de la segunda capa.
}

// ============================================================
// Delta real del modelo entre epocas consecutivas
// ============================================================
void print_epoch_model_delta(
    int epoch_idx,
    const std::vector<latent_t> &previous_weights,
    const std::vector<latent_t> &current_weights,
    const std::vector<bias_t> &previous_biases,
    const std::vector<bias_t> &current_biases
) {
    std::size_t weight_count =
        (previous_weights.size() < current_weights.size()) ? previous_weights.size() : current_weights.size();
    // Se usa el minimo tamano comun para comparar snapshots de pesos sin asumir buffers identicos.

    std::size_t bias_count =
        (previous_biases.size() < current_biases.size()) ? previous_biases.size() : current_biases.size();
    // Se usa el minimo tamano comun para comparar snapshots de bias sin asumir buffers identicos.

    int changed_weights = 0;
    // Se inicializa el conteo de pesos cuyo valor fijo cambio entre snapshots.

    int sign_flip_weights = 0;
    // Se inicializa el conteo de pesos que cambiaron de signo efectivo entre snapshots.

    double max_abs_weight_delta = 0.0;
    // Se inicializa el mayor cambio absoluto observado en los pesos.

epoch_weight_delta_loop:
    for (std::size_t i = 0; i < weight_count; i++) {
        latent_t previous_value = previous_weights[i];
        // Se recupera el peso previo en formato fijo.

        latent_t current_value = current_weights[i];
        // Se recupera el peso actual en formato fijo.

        if (previous_value != current_value) {
            changed_weights++;
            // Se contabiliza un peso cuyo valor representable ya cambio.
        }

        bool previous_positive = (previous_value > (latent_t)0);
        // Se identifica si el peso previo aportaba como signo positivo.

        bool previous_negative = (previous_value < (latent_t)0);
        // Se identifica si el peso previo aportaba como signo negativo.

        bool current_positive = (current_value > (latent_t)0);
        // Se identifica si el peso actual aporta como signo positivo.

        bool current_negative = (current_value < (latent_t)0);
        // Se identifica si el peso actual aporta como signo negativo.

        if (((previous_positive == true) && (current_negative == true)) ||
            ((previous_negative == true) && (current_positive == true))) {
            sign_flip_weights++;
            // Se contabiliza un cruce de cero que altera directamente el forward binarizado.
        }

        double delta_value = (current_value - previous_value).to_double();
        // Se convierte el delta a double solo para resumir su magnitud absoluta.

        if (delta_value < 0.0) {
            delta_value = -delta_value;
            // Se toma el valor absoluto del delta del peso.
        }

        if (delta_value > max_abs_weight_delta) {
            max_abs_weight_delta = delta_value;
            // Se conserva el mayor cambio absoluto observado en los pesos.
        }
    }

    int changed_biases = 0;
    // Se inicializa el conteo de bias cuyo valor fijo cambio entre snapshots.

    int sign_flip_biases = 0;
    // Se inicializa el conteo de bias que cambiaron de signo entre snapshots.

    double max_abs_bias_delta = 0.0;
    // Se inicializa el mayor cambio absoluto observado en los bias.

epoch_bias_delta_loop:
    for (std::size_t i = 0; i < bias_count; i++) {
        bias_t previous_value = previous_biases[i];
        // Se recupera el bias previo en formato fijo.

        bias_t current_value = current_biases[i];
        // Se recupera el bias actual en formato fijo.

        if (previous_value != current_value) {
            changed_biases++;
            // Se contabiliza un bias cuyo valor representable ya cambio.
        }

        bool previous_positive = (previous_value > (bias_t)0);
        // Se identifica si el bias previo era positivo.

        bool previous_negative = (previous_value < (bias_t)0);
        // Se identifica si el bias previo era negativo.

        bool current_positive = (current_value > (bias_t)0);
        // Se identifica si el bias actual es positivo.

        bool current_negative = (current_value < (bias_t)0);
        // Se identifica si el bias actual es negativo.

        if (((previous_positive == true) && (current_negative == true)) ||
            ((previous_negative == true) && (current_positive == true))) {
            sign_flip_biases++;
            // Se contabiliza un cruce de cero en el bias entre snapshots.
        }

        double delta_value = (current_value - previous_value).to_double();
        // Se convierte el delta del bias a double solo para resumir su magnitud absoluta.

        if (delta_value < 0.0) {
            delta_value = -delta_value;
            // Se toma el valor absoluto del delta del bias.
        }

        if (delta_value > max_abs_bias_delta) {
            max_abs_bias_delta = delta_value;
            // Se conserva el mayor cambio absoluto observado en los bias.
        }
    }

    std::cout << "epoch_delta[" << epoch_idx << "]"
              << " changed_w=" << changed_weights
              << " sign_flip_w=" << sign_flip_weights
              << " max_abs_dw=" << max_abs_weight_delta
              << " changed_b=" << changed_biases
              << " sign_flip_b=" << sign_flip_biases
              << " max_abs_db=" << max_abs_bias_delta
              << std::endl;
    // Se imprime un resumen compacto para saber si la epoca realmente altero el estado del modelo.
}

// ============================================================
// Justificacion de congelamiento entre epocas
// ============================================================
void print_epoch_freeze_justification(
    const std::vector<loss_t> &epoch_loss_pos,
    const std::vector<loss_t> &epoch_loss_neg,
    const std::vector<goodness_t> &epoch_g_pos,
    const std::vector<goodness_t> &epoch_g_neg,
    const std::vector<goodness_t> &epoch_gap,
    const std::vector<latent_t> &weights,
    const std::vector<bias_t> &biases,
    int epochs_to_check
) {
    if (epochs_to_check <= 1) {
        return;
        // No tiene sentido diagnosticar congelamiento si solo existe una epoca visible.
    }

    int effective_epochs = epochs_to_check;
    // Se crea una copia local de la cantidad de epocas que se desea revisar.

    if (effective_epochs > (int)epoch_gap.size()) {
        effective_epochs = (int)epoch_gap.size();
        // Se recorta la revision al tamano real de los historiales.
    }

    bool history_is_frozen = true;
    // Se asume congelamiento hasta encontrar alguna diferencia real entre epocas.

freeze_check_loop:
    for (int epoch = 1; epoch < effective_epochs; epoch++) {
        if ((epoch_loss_pos[epoch] != epoch_loss_pos[0]) ||
            (epoch_loss_neg[epoch] != epoch_loss_neg[0]) ||
            (epoch_g_pos[epoch] != epoch_g_pos[0]) ||
            (epoch_g_neg[epoch] != epoch_g_neg[0]) ||
            (epoch_gap[epoch] != epoch_gap[0])) {
            history_is_frozen = false;
            // Se desactiva el diagnostico de congelamiento si alguna metrica cambia entre epocas.
        }
    }

    if (history_is_frozen == false) {
        return;
        // Solo se imprime la justificacion cuando el historial realmente esta congelado.
    }

    bool all_biases_zero = true;
    // Se asume que todos los bias siguen en cero hasta encontrar lo contrario.

bias_zero_check_loop:
    for (std::size_t i = 0; i < biases.size(); i++) {
        if (biases[i] != (bias_t)0) {
            all_biases_zero = false;
            // Se detecta que al menos un bias ya cambio respecto al estado inicial.
        }
    }

    bool weights_still_in_seed_levels = true;
    // Se asume que los pesos visibles siguen anclados a las magnitudes de inicializacion.

weight_seed_check_loop:
    for (std::size_t i = 0; i < weights.size(); i++) {
        latent_t value = weights[i];
        // Se toma el peso actual en formato fijo para compararlo sin conversiones innecesarias.

        bool is_seed_level =
            (value == (latent_t)(LATENT_INIT_MAG)) ||
            (value == (latent_t)(-LATENT_INIT_MAG)) ||
            (value == (latent_t)(LATENT_INIT_MAG * (latent_t)2)) ||
            (value == (latent_t)(-LATENT_INIT_MAG * (latent_t)2));
        // Se comprueba si el peso sigue exactamente en alguno de los cuatro niveles de inicializacion.

        if (is_seed_level == false) {
            weights_still_in_seed_levels = false;
            // Se detecta que al menos un peso salio de la cuadricula de inicializacion.
        }
    }

    print_separator("JUSTIFICACION DEL CONGELAMIENTO ENTRE EPOCAS");
    // Se abre una seccion dedicada a explicar por que las epocas quedaron identicas.

    std::cout << "diagnostico = las metricas de entrenamiento y validacion no cambiaron entre epocas consecutivas." << std::endl;
    // Se deja explicitamente registrado que el historial esta congelado.

    std::cout << "causa_1 = el modelo no se resetea en cada epoca; si el resultado se repite, el problema no es el control del testbench sino la falta de cambio efectivo en el estado entrenable." << std::endl;
    // Se aclara que reset_model solo ocurre en la primera llamada y no explica el congelamiento observado.

    std::cout << "causa_2 = el forward usa solo el signo del peso latente para decidir si suma o resta cada entrada; mientras un peso no cruce cero, el comportamiento visible de la red no cambia." << std::endl;
    // Se explica que pequenas variaciones latentes no alteran la mascara binaria efectiva del forward.

    std::cout << "causa_3 = la goodness observada y el threshold del experimento quedaron en escalas incompatibles; cuando el margen entra saturado a sigmoid y softplus, cada batch recibe casi la misma senal de correccion." << std::endl;
    // Se explica que el congelamiento aparece cuando threshold y goodness quedan muy desalineados en escala.

    if (all_biases_zero == true) {
        std::cout << "evidencia_bias = todos los bias visibles terminaron en cero, asi que la ruta de actualizacion del bias no esta alterando el estado representable del modelo." << std::endl;
        // Se destaca la evidencia directa de que los bias no estan cambiando entre epocas.
    } else {
        std::cout << "evidencia_bias = hay bias no nulos, pero aun asi el forward binarizado sigue viendo la misma separacion global." << std::endl;
        // Se cubre el caso alternativo en que algun bias si se haya movido.
    }

    if (weights_still_in_seed_levels == true) {
        std::cout << "evidencia_pesos = los pesos visibles siguen anclados a los niveles discretos de inicializacion (+/-"
                  << LATENT_INIT_MAG.to_double()
                  << " y +/-"
                  << (LATENT_INIT_MAG * (latent_t)2).to_double()
                  << "), lo que indica que la actualizacion no esta sacando al modelo de su cuadricula inicial."
                  << std::endl;
        // Se deja constancia de que el snapshot final sigue anclado a los valores de semilla vigentes.
    } else {
        std::cout << "evidencia_pesos = algunos pesos ya salieron de la cuadricula inicial, pero su signo efectivo sigue sin cambiar lo suficiente para modificar el forward." << std::endl;
        // Se cubre el caso alternativo en que existan cambios latentes sin impacto binario visible.
    }

    std::cout << "conclusion = por eso cambiar dimensiones, threshold o label_scale por si solos no mejora la accuracy: primero hay que conseguir que el update modifique el estado binario efectivo que usa el forward." << std::endl;
    // Se resume la razon principal por la que el accuracy no mejora con las pruebas actuales.
}
