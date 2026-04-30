#ifndef FORWARD_FW_HPP
#define FORWARD_FW_HPP

// El modo 28x28 con 6 bits por pixel requiere vectores anchos de mas de
// 4700 bits. Se amplia el maximo antes de incluir ap_int.
#ifndef AP_INT_MAX_W
#define AP_INT_MAX_W 8192
#endif

#include <ap_fixed.h>
#include <ap_int.h>
#include <stdint.h>

// ============================================================
// Configuracion global del dataset y del experimento
// ============================================================
// Estas macros seleccionan la build activa del dataset.
// La resolucion y los bits por pixel deben coincidir con la cabecera
// del binario que se quiera probar desde train_tb.cpp.
#ifndef FF_IMAGE_WIDTH_CFG
#define FF_IMAGE_WIDTH_CFG 28
#endif

#ifndef FF_IMAGE_HEIGHT_CFG
#define FF_IMAGE_HEIGHT_CFG 28
#endif

#ifndef FF_INPUT_BITS_PER_PIXEL_CFG
#define FF_INPUT_BITS_PER_PIXEL_CFG 1
#endif

#if (FF_IMAGE_WIDTH_CFG <= 0) || (FF_IMAGE_HEIGHT_CFG <= 0)
#error "FF_IMAGE_WIDTH_CFG y FF_IMAGE_HEIGHT_CFG deben ser positivos."
#endif

#if (FF_INPUT_BITS_PER_PIXEL_CFG < 1) || (FF_INPUT_BITS_PER_PIXEL_CFG > 8)
#error "FF_INPUT_BITS_PER_PIXEL_CFG debe estar entre 1 y 8."
#endif

// ============================================================
// Constantes del dataset hardware
// ============================================================
static const int IMAGE_WIDTH = FF_IMAGE_WIDTH_CFG;
static const int IMAGE_HEIGHT = FF_IMAGE_HEIGHT_CFG;
static const int NUM_CLASSES = 10;
static const int WORD_BITS = 32;
static const int LABEL_BITS = NUM_CLASSES;
static const int NUM_LOGICAL_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
static const int INPUT_BITS_PER_PIXEL = FF_INPUT_BITS_PER_PIXEL_CFG;
static const int PIXEL_MAX_STORED_VALUE =
    (1 << INPUT_BITS_PER_PIXEL) - 1;

static const int PIXEL_STORAGE_BITS =
    NUM_LOGICAL_PIXELS * INPUT_BITS_PER_PIXEL;
static const int INPUT_STORAGE_BITS = LABEL_BITS + PIXEL_STORAGE_BITS;
static const int WORDS_PER_SAMPLE =
    (INPUT_STORAGE_BITS + WORD_BITS - 1) / WORD_BITS;
static const int TOTAL_BITS = WORDS_PER_SAMPLE * WORD_BITS;
static const int PADDING_BITS = TOTAL_BITS - INPUT_STORAGE_BITS;

// Se mantienen estos alias para conservar compatibilidad con el codigo
// existente del kernel y del testbench.
static const int PIXEL_BITS = PIXEL_STORAGE_BITS;
static const int MODEL_INPUT_BITS = LABEL_BITS + NUM_LOGICAL_PIXELS;

// ============================================================
// Constantes del modelo entrenable de Etapa C
// ============================================================
static const int MODEL_NUM_LAYERS = 2;
static const int MODEL_PARALLEL_NEURONS = 8;

#ifndef FF_MODEL_LAYER1_NEURONS_CFG
#define FF_MODEL_LAYER1_NEURONS_CFG 64
#endif

#ifndef FF_MODEL_LAYER2_NEURONS_CFG
#define FF_MODEL_LAYER2_NEURONS_CFG 32
#endif

#ifndef FF_MODEL_BATCH_SIZE_CFG
#define FF_MODEL_BATCH_SIZE_CFG 64
#endif

#ifndef FF_LATENT_INIT_MAG_CFG
#define FF_LATENT_INIT_MAG_CFG 0.03125
#endif

#ifndef FF_LEARNING_RATE_CFG
#define FF_LEARNING_RATE_CFG 0.02
#endif

#ifndef FF_GOODNESS_THRESHOLD_CFG
#define FF_GOODNESS_THRESHOLD_CFG 512.0
#endif

#ifndef FF_LABEL_SCALE_CFG
#define FF_LABEL_SCALE_CFG 3.0
#endif

#ifndef FF_PIXEL_SCALE_CFG
#define FF_PIXEL_SCALE_CFG 1.0
#endif

#ifndef FF_NONLINEAR_CLIP_CFG
#define FF_NONLINEAR_CLIP_CFG 32.0
#endif

static const int MODEL_LAYER1_INPUT_BITS = MODEL_INPUT_BITS;
static const int MODEL_LAYER1_NEURONS = FF_MODEL_LAYER1_NEURONS_CFG;
static const int MODEL_LAYER1_TILES =
    MODEL_LAYER1_NEURONS / MODEL_PARALLEL_NEURONS;
static const int MODEL_LAYER1_WEIGHT_COUNT =
    MODEL_LAYER1_NEURONS * MODEL_LAYER1_INPUT_BITS;

static const int MODEL_LAYER2_INPUT_BITS = MODEL_LAYER1_NEURONS;
static const int MODEL_LAYER2_NEURONS = FF_MODEL_LAYER2_NEURONS_CFG;
static const int MODEL_LAYER2_TILES =
    MODEL_LAYER2_NEURONS / MODEL_PARALLEL_NEURONS;
static const int MODEL_LAYER2_WEIGHT_COUNT =
    MODEL_LAYER2_NEURONS * MODEL_LAYER2_INPUT_BITS;

static const int MODEL_TOTAL_NEURONS =
    MODEL_LAYER1_NEURONS + MODEL_LAYER2_NEURONS;
static const int MODEL_WEIGHT_COUNT =
    MODEL_LAYER1_WEIGHT_COUNT + MODEL_LAYER2_WEIGHT_COUNT;
static const int MODEL_BIAS_COUNT = MODEL_TOTAL_NEURONS;
static const int MODEL_MAX_LAYER_NEURONS = MODEL_LAYER1_NEURONS;
static const int MODEL_BATCH_SIZE = FF_MODEL_BATCH_SIZE_CFG;
static const int TRAIN_MAX_EPOCHS = 50;

// ============================================================
// Tipos base del proyecto
// ============================================================
typedef ap_uint<WORD_BITS> word_t;
typedef ap_uint<TOTAL_BITS> raw_sample_t;
typedef ap_uint<LABEL_BITS> label_oh_t;
typedef ap_uint<PIXEL_STORAGE_BITS> pixels_t;
typedef ap_uint<PADDING_BITS> padding_t;
typedef ap_uint<INPUT_STORAGE_BITS> ff_input_t;
typedef ap_uint<4> label_idx_t;
typedef ap_uint<16> lfsr_t;

typedef ap_fixed<16, 8, AP_RND, AP_SAT> feature_t;
typedef ap_fixed<20, 4, AP_RND, AP_SAT> latent_t;
typedef ap_fixed<24, 8, AP_RND, AP_SAT> bias_t;
typedef ap_fixed<24, 14, AP_RND, AP_SAT> preact_t;
typedef ap_fixed<24, 14, AP_RND, AP_SAT> activation_t;
typedef ap_fixed<36, 24, AP_RND, AP_SAT> goodness_t;
typedef ap_fixed<24, 12, AP_RND, AP_SAT> loss_t;
typedef ap_fixed<24, 12, AP_RND, AP_SAT> scale_t;
typedef ap_fixed<40, 28, AP_RND, AP_SAT> stat_accum_t;
typedef ap_fixed<32, 20, AP_RND, AP_SAT> update_accum_t;
typedef ap_fixed<32, 8, AP_RND, AP_SAT> learning_rate_t;

// ============================================================
// Hiperparametros hardware iniciales
// ============================================================
static const latent_t LATENT_WEIGHT_CLIP = (latent_t)2.5;
static const bias_t LATENT_BIAS_CLIP = (bias_t)8.0;
static const latent_t LATENT_INIT_MAG =
    (latent_t)FF_LATENT_INIT_MAG_CFG;
static const learning_rate_t LEARNING_RATE_HW =
    (learning_rate_t)FF_LEARNING_RATE_CFG;
static const goodness_t GOODNESS_THRESHOLD_HW =
    (goodness_t)FF_GOODNESS_THRESHOLD_CFG;
static const feature_t LABEL_SCALE_HW =
    (feature_t)FF_LABEL_SCALE_CFG;
static const feature_t PIXEL_SCALE_HW =
    (feature_t)FF_PIXEL_SCALE_CFG;
static const goodness_t NONLINEAR_CLIP_HW =
    (goodness_t)FF_NONLINEAR_CLIP_CFG;
static const feature_t PIXEL_NORMALIZATION_HW =
    (feature_t)(1.0 / (double)PIXEL_MAX_STORED_VALUE);

/**
 * @brief Extrae el valor entero de un pixel almacenado dentro de pixels_t.
 *
 * @param pixels Contiene el bloque de pixeles empaquetados de una muestra.
 * @param pixel_idx Indica el indice logico del pixel que se desea leer.
 *
 * @return Retorna el valor entero almacenado para el pixel solicitado.
 *
 * @note Esta funcion se usa tanto en el kernel como en las utilidades de
 * depuracion. El orden interno sigue el packing little-endian por pixel.
 */
static inline ap_uint<8> get_packed_pixel_value(
    pixels_t pixels,
    int pixel_idx
) {
#pragma HLS inline
    int lsb = pixel_idx * INPUT_BITS_PER_PIXEL;
    return (ap_uint<8>)pixels.range(
        lsb + INPUT_BITS_PER_PIXEL - 1,
        lsb
    );
}

/**
 * @brief Extrae el valor entero de un pixel desde la entrada FF empaquetada.
 *
 * @param ff_input Contiene la entrada logica de una muestra sin el padding.
 * @param pixel_idx Indica el indice logico del pixel que se desea leer.
 *
 * @return Retorna el valor entero almacenado para el pixel solicitado.
 *
 * @note Esta funcion permite que la primera capa vea un pixel cuantizado
 * como un unico valor numerico, en lugar de recorrer sus bitplanes por
 * separado.
 */
static inline ap_uint<8> get_packed_pixel_value_from_input(
    ff_input_t ff_input,
    int pixel_idx
) {
#pragma HLS inline
    int lsb = LABEL_BITS + (pixel_idx * INPUT_BITS_PER_PIXEL);
    return (ap_uint<8>)ff_input.range(
        lsb + INPUT_BITS_PER_PIXEL - 1,
        lsb
    );
}

/**
 * @brief Determina si el modo de entrada activo es binario.
 *
 * @return Retorna true cuando el modo activo usa un solo bit por pixel.
 */
static inline bool ff_is_binary_input_mode() {
    return (INPUT_BITS_PER_PIXEL == 1);
}

/**
 * @brief Avanza el LFSR de 16 bits usado por el flujo de entrenamiento.
 *
 * @param state Contiene el estado actual del generador pseudoaleatorio.
 *
 * @return Retorna el siguiente estado del LFSR.
 */
lfsr_t lfsr16_next(lfsr_t state);

/**
 * @brief Verifica si un vector one-hot contiene exactamente un bit activo.
 *
 * @param label_onehot Contiene la etiqueta codificada en formato one-hot.
 *
 * @return Retorna true cuando el vector es un one-hot valido.
 */
bool is_valid_onehot(label_oh_t label_onehot);

/**
 * @brief Convierte una etiqueta one-hot al indice de clase correspondiente.
 *
 * @param label_onehot Contiene la etiqueta codificada en formato one-hot.
 *
 * @return Retorna el indice de clase asociado a la entrada.
 */
label_idx_t decode_onehot(label_oh_t label_onehot);

/**
 * @brief Convierte un indice de clase al formato one-hot.
 *
 * @param label_idx Indica la clase que se desea codificar.
 *
 * @return Retorna la etiqueta codificada en formato one-hot.
 */
label_oh_t encode_onehot(label_idx_t label_idx);

/**
 * @brief Genera una etiqueta negativa distinta de la clase verdadera.
 *
 * @param true_label Indica la clase correcta de la muestra actual.
 * @param state Contiene y actualiza el estado del LFSR.
 *
 * @return Retorna una clase incorrecta y distinta de la verdadera.
 */
label_idx_t generate_negative_label(label_idx_t true_label, lfsr_t &state);

/**
 * @brief Reconstruye una muestra fisica a partir del buffer lineal de words.
 *
 * @param mem Apunta al buffer lineal que contiene el payload del dataset.
 * @param sample_idx Indica la muestra que se desea reconstruir.
 *
 * @return Retorna la muestra empaquetada como un vector ancho.
 */
raw_sample_t load_sample_from_words(const word_t *mem, int sample_idx);

/**
 * @brief Escribe una muestra fisica dentro del buffer lineal de words.
 *
 * @param mem Apunta al buffer lineal que recibira la muestra.
 * @param sample_idx Indica la posicion de la muestra dentro del buffer.
 * @param sample Contiene la muestra empaquetada como vector ancho.
 */
void store_sample_to_words(word_t *mem, int sample_idx, raw_sample_t sample);

/**
 * @brief Separa una muestra fisica en etiqueta, pixeles y padding.
 *
 * @param sample Contiene la muestra fisica empaquetada.
 * @param label_onehot Retorna la etiqueta one-hot de la muestra.
 * @param pixels Retorna el bloque de pixeles empaquetados.
 * @param padding Retorna el padding fisico de la muestra.
 */
void unpack_sample(
    raw_sample_t sample,
    label_oh_t &label_onehot,
    pixels_t &pixels,
    padding_t &padding
);

/**
 * @brief Empaqueta etiqueta, pixeles y padding en una sola muestra fisica.
 *
 * @param label_onehot Contiene la etiqueta one-hot de la muestra.
 * @param pixels Contiene el bloque de pixeles empaquetados.
 * @param padding Contiene el padding fisico de la muestra.
 *
 * @return Retorna la muestra fisica completa.
 */
raw_sample_t pack_sample(
    label_oh_t label_onehot,
    pixels_t pixels,
    padding_t padding
);

/**
 * @brief Construye la entrada logica usada por el algoritmo Forward-Forward.
 *
 * @param label_onehot Contiene la etiqueta incrustada de la muestra.
 * @param pixels Contiene el bloque de pixeles empaquetados.
 *
 * @return Retorna la entrada logica sin padding.
 */
ff_input_t build_ff_input(label_oh_t label_onehot, pixels_t pixels);

/**
 * @brief Separa la entrada logica FF en etiqueta y pixeles empaquetados.
 *
 * @param ff_input Contiene la entrada logica sin padding.
 * @param label_onehot Retorna la etiqueta one-hot incrustada.
 * @param pixels Retorna el bloque de pixeles empaquetados.
 */
void unpack_ff_input(
    ff_input_t ff_input,
    label_oh_t &label_onehot,
    pixels_t &pixels
);

/**
 * @brief Prepara los pares positivos y negativos heredados de la Etapa B.
 *
 * @param in_mem Apunta al payload del dataset de entrada.
 * @param pos_mem Retorna el buffer de muestras positivas.
 * @param neg_mem Retorna el buffer de muestras negativas.
 * @param true_label_mem Retorna las etiquetas verdaderas por muestra.
 * @param neg_label_mem Retorna las etiquetas negativas por muestra.
 * @param n_samples Indica cuantas muestras se deben procesar.
 * @param seed Indica la semilla del LFSR usada por la etapa.
 */
void forward_fw_top(
    const word_t *in_mem,
    word_t *pos_mem,
    word_t *neg_mem,
    label_idx_t *true_label_mem,
    label_idx_t *neg_label_mem,
    int n_samples,
    uint16_t seed
);

/**
 * @brief Ejecuta entrenamiento e inferencia del modelo FF sobre el payload.
 *
 * @param in_mem Apunta al payload del dataset de entrada.
 * @param true_label_mem Retorna las etiquetas verdaderas de inferencia.
 * @param pred_label_mem Retorna las predicciones de inferencia.
 * @param weight_mem_out Retorna el snapshot de pesos del modelo.
 * @param bias_mem_out Retorna el snapshot de bias del modelo.
 * @param g_pos_mem Retorna la goodness positiva por muestra.
 * @param g_neg_mem Retorna la goodness negativa por muestra.
 * @param gap_mem Retorna la separacion positiva-negativa por muestra.
 * @param epoch_loss_pos_mem Retorna el historial de perdida positiva.
 * @param epoch_loss_neg_mem Retorna el historial de perdida negativa.
 * @param epoch_g_pos_mem Retorna el historial de goodness positiva.
 * @param epoch_g_neg_mem Retorna el historial de goodness negativa.
 * @param epoch_gap_mem Retorna el historial de gap por epoca.
 * @param correct_count_mem Retorna la cantidad de aciertos de inferencia.
 * @param n_samples Indica cuantas muestras se infieren.
 * @param n_train_samples Indica cuantas muestras se entrenan.
 * @param n_epochs Indica cuantas epocas se ejecutan.
 * @param seed Indica la semilla del LFSR usada por el kernel.
 * @param reset_model Indica si el modelo debe reinicializarse.
 */
void ff_train_top(
    const word_t *in_mem,
    label_idx_t *true_label_mem,
    label_idx_t *pred_label_mem,
    latent_t *weight_mem_out,
    bias_t *bias_mem_out,
    goodness_t *g_pos_mem,
    goodness_t *g_neg_mem,
    goodness_t *gap_mem,
    loss_t *epoch_loss_pos_mem,
    loss_t *epoch_loss_neg_mem,
    goodness_t *epoch_g_pos_mem,
    goodness_t *epoch_g_neg_mem,
    goodness_t *epoch_gap_mem,
    ap_uint<32> *correct_count_mem,
    int n_samples,
    int n_train_samples,
    int n_epochs,
    uint16_t seed,
    bool reset_model
);

#endif
