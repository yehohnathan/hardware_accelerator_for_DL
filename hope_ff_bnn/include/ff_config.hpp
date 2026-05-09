
/**
 * @file ff_config.hpp
 * @brief Parámetros estáticos de la BNN Forward-Forward para C++/HLS.
 *
 * Este archivo pertenece a la capa de configuración del kernel HLS. Las macros
 * definen resolución de MNIST, cuantización, arquitectura, paralelismo y rangos
 * numéricos antes de compilar. Mantener estos valores en tiempo de compilación
 * permite que Vitis/Vivado HLS reserve arreglos de tamaño fijo y evalúe
 * particiones, pipelines y uso de memoria sin depender de asignación dinámica.
 *
 * En el flujo Forward-Forward, estos parámetros determinan la forma del vector
 * [etiqueta one-hot | imagen], el número de capas entrenables, el umbral de
 * goodness y el tamaño máximo del modelo exportado a memoria externa.
 */
#ifndef HOPE_FF_CONFIG_HPP
#define HOPE_FF_CONFIG_HPP

#include <cstdint>

/*
 * Las macros FF_IMAGE_* y FF_PIXEL_BITS son inyectadas normalmente desde el
 * Makefile para que cada binario MNIST se compile con una vista consistente del
 * layout. Los valores por defecto permiten compilar un caso mínimo sin pasar
 * variables adicionales.
 */
#ifndef FF_IMAGE_WIDTH
#define FF_IMAGE_WIDTH 20
#endif

#ifndef FF_IMAGE_HEIGHT
#define FF_IMAGE_HEIGHT 20
#endif

#ifndef FF_PIXEL_BITS
#define FF_PIXEL_BITS 1
#endif

#ifndef FF_NUM_CLASSES
#define FF_NUM_CLASSES 10
#endif

#ifndef FF_HIDDEN_LAYERS
#define FF_HIDDEN_LAYERS 1
#endif

#ifndef FF_HIDDEN_0
#define FF_HIDDEN_0 64
#endif

#ifndef FF_HIDDEN_1
#define FF_HIDDEN_1 1
#endif

#ifndef FF_HIDDEN_2
#define FF_HIDDEN_2 1
#endif

#ifndef FF_HIDDEN_3
#define FF_HIDDEN_3 1
#endif

#ifndef FF_PARALLEL_NEURONS
#define FF_PARALLEL_NEURONS 8
#endif

/*
 * Layout físico del dataset: una cabecera de 16 palabras describe el archivo y
 * cada muestra se guarda en palabras de 32 bits con convención LSB-first. La
 * etiqueta one-hot ocupa los primeros 10 bits y después aparecen los pixeles.
 */
static const int FF_WORD_BITS = 32;
static const int FF_HEADER_WORDS = 16;
static const uint32_t FF_HEADER_MAGIC = 0x4D4E4953u;
static const uint32_t FF_HEADER_VERSION = 1u;

static const int FF_NUM_PIXELS = FF_IMAGE_WIDTH * FF_IMAGE_HEIGHT;
static const int FF_LABEL_BITS = FF_NUM_CLASSES;
static const int FF_PIXEL_STORAGE_BITS = FF_NUM_PIXELS * FF_PIXEL_BITS;
static const int FF_USEFUL_BITS = FF_LABEL_BITS + FF_PIXEL_STORAGE_BITS;
static const int FF_WORDS_PER_SAMPLE =
    (FF_USEFUL_BITS + FF_WORD_BITS - 1) / FF_WORD_BITS;
static const int FF_TOTAL_BITS = FF_WORDS_PER_SAMPLE * FF_WORD_BITS;
static const int FF_PADDING_BITS = FF_TOTAL_BITS - FF_USEFUL_BITS;

/*
 * La primera capa recibe una característica por clase y una por pixel. Aunque
 * un pixel pueda estar cuantizado con varios bits, la capa ve el valor ya
 * reconstruido por el decodificador para mantener una interfaz de entrada fija.
 */
static const int FF_INPUT_DIM = FF_NUM_CLASSES + FF_NUM_PIXELS;

/*
 * La arquitectura configurable se implementa con máximos estáticos. Esto evita
 * memoria dinámica dentro del kernel HLS: las capas inactivas o las posiciones
 * sobrantes se mantienen sin uso, pero el compilador conoce el tamaño total.
 */
static const int FF_MAX_HIDDEN_LAYERS = 4;
static const int FF_LAYER_NEURONS[FF_MAX_HIDDEN_LAYERS] = {
    FF_HIDDEN_0,
    FF_HIDDEN_1,
    FF_HIDDEN_2,
    FF_HIDDEN_3
};

static const int FF_MAX_NEURONS_01 =
    (FF_HIDDEN_0 > FF_HIDDEN_1) ? FF_HIDDEN_0 : FF_HIDDEN_1;
static const int FF_MAX_NEURONS_23 =
    (FF_HIDDEN_2 > FF_HIDDEN_3) ? FF_HIDDEN_2 : FF_HIDDEN_3;
static const int FF_MAX_LAYER_NEURONS =
    (FF_MAX_NEURONS_01 > FF_MAX_NEURONS_23) ?
        FF_MAX_NEURONS_01 : FF_MAX_NEURONS_23;
static const int FF_MAX_LAYER_INPUT_DIM =
    (FF_INPUT_DIM > FF_MAX_LAYER_NEURONS) ?
        FF_INPUT_DIM : FF_MAX_LAYER_NEURONS;
static const int FF_MODEL_WEIGHT_STRIDE =
    FF_MAX_LAYER_NEURONS * FF_MAX_LAYER_INPUT_DIM;
static const int FF_MODEL_WEIGHT_COUNT =
    FF_HIDDEN_LAYERS * FF_MODEL_WEIGHT_STRIDE;
static const int FF_MODEL_BIAS_COUNT =
    FF_HIDDEN_LAYERS * FF_MAX_LAYER_NEURONS;

/*
 * Las escalas enteras reemplazan normalización en punto flotante. La etiqueta
 * se amplifica porque un solo bit one-hot debe competir contra cientos de
 * pixeles durante la comparación de goodness positiva y negativa.
 */
#ifndef FF_LABEL_SCALE
#define FF_LABEL_SCALE 8
#endif

#ifndef FF_PIXEL_SCALE
#define FF_PIXEL_SCALE 1
#endif

/*
 * El entrenamiento usa una regla tipo hinge: se empuja hacia arriba el ejemplo
 * positivo si su goodness queda bajo el umbral y se empuja hacia abajo el
 * ejemplo negativo si lo supera. Esto evita sigmoid/softplus, que son costosas
 * para una primera implementación en Artix-7.
 */
#ifndef FF_GOODNESS_THRESHOLD
#define FF_GOODNESS_THRESHOLD 72
#endif

/*
 * Los límites de clipping controlan crecimiento de pesos latentes, bias y
 * activaciones. Además de proteger contra overflow en simulación C++, ayudan a
 * acotar anchos efectivos y recursos al migrar a HLS.
 */
#ifndef FF_LATENT_INIT_MAG
#define FF_LATENT_INIT_MAG 3
#endif

#ifndef FF_LATENT_CLIP
#define FF_LATENT_CLIP 127
#endif

#ifndef FF_BIAS_CLIP
#define FF_BIAS_CLIP 2047
#endif

#ifndef FF_ACTIVATION_CLIP
#define FF_ACTIVATION_CLIP 255
#endif

#ifndef FF_PIXEL_LR_STEP
#define FF_PIXEL_LR_STEP 1
#endif

#ifndef FF_LABEL_LR_STEP
#define FF_LABEL_LR_STEP 4
#endif

#ifndef FF_BIAS_LR_STEP
#define FF_BIAS_LR_STEP 1
#endif

#ifndef FF_MAX_EPOCHS
#define FF_MAX_EPOCHS 64
#endif

/*
 * Las comprobaciones de compilación fallan temprano cuando una combinación de
 * Makefile no puede sintetizarse de forma coherente. Esto es preferible a
 * descubrir errores de dimensión durante csim o, peor, durante síntesis.
 */
static_assert(FF_IMAGE_WIDTH > 0, "FF_IMAGE_WIDTH debe ser positivo.");
static_assert(FF_IMAGE_HEIGHT > 0, "FF_IMAGE_HEIGHT debe ser positivo.");
static_assert(FF_PIXEL_BITS >= 1 && FF_PIXEL_BITS <= 8,
              "FF_PIXEL_BITS debe estar entre 1 y 8.");
static_assert(FF_NUM_CLASSES == 10,
              "Esta version inicial asume MNIST con 10 clases.");
static_assert(FF_HIDDEN_LAYERS >= 1 &&
              FF_HIDDEN_LAYERS <= FF_MAX_HIDDEN_LAYERS,
              "FF_HIDDEN_LAYERS debe estar entre 1 y 4.");
static_assert(FF_HIDDEN_0 > 0 && FF_HIDDEN_1 > 0 &&
              FF_HIDDEN_2 > 0 && FF_HIDDEN_3 > 0,
              "Todos los FF_HIDDEN_N deben ser positivos.");
static_assert(FF_PARALLEL_NEURONS > 0,
              "FF_PARALLEL_NEURONS debe ser mayor que cero.");
static_assert(FF_PARALLEL_NEURONS <= FF_MAX_LAYER_NEURONS,
              "FF_PARALLEL_NEURONS no puede exceder FF_MAX_LAYER_NEURONS.");

#endif
