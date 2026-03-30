#ifndef FORWARD_FW_HPP
#define FORWARD_FW_HPP

#include <ap_int.h>     // Se incluyen enteros arbitrarios de HLS para manejar vectores binarios anchos.
#include <ap_fixed.h>   // Se incluyen tipos de punto fijo para pesos latentes, bias y métricas.
#include <stdint.h>     // Se incluyen tipos estándar como uint16_t para semilla y control.

// ============================================================
// Constantes del formato del dataset hardware ya congelado
// ============================================================
static const int NUM_CLASSES      = 10;   // Cantidad de clases de MNIST de 0 a 9.
static const int WORD_BITS        = 32;   // Anchura de cada palabra del binario empaquetado.
static const int WORDS_PER_SAMPLE = 25;   // Cantidad de words de 32 bits por muestra.
static const int TOTAL_BITS       = 800;  // Anchura física total de cada muestra empaquetada.

static const int LABEL_BITS       = 10;   // Anchura del label en codificación one-hot.
static const int PIXEL_BITS       = 784;  // Cantidad de píxeles binarios de la imagen 28x28.
static const int PADDING_BITS     = 6;    // Cantidad de bits de relleno hasta completar 800 bits.

// ============================================================
// Constantes del modelo entrenable de la Etapa C
// ============================================================
static const int MODEL_INPUT_BITS       = LABEL_BITS + PIXEL_BITS;   // Entrada lógica útil de la capa FF: 10 + 784 = 794 bits.
static const int MODEL_NEURONS          = 64;                        // Cantidad total de neuronas de la primera capa entrenable.
static const int MODEL_PARALLEL_NEURONS = 8;                         // Cantidad de neuronas procesadas en paralelo por tile.

static const int MODEL_NEURON_TILES     = MODEL_NEURONS / MODEL_PARALLEL_NEURONS; 	// Cantidad de tiles de 8 neuronas requeridos.
static const int MODEL_WEIGHT_COUNT     = MODEL_NEURONS * MODEL_INPUT_BITS;        	// Cantidad total de pesos latentes del modelo.

// ============================================================
// Hiperparámetros hardware iniciales de entrenamiento FF
// ============================================================
static const int TRAIN_ACTIVATION_CLAMP_INT = 16;  // Saturación usada para normalizar activaciones en la actualización.
static const int TRAIN_SIGNAL_CLAMP_INT     = 16;  // Saturación usada para normalizar la señal de error FF.

// ============================================================
// Tipos base del diseño
// ============================================================
typedef ap_uint<WORD_BITS>       word_t;       // Tipo de una palabra individual de 32 bits.
typedef ap_uint<TOTAL_BITS>      raw_sample_t; // Tipo de la muestra física completa de 800 bits.
typedef ap_uint<LABEL_BITS>      label_oh_t;   // Tipo del label en formato one-hot.
typedef ap_uint<PIXEL_BITS>      pixels_t;     // Tipo del vector de píxeles binarios.
typedef ap_uint<PADDING_BITS>    padding_t;    // Tipo del padding físico.
typedef ap_uint<MODEL_INPUT_BITS> ff_input_t;  // Tipo de la entrada lógica FF sin padding.
typedef ap_uint<4>               label_idx_t;  // Tipo del índice de clase de 0 a 9.
typedef ap_uint<16>              lfsr_t;       // Tipo del estado del LFSR pseudoaleatorio.

typedef ap_fixed<8, 3, AP_RND, AP_SAT>   latent_t;     // Tipo fijo pequeño para los pesos latentes almacenados en BRAM.
typedef ap_fixed<12, 6, AP_RND, AP_SAT>  bias_t;       // Tipo fijo usado para el bias entrenable de cada neurona.
typedef ap_fixed<16, 12, AP_RND, AP_SAT> preact_t;     // Tipo fijo para el pre-activado local de cada neurona.
typedef ap_fixed<16, 12, AP_RND, AP_SAT> activation_t; // Tipo fijo para la activación ReLU local.
typedef ap_fixed<24, 12, AP_RND, AP_SAT> goodness_t;   // Tipo fijo para la goodness y el gap de entrenamiento.
typedef ap_fixed<16, 6, AP_RND, AP_SAT>  signal_t;     // Tipo fijo para la señal local de aprendizaje FF.
typedef ap_fixed<16, 3, AP_RND, AP_SAT>  gain_t;       // Tipo fijo para la magnitud de actualización de pesos y bias.

// ============================================================
// Constantes de entrenamiento en punto fijo
// ============================================================
static const latent_t   	LATENT_WEIGHT_CLIP    	= (latent_t)2.5;     // ímite absoluto de clipping para pesos latentes.
static const bias_t     	LATENT_BIAS_CLIP      	= (bias_t)4.0;       // Límite absoluto de clipping para bias latente.
static const latent_t   	LATENT_INIT_MAG       	= (latent_t)0.25;    // Magnitud base de inicialización pseudoaleatoria.
static const gain_t     	LEARNING_RATE_HW      	= (gain_t)0.125;     // Tasa de aprendizaje hardware en formato fijo.
static const goodness_t 	GOODNESS_THRESHOLD_HW 	= (goodness_t)64.0;  // Umbral hardware inicial para separar positivos y negativos.
static const activation_t 	ACTIVATION_CLAMP_HW 	= (activation_t)TRAIN_ACTIVATION_CLAMP_INT; 	// Clamp de activación usado para normalización.
static const signal_t     	SIGNAL_CLAMP_HW     	= (signal_t)TRAIN_SIGNAL_CLAMP_INT;          	// Clamp de señal usado para normalización.

// ============================================================
// Prototipos de funciones auxiliares sintetizables de Etapas A y B
// ============================================================
// Se declara el avance del LFSR de 16 bits.
lfsr_t lfsr16_next(lfsr_t state);

// Se declara la validación de un vector one-hot.
bool is_valid_onehot(label_oh_t label_onehot);

// Se declara la decodificación de one-hot a índice.
label_idx_t decode_onehot(label_oh_t label_onehot);

// Se declara la codificación de índice a one-hot.
label_oh_t encode_onehot(label_idx_t label_idx);

// Se declara la generación de una etiqueta negativa excluyente.
label_idx_t generate_negative_label(label_idx_t true_label, lfsr_t &state);

// Se declara la carga de una muestra desde 25 words externas.
raw_sample_t load_sample_from_words(const word_t *mem, int sample_idx);

// Se declara el almacenamiento de una muestra en 25 words externas.
void store_sample_to_words(word_t *mem, int sample_idx, raw_sample_t sample);

// Se declara el desempaquetado de la muestra física.
void unpack_sample(raw_sample_t sample, label_oh_t &label_onehot, pixels_t &pixels, padding_t &padding);

// Se declara el empaquetado de la muestra física.
raw_sample_t pack_sample(label_oh_t label_onehot, pixels_t pixels, padding_t padding);

// ============================================================
// Prototipos de funciones auxiliares de la Etapa C
// ============================================================
// Se declara la construcción de la entrada lógica FF de 794 bits.
ff_input_t build_ff_input(label_oh_t label_onehot, pixels_t pixels);

// Se declara el desempaquetado de la entrada lógica FF.
void unpack_ff_input(ff_input_t ff_input, label_oh_t &label_onehot, pixels_t &pixels);

// ============================================================
// Top sintetizable heredado de la Etapa B
// ============================================================
// Se declara el top heredado que conserva la preparación hardware de pares FF.
void forward_fw_top(
    const word_t *in_mem,
    word_t *pos_mem,
    word_t *neg_mem,
    label_idx_t *true_label_mem,
    label_idx_t *neg_label_mem,
    int n_samples,
    uint16_t seed
);

// ============================================================
// Top sintetizable nuevo de la Etapa C
// ============================================================
// Se declara el top que introduce entrenamiento local FF con pesos mutables dentro de la FPGA.
void ff_train_top(
    const word_t *in_mem,
    label_idx_t *true_label_mem,
    label_idx_t *pred_label_mem,
    latent_t *weight_mem_out,
    bias_t *bias_mem_out,
    goodness_t *g_pos_mem,
    goodness_t *g_neg_mem,
    goodness_t *gap_mem,
    ap_uint<32> *correct_count_mem,
    int n_samples,
    int n_train_samples,
    int n_epochs,
    uint16_t seed,
    bool reset_model
);

#endif
