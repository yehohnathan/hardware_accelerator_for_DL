#ifndef FORWARD_FW_HPP
#define FORWARD_FW_HPP

#include <ap_fixed.h>  // Se incluyen tipos de punto fijo sintetizables para pesos, bias y metricas.
#include <ap_int.h>    // Se incluyen enteros arbitrarios para empaquetado binario y control.
#include <stdint.h>    // Se incluyen tipos estandar para semilla y contadores.

// ============================================================
// Constantes del dataset hardware congelado
// ============================================================
static const int NUM_CLASSES = 10;       // Se fija la cantidad de clases de MNIST.
static const int WORD_BITS = 32;         // Se fija el ancho de cada palabra empaquetada.
static const int WORDS_PER_SAMPLE = 25;  // Se fija la cantidad de palabras por muestra.
static const int TOTAL_BITS = 800;       // Se fija el ancho fisico total de cada muestra.

static const int LABEL_BITS = 10;        // Se fija el ancho del label one-hot.
static const int PIXEL_BITS = 784;       // Se fija el ancho de la imagen binaria 28x28.
static const int PADDING_BITS = 6;       // Se fija el padding fisico hasta completar 800 bits.

// ============================================================
// Constantes del modelo entrenable de Etapa C
// ============================================================
static const int MODEL_INPUT_BITS = LABEL_BITS + PIXEL_BITS;  // Se fija la entrada logica util del modelo.
static const int MODEL_NUM_LAYERS = 2;                        // Se fija una red FF de dos capas ocultas.
static const int MODEL_PARALLEL_NEURONS = 8;                  // Se fija el paralelismo base por tile para ambas capas.

#ifndef FF_MODEL_LAYER1_NEURONS_CFG
#define FF_MODEL_LAYER1_NEURONS_CFG 64
#endif
// Se permite sobreescribir la primera capa desde compilacion para barrer hypervalores sin editar el archivo.

#ifndef FF_MODEL_LAYER2_NEURONS_CFG
#define FF_MODEL_LAYER2_NEURONS_CFG 32
#endif
// Se permite sobreescribir la segunda capa desde compilacion para barrer hypervalores sin editar el archivo.

#ifndef FF_MODEL_BATCH_SIZE_CFG
#define FF_MODEL_BATCH_SIZE_CFG 64
#endif
// Se fija por defecto un mini-batch menor porque en las pruebas dio una dinamica de update mas visible y estable.

#ifndef FF_LATENT_INIT_MAG_CFG
#define FF_LATENT_INIT_MAG_CFG 0.03125
#endif
// Se permite sobreescribir la magnitud base de inicializacion desde compilacion.

#ifndef FF_LEARNING_RATE_CFG
#define FF_LEARNING_RATE_CFG 0.02
#endif
// Se fija por defecto una tasa mayor porque el preset anterior no llegaba a modificar ningun valor representable.

#ifndef FF_GOODNESS_THRESHOLD_CFG
#define FF_GOODNESS_THRESHOLD_CFG 512.0
#endif
// Se fija por defecto un threshold alineado con la escala real de goodness observada en hardware.

#ifndef FF_LABEL_SCALE_CFG
#define FF_LABEL_SCALE_CFG 3.0
#endif
// Se permite sobreescribir la escala de etiqueta incrustada desde compilacion.

#ifndef FF_PIXEL_SCALE_CFG
#define FF_PIXEL_SCALE_CFG 1.0
#endif
// Se permite sobreescribir la escala de pixeles desde compilacion.

#ifndef FF_NONLINEAR_CLIP_CFG
#define FF_NONLINEAR_CLIP_CFG 32.0
#endif
// Se fija por defecto un clip mas amplio para no saturar demasiado pronto las no linealidades suaves.

////////
static const int MODEL_LAYER1_INPUT_BITS = MODEL_INPUT_BITS;                            // Se fija la entrada de la primera capa en 794.
static const int MODEL_LAYER1_NEURONS = FF_MODEL_LAYER1_NEURONS_CFG;                    // Se adopta una primera capa compacta pero configurable para pruebas.
static const int MODEL_LAYER1_TILES = MODEL_LAYER1_NEURONS / MODEL_PARALLEL_NEURONS;    // Se fija la cantidad de tiles de la primera capa.
static const int MODEL_LAYER1_WEIGHT_COUNT = MODEL_LAYER1_NEURONS * MODEL_LAYER1_INPUT_BITS;    // Se fija la cantidad total de pesos de la primera capa.

////////
static const int MODEL_LAYER2_INPUT_BITS = MODEL_LAYER1_NEURONS;                        // Se fija la entrada de la segunda capa en la salida de la primera.
static const int MODEL_LAYER2_NEURONS = FF_MODEL_LAYER2_NEURONS_CFG;                    // Se adopta una segunda capa compacta pero configurable para pruebas.
static const int MODEL_LAYER2_TILES = MODEL_LAYER2_NEURONS / MODEL_PARALLEL_NEURONS;    // Se fija la cantidad de tiles de la segunda capa.
static const int MODEL_LAYER2_WEIGHT_COUNT = MODEL_LAYER2_NEURONS * MODEL_LAYER2_INPUT_BITS;    // Se fija la cantidad total de pesos de la segunda capa.

static const int MODEL_TOTAL_NEURONS = MODEL_LAYER1_NEURONS + MODEL_LAYER2_NEURONS;     // Se fija la cantidad total de neuronas latentes del modelo.
static const int MODEL_WEIGHT_COUNT = MODEL_LAYER1_WEIGHT_COUNT + MODEL_LAYER2_WEIGHT_COUNT;    // Se fija la cantidad total de pesos del snapshot externo.
static const int MODEL_BIAS_COUNT = MODEL_TOTAL_NEURONS;                                // Se fija la cantidad total de bias del snapshot externo.
static const int MODEL_MAX_LAYER_NEURONS = MODEL_LAYER1_NEURONS;                        // Se fija la capa mas grande para buffers auxiliares.

static const int MODEL_BATCH_SIZE = FF_MODEL_BATCH_SIZE_CFG;  // Se mantiene el mini-batch configurable para barrer costo vs dinamica de aprendizaje.
static const int TRAIN_MAX_EPOCHS = 50;  // Se eleva el tope para poder replicar el notebook de referencia cuando sea necesario.

// ============================================================
// Tipos base del proyecto
// ============================================================
typedef ap_uint<WORD_BITS> word_t;               // Se define el tipo de una palabra externa del dataset.
typedef ap_uint<TOTAL_BITS> raw_sample_t;        // Se define el tipo de la muestra fisica completa.
typedef ap_uint<LABEL_BITS> label_oh_t;          // Se define el tipo del label one-hot empaquetado.
typedef ap_uint<PIXEL_BITS> pixels_t;            // Se define el tipo del vector de pixeles binarios.
typedef ap_uint<PADDING_BITS> padding_t;         // Se define el tipo del padding fisico.
typedef ap_uint<MODEL_INPUT_BITS> ff_input_t;    // Se define el tipo de la entrada logica FF sin padding.
typedef ap_uint<4> label_idx_t;                  // Se define el tipo del indice de clase de 0 a 9.
typedef ap_uint<16> lfsr_t;                      // Se define el tipo del estado del generador pseudoaleatorio.

typedef ap_fixed<8, 4, AP_RND, AP_SAT> feature_t;           // Se define el tipo fijo de una caracteristica de entrada.
typedef ap_fixed<20, 4, AP_RND, AP_SAT> latent_t;           // Se amplian los bits fraccionales para que updates pequenos sigan siendo representables.
typedef ap_fixed<24, 8, AP_RND, AP_SAT> bias_t;             // Se amplian los bits fraccionales del bias para evitar que la correccion media se redondee a cero.
typedef ap_fixed<20, 11, AP_RND, AP_SAT> preact_t;          // Se define el tipo fijo del preactivado local.
typedef ap_fixed<20, 11, AP_RND, AP_SAT> activation_t;      // Se define el tipo fijo de la activacion ReLU.
typedef ap_fixed<36, 22, AP_RND, AP_SAT> goodness_t;        // Se define el tipo fijo de la goodness por muestra.
typedef ap_fixed<24, 12, AP_RND, AP_SAT> loss_t;            // Se define el tipo fijo de la perdida FF suave.
typedef ap_fixed<24, 12, AP_RND, AP_SAT> scale_t;           // Se define el tipo fijo de las escalas sigmoidales.
typedef ap_fixed<40, 24, AP_RND, AP_SAT> stat_accum_t;      // Se define el tipo fijo de acumulacion de metricas.
typedef ap_fixed<32, 20, AP_RND, AP_SAT> update_accum_t;    // Se define el tipo fijo de acumulacion para dW y db.
typedef ap_fixed<32, 8, AP_RND, AP_SAT> learning_rate_t;    // Se amplia la precision de la tasa para preservar lr y lr por batch con mas fidelidad.

// ============================================================
// Hiperparametros hardware iniciales
// ============================================================
static const latent_t LATENT_WEIGHT_CLIP = (latent_t)2.5;                          // Se fija el clip absoluto de pesos latentes.
static const bias_t LATENT_BIAS_CLIP = (bias_t)8.0;                                // Se fija el clip absoluto de bias latentes.
static const latent_t LATENT_INIT_MAG = (latent_t)FF_LATENT_INIT_MAG_CFG;          // Se hace configurable la magnitud base de inicializacion.
static const learning_rate_t LEARNING_RATE_HW = (learning_rate_t)FF_LEARNING_RATE_CFG; // Se hace configurable la tasa de aprendizaje base.
static const goodness_t GOODNESS_THRESHOLD_HW = (goodness_t)FF_GOODNESS_THRESHOLD_CFG; // Se hace configurable el threshold FF para ajustar la escala hardware.
static const feature_t LABEL_SCALE_HW = (feature_t)FF_LABEL_SCALE_CFG;              // Se hace configurable la escala de la etiqueta incrustada.
static const feature_t PIXEL_SCALE_HW = (feature_t)FF_PIXEL_SCALE_CFG;              // Se hace configurable la escala de los pixeles binarios.
static const goodness_t NONLINEAR_CLIP_HW = (goodness_t)FF_NONLINEAR_CLIP_CFG;     // Se hace configurable el clip de margen previo a sigmoid y softplus.

// ============================================================
// Prototipos sintetizables de Etapas A y B
// ============================================================
lfsr_t lfsr16_next(lfsr_t state);  // Se declara el avance del LFSR de 16 bits.

bool is_valid_onehot(label_oh_t label_onehot);  // Se declara la validacion de un vector one-hot.

label_idx_t decode_onehot(label_oh_t label_onehot);  // Se declara la decodificacion one-hot -> indice.

label_oh_t encode_onehot(label_idx_t label_idx);  // Se declara la codificacion indice -> one-hot.

label_idx_t generate_negative_label(label_idx_t true_label, lfsr_t &state);  // Se declara la etiqueta negativa excluyente.

raw_sample_t load_sample_from_words(const word_t *mem, int sample_idx);  // Se declara la carga de una muestra desde memoria lineal.

void store_sample_to_words(word_t *mem, int sample_idx, raw_sample_t sample);  // Se declara la escritura de una muestra en memoria lineal.

void unpack_sample(raw_sample_t sample, label_oh_t &label_onehot, pixels_t &pixels, padding_t &padding);  // Se declara el desempaquetado fisico.

raw_sample_t pack_sample(label_oh_t label_onehot, pixels_t pixels, padding_t padding);  // Se declara el empaquetado fisico.

// ============================================================
// Prototipos sintetizables de Etapa C
// ============================================================
ff_input_t build_ff_input(label_oh_t label_onehot, pixels_t pixels);  // Se declara la construccion de la entrada FF.

void unpack_ff_input(ff_input_t ff_input, label_oh_t &label_onehot, pixels_t &pixels);  // Se declara el desempaquetado de la entrada FF.

// ============================================================
// Top heredado de Etapa B
// ============================================================
void forward_fw_top(
    const word_t *in_mem,
    word_t *pos_mem,
    word_t *neg_mem,
    label_idx_t *true_label_mem,
    label_idx_t *neg_label_mem,
    int n_samples,
    uint16_t seed
);  // Se declara el top que prepara pares positivos y negativos.

// ============================================================
// Top entrenable de Etapa C
// ============================================================
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
);  // Se declara el top que entrena el modelo en hardware y luego ejecuta inferencia.

#endif
