// forward_fw.cpp
#include "forward_fw.hpp"

// Kernel muy simple y 100% sintetizable.
// Suma módulo 256 los 'feat_dim' bytes de cada muestra y añade label[i].
void forward_fw_top(const uint8_t* inputs,
                    const uint8_t* labels,
                    uint8_t* outputs,
                    uint32_t feat_dim,
                    uint32_t batch) {
#pragma HLS INLINE off
    for (uint32_t i = 0; i < batch; ++i) {
        uint32_t acc = 0;
        for (uint32_t j = 0; j < feat_dim; ++j) {
#pragma HLS PIPELINE II=1
            acc += inputs[i * feat_dim + j];
        }
        acc += labels[i];
        outputs[i] = static_cast<uint8_t>(acc & 0xFFu);
    }
}
