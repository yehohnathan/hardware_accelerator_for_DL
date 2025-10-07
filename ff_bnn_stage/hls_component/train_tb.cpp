#include "forward_fw.hpp"
#include <iostream>

int main(){
  // Ruta fija (Windows)
  const std::string BASE_DIR = R"(D:\TFG\hardware_accelerator_for_DL\ff_bnn_stage\mnist_data\build\pack_routeA)";

  MetaFF meta;
  if(!load_meta(BASE_DIR, meta)) return 1;

  std::cout << "dtype=" << meta.dtype
            << " D=" << meta.input_dim
            << " B=" << meta.batch_size
            << " n_pos=" << meta.n_pos
            << " n_neg=" << meta.n_neg
            << " bytes_per_row=" << meta.bytes_per_row
            << " word_bits=" << meta.word_bits << "\n";

  // POS
  std::vector<uint8_t> raw_pos, y_pos;
  if(!load_inputs_raw(join_path(BASE_DIR, meta.f_inputs_pos), meta.n_pos, meta.bytes_per_row, raw_pos)) return 1;
  (void)load_labels_u8(join_path(BASE_DIR, meta.f_labels_pos), meta.n_pos, y_pos);

  // NEG
  std::vector<uint8_t> raw_neg, y_neg;
  if(!load_inputs_raw(join_path(BASE_DIR, meta.f_inputs_neg), meta.n_neg, meta.bytes_per_row, raw_neg)) return 1;
  (void)load_labels_u8(join_path(BASE_DIR, meta.f_labels_neg), meta.n_neg, y_neg);

  // Palabras LE32 y bits {0,1}
  std::vector<u32> pos_words, neg_words;
  raw_to_words_le32(raw_pos, pos_words);
  raw_to_words_le32(raw_neg, neg_words);

  std::vector<uint8_t> x_pos01, x_neg01;
  unpack_dataset_bits_lsb0(raw_pos, meta.n_pos, meta.input_dim, meta.bytes_per_row, x_pos01);
  unpack_dataset_bits_lsb0(raw_neg, meta.n_neg, meta.input_dim, meta.bytes_per_row, x_neg01);

  std::cout << "words_per_row=" << meta.words_per_row()
            << " POS_words_total=" << pos_words.size()
            << " NEG_words_total=" << neg_words.size() << "\n";

  std::cout << "POS[0] bits (primeros 32): ";
  for(int i=0;i<std::min(32, meta.input_dim);++i) std::cout << int(x_pos01[i]);
  std::cout << "\n";

  return 0;
}
