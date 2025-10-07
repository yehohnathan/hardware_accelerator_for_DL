#pragma once
#include <cstdint>
#include <string>
#include <vector>

#ifdef __SYNTHESIS__
  #include "ap_int.h"
  using u32 = ap_uint<32>;
#else
  using u32 = uint32_t;
#endif

struct MetaFF {
  std::string format, dtype;
  int input_dim = 0, batch_size = 1;
  bool drop_last = false;
  int n_pos = 0, n_neg = 0, n_batches_pos = 0, n_batches_neg = 0;

  std::string f_inputs_pos, f_inputs_neg, f_labels_pos, f_labels_neg;

  int word_bits = 32, bytes_per_row = 0, features_padded = 0;
  std::string bitorder_in_byte = "lsb0", endianness = "little";
  bool row_major = true;

  int rows = -1, cols = -1, token_dim = 10;
  std::string where_tokens = "prefix";

  int words_per_row() const { return (word_bits/8) ? (bytes_per_row / (word_bits/8)) : 0; }
  int total_pos()     const { return n_pos; }
  int total_neg()     const { return n_neg; }
};

// Carga de meta y datos
bool load_meta(const std::string& dir, MetaFF& out);
bool load_inputs_raw(const std::string& path, size_t n, size_t bytes_per_row, std::vector<uint8_t>& raw);
bool load_labels_u8(const std::string& path, size_t n, std::vector<uint8_t>& y);

// Conversión a palabras LE32 (cómodo para AXI/BRAM)
void raw_to_words_le32(const std::vector<uint8_t>& raw, std::vector<u32>& words);

// Desempaquetado {0,1} LSB0
void unpack_dataset_bits_lsb0(const std::vector<uint8_t>& raw, int n, int d, int bytes_per_row, std::vector<uint8_t>& x01);

// Mapeo {0,1}->{-1,+1}
void map01_to_pm1(const std::vector<uint8_t>& x01, std::vector<int8_t>& xpm1);

// util
std::string join_path(const std::string& dir, const std::string& file);
