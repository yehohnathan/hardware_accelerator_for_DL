// train_tb.cpp
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "forward_fw.hpp"

// Ruta a tus archivos (ajústala).
static const char* PACK_DIR =
  R"(D:\TFG\hardware_accelerator_for_DL\ff_bnn_stage\mnist_data\build\pack_routeA)";

static const char* FILE_META  = "meta.json";
static const char* FILE_POS   = "inputs_pos.bin";
static const char* FILE_NEG   = "inputs_neg.bin";
static const char* FILE_LPOS  = "labels_pos.bin";
static const char* FILE_LNEG  = "labels_neg.bin";

// Une directorio + nombre con separador de Windows.
static std::string join_path(const char* dir, const char* name) {
    std::string s(dir);
    if (!s.empty()) {
        char last = s.back();
        if (last != '\\' && last != '/') s.push_back('\\');
    }
    s += name;
    return s;
}

// Lee archivo de texto completo (para meta.json).
static std::string read_text(const std::string& path) {
    std::ifstream ifs(path.c_str());
    if (!ifs) { std::cerr << "ERROR abriendo " << path << "\n"; std::exit(1); }
    std::ostringstream oss; oss << ifs.rdbuf(); return oss.str();
}

// Lee archivo binario completo a memoria.
static std::vector<uint8_t> read_bin(const std::string& path) {
    std::ifstream ifs(path.c_str(), std::ios::binary);
    if (!ifs) { std::cerr << "ERROR abriendo " << path << "\n"; std::exit(1); }
    ifs.seekg(0, std::ios::end);
    std::streamoff len = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf((size_t)len);
    if (len > 0) ifs.read((char*)buf.data(), len);
    return buf;
}

// Parser JSON mínimo (sin dependencias):
// Busca "clave": valor (numérica) o "clave": "texto".
static uint32_t parse_uint(const std::string& js, const char* key) {
    std::string kw = std::string("\"") + key + "\"";
    size_t p = js.find(kw);
    if (p == std::string::npos) return 0;
    p = js.find(':', p); if (p == std::string::npos) return 0; ++p;
    while (p < js.size() && (js[p]==' '||js[p]=='\t')) ++p;
    uint32_t v = 0;
    while (p < js.size() && (js[p]>='0' && js[p]<='9')) {
        v = v*10u + (uint32_t)(js[p]-'0'); ++p;
    }
    return v;
}
static bool parse_string(const std::string& js, const char* key,
                         char* out, size_t out_sz) {
    std::string kw = std::string("\"") + key + "\"";
    size_t p = js.find(kw);
    if (p == std::string::npos) return false;
    p = js.find(':', p); if (p == std::string::npos) return false;
    p = js.find('\"', p); if (p == std::string::npos) return false;
    size_t q = js.find('\"', p+1); if (q == std::string::npos) return false;
    std::string val = js.substr(p+1, q-(p+1));
    std::memset(out, 0, out_sz);
    std::strncpy(out, val.c_str(), out_sz-1);
    return true;
}

// Carga meta.json soportando las claves reales del empaquetador.
static Meta load_meta(const std::string& meta_path) {
    Meta m{};
    std::string js = read_text(meta_path);

    // Nombres reales (PackInfo): input_dim, token_dim, rows, cols,
    // where_tokens, dtype, word_bits, bytes_per_row, n_pos, n_neg,
    // batch_size, row_major (opcional).
    m.input_dim     = parse_uint(js, "input_dim");
    m.token_dim     = parse_uint(js, "token_dim");
    if (!m.token_dim) m.token_dim = 10; // por defecto

    m.rows          = parse_uint(js, "rows");
    m.cols          = parse_uint(js, "cols");

    parse_string(js, "where_tokens", m.where_tokens, sizeof m.where_tokens);
    if (m.where_tokens[0] == 0) std::strcpy(m.where_tokens, "suffix");

    parse_string(js, "dtype", m.dtype, sizeof m.dtype);

    m.word_bits     = parse_uint(js, "word_bits");
    m.bytes_per_row = parse_uint(js, "bytes_per_row");

    m.n_pos         = parse_uint(js, "n_pos");
    m.n_neg         = parse_uint(js, "n_neg");
    m.batch_size    = parse_uint(js, "batch_size");

    return m;
}

// Desempaqueta una fila bitpacked (LSB-first por byte) a {0,1}.
static void unpack_row_to01(const uint8_t* row_bytes,
                            uint32_t bytes_per_row,
                            uint32_t D,
                            std::vector<uint8_t>& out01) {
    out01.assign(D, 0);
    uint32_t f = 0;
    for (uint32_t i = 0; i < bytes_per_row && f < D; ++i) {
        uint8_t byte = row_bytes[i];
        for (uint32_t b = 0; b < 8 && f < D; ++b) {
            out01[f++] = (byte >> b) & 0x1u; // LSB -> feature 0
        }
    }
}

// Imprime hasta 'max_cols' valores de un vector uint8_t.
static void print_vec_u8(const uint8_t* v, uint32_t len, uint32_t max_cols) {
    uint32_t n = (len < max_cols) ? len : max_cols;
    for (uint32_t i = 0; i < n; ++i) {
        std::cout << +v[i] << (i+1<n ? ' ' : '\n');
    }
    if (n < len) std::cout << "...\n";
}

int main() {
    // ---------- 1) Carga metadatos ----------
    const std::string path_meta = join_path(PACK_DIR, FILE_META);
    Meta meta = load_meta(path_meta);

    const uint32_t D = meta.input_dim;
    const uint32_t T = meta.token_dim;
    const uint32_t feat_dim = D; // ya incluye tokens

    std::cout << "META\n";
    std::cout << " input_dim=" << D
              << " token_dim=" << T
              << " rows=" << meta.rows
              << " cols=" << meta.cols
              << " where=" << meta.where_tokens << "\n";
    std::cout << " dtype=" << meta.dtype
              << " word_bits=" << meta.word_bits
              << " bytes_per_row=" << meta.bytes_per_row << "\n";
    std::cout << " n_pos=" << meta.n_pos
              << " n_neg=" << meta.n_neg
              << " batch_size=" << meta.batch_size << "\n\n";

    // ---------- 2) Abre binarios ----------
    const std::string p_inputs_pos = join_path(PACK_DIR, FILE_POS);
    const std::string p_inputs_neg = join_path(PACK_DIR, FILE_NEG);
    const std::string p_labels_pos = join_path(PACK_DIR, FILE_LPOS);
    const std::string p_labels_neg = join_path(PACK_DIR, FILE_LNEG);

    auto bin_pos = read_bin(p_inputs_pos);
    auto bin_neg = read_bin(p_inputs_neg);
    auto lpos    = read_bin(p_labels_pos);
    auto lneg    = read_bin(p_labels_neg);

    // ---------- 3) Chequeo de tamaños esperados ----------
    // Stride por muestra según dtype.
    size_t stride_pos = 0, stride_neg = 0;
    bool is_bitpacked = (std::string(meta.dtype) == "bitpacked");

    if (is_bitpacked) {
        if (!meta.bytes_per_row) {
            std::cerr << "ERROR: bytes_per_row=0 en meta.json\n";
            return 1;
        }
        stride_pos = stride_neg = meta.bytes_per_row;
    } else if (std::string(meta.dtype) == "float32") {
        stride_pos = stride_neg = (size_t)feat_dim * sizeof(float);
    } else {
        // fallback (por si guardaste uint8 sin packing)
        stride_pos = stride_neg = (size_t)feat_dim;
    }

    const size_t need_pos = (size_t)meta.n_pos * stride_pos;
    const size_t need_neg = (size_t)meta.n_neg * stride_neg;
    if (bin_pos.size() != need_pos) {
        std::cerr << "ERROR: inputs_pos.bin tamaño inesperado. got="
                  << bin_pos.size() << " expect=" << need_pos << "\n";
        return 1;
    }
    if (bin_neg.size() != need_neg) {
        std::cerr << "ERROR: inputs_neg.bin tamaño inesperado. got="
                  << bin_neg.size() << " expect=" << need_neg << "\n";
        return 1;
    }
    if (lpos.size() != meta.n_pos || lneg.size() != meta.n_neg) {
        std::cerr << "ERROR: labels size no coincide.\n";
        return 1;
    }

    // ---------- 4) Imprime las primeras 10 muestras ----------
    const uint32_t to_show = 10;
    std::cout << "PRIMERAS " << to_show << " POSITIVAS\n";
    for (uint32_t i = 0; i < to_show && i < meta.n_pos; ++i) {
        const uint8_t* row = bin_pos.data() + i * stride_pos;
        std::cout << " idx=" << i << " label=" << +lpos[i] << " : ";

        if (is_bitpacked) {
            std::vector<uint8_t> bits01;
            unpack_row_to01(row, meta.bytes_per_row, feat_dim, bits01);
            print_vec_u8(bits01.data(), bits01.size(), /*max_cols=*/32);
        } else if (std::string(meta.dtype) == "float32") {
            const float* f = reinterpret_cast<const float*>(row);
            uint32_t n = (feat_dim < 16u) ? feat_dim : 16u;
            for (uint32_t k = 0; k < n; ++k) {
                std::cout.setf(std::ios::fixed);
                std::cout.precision(2);
                std::cout << f[k] << (k+1<n ? ' ' : '\n');
            }
            if (n < feat_dim) std::cout << "...\n";
        } else {
            print_vec_u8(row, feat_dim, /*max_cols=*/32);
        }
    }

    std::cout << "\nPRIMERAS " << to_show << " NEGATIVAS\n";
    for (uint32_t i = 0; i < to_show && i < meta.n_neg; ++i) {
        const uint8_t* row = bin_neg.data() + i * stride_neg;
        std::cout << " idx=" << i << " label=" << +lneg[i] << " : ";

        if (is_bitpacked) {
            std::vector<uint8_t> bits01;
            unpack_row_to01(row, meta.bytes_per_row, feat_dim, bits01);
            print_vec_u8(bits01.data(), bits01.size(), /*max_cols=*/32);
        } else if (std::string(meta.dtype) == "float32") {
            const float* f = reinterpret_cast<const float*>(row);
            uint32_t n = (feat_dim < 16u) ? feat_dim : 16u;
            for (uint32_t k = 0; k < n; ++k) {
                std::cout.setf(std::ios::fixed);
                std::cout.precision(2);
                std::cout << f[k] << (k+1<n ? ' ' : '\n');
            }
            if (n < feat_dim) std::cout << "...\n";
        } else {
            print_vec_u8(row, feat_dim, /*max_cols=*/32);
        }
    }

    return 0;
}
