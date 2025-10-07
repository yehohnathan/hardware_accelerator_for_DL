#include "forward_fw.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <cstdio>   // fprintf, stderr

// --- helpers mínimos de JSON (ad hoc para meta.json) ---
namespace {
static inline std::string trim(std::string s){
  auto issp=[](unsigned char c){return std::isspace(c);};
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [&](unsigned char c){return !issp(c);}));
  s.erase(std::find_if(s.rbegin(), s.rend(), [&](unsigned char c){return !issp(c);} ).base(), s.end());
  return s;
}
static bool find_block(const std::string& t, const std::string& key, size_t& s, size_t& e){
  std::string pat="\""+key+"\""; size_t k=t.find(pat); if(k==std::string::npos) return false;
  size_t b=t.find('{',k); if(b==std::string::npos) return false; int d=0;
  for(size_t i=b;i<t.size();++i){ if(t[i]=='{') d++; else if(t[i]=='}'){ d--; if(d==0){s=b; e=i; return true;}}}
  return false;
}
static bool extract_string(const std::string& t, const std::string& key, std::string& out){
  std::string pat="\""+key+"\""; size_t k=t.find(pat); if(k==std::string::npos) return false;
  size_t q1=t.find('"', t.find(':',k)); if(q1==std::string::npos) return false;
  size_t q2=t.find('"', q1+1); if(q2==std::string::npos) return false;
  out=t.substr(q1+1, q2-(q1+1)); return true;
}
static bool extract_int(const std::string& t, const std::string& key, int& out){
  std::string pat="\""+key+"\""; size_t k=t.find(pat); if(k==std::string::npos) return false;
  size_t c=t.find(':',k); if(c==std::string::npos) return false; size_t p=c+1; while(p<t.size()&&std::isspace((unsigned char)t[p])) ++p;
  size_t e=p; while(e<t.size()&&(std::isdigit((unsigned char)t[e])||t[e]=='-')) ++e; if(e==p) return false;
  out=std::stoi(t.substr(p,e-p)); return true;
}
static bool extract_bool(const std::string& t, const std::string& key, bool& out){
  std::string pat="\""+key+"\""; size_t k=t.find(pat); if(k==std::string::npos) return false;
  size_t c=t.find(':',k); if(c==std::string::npos) return false; size_t p=c+1; while(p<t.size()&&std::isspace((unsigned char)t[p])) ++p;
  if(t.compare(p,4,"true")==0){out=true; return true;} if(t.compare(p,5,"false")==0){out=false; return true;} return false;
}
static bool extract_string_in_object(const std::string& t,const std::string& obj,const std::string& key,std::string& out){
  size_t s=0,e=0; if(!find_block(t,obj,s,e)) return false; std::string blk=t.substr(s,e-s+1); return extract_string(blk,key,out);
}
} // ns

std::string join_path(const std::string& dir, const std::string& file){
  if(dir.empty()) return file;
  if(dir.back()=='/'||dir.back()=='\\') return dir+file;
#ifdef _WIN32
  return dir + "\\" + file;
#else
  return dir + "/" + file;
#endif
}

bool load_meta(const std::string& dir, MetaFF& m){
  const std::string path = join_path(dir, "meta.json");
  std::ifstream f(path);
  if(!f){
    std::fprintf(stderr, "[meta] No se pudo abrir: %s\n", path.c_str());
    return false;
  }
  std::stringstream ss; ss<<f.rdbuf(); const std::string txt=ss.str();

  // Campos básicos
  if(!extract_string(txt,"format",m.format)) m.format.clear();
  if(!extract_string(txt,"dtype",m.dtype))   m.dtype.clear();
  extract_int   (txt,"input_dim",m.input_dim);
  extract_int   (txt,"batch_size",m.batch_size);
  extract_bool  (txt,"drop_last",m.drop_last);
  extract_int   (txt,"n_pos",m.n_pos);
  extract_int   (txt,"n_neg",m.n_neg);
  extract_int   (txt,"n_batches_pos",m.n_batches_pos);
  extract_int   (txt,"n_batches_neg",m.n_batches_neg);

  // Archivos
  extract_string_in_object(txt,"files","inputs_pos",m.f_inputs_pos);
  extract_string_in_object(txt,"files","inputs_neg",m.f_inputs_neg);
  extract_string_in_object(txt,"files","labels_pos",m.f_labels_pos);
  extract_string_in_object(txt,"files","labels_neg",m.f_labels_neg);

  // Bitpacked
  extract_int   (txt,"word_bits",m.word_bits);
  extract_int   (txt,"bytes_per_row",m.bytes_per_row);
  extract_int   (txt,"features_padded",m.features_padded);
  extract_string(txt,"bitorder_in_byte",m.bitorder_in_byte);
  extract_string(txt,"endianness",m.endianness);
  extract_bool  (txt,"row_major",m.row_major);

  // Imagen/tokens
  extract_int   (txt,"rows",m.rows);
  extract_int   (txt,"cols",m.cols);
  extract_int   (txt,"token_dim",m.token_dim);
  extract_string(txt,"where_tokens",m.where_tokens);

  // Validaciones mínimas (sin throw)
  if(m.dtype.empty()){
    std::fprintf(stderr, "[meta] falta 'dtype'\n"); return false;
  }
  if(m.input_dim <= 0){
    std::fprintf(stderr, "[meta] 'input_dim' invalido\n"); return false;
  }
  if(m.dtype=="bitpacked"){
    if(m.bytes_per_row <= 0 || (m.word_bits!=32 && m.word_bits!=64)){
      std::fprintf(stderr, "[meta] bitpacked: 'bytes_per_row'/'word_bits' invalidos\n");
      return false;
    }
    if(m.bitorder_in_byte!="lsb0" || m.endianness!="little"){
      std::fprintf(stderr, "[meta] Se espera bitorder=lzb0 y endianness=little\n");
      return false;
    }
  }
  return true;
}

bool load_inputs_raw(const std::string& path, size_t n, size_t bytes_per_row, std::vector<uint8_t>& raw){
  const size_t total = n*bytes_per_row; raw.resize(total);
  std::ifstream f(path, std::ios::binary);
  if(!f){
    std::fprintf(stderr, "[data] No se pudo abrir: %s\n", path.c_str());
    return false;
  }
  f.read(reinterpret_cast<char*>(raw.data()), total);
  if(!f.good()){
    std::fprintf(stderr, "[data] Lectura incompleta: %s\n", path.c_str());
    return false;
  }
  return true;
}

bool load_labels_u8(const std::string& path, size_t n, std::vector<uint8_t>& y){
  y.resize(n);
  std::ifstream f(path, std::ios::binary);
  if(!f){
    std::fprintf(stderr, "[labels] No se pudo abrir: %s\n", path.c_str());
    return false;
  }
  f.read(reinterpret_cast<char*>(y.data()), n);
  if(!f.good()){
    std::fprintf(stderr, "[labels] Lectura incompleta: %s\n", path.c_str());
    return false;
  }
  return true;
}

// ---- Conversión crudo -> palabras LE32 ----
void raw_to_words_le32(const std::vector<uint8_t>& raw, std::vector<u32>& words){
  const size_t W = raw.size()/4; words.resize(W);
  for(size_t i=0;i<W;++i){
    uint32_t v =  (uint32_t)raw[4*i+0]
                | ((uint32_t)raw[4*i+1] << 8)
                | ((uint32_t)raw[4*i+2] << 16)
                | ((uint32_t)raw[4*i+3] << 24);
    words[i] = (u32)v;
  }
}

// ---- Desempaquetado LSB0 a {0,1} ----
void unpack_dataset_bits_lsb0(const std::vector<uint8_t>& raw, int n, int d, int bytes_per_row, std::vector<uint8_t>& x01){
  x01.assign((size_t)n*(size_t)d, 0u);
  for(int r=0;r<n;++r){
    const uint8_t* row = &raw[(size_t)r*(size_t)bytes_per_row];
    uint8_t* out = &x01[(size_t)r*(size_t)d];
    int out_i=0;
    for(int b=0;b<bytes_per_row && out_i<d;++b){
      uint8_t byte = row[b];
      for(int k=0;k<8 && out_i<d;++k) out[out_i++] = (byte>>k)&1u; // LSB0
    }
  }
}

void map01_to_pm1(const std::vector<uint8_t>& x01, std::vector<int8_t>& xpm1){
  xpm1.resize(x01.size());
  for(size_t i=0;i<x01.size();++i) xpm1[i] = x01[i] ? 1 : -1;
}
