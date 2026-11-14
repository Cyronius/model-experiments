// tde: torch-free document embedder (Plan A). C ABI, single-threaded per call, no OpenMP.
#pragma once
#include <stdint.h>

// Integer config, in this order (TDE_CFG_N entries).
enum {
  TDE_VOCAB, TDE_D, TDE_HEADS, TDE_FFN, TDE_LAYERS, TDE_MAX_POS, TDE_WINDOW, TDE_OUT,
  TDE_HEAD_HIDDEN,   // 0: linear head (skip + bias only)
  TDE_EMB_INT8,      // 1: word table int8 + per-row float scale; 0: float16 table (uint16 bits)
  TDE_CLS, TDE_SEP, TDE_MAX_WINDOWS,
  TDE_CFG_N
};

// Pointer table (tde_n_ptrs entries), every array C-contiguous:
//   0 word        int8 [vocab][d] or uint16 (fp16 bits) [vocab][d]
//   1 word_scale  float [vocab] (null when the table is fp16)
//   2 pos         float [max_pos][d]
//   3 tok_type0   float [d]
//   4 ln_e_w, 5 ln_e_b  float [d]
//   per layer l, base = 6 + 22*l, for linear j in (q, k, v, o, fc1, fc2):
//     base+3j   W  uint8, tiled [N/32][K/4][32][4], value = code + 64, code in [-64, 63]
//     base+3j+1 S  float [N] per-output-row scale
//     base+3j+2 B  float [N] bias
//   base+18 ln1_w, base+19 ln1_b, base+20 ln2_w, base+21 ln2_b
//   head, base = 6 + 22*layers (standardisation folded in):
//     +0 skipW float [d][out], +1 bias float [out],
//     +2 W1 float [d][hidden], +3 b1 float [hidden], +4 W2 float [hidden][out] (null when hidden = 0)
#ifdef __cplusplus
extern "C" {
#endif

typedef struct tde_model tde_model;

int tde_n_ptrs(const int32_t* cfg);
tde_model* tde_load(const int32_t* cfg, const void* const* ptrs);
void tde_free(tde_model* m);

// 2 = AVX-512 VNNI path, 1 = AVX2 path. Chosen once at load; env TDE_FORCE_AVX2=1 forces 1.
int tde_isa(const tde_model* m);
int tde_cpu_has_vnni(void);
int tde_cpu_has_avx2(void);

// Encode n_docs documents. ids_flat holds content-token ids (no [CLS]/[SEP]); doc i is
// ids_flat[offsets[i] .. offsets[i+1]), at most max_windows * window ids (extra ids are ignored).
// out: [n_docs][out] unit vectors; conf: [n_docs] norm of the head output before normalisation
// (may be null); feats: [n_docs][d] pooled backbone features (may be null). Returns 0 on success.
int tde_encode(const tde_model* m, const int32_t* ids_flat, const int64_t* offsets, int n_docs,
               float* out, float* conf, float* feats);

#ifdef __cplusplus
}
#endif
