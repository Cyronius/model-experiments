// tde forward pass (x86-64-v3 baseline: AVX2 + FMA). Matches qwen-space reference:
// BERT post-norm layers, int8 activations (power-of-two scale per token per 256 channels, or whole row
// when K is not a multiple of 256), 7-bit per-row weight codes, float32 attention/LayerNorm/GELU(tanh),
// mean pooling over [CLS] content [SEP], windows of `window` content tokens, token-weighted window mean,
// head (linear skip + optional GELU MLP), L2 normalisation.
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "kernels.h"
#include "tde.h"

namespace {

struct Lin { const uint8_t* W; const float* S; const float* B; int K, N; };
struct LayerW { Lin l[6]; const float *ln1w, *ln1b, *ln2w, *ln2b; };

}  // namespace

struct tde_model {
  int cfg[TDE_CFG_N];
  const void* word; const float* word_scale; const float* pos; const float* tok0;
  const float *lnew, *lneb;
  LayerW layers[16];
  int n_layers;
  const float *skipW, *bias, *W1, *b1, *W2;
  int isa;
  lin_fn lin;
};

namespace {

// ---- vector math (AVX2) -------------------------------------------------------------------------
inline __m256 exp256(__m256 x) {
  // Cephes-style expf: x = n ln2 + r, |r| <= ln2/2, degree-6 polynomial; relative error ~1e-7.
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.3f)), _mm256_set1_ps(88.3f));
  __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
  r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), r);
  __m256 y = _mm256_set1_ps(1.9875691500e-4f);
  y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.3981999507e-3f));
  y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(8.3334519073e-3f));
  y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(4.1665795894e-2f));
  y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.6666665459e-1f));
  y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(5.0000001201e-1f));
  y = _mm256_fmadd_ps(y, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
  __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

inline float hsum(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}
inline float hmax(__m256 v) {
  __m128 s = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_max_ps(s, _mm_movehl_ps(s, s));
  s = _mm_max_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}

// GELU, tanh approximation (torch approximate='tanh'); tanh(y) = 1 - 2 / (exp(2y) + 1)
void gelu_tanh(float* x, size_t n) {
  const __m256 c0 = _mm256_set1_ps(0.7978845608028654f), c1 = _mm256_set1_ps(0.044715f);
  const __m256 half = _mm256_set1_ps(0.5f), one = _mm256_set1_ps(1.0f), two = _mm256_set1_ps(2.0f);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(x + i);
    __m256 y = _mm256_mul_ps(c0, _mm256_fmadd_ps(_mm256_mul_ps(c1, v), _mm256_mul_ps(v, v), v));
    __m256 t = _mm256_sub_ps(one, _mm256_div_ps(two, _mm256_add_ps(exp256(_mm256_mul_ps(two, y)), one)));
    _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_mul_ps(half, v), _mm256_add_ps(one, t)));
  }
  for (; i < n; i++) {
    float v = x[i];
    x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
  }
}

// LayerNorm over d (multiple of 8), biased variance, in place: x = LN(x + r) when r != null.
void layernorm(float* x, const float* r, int T, int d, const float* w, const float* b, float eps) {
  for (int t = 0; t < T; t++) {
    float* p = x + (size_t)t * d;
    if (r) {
      const float* q = r + (size_t)t * d;
      for (int i = 0; i < d; i += 8) _mm256_storeu_ps(p + i, _mm256_add_ps(_mm256_loadu_ps(p + i), _mm256_loadu_ps(q + i)));
    }
    __m256 s = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 8) s = _mm256_add_ps(s, _mm256_loadu_ps(p + i));
    const float mean = hsum(s) / d;
    const __m256 mv = _mm256_set1_ps(mean);
    __m256 v = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 8) { __m256 c = _mm256_sub_ps(_mm256_loadu_ps(p + i), mv); v = _mm256_fmadd_ps(c, c, v); }
    const float inv = 1.0f / sqrtf(hsum(v) / d + eps);
    const __m256 iv = _mm256_set1_ps(inv);
    for (int i = 0; i < d; i += 8) {
      __m256 c = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(p + i), mv), iv);
      _mm256_storeu_ps(p + i, _mm256_fmadd_ps(c, _mm256_loadu_ps(w + i), _mm256_loadu_ps(b + i)));
    }
  }
}

// int8 activations: power-of-two scale per row per group (tt_qa / quant_act).
void quant_rows(const float* x, int M, int K, int gs, int8_t* q, float* sc, int32_t* asum) {
  const int G = K / gs;
  const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  for (int m = 0; m < M; m++) {
    for (int g = 0; g < G; g++) {
      const float* p = x + (size_t)m * K + (size_t)g * gs;
      __m256 mx = _mm256_setzero_ps();
      for (int i = 0; i < gs; i += 8) mx = _mm256_max_ps(mx, _mm256_and_ps(_mm256_loadu_ps(p + i), absmask));
      float a = hmax(mx);
      if (a < 1e-5f) a = 1e-5f;
      int k; float f = frexpf(a, &k);
      const int e = k - 7 + (2.0f * f > 127.0f / 64.0f ? 1 : 0);
      const __m256 vi = _mm256_set1_ps(ldexpf(1.0f, -e));
      int8_t* o = q + (size_t)m * K + (size_t)g * gs;
      __m256i sum = _mm256_setzero_si256();
      for (int i = 0; i < gs; i += 8) {
        __m256i r = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + i), vi));  // nearest even
        r = _mm256_max_epi32(_mm256_min_epi32(r, _mm256_set1_epi32(127)), _mm256_set1_epi32(-128));
        sum = _mm256_add_epi32(sum, r);
        alignas(32) int32_t t[8];
        _mm256_store_si256((__m256i*)t, r);
        for (int j = 0; j < 8; j++) o[i + j] = (int8_t)t[j];
      }
      sc[m * G + g] = ldexpf(1.0f, e);
      __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
      s4 = _mm_hadd_epi32(s4, s4); s4 = _mm_hadd_epi32(s4, s4);
      asum[m * G + g] = _mm_cvtsi128_si32(s4);
    }
  }
}

inline int group_of(int K) { return (K > 256 && K % 256 == 0) ? 256 : K; }

template <class T> struct Buf {
  T* p = nullptr;
  void resize(size_t n, T v = T()) { p = (T*)aligned_alloc(64, (n * sizeof(T) + 63) / 64 * 64); for (size_t i = 0; i < n; i++) p[i] = v; }
  T* data() { return p; }
  T& operator[](size_t i) { return p[i]; }
  ~Buf() { free(p); }
};
struct Scratch {
  Buf<float> x, qkv, att, h, tmp, kt, prob;
  Buf<int8_t> q;
  Buf<float> sc;
  Buf<int32_t> as;
};

void linear(const tde_model* m, const Lin& L, const float* x, int T, int Tp, Scratch& s, float* out) {
  const int gs = group_of(L.K), G = L.K / gs;
  memset(s.q.data(), 0, (size_t)Tp * L.K);
  quant_rows(x, T, L.K, gs, s.q.data(), s.sc.data(), s.as.data());
  for (int t = T; t < Tp; t++) for (int g = 0; g < G; g++) { s.sc[t * G + g] = 0.f; s.as[t * G + g] = 0; }
  m->lin(s.q.data(), s.sc.data(), s.as.data(), Tp, L.K, gs, L.W, L.S, L.B, L.N, out);
}

// One window: ids (with [CLS]/[SEP]) of length T -> pooled d-vector.
void window(const tde_model* m, const int32_t* ids, int T, Scratch& s, float* pooled) {
  const int d = m->cfg[TDE_D], H = m->cfg[TDE_HEADS], hd = d / H, F = m->cfg[TDE_FFN];
  const int Tp = (T + 7) / 8 * 8;
  const float eps = 1e-12f;
  float* x = s.x.data();
  // embeddings
  for (int t = 0; t < T; t++) {
    float* p = x + (size_t)t * d;
    const float* pp = m->pos + (size_t)t * d;
    if (m->cfg[TDE_EMB_INT8]) {
      const int8_t* w = (const int8_t*)m->word + (size_t)ids[t] * d;
      const float sc = m->word_scale[ids[t]];
      for (int i = 0; i < d; i++) p[i] = (float)w[i] * sc + pp[i] + m->tok0[i];
    } else {
      const uint16_t* w = (const uint16_t*)m->word + (size_t)ids[t] * d;
      for (int i = 0; i < d; i += 8) {
        __m256 v = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w + i)));
        _mm256_storeu_ps(p + i, _mm256_add_ps(_mm256_add_ps(v, _mm256_loadu_ps(pp + i)), _mm256_loadu_ps(m->tok0 + i)));
      }
    }
  }
  layernorm(x, nullptr, T, d, m->lnew, m->lneb, eps);
  float* Q = s.qkv.data();
  float* K = Q + (size_t)Tp * d;
  float* V = K + (size_t)Tp * d;
  float* A = s.att.data();
  float* Ht = s.h.data();
  float* O = s.tmp.data();
  float* Kt = s.kt.data();
  float* P = s.prob.data();
  const float scale = 1.0f / sqrtf((float)hd);
  for (int li = 0; li < m->n_layers; li++) {
    const LayerW& L = m->layers[li];
    linear(m, L.l[0], x, T, Tp, s, Q);
    linear(m, L.l[1], x, T, Tp, s, K);
    linear(m, L.l[2], x, T, Tp, s, V);
    const int T8 = (T + 7) / 8 * 8;
    for (int h = 0; h < H; h++) {
      // Kt[c][j] = K[j][h*hd + c], padded columns zero
      for (int c = 0; c < hd; c++) {
        float* row = Kt + (size_t)c * T8;
        for (int j = 0; j < T; j++) row[j] = K[(size_t)j * d + h * hd + c];
        for (int j = T; j < T8; j++) row[j] = 0.f;
      }
      for (int i = 0; i < T; i++) {
        float* pr = P + (size_t)i * T8;
        for (int j = 0; j < T8; j += 8) _mm256_storeu_ps(pr + j, _mm256_setzero_ps());
        const float* qi = Q + (size_t)i * d + h * hd;
        for (int c = 0; c < hd; c++) {
          const __m256 qc = _mm256_set1_ps(qi[c] * 1.0f);
          const float* kr = Kt + (size_t)c * T8;
          for (int j = 0; j < T8; j += 8) _mm256_storeu_ps(pr + j, _mm256_fmadd_ps(qc, _mm256_loadu_ps(kr + j), _mm256_loadu_ps(pr + j)));
        }
        // softmax over j < T of pr * scale
        float mx = -INFINITY;
        for (int j = 0; j < T; j++) { pr[j] *= scale; mx = pr[j] > mx ? pr[j] : mx; }
        for (int j = T; j < T8; j++) pr[j] = -INFINITY;
        const __m256 mv = _mm256_set1_ps(mx);
        __m256 sum = _mm256_setzero_ps();
        for (int j = 0; j < T8; j += 8) {
          __m256 e = exp256(_mm256_sub_ps(_mm256_loadu_ps(pr + j), mv));
          if (j + 8 > T) {  // zero the padded lanes
            alignas(32) float t[8]; _mm256_store_ps(t, e);
            for (int z = T - j; z < 8; z++) t[z] = 0.f;
            e = _mm256_load_ps(t);
          }
          _mm256_storeu_ps(pr + j, e); sum = _mm256_add_ps(sum, e);
        }
        const float inv = 1.0f / hsum(sum);
        // A[i][h*hd..] = sum_j p_ij V[j]
        __m256 o[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (int j = 0; j < T; j++) {
          const __m256 pj = _mm256_set1_ps(pr[j] * inv);
          const float* vj = V + (size_t)j * d + h * hd;
          for (int c = 0; c < hd / 8; c++) o[c] = _mm256_fmadd_ps(pj, _mm256_loadu_ps(vj + 8 * c), o[c]);
        }
        for (int c = 0; c < hd / 8; c++) _mm256_storeu_ps(A + (size_t)i * d + h * hd + 8 * c, o[c]);
      }
    }
    linear(m, L.l[3], A, T, Tp, s, O);
    layernorm(x, O, T, d, L.ln1w, L.ln1b, eps);
    linear(m, L.l[4], x, T, Tp, s, Ht);
    gelu_tanh(Ht, (size_t)T * F);
    linear(m, L.l[5], Ht, T, Tp, s, O);
    layernorm(x, O, T, d, L.ln2w, L.ln2b, eps);
  }
  for (int i = 0; i < d; i++) pooled[i] = 0.f;
  for (int t = 0; t < T; t++) for (int i = 0; i < d; i++) pooled[i] += x[(size_t)t * d + i];
  for (int i = 0; i < d; i++) pooled[i] /= (float)T;
}

// out[n] += sum_i x[i] * W[i][n]  (W row-major [in][n], n multiple of 8)
void gemv_acc(const float* x, int in, const float* W, int n, float* out) {
  for (int i = 0; i < in; i++) {
    const __m256 xi = _mm256_set1_ps(x[i]);
    const float* w = W + (size_t)i * n;
    for (int j = 0; j < n; j += 8) _mm256_storeu_ps(out + j, _mm256_fmadd_ps(xi, _mm256_loadu_ps(w + j), _mm256_loadu_ps(out + j)));
  }
}

void head(const tde_model* m, const float* f, Buf<float>& hid, float* out, float* conf) {
  const int d = m->cfg[TDE_D], O = m->cfg[TDE_OUT], Hh = m->cfg[TDE_HEAD_HIDDEN];
  memcpy(out, m->bias, sizeof(float) * O);
  gemv_acc(f, d, m->skipW, O, out);
  if (Hh) {
    memcpy(hid.data(), m->b1, sizeof(float) * Hh);
    gemv_acc(f, d, m->W1, Hh, hid.data());
    gelu_tanh(hid.data(), Hh);
    gemv_acc(hid.data(), Hh, m->W2, O, out);
  }
  double ss = 0;
  for (int j = 0; j < O; j++) ss += (double)out[j] * out[j];
  const float nrm = (float)sqrt(ss);
  if (conf) *conf = nrm;
  const float inv = nrm > 0 ? 1.0f / nrm : 0.f;
  for (int j = 0; j < O; j++) out[j] *= inv;
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) int tde_cpu_has_vnni(void) {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni");
}
__attribute__((visibility("default"))) int tde_cpu_has_avx2(void) {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
}

__attribute__((visibility("default"))) int tde_n_ptrs(const int32_t* cfg) { return 6 + 22 * cfg[TDE_LAYERS] + 5; }

__attribute__((visibility("default"))) tde_model* tde_load(const int32_t* cfg, const void* const* p) {
  if (!tde_cpu_has_avx2()) return nullptr;
  if (cfg[TDE_LAYERS] > 16) return nullptr;
  tde_model* m = (tde_model*)calloc(1, sizeof(tde_model));
  memcpy(m->cfg, cfg, sizeof(m->cfg));
  const int d = cfg[TDE_D], F = cfg[TDE_FFN];
  if (d % 32 || F % 32 || cfg[TDE_OUT] % 8 || cfg[TDE_HEAD_HIDDEN] % 8 || (d / cfg[TDE_HEADS]) % 8) { free(m); return nullptr; }
  m->word = p[0]; m->word_scale = (const float*)p[1]; m->pos = (const float*)p[2]; m->tok0 = (const float*)p[3];
  m->lnew = (const float*)p[4]; m->lneb = (const float*)p[5];
  const int KN[6][2] = {{d, d}, {d, d}, {d, d}, {d, d}, {d, F}, {F, d}};
  for (int l = 0; l < cfg[TDE_LAYERS]; l++) {
    const int b = 6 + 22 * l;
    LayerW L;
    for (int j = 0; j < 6; j++)
      L.l[j] = Lin{(const uint8_t*)p[b + 3 * j], (const float*)p[b + 3 * j + 1], (const float*)p[b + 3 * j + 2], KN[j][0], KN[j][1]};
    L.ln1w = (const float*)p[b + 18]; L.ln1b = (const float*)p[b + 19];
    L.ln2w = (const float*)p[b + 20]; L.ln2b = (const float*)p[b + 21];
    m->layers[m->n_layers++] = L;
  }
  const int hb = 6 + 22 * cfg[TDE_LAYERS];
  m->skipW = (const float*)p[hb]; m->bias = (const float*)p[hb + 1];
  m->W1 = (const float*)p[hb + 2]; m->b1 = (const float*)p[hb + 3]; m->W2 = (const float*)p[hb + 4];
  const char* force = getenv("TDE_FORCE_AVX2");
  const bool forced = force && force[0] && force[0] != '0';
  m->isa = (!forced && tde_cpu_has_vnni()) ? 2 : 1;
  m->lin = m->isa == 2 ? lin_vnni : lin_avx2;
  return m;
}

__attribute__((visibility("default"))) void tde_free(tde_model* m) { free(m); }
__attribute__((visibility("default"))) int tde_isa(const tde_model* m) { return m->isa; }

__attribute__((visibility("default"))) int tde_encode(const tde_model* m, const int32_t* ids_flat, const int64_t* offsets, int n_docs,
               float* out, float* conf, float* feats) {
  const int d = m->cfg[TDE_D], F = m->cfg[TDE_FFN], W = m->cfg[TDE_WINDOW], O = m->cfg[TDE_OUT];
  const int maxT = W + 2, Tp = (maxT + 7) / 8 * 8, maxw = m->cfg[TDE_MAX_WINDOWS];
  if (maxT > m->cfg[TDE_MAX_POS]) return -1;
  Scratch s;
  s.x.resize((size_t)Tp * d); s.qkv.resize((size_t)3 * Tp * d); s.att.resize((size_t)Tp * d, 0.f);
  s.h.resize((size_t)Tp * F); s.tmp.resize((size_t)Tp * (F > d ? F : d));
  s.kt.resize((size_t)(d / m->cfg[TDE_HEADS]) * Tp); s.prob.resize((size_t)Tp * Tp);
  s.q.resize((size_t)Tp * F); s.sc.resize((size_t)Tp * 8); s.as.resize((size_t)Tp * 8);
  Buf<float> pooled, acc, hid;
  pooled.resize(d); acc.resize(d); hid.resize(m->cfg[TDE_HEAD_HIDDEN] ? m->cfg[TDE_HEAD_HIDDEN] : 1);
  Buf<int32_t> ids;
  ids.resize(maxT);
  for (int i = 0; i < n_docs; i++) {
    const int32_t* doc = ids_flat + offsets[i];
    int n = (int)(offsets[i + 1] - offsets[i]);
    if (n > W * maxw) n = W * maxw;
    for (int j = 0; j < d; j++) acc[j] = 0.f;
    float wsum = 0.f;
    int start = 0;
    do {
      const int c = (n - start) < W ? (n - start) : W;
      ids[0] = m->cfg[TDE_CLS];
      for (int j = 0; j < c; j++) {
        const int32_t t = doc[start + j];
        if (t < 0 || t >= m->cfg[TDE_VOCAB]) return -2;
        ids[1 + j] = t;
      }
      ids[1 + c] = m->cfg[TDE_SEP];
      const int T = c + 2;
      window(m, ids.data(), T, s, pooled.data());
      for (int j = 0; j < d; j++) acc[j] += (float)T * pooled[j];
      wsum += (float)T;
      start += W;
    } while (start < n);
    for (int j = 0; j < d; j++) acc[j] /= wsum;
    if (feats) memcpy(feats + (size_t)i * d, acc.data(), sizeof(float) * d);
    head(m, acc.data(), hid, out + (size_t)i * O, conf ? conf + i : nullptr);
  }
  return 0;
}

}  // extern "C"
