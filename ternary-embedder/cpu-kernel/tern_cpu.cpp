// Packed ternary linear layers for AVX512-VNNI CPUs. Matches tern-embed/model.py TernLinear (quant=True):
// int8 activations with a power-of-two scale per 256-channel group (whole row when K is not a multiple of 256),
// ternary codes with one bf16-rounded scale per output row.
#include <immintrin.h>
#include <cmath>
#include <cstdint>

extern "C" {

void quant_rows(const float* x, int M, int K, int gs, int8_t* q, float* sc, int32_t* asum) {
  int G = K / gs;
  #pragma omp parallel for schedule(static) if (M > 16)
  for (int m = 0; m < M; m++) {
    for (int g = 0; g < G; g++) {
      const float* p = x + (size_t)m * K + (size_t)g * gs;
      __m512 mx = _mm512_setzero_ps();
      for (int i = 0; i < gs; i += 16) mx = _mm512_max_ps(mx, _mm512_abs_ps(_mm512_loadu_ps(p + i)));
      float a = _mm512_reduce_max_ps(mx);
      if (a < 1e-5f) a = 1e-5f;
      int k; float f = frexpf(a, &k);
      int e = k - 7 + (2.0f * f > 127.0f / 64.0f ? 1 : 0);
      float inv = ldexpf(1.0f, -e);
      __m512 vi = _mm512_set1_ps(inv);
      int8_t* o = q + (size_t)m * K + (size_t)g * gs;
      __m512i sum = _mm512_setzero_si512();
      for (int i = 0; i < gs; i += 16) {
        __m512i r = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(p + i), vi));  // round to nearest even
        r = _mm512_max_epi32(_mm512_min_epi32(r, _mm512_set1_epi32(127)), _mm512_set1_epi32(-128));
        sum = _mm512_add_epi32(sum, r);
        _mm_storeu_si128((__m128i*)(o + i), _mm512_cvtepi32_epi8(r));
      }
      sc[m * G + g] = ldexpf(1.0f, e);
      asum[m * G + g] = _mm512_reduce_add_epi32(sum);
    }
  }
}

}  // extern C

static inline __m128i reduce4(__m512i a0, __m512i a1, __m512i a2, __m512i a3) {
  auto r = [](__m512i a) {
    __m256i y = _mm256_add_epi32(_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64(a, 1));
    return _mm_add_epi32(_mm256_castsi256_si128(y), _mm256_extracti128_si256(y, 1));
  };
  return _mm_hadd_epi32(_mm_hadd_epi32(r(a0), r(a1)), _mm_hadd_epi32(r(a2), r(a3)));
}

// MODE 0: W is uint8 codes+1 (N x K). MODE 1: W is 2-bit packed codes+1, 32 bytes per 128 weights
// (byte j of a block holds weights j, j+32, j+64, j+96 in bit pairs 0..3).
template <int MODE>
static void lin(const int8_t* q, const float* sc, const int32_t* asum, int M, int K, int gs, const uint8_t* W,
                const float* ws, const float* bias, int N, float* out) {
  int G = K / gs;
  size_t rb = MODE == 0 ? (size_t)K : (size_t)K / 4;
  const __m256i m3 = _mm256_set1_epi8(3);
  #pragma omp parallel for schedule(static)
  for (int n = 0; n < N; n += 4) {
    for (int m = 0; m < M; m++) {
      const int8_t* a = q + (size_t)m * K;
      float r[4] = {0, 0, 0, 0};
      for (int g = 0; g < G; g++) {
        __m512i acc[4] = {_mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512()};
        if (MODE == 0) {
          for (int k = g * gs; k < (g + 1) * gs; k += 64) {
            __m512i av = _mm512_loadu_si512(a + k);
            for (int j = 0; j < 4; j++)
              acc[j] = _mm512_dpbusd_epi32(acc[j], _mm512_loadu_si512(W + (n + j) * rb + k), av);
          }
        } else {
          for (int k = g * gs; k < (g + 1) * gs; k += 128) {
            __m512i a0 = _mm512_loadu_si512(a + k), a1 = _mm512_loadu_si512(a + k + 64);
            for (int j = 0; j < 4; j++) {
              __m256i p = _mm256_loadu_si256((const __m256i*)(W + (n + j) * rb + k / 4));
              __m256i t0 = _mm256_and_si256(p, m3);
              __m256i t1 = _mm256_and_si256(_mm256_srli_epi16(p, 2), m3);
              __m256i t2 = _mm256_and_si256(_mm256_srli_epi16(p, 4), m3);
              __m256i t3 = _mm256_and_si256(_mm256_srli_epi16(p, 6), m3);
              __m512i w0 = _mm512_inserti64x4(_mm512_castsi256_si512(t0), t1, 1);
              __m512i w1 = _mm512_inserti64x4(_mm512_castsi256_si512(t2), t3, 1);
              acc[j] = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(acc[j], w0, a0), w1, a1);
            }
          }
        }
        alignas(16) int32_t s[4];
        _mm_store_si128((__m128i*)s, reduce4(acc[0], acc[1], acc[2], acc[3]));
        float sg = sc[m * G + g]; int as = asum[m * G + g];
        for (int j = 0; j < 4; j++) r[j] += sg * (float)(s[j] - as);
      }
      for (int j = 0; j < 4; j++) out[(size_t)m * N + n + j] = r[j] * ws[n + j] + bias[n + j];
    }
  }
}

extern "C" {
void lin_i8(const int8_t* q, const float* sc, const int32_t* asum, int M, int K, int gs, const uint8_t* W,
            const float* ws, const float* bias, int N, float* out) { lin<0>(q, sc, asum, M, K, gs, W, ws, bias, N, out); }
void lin_p2(const int8_t* q, const float* sc, const int32_t* asum, int M, int K, int gs, const uint8_t* W,
            const float* ws, const float* bias, int N, float* out) { lin<1>(q, sc, asum, M, K, gs, W, ws, bias, N, out); }
}

// ---- v2: output channels across SIMD lanes (no horizontal reductions), 8 rows x 32 outputs register tile. ----
// MODE 0: W[ntile][k4][32 n][4 k] uint8 (codes+1). MODE 1: W[ntile][k/16][32 n][4 bytes], byte b bit pair s = k offset 4s+b.
// M must be a multiple of 8, N a multiple of 32.
template <int MODE>
static void lin2(const int8_t* q, const float* sc, const int32_t* asum, int M, int K, int gs, const uint8_t* W,
                 const float* ws, const float* bias, int N, float* out) {
  int G = K / gs;
  const __m512i m3 = _mm512_set1_epi32(0x03030303);
  #pragma omp parallel for schedule(static)
  for (int nt = 0; nt < N / 32; nt++) {
    const uint8_t* Wt = W + (size_t)nt * 32 * K / (MODE == 0 ? 1 : 4);
    for (int m0 = 0; m0 < M; m0 += 8) {
      for (int g = 0; g < G; g++) {
        __m512i acc[8][2];
        for (int i = 0; i < 8; i++) acc[i][0] = acc[i][1] = _mm512_setzero_si512();
        if (MODE == 0) {
          for (int k4 = g * gs / 4; k4 < (g + 1) * gs / 4; k4++) {
            __m512i w0 = _mm512_loadu_si512(Wt + (size_t)k4 * 128), w1 = _mm512_loadu_si512(Wt + (size_t)k4 * 128 + 64);
            for (int i = 0; i < 8; i++) {
              __m512i a = _mm512_set1_epi32(*(const int32_t*)(q + (size_t)(m0 + i) * K + k4 * 4));
              acc[i][0] = _mm512_dpbusd_epi32(acc[i][0], w0, a);
              acc[i][1] = _mm512_dpbusd_epi32(acc[i][1], w1, a);
            }
          }
        } else {
          for (int c = g * gs / 16; c < (g + 1) * gs / 16; c++) {
            __m512i wl = _mm512_loadu_si512(Wt + (size_t)c * 128), wh = _mm512_loadu_si512(Wt + (size_t)c * 128 + 64);
            #define STEP(S) { \
              __m512i w0 = _mm512_and_si512(_mm512_srli_epi32(wl, 2 * S), m3), w1 = _mm512_and_si512(_mm512_srli_epi32(wh, 2 * S), m3); \
              for (int i = 0; i < 8; i++) { \
                __m512i a = _mm512_set1_epi32(*(const int32_t*)(q + (size_t)(m0 + i) * K + c * 16 + S * 4)); \
                acc[i][0] = _mm512_dpbusd_epi32(acc[i][0], w0, a); \
                acc[i][1] = _mm512_dpbusd_epi32(acc[i][1], w1, a); } }
            STEP(0) STEP(1) STEP(2) STEP(3)
            #undef STEP
          }
        }
        for (int i = 0; i < 8; i++) {
          int m = m0 + i; float sg = sc[m * G + g]; __m512i as = _mm512_set1_epi32(asum[m * G + g]);
          for (int h = 0; h < 2; h++) {
            float* o = out + (size_t)m * N + nt * 32 + h * 16;
            __m512 f = _mm512_mul_ps(_mm512_set1_ps(sg), _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[i][h], as)));
            if (g > 0) f = _mm512_add_ps(f, _mm512_loadu_ps(o));
            if (g == G - 1)
              f = _mm512_fmadd_ps(f, _mm512_loadu_ps(ws + nt * 32 + h * 16), _mm512_loadu_ps(bias + nt * 32 + h * 16));
            _mm512_storeu_ps(o, f);
          }
        }
      }
    }
  }
}
extern "C" {
void lin2_i8(const int8_t* q, const float* sc, const int32_t* asum, int M, int K, int gs, const uint8_t* W,
             const float* ws, const float* bias, int N, float* out) { lin2<0>(q, sc, asum, M, K, gs, W, ws, bias, N, out); }
void lin2_p2(const int8_t* q, const float* sc, const int32_t* asum, int M, int K, int gs, const uint8_t* W,
             const float* ws, const float* bias, int N, float* out) { lin2<1>(q, sc, asum, M, K, gs, W, ws, bias, N, out); }
}
