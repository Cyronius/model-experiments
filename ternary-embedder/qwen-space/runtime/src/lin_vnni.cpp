// AVX-512 VNNI linear kernel. Compiled with -mavx512f -mavx512bw -mavx512vl -mavx512vnni; called only
// after __builtin_cpu_supports checks. Tile: 8 rows x 32 outputs, outputs across SIMD lanes.
#include <immintrin.h>
#include "kernels.h"

void lin_vnni(const int8_t* q, const float* sc, const int32_t* asum, int M, int K, int gs,
              const uint8_t* W, const float* ws, const float* b, int N, float* out) {
  const int G = K / gs;
  for (int nt = 0; nt < N / 32; nt++) {
    const uint8_t* Wt = W + (size_t)nt * 32 * K;
    const __m512 ws0 = _mm512_loadu_ps(ws + nt * 32), ws1 = _mm512_loadu_ps(ws + nt * 32 + 16);
    const __m512 b0 = _mm512_loadu_ps(b + nt * 32), b1 = _mm512_loadu_ps(b + nt * 32 + 16);
    for (int m0 = 0; m0 < M; m0 += 8) {
      __m512 r[8][2];
      for (int i = 0; i < 8; i++) r[i][0] = r[i][1] = _mm512_setzero_ps();
      for (int g = 0; g < G; g++) {
        __m512i acc[8][2];
        for (int i = 0; i < 8; i++) acc[i][0] = acc[i][1] = _mm512_setzero_si512();
        for (int k4 = g * gs / 4; k4 < (g + 1) * gs / 4; k4++) {
          const __m512i w0 = _mm512_loadu_si512(Wt + (size_t)k4 * 128);
          const __m512i w1 = _mm512_loadu_si512(Wt + (size_t)k4 * 128 + 64);
          for (int i = 0; i < 8; i++) {
            const __m512i a = _mm512_set1_epi32(*(const int32_t*)(q + (size_t)(m0 + i) * K + k4 * 4));
            acc[i][0] = _mm512_dpbusd_epi32(acc[i][0], w0, a);
            acc[i][1] = _mm512_dpbusd_epi32(acc[i][1], w1, a);
          }
        }
        for (int i = 0; i < 8; i++) {
          const int m = m0 + i;
          const __m512 sg = _mm512_set1_ps(sc[m * G + g]);
          const __m512i off = _mm512_set1_epi32(64 * asum[m * G + g]);
          r[i][0] = _mm512_fmadd_ps(sg, _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[i][0], off)), r[i][0]);
          r[i][1] = _mm512_fmadd_ps(sg, _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[i][1], off)), r[i][1]);
        }
      }
      for (int i = 0; i < 8; i++) {
        float* o = out + (size_t)(m0 + i) * N + nt * 32;
        _mm512_storeu_ps(o, _mm512_fmadd_ps(r[i][0], ws0, b0));
        _mm512_storeu_ps(o + 16, _mm512_fmadd_ps(r[i][1], ws1, b1));
      }
    }
  }
}
