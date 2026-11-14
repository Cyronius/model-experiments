// AVX2 linear kernel (x86-64-v3). vpmaddubsw (u8 weights x s8 activations, codes + 64 <= 127 so the
// int16 pair sums cannot saturate: 2 * 127 * 128 = 32512) then vpmaddwd with ones. Computes exactly the
// same integers as the VNNI path, and the same float epilogue in the same order.
#include <immintrin.h>
#include "kernels.h"

void lin_avx2(const int8_t* q, const float* sc, const int32_t* asum, int M, int K, int gs,
              const uint8_t* W, const float* ws, const float* b, int N, float* out) {
  const int G = K / gs;
  const __m256i ones = _mm256_set1_epi16(1);
  for (int nt = 0; nt < N / 32; nt++) {
    const uint8_t* Wt = W + (size_t)nt * 32 * K;
    for (int h = 0; h < 2; h++) {
      const int n0 = nt * 32 + h * 16;
      const __m256 ws0 = _mm256_loadu_ps(ws + n0), ws1 = _mm256_loadu_ps(ws + n0 + 8);
      const __m256 b0 = _mm256_loadu_ps(b + n0), b1 = _mm256_loadu_ps(b + n0 + 8);
      for (int m0 = 0; m0 < M; m0 += 4) {
        __m256 r[4][2];
        for (int i = 0; i < 4; i++) r[i][0] = r[i][1] = _mm256_setzero_ps();
        for (int g = 0; g < G; g++) {
          __m256i acc[4][2];
          for (int i = 0; i < 4; i++) acc[i][0] = acc[i][1] = _mm256_setzero_si256();
          const int8_t* qa = q + (size_t)m0 * K;
          for (int k4 = g * gs / 4; k4 < (g + 1) * gs / 4; k4++) {
            const uint8_t* wp = Wt + (size_t)k4 * 128 + h * 64;
            const __m256i w0 = _mm256_loadu_si256((const __m256i*)wp);
            const __m256i w1 = _mm256_loadu_si256((const __m256i*)(wp + 32));
            for (int i = 0; i < 4; i++) {
              const __m256i a = _mm256_set1_epi32(*(const int32_t*)(qa + (size_t)i * K + k4 * 4));
              acc[i][0] = _mm256_add_epi32(acc[i][0], _mm256_madd_epi16(_mm256_maddubs_epi16(w0, a), ones));
              acc[i][1] = _mm256_add_epi32(acc[i][1], _mm256_madd_epi16(_mm256_maddubs_epi16(w1, a), ones));
            }
          }
          for (int i = 0; i < 4; i++) {
            const int m = m0 + i;
            const __m256 sg = _mm256_set1_ps(sc[m * G + g]);
            const __m256i off = _mm256_set1_epi32(64 * asum[m * G + g]);
            r[i][0] = _mm256_fmadd_ps(sg, _mm256_cvtepi32_ps(_mm256_sub_epi32(acc[i][0], off)), r[i][0]);
            r[i][1] = _mm256_fmadd_ps(sg, _mm256_cvtepi32_ps(_mm256_sub_epi32(acc[i][1], off)), r[i][1]);
          }
        }
        for (int i = 0; i < 4; i++) {
          float* o = out + (size_t)(m0 + i) * N + n0;
          _mm256_storeu_ps(o, _mm256_fmadd_ps(r[i][0], ws0, b0));
          _mm256_storeu_ps(o + 8, _mm256_fmadd_ps(r[i][1], ws1, b1));
        }
      }
    }
  }
}

// Q4 timing probe (not used by the model): codes in [0, 15] (code + 8), two per byte. For each
// (tile, half, pair of k4 steps) 64 bytes hold [n 0-7 | n 8-15] x 4 k x 2 steps: low nibble = step 0,
// high nibble = step 1. Unpacked in the loop with and/shift, then the same maddubs path. The offset
// term uses 8 instead of 64; the probe only times the kernel.
void lin_avx2_p4(const int8_t* q, const float* sc, const int32_t* asum, int M, int K, int gs,
                 const uint8_t* W, const float* ws, const float* b, int N, float* out) {
  const int G = K / gs;
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i lo = _mm256_set1_epi8(0x0F);
  for (int nt = 0; nt < N / 32; nt++) {
    const uint8_t* Wt = W + (size_t)nt * 16 * K;
    for (int h = 0; h < 2; h++) {
      const int n0 = nt * 32 + h * 16;
      const __m256 ws0 = _mm256_loadu_ps(ws + n0), ws1 = _mm256_loadu_ps(ws + n0 + 8);
      const __m256 b0 = _mm256_loadu_ps(b + n0), b1 = _mm256_loadu_ps(b + n0 + 8);
      for (int m0 = 0; m0 < M; m0 += 4) {
        __m256 r[4][2];
        for (int i = 0; i < 4; i++) r[i][0] = r[i][1] = _mm256_setzero_ps();
        for (int g = 0; g < G; g++) {
          __m256i acc[4][2];
          for (int i = 0; i < 4; i++) acc[i][0] = acc[i][1] = _mm256_setzero_si256();
          const int8_t* qa = q + (size_t)m0 * K;
          for (int k8 = g * gs / 8; k8 < (g + 1) * gs / 8; k8++) {
            const uint8_t* wp = Wt + (size_t)k8 * 128 + h * 64;
            const __m256i p0 = _mm256_loadu_si256((const __m256i*)wp);
            const __m256i p1 = _mm256_loadu_si256((const __m256i*)(wp + 32));
            const __m256i w00 = _mm256_and_si256(p0, lo), w01 = _mm256_and_si256(_mm256_srli_epi16(p0, 4), lo);
            const __m256i w10 = _mm256_and_si256(p1, lo), w11 = _mm256_and_si256(_mm256_srli_epi16(p1, 4), lo);
            for (int i = 0; i < 4; i++) {
              const __m256i a0 = _mm256_set1_epi32(*(const int32_t*)(qa + (size_t)i * K + k8 * 8));
              const __m256i a1 = _mm256_set1_epi32(*(const int32_t*)(qa + (size_t)i * K + k8 * 8 + 4));
              acc[i][0] = _mm256_add_epi32(acc[i][0], _mm256_madd_epi16(_mm256_maddubs_epi16(w00, a0), ones));
              acc[i][1] = _mm256_add_epi32(acc[i][1], _mm256_madd_epi16(_mm256_maddubs_epi16(w10, a0), ones));
              acc[i][0] = _mm256_add_epi32(acc[i][0], _mm256_madd_epi16(_mm256_maddubs_epi16(w01, a1), ones));
              acc[i][1] = _mm256_add_epi32(acc[i][1], _mm256_madd_epi16(_mm256_maddubs_epi16(w11, a1), ones));
            }
          }
          for (int i = 0; i < 4; i++) {
            const int m = m0 + i;
            const __m256 sg = _mm256_set1_ps(sc[m * G + g]);
            const __m256i off = _mm256_set1_epi32(8 * asum[m * G + g]);
            r[i][0] = _mm256_fmadd_ps(sg, _mm256_cvtepi32_ps(_mm256_sub_epi32(acc[i][0], off)), r[i][0]);
            r[i][1] = _mm256_fmadd_ps(sg, _mm256_cvtepi32_ps(_mm256_sub_epi32(acc[i][1], off)), r[i][1]);
          }
        }
        for (int i = 0; i < 4; i++) {
          float* o = out + (size_t)(m0 + i) * N + n0;
          _mm256_storeu_ps(o, _mm256_fmadd_ps(r[i][0], ws0, b0));
          _mm256_storeu_ps(o + 8, _mm256_fmadd_ps(r[i][1], ws1, b1));
        }
      }
    }
  }
}
