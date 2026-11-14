#pragma once
#include <stdint.h>

// Quantized linear: out[M][N] = (sum_g sc[m,g] * (sum_k q[m,k] * u[n,k] - 64 * asum[m,g])) * ws[n] + b[n]
// q: int8 [M][K] (M a multiple of 8), u = code + 64 tiled [N/32][K/4][32][4], gs = activation group size.
typedef void (*lin_fn)(const int8_t* q, const float* sc, const int32_t* asum, int M, int K, int gs,
                       const uint8_t* W, const float* ws, const float* b, int N, float* out);

void lin_avx2(const int8_t* q, const float* sc, const int32_t* asum, int M, int K, int gs,
              const uint8_t* W, const float* ws, const float* b, int N, float* out);
void lin_vnni(const int8_t* q, const float* sc, const int32_t* asum, int M, int K, int gs,
              const uint8_t* W, const float* ws, const float* b, int N, float* out);

// Q4 timing probe only (Stage 4, 30-minute check): same kernels with weights packed two per byte.
void lin_avx2_p4(const int8_t* q, const float* sc, const int32_t* asum, int M, int K, int gs,
                 const uint8_t* W, const float* ws, const float* b, int N, float* out);
