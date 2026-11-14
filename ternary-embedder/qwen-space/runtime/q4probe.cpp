// Stage 4 Q4 check (30-minute hard stop): time one linear, the 384 -> 1536 FFN matrix over 504 token rows,
// synthetic input, with weights one per byte (lin_avx2 / lin_vnni) vs packed two per byte and unpacked in
// the loop (lin_avx2_p4). Single thread. Prints JSON.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include "src/kernels.h"

static double now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

template <class F> static double median_ms(F f, int reps) {
  double* s = (double*)malloc(sizeof(double) * reps);
  for (int i = 0; i < 20; i++) f();
  for (int i = 0; i < reps; i++) { double t = now(); f(); s[i] = (now() - t) * 1e3; }
  std::sort(s, s + reps);
  double m = s[reps / 2]; free(s); return m;
}

int main(int argc, char** argv) {
  const int M = 504, K = 384, N = 1536, gs = 384, reps = argc > 1 ? atoi(argv[1]) : 400;
  int8_t* q = (int8_t*)aligned_alloc(64, (size_t)M * K);
  uint8_t* W8 = (uint8_t*)aligned_alloc(64, (size_t)N * K);
  uint8_t* W4 = (uint8_t*)aligned_alloc(64, (size_t)N * K / 2);
  float* sc = (float*)malloc(sizeof(float) * M);
  int32_t* as = (int32_t*)malloc(sizeof(int32_t) * M);
  float* ws = (float*)malloc(sizeof(float) * N);
  float* b = (float*)malloc(sizeof(float) * N);
  float* out = (float*)aligned_alloc(64, sizeof(float) * (size_t)M * N);
  srand(1);
  for (size_t i = 0; i < (size_t)M * K; i++) q[i] = (int8_t)(rand() % 256 - 128);
  for (size_t i = 0; i < (size_t)N * K; i++) W8[i] = (uint8_t)(rand() % 128);
  for (size_t i = 0; i < (size_t)N * K / 2; i++) W4[i] = (uint8_t)(rand() % 256);
  for (int i = 0; i < M; i++) { sc[i] = 1.0f / 64; as[i] = 0; }
  for (int i = 0; i < N; i++) { ws[i] = 0.01f; b[i] = 0.f; }
  double a8 = median_ms([&] { lin_avx2(q, sc, as, M, K, gs, W8, ws, b, N, out); }, reps);
  double a4 = median_ms([&] { lin_avx2_p4(q, sc, as, M, K, gs, W4, ws, b, N, out); }, reps);
  double v8 = -1;
  if (__builtin_cpu_supports("avx512vnni")) v8 = median_ms([&] { lin_vnni(q, sc, as, M, K, gs, W8, ws, b, N, out); }, reps);
  printf("{\"rows\": %d, \"K\": %d, \"N\": %d, \"reps\": %d, \"avx2_int8_one_per_byte_ms\": %.4f, "
         "\"avx2_q4_packed_two_per_byte_ms\": %.4f, \"avx2_packed_speedup\": %.4f, \"vnni_int8_one_per_byte_ms\": %.4f}\n",
         M, K, N, reps, a8, a4, a8 / a4, v8);
  return 0;
}
