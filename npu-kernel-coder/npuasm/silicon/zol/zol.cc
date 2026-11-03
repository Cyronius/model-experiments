// SPDX-License-Identifier: Apache-2.0
// One counted scalar loop. Peano turns it into a hardware zero-overhead loop
// (movxm ls / movxm le / add.nc lc); the test checks every iteration ran.
#include <stdint.h>

template <int N>
static inline void zol_impl(const int32_t *__restrict in, int32_t *__restrict out) {
#pragma clang loop unroll(disable) vectorize(disable)
  for (int i = 0; i < N; i++)
    out[i] = in[i] * 3 + i;
}

extern "C" {
void zol_1024(int32_t *in, int32_t *out) { zol_impl<1024>(in, out); }
}
