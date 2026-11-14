// Native XDNA2 ternary primitive: four rows, 32 output channels, exact K.
// Two-bit weights expand into the int4 matrix operands as 4*w. Repetition
// accumulates the result to expose kernel time above the host dispatch floor.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef WIDTH_K
#define WIDTH_K 384
#endif
#ifndef REPEATS
#define REPEATS 1024
#endif
static_assert(WIDTH_K % 16 == 0);
using Mat = aie::mmul<4, 16, 16, int8, int4>;

extern "C" void width_probe(const int8_t *__restrict a,
                            const uint8_t *__restrict w,
                            int32_t *__restrict out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  Mat low, high;
#pragma clang loop unroll(disable)
  for (unsigned rep = 0; rep < REPEATS; ++rep) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(24)
    for (unsigned k = 0; k < WIDTH_K / 16; ++k) {
      const auto av = aie::load_v<64>(a + k * 64);
      const auto q = aie::load_v<128>(w + k * 128);
      const auto l = aie::bit_and((uint8)0xCC,
          aie::upshift(q.cast_to<uint32>(), 2).cast_to<uint8>()).cast_to<int4>();
      const auto h = aie::bit_and((uint8)0xCC, q).cast_to<int4>();
      low.mac(av, l);
      high.mac(av, h);
    }
  }
  aie::store_v(out, low.to_vector<int32>());
  aie::store_v(out + 64, high.to_vector<int32>());
}
