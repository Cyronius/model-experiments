// Four real rows, all output columns, no repeat loop and no hidden width padding.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef STAGE_K
#define STAGE_K 384
#endif
static constexpr unsigned GROUP_K = STAGE_K == 1536 ? 256 : STAGE_K;
static constexpr unsigned GROUPS = STAGE_K / GROUP_K;
using Mat = aie::mmul<4, 16, 16, int8, int4>;

extern "C" void projection(const int8_t *__restrict a,
                            const uint8_t *__restrict w,
                            int32_t *__restrict out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
  for (unsigned group = 0; group < GROUPS; ++group) {
    Mat low, high;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (unsigned k = 0; k < GROUP_K / 16; ++k) {
      const unsigned block = group * (GROUP_K / 16) + k;
      const auto av = aie::load_v<64>(a + block * 64);
      const auto q = aie::load_v<128>(w + block * 128);
      low.mac(av, aie::bit_and((uint8)0xCC,
          aie::upshift(q.cast_to<uint32>(), 2).cast_to<uint8>()).cast_to<int4>());
      high.mac(av, aie::bit_and((uint8)0xCC, q).cast_to<int4>());
    }
    aie::store_v(out + group * 128, low.to_vector<int32>());
    aie::store_v(out + group * 128 + 64, high.to_vector<int32>());
  }
}
