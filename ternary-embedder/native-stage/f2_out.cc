#include "fused2.h"
// Layer trace: the host reads only the first 1536 floats of the 8192-byte packet.
extern "C" void f2_out(const float *x,uint8 *e) {
#ifdef F2_STUB
  return;
#endif
  for(int i=0;i<1536;i+=32) aie::store_v((float*)e+i,aie::load_v<32>(x+i));
}
