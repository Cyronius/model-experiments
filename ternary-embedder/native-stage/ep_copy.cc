#include "fused.h"
extern "C" void ep_copy(const uint8 *e,float *p) {
  for(int i=0;i<1024;i+=32) aie::store_v(p+i,aie::load_v<32>((const float*)e+i));
}
