#include "fused.h"
extern "C" void ep_mask(const uint8 *e,int32 *m) {
  for(int i=0;i<128;i+=32) aie::store_v(m+i,aie::load_v<32>((const int32*)e+i));
}
