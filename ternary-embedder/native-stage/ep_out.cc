#include "fused.h"
extern "C" void ep_out(const float *x,uint8 *e) {
  for(int i=0;i<1536;i+=32) aie::store_v((float*)e+i,aie::load_v<32>(x+i));
  for(int i=1536;i<2048;i+=32) aie::store_v((float*)e+i,aie::zeros<float,32>());
}
