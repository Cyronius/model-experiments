#include "fused.h"
extern "C" void ep_norm(float *x,const uint8 *p) { norm(x,(const float*)p); }
