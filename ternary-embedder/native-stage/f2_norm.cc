#include "fused2.h"
extern "C" void f2_norm(float *x,const uint8 *p) {
#ifdef F2_STUB
  return;
#endif
 norm(x,(const float*)p); }
