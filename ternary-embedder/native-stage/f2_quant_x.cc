#include "fused2.h"
extern "C" void f2_quant_x(const float *x,int8 *q) {
#ifdef F2_STUB
  return;
#endif
 quant(x,q,384,0); }
