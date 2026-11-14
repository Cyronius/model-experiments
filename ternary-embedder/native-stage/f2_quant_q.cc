#include "fused2.h"
extern "C" void f2_quant_q(const bfloat16 *x,int8 *q,int d) {
#ifdef F2_STUB
  return;
#endif
 quant(x,q,d,1); }
