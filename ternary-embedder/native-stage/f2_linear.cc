#include "fused2.h"
extern "C" void f2_linear(const uint8 *w,const int8 *aq,float *x,
                          bfloat16 *q,uint8 *kv,int j,int kind,int d) {
#ifdef F2_STUB
  return;
#endif
  project(w,aq,x,q,kv,j,kind,d);
}
