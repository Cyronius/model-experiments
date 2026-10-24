#include "fused.h"
extern "C" void ep_linear(const uint8 *w,const float *par,const int8 *aq,float *x,
                          bfloat16 *q,uint8 *kv,int j,int kind,int d) {
  project(w,par,aq,x,q,kv,j,kind,d);
}
