#include "fused2.h"
extern "C" void f2_attend(bfloat16 *q,const uint8 *e,float *w,float *ml,
                          const float *bias,int head,int block) {
#ifdef F2_STUB
  return;
#endif
  attend(q,e,w,ml,bias,head,block);
}
