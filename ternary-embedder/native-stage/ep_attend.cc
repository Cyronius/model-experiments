#include "fused.h"
extern "C" void ep_attend(const bfloat16 *q,const uint8 *e,float *w,float *ml,
                          const int32 *mask,int head,int block,int core,int cap) {
  attend(q,e,w,ml,mask,head,block,core,cap);
}
