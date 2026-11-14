#include "fused.h"
extern "C" void ep_attfinish(float *w,bfloat16 *q,int head) { finish_attention(w,q,head); }
