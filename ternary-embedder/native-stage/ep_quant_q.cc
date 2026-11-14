#include "fused.h"
extern "C" void ep_quant_q(const bfloat16 *x,int8 *q,int d) { quant(x,q,d,1); }
