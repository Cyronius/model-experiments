#include "fused.h"
extern "C" void ep_quant_x(const float *x,int8 *q) { quant(x,q,384,0); }
