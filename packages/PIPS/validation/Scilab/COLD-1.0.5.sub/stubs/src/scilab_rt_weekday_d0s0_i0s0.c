
#include <stdlib.h>

void scilab_rt_weekday_d0s0_i0s0(double in0, char* in1, int* out0, char** out1)
{
  *out0 = (int) in0 + *in1;
  *out1 = (char*) malloc(in0);

}
