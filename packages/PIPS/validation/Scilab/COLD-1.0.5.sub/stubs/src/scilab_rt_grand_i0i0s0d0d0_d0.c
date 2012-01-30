#include <stdio.h>


void scilab_rt_grand_i0i0s0d0d0_d0(int in0, int in1, char* in2, double in3, double in4,
        double* out0)
{
  double val0 = 0;

  printf("%s", in2);
  val0 = in0 + in1 + in3 + in4;
  printf("%f", val0);
  *out0 = val0;
}


