
#include <stdio.h>
#include <stdlib.h>

void scilab_rt_interp1_i2d2d0_d0(int sin00, int sin01, int in0[sin00][sin01],
    int sin10, int sin11, double in1[sin10][sin11],
    double in2,
    double* out0)
{
  int i;
  int j;

  int val0 = 0;
  double val1 = 0;
  for (i = 0; i < sin00; ++i) {
    for (j = 0; j < sin01; ++j) {
      val0 += in0[i][j];
    }
  }
  for (i = 0; i < sin10; ++i) {
    for (j = 0; j < sin11; ++j) {
      val1 += in1[i][j];
    }
  }

  *out0 = val0 + val1 + in2;
}
