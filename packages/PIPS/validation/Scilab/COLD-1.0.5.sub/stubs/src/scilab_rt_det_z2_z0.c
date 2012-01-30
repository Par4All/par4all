
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

void scilab_rt_det_z2_z0(int sin00, int sin01, double complex in0[sin00][sin01], double complex* out0)
{
  int i;
  int j;

  double complex val0 = 0;
  for (i = 0; i < sin00; ++i) {
    for (j = 0; j < sin01; ++j) {
      val0 += in0[i][j];
    }
  }

  *out0 = val0;
}
