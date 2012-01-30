
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

double complex scilab_rt_det_z2_(int sin00, int sin01, double complex in0[sin00][sin01])
{
  int i;
  int j;

  double complex val0 = 0;
  for (i = 0; i < sin00; ++i) {
    for (j = 0; j < sin01; ++j) {
      val0 += in0[i][j];
    }
  }

  return val0;
}
