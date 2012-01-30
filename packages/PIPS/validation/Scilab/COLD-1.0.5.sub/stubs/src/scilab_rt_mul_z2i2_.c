
#include <complex.h>

double complex scilab_rt_mul_z2i2_(int sin00, int sin01, double complex in0[sin00][sin01],
    int sin10, int sin11, int in1[sin10][sin11])
{
  int i;
  int j;

  double complex val0 = 0;
  int val1 = 0;
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

  return val0 + val1;
}

