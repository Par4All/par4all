
#include <complex.h>

void scilab_rt_fft_d2d0_z2(int sin00, int sin01, double in0[sin00][sin01], double direction,
    int sout00, int sout01, double complex out0[sout00][sout01])
{
  int i;
  int j;

  double val0 = direction;
  for (i = 0; i < sin00; ++i) {
    for (j = 0; j < sin01; ++j) {
      val0 += in0[i][j];
    }
  }

  for (i = 0; i < sout00; ++i) {
    for (j = 0; j < sout01; ++j) {
      out0[i][j] = val0 + val0*I;
    }
  }
}

