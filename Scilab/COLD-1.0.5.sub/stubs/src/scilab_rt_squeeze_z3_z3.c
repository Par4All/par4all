
#include <complex.h>

void scilab_rt_squeeze_z3_z3(int sin00, int sin01, int sin02, double complex in0[sin00][sin01][sin02],
    int sout00, int sout01, int sout02, double complex out0[sout00][sout01][sout02])
{

  int i;
  int j;
  int k;

  double Rval0 = 0;
  double Ival0 = 0;
  for (i = 0; i < sin00; ++i) {
    for (j = 0; j < sin01; ++j) {
      for (k = 0; k < sin02; ++k) {
        Rval0 += creal(in0[i][j][k]);
        Ival0 += cimag(in0[i][j][k]);
      }
    }
  }

  for (i = 0; i < sout00; ++i) {
    for (j = 0; j < sout01; ++j) {
      for (k = 0; k < sout02; ++k) {
        out0[i][j][k] = Rval0 + Ival0*I;
      }
    }
  }
}

