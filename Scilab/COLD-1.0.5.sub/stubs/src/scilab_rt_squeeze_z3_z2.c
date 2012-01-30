
#include <complex.h>

void scilab_rt_squeeze_z3_z2(int in00, int in01, int in02, double complex matrixin0[in00][in01][in02],
     int out00, int out01, double complex matrixout0[out00][out01])
{
  int i;
  int j;
  int k;

  double Rval0 = 0;
  double Ival0 = 0;
  for (i = 0; i < in00; ++i) {
    for (j = 0; j < in01; ++j) {
      for (k = 0; k < in02; ++k) {
        Rval0 += creal(matrixin0[i][j][k]);
        Ival0 += cimag(matrixin0[i][j][k]);
      }
    }
  }

  for (i = 0; i < out00; ++i) {
    for (j = 0; j < out01; ++j) {
        matrixout0[i][j] = Rval0 + Ival0*I;
    }
  }
}
