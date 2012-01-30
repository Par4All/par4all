
#include <complex.h>

void scilab_rt_matrix_z2i0i0_z2(int nIn, int mIn, double complex in[nIn][mIn],
    int n,  int m,
    int nOut, int mOut, double complex out[nOut][mOut])
{
  int i;
  int j;

  double complex val0 = 0;

  for (i = 0; i < nIn; ++i) {
    for (j = 0; j < mIn; ++j) {
      val0 += in[i][j];
    }
  }
  
  for (i = 0; i < nOut; ++i) {
    for (j = 0; j < mOut; ++j) {
      out[i][j] = val0;
    }
  }
}
