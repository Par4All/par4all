
#include <complex.h>

void scilab_rt_matrix_z2i0i0i0_z3(int nIn, int mIn, double complex in[nIn][mIn],
    int n,  int m, int k,
    int nOut, int mOut, int kOut, double complex out[nOut][mOut][kOut])
{
  int i;
  int j;
  int l;

  double complex val0 = 0;

  for (i = 0; i < nIn; ++i) {
    for (j = 0; j < mIn; ++j) {
      val0 += in[i][j];
    }
  }
  
  for (i = 0; i < nOut; ++i) {
    for (j = 0; j < mOut; ++j) {
      for (l = 0; l < kOut; ++l) {
        out[i][j][l] = val0;
      }
    }
  }

}

