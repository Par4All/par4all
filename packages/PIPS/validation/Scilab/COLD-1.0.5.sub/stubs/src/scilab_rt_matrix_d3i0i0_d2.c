
void scilab_rt_matrix_d3i0i0_d2(int nIn, int mIn, int kIn, double in[nIn][mIn][kIn],
    int n,  int m,
    int nOut, int mOut, double out[nOut][mOut])
{
  int i;
  int j;
  int k;

  double val0 = 0;

  for (i = 0; i < nIn; ++i) {
    for (j = 0; j < mIn; ++j) {
      for (k = 0; k < kIn; ++k) {
        val0 += in[i][j][k];
      }
    }
  }
  
  for (i = 0; i < nOut; ++i) {
    for (j = 0; j < mOut; ++j) {
      out[i][j] = val0;
    }
  }
}

