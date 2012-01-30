
void scilab_rt_matrix_i3i0i0_i2(int nIn, int mIn, int kIn, int in[nIn][mIn][kIn],
    int n,  int m,
    int nOut, int mOut, int out[nOut][mOut])
{
  int i;
  int j;
  int k;

  int val0 = 0;

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
