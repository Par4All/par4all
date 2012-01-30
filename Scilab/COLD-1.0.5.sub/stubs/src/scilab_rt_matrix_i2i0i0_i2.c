

void scilab_rt_matrix_i2i0i0_i2(int nIn, int mIn, int in[nIn][mIn],
    int n,  int m,
    int nOut, int mOut, int out[nOut][mOut])
{
  int i;
  int j;

  int val0 = 0;

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
