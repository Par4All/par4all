
void scilab_rt_matrix_d2i0i0_d2(int nIn, int mIn, double in[nIn][mIn],
    int n,  int m,
    int nOut, int mOut, double out[nOut][mOut])
{
  int i;
  int j;

  double val0 = 0;

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

