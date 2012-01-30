
void scilab_rt_matrix_s3i0i0_s2(int nIn, int mIn, int kIn, char* in[nIn][mIn][kIn],
    int n,  int m,
    int nOut, int mOut, char* out[nOut][mOut])
{
  int i;
  char** pIn  = (char**) in;
  char** pOut = (char**) out;

  for (i = 0; i < nIn*mIn*kIn; ++i) {
    pOut[i] = pIn[i];
  }

}
