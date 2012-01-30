
void scilab_rt_matrix_s2i0i0_s2(int nIn, int mIn, char* in[nIn][mIn],
    int n,  int m,
    int nOut, int mOut, char* out[nOut][mOut])
{
  int i;
  char** pIn  = (char**) in;
  char** pOut = (char**) out;

  for (i = 0; i < nIn*mIn; ++i) {
    pOut[i] = pIn[i];
  }

}

