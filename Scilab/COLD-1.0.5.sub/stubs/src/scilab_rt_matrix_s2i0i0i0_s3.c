
void scilab_rt_matrix_s2i0i0i0_s3(int nIn, int mIn, char* in[nIn][mIn],
    int n,  int m, int k,
    int nOut, int mOut, int kOut, char* out[nOut][mOut][kOut])
{
  int i;
  char** pIn  = (char**) in;
  char** pOut = (char**) out;
  
  for (i = 0; i < nIn*mIn; ++i) {
    pOut[i] = pIn[i];
  }

}

