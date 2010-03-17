#define n 10000
void empty_for()
{
  //  int n = 10000;
  int loops = 10;

  //int numBytes = n * n * sizeof(float);
  /*
  float* h_A = (float *) malloc(numBytes);
  float* h_C = (float *) malloc(numBytes);
  */
  float h_A[n*n];
  float h_C[n*n];

  int i, j;

  for(i=0; i< n; ++i) {
    for(j=0; j<n; ++j) {
      h_A[i*n+j] = 1;
      h_C[i*n+j] = 0;
    }
  }

  for (i=0; i<loops; ++i) {
    //conv_cpu(h_A, h_C, 1, 1, 1, 1, 1, 1, 1, 1, 1);
  }

}
