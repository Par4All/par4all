void main()
{
  int n = 10000;
  int loops = 10;

  float h_A[n*n];
  float h_C[n*n];

  int i, j;

  for(i=0; i< n; ++i) {
    for(j=0; j<n; ++j) {
      h_A[i*n+j] = 1;
      h_C[i*n+j] = 0;
    }
  }
}
