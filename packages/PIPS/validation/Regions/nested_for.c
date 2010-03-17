void calc(int n) {
  float h_A[n][n];

  int i, j;

  for(i=0; i< n; ++i)
    for(j=i; j<n; ++j)
      h_A[i][j] = 1;
}
