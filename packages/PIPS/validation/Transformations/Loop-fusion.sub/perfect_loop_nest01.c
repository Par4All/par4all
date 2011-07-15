

void perfect(int n, int a[n][n], int b[n][n]) {
  int i,j;
  for(i = 0; i<n;i++) {
    for(j = 0; j<n;j++) {
      a[i][j] = 0;
    }
  }
  for(i = 0; i<n;i++) {
    for(j = 0; j<n;j++) {
      a[i][j] = 0;
      // The following prevents inner loop parallelization
      b[i][a[i][j]]++;
    }
  }

}
