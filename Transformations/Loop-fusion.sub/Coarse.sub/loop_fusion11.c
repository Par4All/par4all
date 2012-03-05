

void fusion(int l, int m, int n, int a[l][m][n], int b[l][m][n]) {
  int i,j,k,p;
  for(i=1; i<l; i++) {
    for(j=2; j<m; j++) {
      for(k=2; k<n; k++) {
        a[i-1][j-1][k-1] = 0;
      }
    }
  }
  for(i=1; i<l; i++) {
    for(j=2; j<m; j++) {
      for(k=2; k<n; k++) {
        p = a[i-1][j-1][k-1];
        b[i-1][j-1][k-1] += p;
      }
    }
  }
}
