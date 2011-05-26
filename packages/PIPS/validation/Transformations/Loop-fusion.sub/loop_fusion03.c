
#define N 100
void loop_fusion03( int a[N][N], int b[N][N] ) {
  int i, j;
  int k;

  /* These loop nests can be fused together, even with the reduction on k */
  k = 0;
  for ( i = 0; i < N; i++ ) {
    for ( j = 0; j < N; j++ ) {
      a[i][j] = i + j;
    }
    k += a[i][j-1];
    for ( j = 0; j < N; j++ ) {
      b[i][j] += a[i][j];
    }
  }


}
