

#define N 100
void loop_fusion01( int a[N], int b[N]) {
  int i,j;

  /* The three loops are fusable */
  for( i=0; i<N; i++ ) {
    a[i] = i;
  }

  for( i=0; i<N; i++ ) {
    b[i] += a[i];
  }

  for( i=0; i<N; i++ ) {
    a[i] = b[i];
  }

}
