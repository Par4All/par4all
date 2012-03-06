

#define N 100
void loop_fusion02( int a[N], int b[N]) {
  int i,j;

  /* The first loop can't be fused */
  for( i=0; i<N; i++ ) {
    a[i] = i;
  }

  for( i=0; i<N; i++ ) {
    b[i] += a[i+1];
  }

  for( i=0; i<N; i++ ) {
    a[i] = b[i];
  }

}
