

#define N 100
void loop_fusion06( int a[N], int b[N]) {
  int i,j;

  // This loop is parallel
  // if we fuse we'll lose parallelism !
  for( i=0; i<N; i++ ) {
    a[i] = i;
  }

  /* This loop is not parallel !! */
  for( i=0; i<N; i++ ) {
    a[i] += a[i-1];
  }

}
