#define N 100
void carried_dependence_in_conditional( int a[N], int n ) {
  int j,k;

  // There shouldn't be a backward flow dependence from third
  // statement to the second one because of the kill by the first one.
  for(j = 1; j <= 99; j += 1) {
    if( j<N) {
      k = 0;
    }
  }
}
