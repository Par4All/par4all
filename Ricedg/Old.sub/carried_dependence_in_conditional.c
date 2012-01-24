#define N 100
void carried_dependence_in_conditional( int a[N], int n ) {
  int j,k;

  // The backward output dependence should be marked as carried
  for(j = 1; j <= 99; j += 1) {
    if( j<n) {
      k = 0;
    }
  }
}
