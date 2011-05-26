#define N 100
void backward_flow_dependence( int a[N], int b[N] ) {
  int j, l;

  // There shouldn't be a backward flow dependence from third
  // statement to the second one because of the kill by the first one.
  for(j = 1; j <= 99; j += 1) {
    a[j] = j;
    b[j] = a[j];
    a[j+1] = j;
  }
}
