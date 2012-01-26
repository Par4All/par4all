
void init(int *a, int size) {
  for(int i = 0; i<size; i++) {
    a[i] = i;
  }
}

void cast_implicit() {
  int n = 10;
  int a[n][n];

  init(a,n*n);

}
