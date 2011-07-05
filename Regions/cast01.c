



void cast() {
  int i;
  int a[10];
  // Without cast, it's parallelized
  for(i=0; i<10; i++) {
    a[i] = 0;
  }
  // Cast prevents parallelization
  for(i=0; i<10; i++) {
    a[(int)i] = 0;
  }
}
