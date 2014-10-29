void dependenttype02() {
  int i;
  for (i=0; i<10; i++) {
    int a[i+1];
    for (int j=0; j<i+1; j++) {
      a[j] = j;
    }
  }
}
