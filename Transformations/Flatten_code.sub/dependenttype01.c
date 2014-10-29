void dependenttype01() {
  int i;
  {
    int i;
    i=10;
    int a[i];
    a[0] = 0;
  }
}