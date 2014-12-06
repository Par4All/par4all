void union01(void) {
  int i;
  {
    int i;
    union foo { int a; double b;};
    union foo j;
    j.a = 1;
  }
  return;
}
