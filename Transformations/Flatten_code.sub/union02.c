void union02(void) {
  int i;
  {
    int i;
    union foo { int a; double b;};
    union foo j;
    j.a = 1;
  }
  {
    int i;
    union foo { double a; int b;};
    union foo j;
    j.a = 0.;
  }
  return;
}
