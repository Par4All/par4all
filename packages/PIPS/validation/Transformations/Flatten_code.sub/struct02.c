void struct02(void) {
  int i;
  {
    int i;
    struct foo { int a;};
    struct foo j;
    j.a = 1;
  }
  {
    int i;
    struct foo { double a;};
    struct foo j;
    j.a = 0.;
  }
  return;
}
