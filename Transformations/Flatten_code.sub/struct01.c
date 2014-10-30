void struct01() {
  int i;
  {
    int i;
    struct foo { int a;};
    struct foo j;
    j.a = 1;
  }
}
