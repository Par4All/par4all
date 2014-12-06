void enum01(void) {
  int i;
  {
    int i;
    enum foo { zero, un, deux};
    enum foo j;
    j = un;
  }
  return;
}
