/* Check renaming for conflicting enum */

void enum02(void) {
  int i;
  {
    int i;
    enum foo { zero, un, deux};
    enum foo j;
    j = un;
  }
  {
    int i;
    enum foo { zero, one, two};
    enum foo j;
    j = one;
  }
  return;
}
