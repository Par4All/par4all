/* Simple case for tests in the controlizer: */
void test() {
  int x = 6;
 ici:
  if (x > 4) {
    x = 2;
    goto ici;
  }
  else
    x = 3;
}
