/* Simple case for tests in the controlizer: */

void test() {
  int x = 6;

  if (x > 4) {
    x = 2;
    goto ici;
  }
  else {
    x = 3;
    goto ici;
  }
  x = 5;
 ici:
  return;
}
