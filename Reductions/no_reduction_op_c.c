// The goal of this test case is to check that all basic cases where
// reduction is not possible are well detected by pips. None of the loops in
// the programm should be reduced.

int main () {
  int b = 0;
  int i = 0;

  for (i=0; i < 100; i++) {
    b = i - b;
  }
  for (i=0; i < 100; i++) {
    b = i / b;
  }
}
