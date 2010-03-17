// The goal of this test case is to check that all basic reduction operators
// are well printed by the pips prettyprinter. All the loops in the programm
// should be parallelized with an omp reduction pragma.

int main () {
  int b = 0;
  int i = 0;

  for (i=0; i < 100; i++) {
    b = b + i;
  }
  for (i=0; i < 100; i++) {
    b = b * i;
  }
  for (i=0; i < 100; i++) {
    b = b - i;
  }
  for (i=0; i < 100; i++) {
    b = b / i;
  }
  for (i=0; i < 100; i++) {
    b = b & i;
  }
  for (i=0; i < 100; i++) {
    b = b ^ i;
  }
  for (i=0; i < 100; i++) {
    b = b | i;
  }
  for (i=0; i < 100; i++) {
    b += i;
  }
  for (i=0; i < 100; i++) {
    b *= i;
  }
  for (i=0; i < 100; i++) {
    b -= i;
  }
  for (i=0; i < 100; i++) {
    b /= i;
  }
  for (i=0; i < 100; i++) {
    b &= i;
  }
  for (i=0; i < 100; i++) {
    b ^= i;
  }
  for (i=0; i < 100; i++) {
    b |= i;
  }
  for (i=0; i < 100; i++) {
    b++;
  }
  for (i=0; i < 100; i++) {
    b--;
  }
  for (i=0; i < 100; i++) {
    ++b;
  }
  for (i=0; i < 100; i++) {
    --b;
  }
  for (i=0; i < 100; i++) {
    b = b && (i == i);
  }
  for (i=0; i < 100; i++) {
    b = b || (i == i);
  }

  return 0;
}
