// The goal of this test case is to check that all basic reduction operators
// are well detected by pips

int main () {
  int b = 0;
  int i = 0;
  float y = 10.0;
  float x = 1.0;

  // b should be reduced here
  for (i=0; i < 100; i++) {
    b = b + i;
  }

  // b should be reduced here
  for (i=0; i < 100; i++) {
    b = b * i;
  }

  // b should not be reduced here
  // because "/" on int is not accepted
  for (i=0; i < 100; i++) {
    b = b / i;
  }

  // b should be reduced here
  for (i=0; i < 100; i++) {
    b = b & i;
  }

  // b should be reduced here
  for (i=0; i < 100; i++) {
    b = b ^ i;
  }

  // b should be reduced here
  for (i=0; i < 100; i++) {
    b = b | i;
  }

  // b should be reduced here
  for (i=0; i < 100; i++) {
    b += i;
  }

  // b should be reduced here
  for (i=0; i < 100; i++) {
    b *= i;
  }

  // b should not be reduced here
  // because "/" on int is not accepted
  for (i=0; i < 100; i++) {
    b /= i;
  }

  // b should be reduced here
  for (i=0; i < 100; i++) {
    b -= i;
  }

  // b should be reduced here
  for (i=0; i < 100; i++) {
    b &= i;
  }

  // b should be reduced here
  for (i=0; i < 100; i++) {
    b |= i;
  } 

  // b should be reduced here
  for (i=0; i < 100; i++) {
    b ^= i;
  }

  // b should be reduced here
  // because "/" on float is accepted
  for (i=0; i < 100; i++) {
    y = y / x;
  }

  // b should be reduced here
  // because "/" on float is accepted
  for (i=0; i < 100; i++) {
    y = y / x;
  }

  // b should be reduced here
  for (i=0; i < 100; i++) {
    b++;
  }

  // b should be reduced here
  for (i=0; i < 100; i++) {
    b--;
  }

  // b should be reduced here
  for (i=0; i < 100; i++) {
    ++b;
  }

  // b should be reduced here
  for (i=0; i < 100; i++) {
    --b;
  }

  // b should be reduced here
  for (i=0; i < 100; i++) {
    b = b && (i == i);
  }

  // flg should be reduced here
  for (i=0; i < 100; i++) {
    b = b || (i == i);
  }

  return 0;
}
