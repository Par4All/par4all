int main () {
  float a[10];
  int m = 0;
  int n = 0;

  while (n < 100) {
    for (m = 0; m < 10; m++) {
      n = n+1;
    }
  }

  for (m = 0; m < 10; m++) {
    n = n+1;
  }

  n = n+2;

  return 0;
}
