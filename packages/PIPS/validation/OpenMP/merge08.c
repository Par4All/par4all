const int size = 100;

int main (void) {
  int k = 0, i = 0, l = 0;
  int sum = 0;
  int a[size][size][size];

  for (l = 0; l < size; l++) {
    for (k = 0; k < size; k++) {
      for (i = 0; i < size; i++) {
	a[l][k][i] = 10;
      }
      for (i = 0; i < size; i++) {
	a[l][k][i] += 10;
      }
    }
    for (k = 0; k < size; k++) {
      a[l][k][k] += 10;
    }
  }

  return 0;
}
