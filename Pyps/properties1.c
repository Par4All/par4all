#include <stdio.h>

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
    }
  }

  for (l = 0; l < size; l++) {
    for (k = 0; k < size; k++) {
      for (i = 0; i < size; i++) {
	sum  += a[l][k][i];
      }
    }
  }

  printf ("sum is %d", sum);

  return 0;
}
