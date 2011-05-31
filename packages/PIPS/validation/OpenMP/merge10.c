#include <stdio.h>

int size = 100;

int main (void) {
  int k = 0, i = 0, l = 0;
  float sum = 0;
  float array [size][size][size];

  for (l = 0; l < size; l++) {
    for (k = 0; k < size; k++) {
      for (i = 0; i < size; i++) {
	array[l][k][i] = i + k ;
      }
    }
  }

  l = 4;

  for (i = 0; i < size; i++) {
    for (k = 0; k < size; k++) {
      printf ("array[%d][%d] = %f", i, k, array[l][k][i]);
    }
  }

  return 0;
}
