#include <stdio.h>

int size = 100;

int main (void) {
  int k =0, i = 0;
  float sum = 0;
  float array [size][size];

  for (k = 0; k < size; k++) {
    for (i = 0; i < size; i++) {
      array[k][i] = i + k;
    }
  }

  for (i = 0; i < size; i++) {
    for (k = 0; k < size; k++) {
      printf ("array[%d][%d] = %f", i, k, array[k][i]);
    }
  }

  return 0;
}
