#include <stdio.h>
#include <stdlib.h>

#define TAILLE 500

// Test with condition and non affine expression
int main () {
  int A[TAILLE];
  int i = 0;
  int k = 0;
  
  for (i = 0; i < TAILLE/2 - 3; i++) {
    k = 2*i;
    A[k+3] = 12;
  }
  return 0;
}
