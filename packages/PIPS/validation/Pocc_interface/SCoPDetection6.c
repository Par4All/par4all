#include <stdio.h>
#include <stdlib.h>

#define TAILLE 500

// Test with affine condition and whole condition SCoP
int main () {
  int A[TAILLE];
  int i = 0;
  int condition = 0;
  if (condition) {
    for (i = 0; i < TAILLE; i++)
      A[i] = 405;
  }
  else {
    for (i = 0; i < TAILLE; i++)
      A[i] = 456;
  }
  return 0;
}
