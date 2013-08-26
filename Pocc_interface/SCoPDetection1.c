#include <stdio.h>
#include <stdlib.h>

#define TAILLE 500

// Test with an affine expression
int main () {
  int A[TAILLE];
  int i = 0;
  
  for (int i = 0; i < TAILLE/2 - 3; i++)
    A[2*i + 3] = 1234;
  
  return A[0];
}
