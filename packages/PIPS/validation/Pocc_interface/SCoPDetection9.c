#include <stdio.h>
#include <stdlib.h>

#define TAILLE 500

// Test with non affine condition within a loop
int main () {
  int A[TAILLE];
  int i = 0;
  int condition;
  
  for (i = 0; i < TAILLE; i++) {
    condition = 2*i + i*i;
    if (condition-500 == 0)
      A[i] = 40;
  }
  
  return A[0];
}
