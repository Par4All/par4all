#include <stdio.h>
#include <stdlib.h>

#define TAILLE 500

// Test with an affine condition inside a loop
int main () {
  int A[TAILLE];
  int i = 0;
  int condition;
  
  for (i = 0; i < TAILLE; i++) {
    condition = 2*i;
    if (condition - TAILLE == 0)
      A[i] = 40;
  }
   
    
  return A[0];
}
