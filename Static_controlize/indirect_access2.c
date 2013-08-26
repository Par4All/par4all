#include <stdlib.h>
#include <stdio.h>

#define TAILLE 500

int main () {
  int A[TAILLE];
  int i = 0;
  int ind;
  // Compared to test 1, allows to check if we did not prevent access using a loop-bound affine dependent variable
  for (i = 0; i < (TAILLE/2 - 3); i++) {
    ind = 2*i + 3;
    A[ind] = 1234;
  }
    
  return A[0];
}
