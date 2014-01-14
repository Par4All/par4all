#include <stdio.h>
#include <stdlib.h>

#define TAILLE 500

// Test with affine expression but using a variable defined outside a loop;
// Does not work, need to call PARTIAL_EVAL[main] before calling this pass in this case
int main () {
  int A[TAILLE];
  int i = 0;
  int k = 3;
  for (i = 1; i < TAILLE/2; i++)
    A[k*i] = 400;
  
  return A[0];
}
