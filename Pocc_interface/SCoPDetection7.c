#include <stdio.h>
#include <stdlib.h>

#define TAILLE 500

// Test with affine condition but only one SCoP
int main () {
  int A[TAILLE];
  int i = 0;
  int condition = 0;
  
  if (condition) {
    for (i = 0; i < TAILLE; i++)
      A[i] = 0;
  }
  else {
    for (i = 0; i < TAILLE; i++)
      A[i*i] = 0;
  }
    
  return 0;
}
