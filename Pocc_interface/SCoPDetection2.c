#include <stdio.h>
#include <stdlib.h>

#define TAILLE 500
#define SQRTTAILLE 22

// Test with a non affine expression
int main () {
  int A[TAILLE];
  int i = 0;
  
  for (int i = 0; i < SQRTTAILLE; i++)
    A[i*i] = 400;
  
  return A[0];
}
