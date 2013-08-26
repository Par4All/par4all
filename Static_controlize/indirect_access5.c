#include <stdlib.h>
#include <stdio.h>

#define TAILLE 500

int main () {
  int A[TAILLE][TAILLE];
  int B[TAILLE];
  int i = 0;
  int j = 0;
  int ind;

  // Testing using multidimensional array (just in case)
  for (j = 0; j < TAILLE; j++)
    B[j] = j;
  for (i = 0; i < TAILLE; i++) {
    ind = 4 + B[i];
    A[i][ind] = 1234;
  }
  
  return A[0][0];
}
