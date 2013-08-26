#include <stdlib.h>
#include <stdio.h>

#define TAILLE 500

int main () {
  int A[TAILLE];
  int B[TAILLE];
  int i = 0;
  int j = 0;
  int ind;

  // Testing access to an array using a variable defined using another array
  for (j = 0; j < TAILLE; j++)
    B[j] = j;
  for (i = 0; i < TAILLE; i++) {
    ind = B[i];
    A[ind] = 123;
  }
  
  return A[0];
}
