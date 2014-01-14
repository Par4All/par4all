#include <stdio.h>
#include <stdlib.h>

#define TAILLE 500

// Test for loop nest for which only one loop is static control 
int main () {
  int A[TAILLE];
  int B[TAILLE];
  int i = 0;
  int j = 0;
  for (i = 0; i < TAILLE; i++) {
    A[i*i] = 123;
    for (j = 0; j < TAILLE; j++) {
      B[j] = 123;
    }
  }
  return A[0];
}
