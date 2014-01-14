#include <stdio.h>
#include <stdlib.h>

#define TAILLE 500

void function_test (int A[TAILLE][TAILLE], int B[TAILLE], int i, int j) {
  int t;
  for (i = 0; i < TAILLE; i++) {
    t = B[i];
    for(j = 0; j < TAILLE; j++) {
      A[t][j] = 1234;
    }
  }
}

int main () {
  int A[TAILLE][TAILLE];
  int B[TAILLE];
  int i = 0;
  int j = 0;
  function_test(A, B, i, j);
  return 0;
}
