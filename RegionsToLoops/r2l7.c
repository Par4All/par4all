#include <stdio.h>
#include <stdlib.h>

#define TAILLE 500

void function_test (int A[TAILLE][TAILLE], int i, int j) {
  for(i = 0; i < TAILLE; i++)
    for(j = 0; j < TAILLE; j++)
      A[i][j] = 1234;
}

int main () {
  int A[TAILLE][TAILLE];
  int i = 0;
  int j = 0;
  function_test(A, i, j);
  return 0;
}
