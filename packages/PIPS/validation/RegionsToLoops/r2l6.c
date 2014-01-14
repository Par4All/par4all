#include <stdio.h>
#include <stdlib.h>

#define TAILLE 500

void function_test (int A[TAILLE], int i) {
  int t = 0;
  for (i = 0; i < TAILLE; i++)
    t = A[i];
}

int main () {
  int A[TAILLE];
  int i = 0;
  function_test(A, i);
  return 0;
}
