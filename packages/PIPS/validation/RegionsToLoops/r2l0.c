#include <stdio.h>
#include <stdlib.h>

#define TAILLE 500

void function_test (int A[TAILLE], int i) {
  for(i = 0; i < TAILLE; i++)
    A[i] = 12345;
}

int main () {
  int A[TAILLE];
  int i = 0;
  function_test(A, i);
  return 0;
}
