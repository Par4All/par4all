#include <stdio.h>
#include <stdlib.h>

#define TAILLE 500
#define SQRTTAILLE 22

void function_test (int A[TAILLE], int i) {
  for(i = 0; i < SQRTTAILLE; i++)
    A[i*i] = 30;
}

int main () {
  int A[TAILLE];
  int i = 0;
  function_test(A, i);
  return 0;
}
