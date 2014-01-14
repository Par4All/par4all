#include <stdio.h>
#include <stdlib.h>

#define TAILLE 500
#define SQRTTAILLE 22

void function_test (int A[TAILLE][TAILLE][TAILLE], int B[TAILLE], int i, int j, int k) {
  for (i = 0; i < TAILLE; i++)
    A[i][1][1] = B[i];
	
  for (i = 0; i < TAILLE/2; i++)
    for (j = 0; j < SQRTTAILLE; j++)
      for(k = 0; k < TAILLE; k++)
	A[(i*j*k%2==0)?i:2*i][j*j][k] = 1234;
}

int main () {
  int A[TAILLE][TAILLE][TAILLE];
  int B[TAILLE];
  int i = 0;
  int j = 0;
  int k = 0;
  function_test(A, B, i, j, k);
  return 0;
}
