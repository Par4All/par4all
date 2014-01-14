#include <stdlib.h>
#include <stdio.h>

#define TAILLE 500

int main () {
  int A[TAILLE];
  int B[TAILLE];
  int i = 0;
  int j = 0;
  int ind;
  // Test using assignment within an enclosed loop
  for (j = 0; j < TAILLE; j++) {
    B[j] = j;
    ind = B[j];
    for (i = 0; i < TAILLE; i++) {
      A[ind] = 1234;
    }
  }
    
  return 0;
}
