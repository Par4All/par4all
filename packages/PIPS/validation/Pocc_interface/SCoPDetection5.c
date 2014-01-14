#include <stdio.h>
#include <stdlib.h>

#define TAILLE 500

// Test with while
int main () {
  int A[TAILLE];
  int i = 0;
  while (i < TAILLE) {
    A[i] = 400;
    i++;
  }
  return A[0];
}
