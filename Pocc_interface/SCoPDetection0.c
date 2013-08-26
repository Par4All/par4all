#include <stdio.h>
#include <stdlib.h>

#define TAILLE 500

// Test for a simple loop
int main () {
  int A[TAILLE];
  int i = 0;

  for (int i = 0; i < TAILLE; i++)
    A[i] = 230;
  
  return A[0];
}
