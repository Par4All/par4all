#include <stdio.h>
#include <stdlib.h>

#define TAILLE 500

void print() {
  printf("Hello World \n");
}

// Test for calls with STATIC_CONTROLIZE_ACROSS_USER_CALL TRUE
int main () {
  int A[TAILLE];
  int i = 0;
  for (i = 0; i < TAILLE; i++)
    A[i] = 0;
  print();
  for (i = 1; i < TAILLE; i++)
    A[i-1] = 1;
  
  return A[0];
}
