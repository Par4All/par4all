#include <stdlib.h>

void function_test(float A[500], int i)  {
  int x;
#pragma scop
  x = 0;

  for (i = 0; i < 400; i++) {
    x += 2;
    A[i] = 2;
  }

#pragma endscop
  return;

}

