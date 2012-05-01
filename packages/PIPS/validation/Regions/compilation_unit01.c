// goes with compilation_unit01_bis

#include <stdio.h>

typedef struct {
  float re;
  float im;
} CplFloat;

int main()
{
  CplFloat A[10];

  init(A);

  for (int i =0; i<10; i++)
    printf("A[%d] = %f + i %f\n", i, A[i].re, A[i].im);

  return 0;
}
