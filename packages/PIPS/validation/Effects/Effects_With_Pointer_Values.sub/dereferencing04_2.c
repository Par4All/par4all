#include <stdlib.h>

#define N 10

void dereferencing04()
{
  double * p = (double *) malloc (N*sizeof(double));
  double * q = (double *) malloc (N*sizeof(double));
  double * r = (double *) malloc (N*sizeof(double));

  *(p+(q-r)) = 0.;
}

int main()
{
  dereferencing04();
  return 1;
}

