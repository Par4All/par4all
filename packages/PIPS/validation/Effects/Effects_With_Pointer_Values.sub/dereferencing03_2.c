#include <stdlib.h>

void dereferencing03()
{
  double * p;
  int i = 5;

  p = (double *) malloc(10*sizeof(double));

  *(p+1) = 0.;
  *(p+i) = 0.;
}

int main()
{
  dereferencing03();
  return 1;
}
