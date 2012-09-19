#include <stdio.h>

void dereferencing06()
{
  double x[3] = {1., 2., 3.};
  double * p = x;

  printf("%f\n", x[2]);
  *p++=0.;
  *p++=1.;
  *p=2.;
  printf("%f\n", x[2]);
}

int main()
{
  dereferencing06();
  return 0;
}
