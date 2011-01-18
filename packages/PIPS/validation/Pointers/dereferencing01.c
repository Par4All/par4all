#include<stdio.h>

int main()
{
  double x = 1.;
  double * p = &x;
  double * q = 0;
  printf("q = %p",q);
  q = p;
  *p = 2.;
  return 0;
}
