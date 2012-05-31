#include<stdio.h>

int main()
{
  int i=5;
  int j = 6;
  float x;
  double y[10];
  char * fmt = "i=%d, %d, %f\n";

  y[0]=0.;
  printf(fmt, i, i+j, x);
  return 0;
}
