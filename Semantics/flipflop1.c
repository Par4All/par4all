#include <stdio.h>

double g(double n)
{
  return n;
}

void flipflop(n,m)
{
  double x[2][m];
  int old = 0, new = 1, i, t;

  for(t=0; t<n;t++) {
    //   for(i=0;i<m;i++)
    //    x[new][i] = g(x[old][i]);
    old = new;
    new = 1 - old;
    new=new;
  }

  if (new+old==1)
    printf("Property verified");
}

void main(){
  flipflop(1000,10);
}

