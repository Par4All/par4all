// Code C, non parallelisable sans les effets constants
#include<stdlib.h>
void pointer07(double *a, double *b,double *c, int cnt)
{
  double *w = NULL; 
  int i;
   
  w = (double *) malloc(cnt * sizeof(double));
  for (i = 0; i < cnt; i++)
    w[i]  = c[i] + a[i] * b[i];
}

