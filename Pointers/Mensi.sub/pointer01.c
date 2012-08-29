/* Parallelization with pointers */

#include <stdio.h>

void pointer01(int n, float *p, float *q)
{
  int i;

  for(i=0; i<n; i++)
      p[i] = q[i];
 
}
