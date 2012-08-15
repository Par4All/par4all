/* Exmaple taken from Rice */

/* Parallelizarion with pointers */

#include <assert.h>
#include <stdio.h>

void pointer02(int n, float *p)
{
  int i;
  assert(p!=NULL);
  float * r = p-1;

  for(i=0; i<n; i++)
    p[i] = r[i];
  return;
}
