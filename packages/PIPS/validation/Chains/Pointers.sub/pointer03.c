/* Parallelizarion with pointers */
#include<stdlib.h>
void pointer03(int n)
{
  int i;
  int *p = malloc(10*sizeof(*p));
  int *r = p;

  for(i=0; i<n; i++)
    p[i] = r[i];
}
