#include<stdio.h>
int struct03()
{
  struct {
    int **q;
    int **p;
  }m, n;
  
  int i=0, j=1, k=2;
  int *r;
  r = &k;
  m.q = &i;
  m.p = &j;
  n = m;
  r = n.p;
  return 0;
}
