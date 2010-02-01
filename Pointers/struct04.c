#include<stdio.h>
int struct004()
{
  struct {
    int **q;
    int **p;
  }m, n;
  
  int i=0, j=1, k=2;
  int *r, *s;
  r = &k;
  s = &j;
  m.q = &r;
  m.p = &s;
  m.q = m.p;
  *m.p = &k;
  n = m;
  return 0;
}
