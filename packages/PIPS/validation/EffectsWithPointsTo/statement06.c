#include <stdio.h>
int main() {
  int a, b, c, d;
  int *pa, *pb, *pc, *pd;
  int **ppa, **ppb, **ppc, **ppd;
  a = 1;
  b = 2;
  c = 3;
  d = 4;

  pa = &a;
  ppa = &pa;
  *ppa = &b;

  pb = &b;
  ppb = &pb;
  *ppb = &a;

  pc = &c;
  ppc = &pc;
  *ppc = &a;

  pd = &d;
  ppd = &pd;
  *ppd = &c;

  pc = pd;

  *ppc = &b;

  *ppa = pc;

  pa = *ppd;

  *ppa = *ppd;

  return 0;
}
