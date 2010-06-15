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
  ppa = &b;
  *ppa = &pa;

  pb = &b;
  ppb = &a;
  *ppb = &pb;

  pc = &c;
  ppc = &d;
  *ppc = &pc;

  pd = &d;
  ppd = &c;
  *ppd = &pd;

  pc = pd;

  *ppc = &a;

  *ppa = pc;

  pa = *ppd;

  *ppa = *ppd;

  return 0;
}
