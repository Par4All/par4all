// after call to exchange, pa points to b and pb points to a
#include <stdlib.h>
#include <stdio.h>

void exchange(int **p, int **q)
{
  int * r;

  r= *p;
  *p = *q;
  *q = r;
}


int main()
{
  int * pa, *pb, a, b;

  a = 1;
  b = 2;

  pa = &a;
  pb = &b;

  printf("*pa = %d, *pb = %d\n", *pa, *pb);

  exchange (&pa, &pb);

  printf("*pa = %d, *pb = %d\n", *pa, *pb);
  return 0;
}
