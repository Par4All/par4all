#include <stdio.h>
/* we must start using predicates on the approximations when we compute points-to relations */
#include <stdlib.h>
#include <time.h>

static int alea(void)
{
  int r = rand();
  printf("%d\n", r);
  return rand()%2 == 1;
}

int main(void)
{
  int i1, i2;
  int * pi1 = &i1;
  int * pi2 = &i2;
  int **pp;
  srand(time(NULL));
  if (alea())
    pp = &pi1;
  else
    pp = &pi2;
  *pp = NULL;
  printf("%p %p %p\n", pp, pi1, pi2);
  return 0;
}
