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
  int i1;
  int * pi1 = &i1;
  int **pp;
  srand(time(NULL));
  if (alea())
    pp = &pi1;
  
  *pp = NULL;
  printf("%p %p \n", pp, pi1);
  return 0;
}
