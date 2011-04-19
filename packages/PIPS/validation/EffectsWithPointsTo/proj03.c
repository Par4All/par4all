#include <stdio.h>
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
  int i = 1;
  int * pi = &i;
  int **pp = &pi;
  int j = 2;

  srand(time(NULL));

  if (alea())
  {
    int * pj = &j;
    pp = &pj;
    **pp = 0;
  }

  // pp may points to a dead value in the stack
  printf("%p\n", pp);

  // possible segfault.
  // should generate an error while analyzing?
  **pp = 0;
  return 0;
}
