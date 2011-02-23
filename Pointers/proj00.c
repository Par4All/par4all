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
  int i;
  int **pp;
  srand(time(NULL));

  if (alea())
  {
    int * pi = &i;
    pp = &pi;

    **pp = 1;
  }

  // pp points to a dead value in the stack
  // should generate an error
  // **pp = 0; // segfault?

  printf("%p\n", pp);
  return 0;
}
