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
  int i = 0, j = 0;
  int **pp = NULL;

  srand(time(NULL));
  if (alea())
  {
    int * pi = &i;
    pp = &pi;
  }
  else
  {
    int * pj = &j;
    pp = &pj;
  }

  printf("pp=%p\n", pp);

  // possible segfault.
  // should generate an error while analyzing?
  **pp = 0;
  return 0;
}
