#include <stdio.h>
#include <time.h>

int main(void)
{
  int i, j;
  int **pp;
  
  {
    int * pi = &i;
    pp = &pi;
    **pp = 1;
  }

  // pp points to a dead value in the stack
  // should generate an error
  **pp = 0; // segfault?
  *pp = &j;
  return 0;
}
