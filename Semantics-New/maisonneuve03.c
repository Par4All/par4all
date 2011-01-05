// This example makes PIPS segfault (at least revisions 18437 & 18499)

// $Id$

#include <stdlib.h>

int flip(void)
{
  return rand() % 2;
}

void run(void)
{
  int x;

  while (flip()) {
    if (flip()) {
      if (x < 0) {
	x++;
      }
    }
  }
  /* We'd like some information about x at loop exit, if any. */
  ;
}

int main(void)
{
  run();
  return 0;
}

