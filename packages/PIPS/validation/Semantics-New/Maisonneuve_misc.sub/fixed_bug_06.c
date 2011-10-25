// Wrong preconditions found by PIPS r20573 (i == 0 instead of i >= 0),
// using transformer lists.

// $Id$

#include <stdlib.h>

void run(void)
{
  int i = 0;
  while (rand()) {
    if (rand()) goto incr;
  incr:
    i++;
  }
  return;
}

int main(void)
{
  run();
  return 0;
}
