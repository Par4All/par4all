// Check that useless until loops are removed

// Check that labels are preserved...

#include <stdio.h>

void foo(int *i)
{
  *i = 0;
}

void until05()
{
  int old, new;
 do {
   old = new;
   foo(&new);
  } while(old!=new);
  printf("%d", old);
}
