// Check that useless until loops are removed

// Check that effects are taken into account when available

// See also until05.c. Only a tpips option is different to take potential effects of bar into account

#include <stdio.h>

void foo(int *i)
{
  *i = 0;
}

void bar(int *i)
{
}

void until06()
{
  int old, new;
 do {
   old = new;
   foo(&new);
  } while(old!=new);

 do {
   old = new;
   bar(&new);
  } while(old!=new);
  printf("%d", old);
}
