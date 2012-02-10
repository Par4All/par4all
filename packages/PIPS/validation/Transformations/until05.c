// Check that useless until loops are removed

// Check that effects are taken into account when available

// See also until06.c

#include <stdio.h>

void foo(int *i)
{
  *i = 0;
}

void bar(int *i)
{
  ;
}

void until05()
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
