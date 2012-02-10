// Check that useless until loops are removed

// Check that effects are taken into account when available

// See also until05.c and until06.c. Only a tpips option is different to take
// potential effects of bar into account

#include <stdio.h>

void foo(int *i)
{
  *i = 0;
}

void bar(int *i)
{
}

void until07()
{
  int old, new;
 do {
   int i;
   old = new;
   foo(&i);
  } while(old!=new);

 do {
   int i;
   old = new;
   bar(&i);
  } while(old!=new);
  printf("%d", old);
}
