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

void while07()
{
  int old, new;
  int i;

  while(old!=new) {
    old = new;
    foo(&i);
  } 

  foo(&old);

  while(old!=new) {
    old = new;
    bar(&i);
  }
  printf("%d", old);
}
