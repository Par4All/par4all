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

void while05()
{
  int old, new;
  while(old!=new) {
    old = new;
    foo(&new);
  } 

  foo(&old);

  while(old!=new) {
    old = new;
    bar(&new);
  }
  printf("%d", old);
}
