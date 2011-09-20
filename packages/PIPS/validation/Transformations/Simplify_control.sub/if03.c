// Simplify control directly: always true test

#include <stdio.h>

void if03()
{
  int i = 0;

  if(1<2)
    printf("%d\n", i);

  return;
}
