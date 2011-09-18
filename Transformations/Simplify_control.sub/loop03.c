// Simplify control: exactly one iteration

#include <stdio.h>

void loop03()
{
  int i = 0;

  for(i=0;i<1;i++)
    /* This loop is executed exactly once */
    printf("%d\n", i);

  return;
}
