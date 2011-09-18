// Simplify control: no iterations

#include <stdio.h>

void loop04()
{
  int i = 0;

  for(i=3;i<1;i++)
    /* This statement is never executed */
    printf("%d\n", i);

  return;
}
