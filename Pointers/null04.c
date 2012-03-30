// Check that an assignment by NULL kills previous MAY points-to

// Issue observed in rand04

#include <stdio.h>

int main()
{
  int * p;
  int i, j;

  if(i==j)
    p = &i;
  else
    p = &j;

  p = NULL;

  return 0;
}
