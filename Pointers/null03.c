// Check kill by a pointer assignment

// Issue observed in rand04

#include <stdio.h>

int main()
{
  int * p;
  int i;
  p = &i;
  p = NULL;

  return 0;
}
