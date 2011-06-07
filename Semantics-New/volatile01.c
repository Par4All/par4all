// Check volatile qualifier

#include <stdio.h>

void volatile01()
{
  int volatile i = 42;

  printf("%d\n", i);
  return;
}

