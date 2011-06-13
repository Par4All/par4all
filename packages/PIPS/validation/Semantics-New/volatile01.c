// Check volatile qualifier

#include <stdio.h>

void volatile01()
{
  int volatile i = 42;
  int j = 84;

  printf("%d, %d\n", i, j);
  return;
}

