// Check that useless until loops are removed

#include <stdio.h>

void until01()
{
  int i = 1;
  do {
    i++;
  } while(0);
  printf("%d", i);
}
