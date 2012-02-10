// Check that useless until loops are removed without preconditions

#include <stdio.h>
#include <stdbool.h>

void until08()
{
  int i = 1;
  do {
    i++;
  } while(0);
  printf("%d", i);
  do {
    i++;
  } while(false);
  printf("%d", i);
}
