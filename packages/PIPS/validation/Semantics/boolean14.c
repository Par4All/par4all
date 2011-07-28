// Check issue with boolean variables used as integer variables

// Subset of boolean13.c

#include <stdbool.h>
#include <stdio.h>

int main(void)
{
  bool stabilize = 1;
  int k;

  k = 3*stabilize;

  printf("k=%d\n", k);

  return k;
}
