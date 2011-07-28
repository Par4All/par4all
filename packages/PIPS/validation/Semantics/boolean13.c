// Check issue with boolean variables used as integer variables

// Subset of boolean12.c

#include <stdbool.h>
#include <stdio.h>

int main(void)
{
  bool stabilize = 1;
  int j, k;

  j = stabilize+stabilize;
  k = 3*stabilize;

  printf("j=%d, k=%d\n", j, k);

  return j+k;
}
