// Check issue with boolean variables used as integer variables

#include <stdbool.h>
#include <stdio.h>

int main(void)
{
  bool stabilize = 1;
  int i = 1, j, k;

  if(stabilize==1)
    i = 2;

  if(stabilize) {
    i = 0;
    j = stabilize+stabilize;
    k = 3*stabilize;
  }

  printf("i=%d, j=%d, k=%d\n", i, j, k);

  return i+j+k;
}
