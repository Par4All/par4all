// global variable used as loop index
// re-used afterwards in the same module -> loop must not be parallelized

#include <stdio.h>
int i;

int main()
{
  int j,a[10];

  for (i = 0; i <10; i++)
      a[i] = i;

  for (j = 0; j<10; j++)
    printf("a[%d] = %d\n", j, a[j]);

  return i;
}
