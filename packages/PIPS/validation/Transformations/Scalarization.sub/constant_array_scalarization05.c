// Check that array initializations are properly processed

#include <stdio.h>

void constant_array_scalarization05()
{
  int a[3] = { 1, 2, 3}, b[3], c[3], i;

  b[2] = a[2];
  b[1] = a[1];
  b[0] = a[0];

  c[i] = 0;

  printf("%d %d %d %d\n", b[0], b[1], b[2], c[i]);
}
