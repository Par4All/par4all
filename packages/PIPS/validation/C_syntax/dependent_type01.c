/* This code is C99. It is not parsed by the PIPS C parser. */

#include <stdio.h>

main()
{
  int m = 3;
  int n = 2;
  double a[m][n];
  m++;
  n++;
  double b[m][n];
  printf("size of a = %d\n", (int) sizeof(a));
  printf("size of b = %d\n", (int) sizeof(b));
  return 0;
}
