#include <stdio.h>

main()
{
  int m = 3;
  int n = 2;
  double a[m][n];
  m++;
  n++;
  double b[m][n];
  printf("size of a = %d\n", sizeof(a));
  printf("size of b = %d\n", sizeof(b));
}
