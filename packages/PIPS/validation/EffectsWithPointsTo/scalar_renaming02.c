#include <stdio.h>
int scalar_renaming02()
{
  int *p, a, b, c;
  p = &a;
  a=1;
  a=2;
  b=a;
  a=3;
  *p = 4;
  c=a;
  printf("%d-%d-%d",a,b,c);
  return 0;
}
