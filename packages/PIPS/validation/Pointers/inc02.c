/* test case to illustrate how use-def can't be computed without
   points-to */

#include <stdio.h>

int main()
{
  int i = 0;
  int * q = &i;
  int * p = &i;
  // Increment i
  (*q)++;
  // increment q
  *q++; // The loaded value is not used
  printf("i=%d, p=%p, q=%p\n", i, p, q);
  return 0;
}
