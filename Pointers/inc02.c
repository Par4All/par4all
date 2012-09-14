/* test case to illustrate how use-def can't be computed without
   points-to */

#include <stdio.h>

int main()
{
  int i[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int * q = &i[0];
  int * p = &i[0];
  // Increment i[0]
  (*q)++;
  // increment q
  int j = *q++; // The loaded value is not used
  printf("i=%d, p=%p, q=%p\n", i[0], p, q);
  return j;
}
