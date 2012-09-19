/* Check the handling of an update */

#include <stdio.h>

void update01()
{
  int i[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int * p, *q;

  p = i;
  q= &i[0];
  p++;
  q++;
  *p = 2;
  *q = 3;
  printf("a[1]=%d\n", i[1]);
}

int main()
{
  update01();
}
