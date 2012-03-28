/* Check pointer arithmetic
 *
 */

#include <stdio.h>

int main()
{
  int a[10];
  int * q = &a[0];
  int * p = &a[0];
  int i;

  for(i=0;i<10;i++)
    a[i]=i;

  *q++;
  printf("p=%p, q=%p, *q=%d\n", p, q, *q);
  return 0;
}
