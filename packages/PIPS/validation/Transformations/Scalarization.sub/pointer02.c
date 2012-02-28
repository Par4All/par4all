/* Check problems with pointers... */

#include <stdio.h>

int main()
{
  int * p;
  int a[1000];
  int i=0;

    p = &a[i];
    a[i]=1;
    *p = 0;
    a[i]++;

  printf("a[0]=%d\n", a[0]);
}
