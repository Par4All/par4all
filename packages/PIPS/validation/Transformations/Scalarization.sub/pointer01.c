/* Check problems with pointers... */

#include <stdio.h>

int main()
{
  int * p;
  int a[1000];
  int i;

  for(i=0;i<1000;i++) {
    p = &a[i];
    a[i]=1;
    *p = 0;
    a[i]++;
  }

  printf("a[0]=%d\n", a[0]);
}
