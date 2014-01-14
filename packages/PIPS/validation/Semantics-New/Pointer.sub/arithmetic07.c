//same than arithmetic05 but with a var instead of a constant

#include <stdlib.h>

int main()
{
  int a[20], i=2, j=3;
  int *p, *q;
  
  q=&a[0];
  p=q+i*j;
  
  return 0;
}
