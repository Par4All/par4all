//same than arithmetic05 but with a var instead of a constant

#include <stdlib.h>

int main()
{
  int a[10], i=2;
  int *p, *q;
  
  q=&a[0];
  p=q+i;
  
  return 0;
}
