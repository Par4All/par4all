#include <stdlib.h>

int main()
{
  int i, a[10], *p;
  p = &a[0];
  i=0;
  
  while (p<&a[10]) {
    p++;
    i++;
    i++;
  }
  
  return i;
}
