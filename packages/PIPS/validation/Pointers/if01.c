#include <stdio.h>
int main() {
  int *x, x1, x2, y;
  x1 = 1;
  x2 = 2;

  if(0==0)
    {
      x = &x1;
    }
  else
    {
      x =&x2;
    }
  
  y = *x;
  return 0;
}
