#include <stdio.h>

int main() {
  int **x, **y, *z, *t;
  int i, j;
  i=1;
  j=2;

  z = &i;
  t = &j;
  
  z = t;
  *x = z;

  printf(" x = %p\n",x);

  return 0;
}
