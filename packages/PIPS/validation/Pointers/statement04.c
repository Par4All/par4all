/* Buggy source code: x is not initialized. */

#include <stdio.h>
int main() {
  int **x, *y, k, h;
  k=1;
  h=2;
  y = &h;
  *x = y;
  printf("x = %p \n", x);
  return 0;
}
