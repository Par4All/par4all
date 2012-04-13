
#include <stdio.h>

int main() {
  int *x, x1, **y, *y1, y2;
  
  x1 = 0;
  y2 = 1;
  x = &x1;
  y1 = &y2;
  y = &y1;
  x = *y;
  printf("*x=%d\n", *x);
  *y = x;
  return 0;
}
