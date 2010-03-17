#include <stdio.h>
int main() {
  int *x, *y, k[5], h;
  k=1;
  h=2;

  x = &k[2];
  y = &h;
  
  x = y;

  return 0;
}
