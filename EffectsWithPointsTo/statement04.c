/* This code is buggy because x is used uninitialized. This is
 * detected by the points-to analysis, but not fully acknowledge by
 * the proper effect analysis.
 */

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
