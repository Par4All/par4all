
/* Check pointers to arrays */

#include <stdio.h>

int main() {
  int t[10][10][10];

  int *p1 = t[10][10];
  //int **p2 = t[10];
  int (*p2)[10] = t[10];
  //int ***p3 = t;
  int (*p3)[10][10] = t;

  //int ***q = &t;
  int (*q)[10][10][10] = &t;

  int *r1 = &t[10][10][0];
  //int **r2 = &t[10][0];
  int (*r2)[10] = &t[10][0];
  //int ***r3 = &t[0];
  int (*r3)[10][10] = &t[0];

  return 0;
}
