
/* From Ronan Keryell (see trac ticket #150) */

#include <stdio.h>

int main() {
  int t[100][10][3];

  int (*p)[3];
  int *pu;

  int (*q)[10][3];
  int (*qu)[3];

  int (*r)[100][10][3];
  int (*ru)[10][3];

  printf("sizeof(t) = %zd\n", sizeof(t));
  printf("sizeof(t[1]) = %zd\n", sizeof(t[1]));
  printf("sizeof(t[1][2]) = %zd\n", sizeof(t[1][2]));
  printf("sizeof(t[1][2][3]) = %zd\n", sizeof(t[1][2][3]));

  p = &t[1][2];
  pu = t[1][2];
  pu = &t[1][2][0];

  q = &t[1];
  qu = t[1];
  qu = &t[1][0];

  r = &t;
  ru = t;
  ru = &t[0];
  return 0;
}
