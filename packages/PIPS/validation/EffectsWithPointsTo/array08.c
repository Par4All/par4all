
#include <stdio.h>

int main() {
  int t[100][10][3];

  int (*p)[3];
  int (*r)[100][10][3];

  p = &t[1][2];
  r = &t;
  (*r)[0][0][0] = 0;
  return 0;
}
