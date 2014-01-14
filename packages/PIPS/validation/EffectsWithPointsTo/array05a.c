
/* Excerpt of array05: issue with variable r */

#include <stdio.h>

int main() {
  int t[100][10][3];

  int (*r)[100][10][3];

  r = &t;
  (*r)[0][0][0] = 0;
  (*r)[1][2][3] = 0;

  return 0;
}
