/* Excerpt from array05, used for debugging */

#include <stdio.h>

int main() {
  int t[100][10][3];

  int (*p)[3];

  p = &t[1][2];
  (*p)[1] = 0;

  return 0;
}
