/* Check naming of stubs according to fields used */

#include <stdio.h>

struct one {
  int **first;
  char **second;
  float **third;
} x;

int main() {
  int y[10];
  char z[10];
  float v[10];

  *(x.first) = &y[5];
  *(x.second) = &z[6];
  *(x.third) = &v[7];

  return 0;
}
