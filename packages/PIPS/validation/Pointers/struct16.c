/* Same as struct15.c, but with a non-strict type property in struct16.tpips */

#include <stdio.h>

struct one {
  int **first;
  int **second;
  int **third;
} x;

int main() {
  int y[10];

  *(x.first) = &y[5];
  *(x.second) = &y[6];
  *(x.third) = &y[7];

  return 0;
}
