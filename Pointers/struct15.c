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
