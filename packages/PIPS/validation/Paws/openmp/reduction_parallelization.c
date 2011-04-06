#include <stdio.h>

int reduction_parallelization (void) {
  int i;
  int a[100], s;

  for (i = 0; i < 100; ++i) {
    s = s + a[i];
  }

  return s;
}
