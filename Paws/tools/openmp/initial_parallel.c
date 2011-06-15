#include <stdio.h>

int initial_parallel (void) {
  int i;
  double t, s, a[100];

  for (i = 0; i < 50; ++i) {
    t = a[i];
    a[i+50] = t + (a[i] + a[i + 50]) / 2.0;
    s = s + 2*a[i];
  }

  return 0;
}
