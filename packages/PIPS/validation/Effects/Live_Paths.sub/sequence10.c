// test liveness analysis on pointer with aliasing

#include <stdlib.h>

int main () {
  int i;
  int* p;
  int* q;

  p = malloc (5 * sizeof (int));
  q = &(p[2]);

  for (i = 0; i < 5; i++) {
    p[i] = i;
  }

  return p[1] + q[1];
}
