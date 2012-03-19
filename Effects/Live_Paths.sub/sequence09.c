// test liveness analysis on pointer;
#include <stdlib.h>

int main () {
  int i;
  int* p;

  p = malloc (5 * sizeof (int));

  for (i = 0; i < 5; i++) {
    p[i] = i;
  }

  return p[1];
}
