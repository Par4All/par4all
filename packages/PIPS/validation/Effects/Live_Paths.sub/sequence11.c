// test liveness analysis on pointer

#include <stdlib.h>

int main () {
  int i;
  int* p;
  int** q;

  p = malloc (5 * sizeof (int));
  q = malloc (5 * sizeof (int*));

  for (i = 0; i < 5; i++) {
    p[i] = i;
    q[i]= &(p[i]);
  }

  return q[2][1];
}
