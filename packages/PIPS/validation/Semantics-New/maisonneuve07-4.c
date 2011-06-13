// This program has an invariant of kind -b<=x<=b which cannot be
// derived directly from T'(dx)
//
// This is a copy of maisonneuve07-1.c, but b is no longer a known
// constant as in maisonneuve07-3.c; however, x is still initialized
// to 0

// $Id$

#include <stdlib.h>

void run(void) {
  int x = 0;
  int t = 0;
  int b = rand(); // b > 0 should be enough
  while(1) {
    if(rand() % 2 && x >= -b) x--;
    if(rand() % 2 && x<=b) x++;
    t++;
  }
}

int main(void) {
  run();
  return 0;
}

