// Rewritting of maisonneuve07-1 as a single state automaton with 3 loops
// PIPS is still not able to find the invariant

// $Id$

#include <stdlib.h>

void run(void) {
  int x = 0;
  int t = 0;
  int b = 10; // b > 0 should be enough
  while(1) {
    while(rand() % 2 && x >= -b) x--, t++;
    while(rand() % 2 && x<=b) x++, t++;
		while(rand() % 2) t++;
  }
}

int main(void) {
  run();
  return 0;
}

