// Segfaults in deprecated versions of PIPS.

// $Id$

#include <stdlib.h>

void run(void) {
  int x;
  while (rand() % 2) {
    if (rand() % 2) {
      if (x < 0) {
        x++;
      }
    }
  }
}

int main(void) {
  run();
  return 0;
}

