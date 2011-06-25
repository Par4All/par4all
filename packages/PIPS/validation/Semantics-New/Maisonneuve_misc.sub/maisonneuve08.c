// This program combine three different uses of b: identity, reset
// (involutive) and increment, but in the loop b is always positive

// $Id$

#include <stdlib.h>

void run(void) {
  int b = 0;
  while(1) {
    if (rand() % 2)
      ;
    else if (rand() % 2)
      b = 0;
    else
      b++;
  }
}

int main(void) {
  run();
  return 0;
}

