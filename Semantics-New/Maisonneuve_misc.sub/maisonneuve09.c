// This program combine three different uses of b: identity, reset and
// increment, but in the loop b is always positive. Unlike
// maisonneuve08, another variable, i, muddles the situation

// $Id$

#include <stdlib.h>

int flip() {
  return rand() % 2;
}

void run(void)
{
  int b = 0;
  int i = 0;
  while(1) {
    if(flip())
      i++;
    else if(flip())
      b = 0, i++;
    else
      b++, i++;
  }
}

int main(void) {
  run();
  return 0;
}

