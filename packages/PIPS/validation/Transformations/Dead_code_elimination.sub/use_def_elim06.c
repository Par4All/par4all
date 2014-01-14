// Test case added by Francois Irigoin
//
// This test case was designed to see if control dependences are taken
// into account, but they are not.
//
// The output could be OK, if suppress_dead_code were applied first.

#include "stdio.h"

int main(void) {
  int i = 1;
  exit(1);
  printf("%d\n", i);
}

