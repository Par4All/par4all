// this program has an invariant of kind -b<=a.s<=b which cannot be
// derived from T'(dx)

// New version of maisonneuve07.c, implemented with nested whiles
// instead of while, and some more modifications.

// $Id: $

#include <stdlib.h>
#include <stdio.h>

int alea() {
  return rand() % 2;
}

void run(void)
{
  int x = 0;
  int t = 0;
  int b = 10; // b>0 should be enough
  // while(x >= -b || x<=b) { no results at all with an or which is
  // always true...
  while(x >= -b && x<=b) { // this restrict the initial state and the
			   // boundaries of maisonneuve07.c
    if(1) {
      if(1) while(alea() && x>=-b) x--, t++;
      if(1) while(alea() && x<=b) x++, t++;
      if(1) while(alea()) t++;
    }
  }
  return;
}

int main(void)
{
  run();
  return 0;
}

