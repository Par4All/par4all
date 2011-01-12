// rewritting of maisonneuve07.c as a single state automaton with 3 loops
// still not able to find the invariant

// $Id$

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
  while(1) {
    while(alea() && x >= -b) x--, t++;
    while(alea() && x<=b) x++, t++;
		while(alea()) t++;
  }

}

int main(void)
{
  run();
  return 0;
}

