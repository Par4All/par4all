// this program has an invariant of kind -b<=a.s<=b which cannot be
// derived from T'(dx)

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
  while(1) {
    if(alea() && x >= -b) x--;
    if(alea() && x<=b) x++;
    t++;
  }

}

int main(void)
{
  run();
  return 0;
}

