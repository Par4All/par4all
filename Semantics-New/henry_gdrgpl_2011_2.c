// Julien Henry (Verimag)
// Path focusing...
// GDR GPL 2011, Lille 2011-06-08, slide 19
// x oscillations between 0 and 1000

#include <stdio.h>
#define true 1

int main(void)
{
  int x = 0;
  int d = 1;
  // infinite loop
  while (true)
  {
    fprintf(stdout, "loop: x=%d d=%d\n", x, d);
    if (x==0) d=1;
    if (x==1000) d=-1;
    x += d;
  }
  // obviously never reached:
  // there is no exit from the previous while loop.
  return 0;
}
