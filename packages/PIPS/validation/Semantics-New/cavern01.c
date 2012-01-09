// STIC 2012, CAVERN presentation

// Modeling non-linear computation

#include <assert.h>

int cavern01(int x1, int x2, int x3)
{
  int r = x2;
  // assert(x1>=2); // statement r=x1 is found not reachable
  // assert(x1<=-2); // statement r=x1 is found not reachable
  // assert(x1>=1);
  if(x1==x2 && x2==x3)
    if(x1*x2==x3)
      r = x1;
  if(x1*x1==x1)
      r = x1;
  return r;
}
