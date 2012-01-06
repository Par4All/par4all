// STIC 2012, expose CAVERN

// Disjunction about i, loop entered or not: if the convex hull is
// used, the information r>=402 is not found

// Should we use transformer lists on sequences? What is the issue?

#include <assert.h>

int cavern02(int i)
{
  // assert(i<=1); // The loop is not entered and j==100
  // assert(i>1); // The loop is entered
  int r = i;
  int j = 100;
  while(i>1)
    j++, i--;
  if(j>500)
    r = i;
  return r;
}
