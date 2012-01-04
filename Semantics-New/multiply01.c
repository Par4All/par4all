// Bug in simplification of multiplication constraints.

// The bug was in sc_bounded_normalization

// First observed with linked_regions02. see multiply01.

#include "assert.h"

void multiply01(int i, int j, int N)
{
  int k;
  assert(1<=i && i<=100 && 1<=j && j<=100 && N==100);

  if(2*i*j<N)
    k++;
}
