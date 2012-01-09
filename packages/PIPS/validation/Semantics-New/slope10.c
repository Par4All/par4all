// Goal: check constraint simplification by function
// small_positive_slope_reduce_coefficients_with_bounding_box()

// Bug found in linked_regions02

#include <assert.h>

void slope10()
{
  int x, y, k;

  assert(0<=x && x<=50);

  if(2*x-99*y<=0) {
    k = x + y;
  }
  else {
    k = x + y;
  }

  return;
}
