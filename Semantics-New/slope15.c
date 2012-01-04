// Goal: check constraint simplification by function
// small_positive_slope_reduce_coefficients_with_bounding_box()

// Synthetic case: how to find the intermediate points? Here, three
// intermediate points are needed

#include <assert.h>

void slope15()
{
  int x, y, k;

  assert(2<=x && x<=11);

  if(12*y<=5*x) {
    /* (2,0), (3,1), (5,2), (10,4), (11,4) */
    k = x + y;
  }
  else {
    k = x + y;
  }

  return;
}
