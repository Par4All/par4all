// Goal: check constraint simplification by function
// small_positive_slope_reduce_coefficients_with_bounding_box()

// Bug found in linked_regions02: exploration of the slope11 case on a
// reduced interval

#include <assert.h>

void slope12()
{
  int x, y, k;

  assert(-197<=x && x<=50);

  if(-1551*x+76799*y-7525551<=0) {
    k = x + y;
  }
  else {
    k = x + y;
  }

  return;
}
