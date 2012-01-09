// Goal: check constraint simplification by function
// small_positive_slope_reduce_coefficients_with_bounding_box()

// Bug found in linked_regions02

#include <assert.h>

void slope11()
{
  int x, y, k;

  assert(-198<=x && x<=99);

  if(-1551*x+76799*y-7525551<=0) {
    /* Apparently, there are *three* intermediate points between
       (-198,93) and (99,99): (-197,94), (-148,95) and (50,99) */
    k = x + y;
  }
  else {
    k = x + y;
  }

  return;
}
