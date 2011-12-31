// Goal: check constraint simplification by function
// small_positive_slope_reduce_coefficients_with_bounding_box()

#include <assert.h>

void slope02()
{
  int x, y, k;

  /* create a bounding box */
  //assert(0<=x && x<=10 && 0<=y && 0<=10);
  assert(0<=x && x<=10);

  /* Define a horizontal constraint that is slightly increasing in
   * rationals.  It links point (-1, 5) to point(11,7), 6y=x+31. It
   * should be simplified into y<=6 and 5y<x+25.
   *
   * Due to division, the left point must be moved to (-2,5) for dx=13
   */
  //if(6*y<=x+31)
  if(13*y<=2*x+69)
    k = x + y;
  return;
}
