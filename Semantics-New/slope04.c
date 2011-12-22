// Goal: check constraint simplification by function
// small_positive_slope_reduce_coefficients_with_bounding_box()

#include <assert.h>

void slope04()
{
  int x, y, k;

  /* create a minimal bounding box */
  assert(0<=x && x<=5);

  /* Define a horizontal constraint that is slightly increasing in
   * rationals.  It links point (-3, 0) to point(7,3), 10y<=3x+9. It
   * should be simplified into y<=x, y<=2, and 3*y<=x+2, using
   * intermediates integer points (1,1) and (4,2) and extreme points
   * (0,0) and (5, 2).
   */
  if(-3*x+10*y-9<=0)
    k = x + y;
  return;
}
