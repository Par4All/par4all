// Goal: check constraint simplification by function
// small_positive_slope_reduce_coefficients_with_bounding_box()

#include <assert.h>

void slope03()
{
  int x, y, k;

  /* create a minimal bounding box */
  // assert(0<=x && x<=10 && 0<=y && 0<=10);
  assert(0<=x && x<=10);

  /* Define a horizontal constraint that is slightly increasing in
   * rationals.  It links point (-1, 5) to point(21,7), 21y=2x+107. It
   * should be simplified into 10y=x+50.
   */
  if(12*y<=x+61) 
    k = x + y;
  return;
}
