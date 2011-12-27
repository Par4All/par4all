// Goal: check constraint simplification by function
// small_positive_slope_reduce_coefficients_with_bounding_box()

// First part of slope07.c

#include <assert.h>

void slope08()
{
  int x, y, k;

  /* create a bounding box */
  // assert(0<=x && x<=10 && 0<=y && 0<=10);
  assert(0<=x && x<=58);

  if(61*y<=55*x+54) {
    /* x=9, y=9 is reachable */
    /* Three new constraints are derived from points (0,0), (9,9),
       (49,45) and (58,53) with slopes 1, 9/10 and 8/9 */
    k = x + y;
  }
  else {
    /* Two new constraints are derived from points (1,0), (50,46) and
       (58,54) with slopes 9/10 and 1 */
    k = x + y;
  }

  return;
}
