// Goal: check constraint simplification by function
// small_positive_slope_reduce_coefficients_with_bounding_box()

#include <assert.h>

void slope01()
{
  int x, y, k;

  /* create a bounding box */
  // assert(0<=x && x<=10 && 0<=y && 0<=10);
  assert(0<=x && x<=10);

  /* Define a horizontal constraint that is slightly increasing in
   * rationals.  It links point (-1, 5) to point(11,6), 12y=x+61. It
   * should be simplified into y<=5.
   *
   * Problem for testing: the redundancy elimination kill one of the
   * bounding box inequalities... The analysis was wrong: you do not
   * need a bounding box but simply bounds for the x dimension if the
   * y coefficient is large and vice versa.
   */
  if(-x+12*y-61<=0) 
    k = x + y;
  return;
}
