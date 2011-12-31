// Goal: check constraint simplification by function
// small_positive_slope_reduce_coefficients_with_bounding_box()

// Extension of slope01.c: new constraints are derived by changing the
// coefficient signs; too much of an extension for debugging! so
// slope06.c was derived from slope05.c

#include <assert.h>

void slope05()
{
  int x, y, k;

  /* create a bounding box */
  // assert(0<=x && x<=10 && 0<=y && 0<=10);
  assert(0<=x && x<=10);

  /* Define a horizontal constraint that is slightly increasing in
   * rationals.  It links point (-1, 5) to point(11,6), 12y=x+61. It
   * should be simplified into y<=5.
   */
  if(-x+12*y-61<=0) 
    k = x + y;
  else
    k = x + y;

  if(-x-12*y-61<=0) 
    k = x + y;
  else
    k = x + y;

  if(x-12*y-61<=0) 
    k = x + y;
  else
    k = x + y;

  if(x+12*y-61<=0) 
    k = x + y;
  else
    k = x + y;

  return;
}
