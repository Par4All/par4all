// Goal: check constraint simplification by function
// small_positive_slope_reduce_coefficients_with_bounding_box()

// Extension of slope01.c: one new constraint is derived by changing the
// coefficient signs

#include <assert.h>

void slope06()
{
  int x, y, k;

  /* create a bounding box */
  // assert(0<=x && x<=10 && 0<=y && 0<=10);
  assert(0<=x && x<=10);

  if(-x-12*y-61<=0) {
    /* x==0 -> y>=-5; x==10 -> 12y>=-71 -> y>=-5 */
    k = x + y;
  }
  else {
    /* -x-12y-61>=1, x+12y+62<=0*/
    /* x==0 -> y<=-6; x==10 -> 12y<=72 -> y<=-6 */
    k = x + y;
  }

  return;
}
